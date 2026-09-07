use serde::{Deserialize, Serialize};

/// One JSONL row per evaluated candidate + summary rows.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct IterationRecord {
    pub iteration: u32,
    pub agent_idx: u32,
    pub kind: RecordKind,
    pub commit: Option<String>,
    pub worktree: Option<String>,
    pub samples: Vec<Option<f64>>,
    /// Per-sample eval process exit codes (None = timeout/spawn failure).
    /// `#[serde(default)]` keeps old JSONL files readable.
    #[serde(default)]
    pub exits: Vec<Option<i32>>,
    /// Per-sample failure flags: true on non-zero exit, timeout, or
    /// unparsable/missing score. See `crate::eval::sample_failed`.
    #[serde(default)]
    pub failed: Vec<bool>,
    pub agg: Option<f64>,
    pub best: Option<f64>,
    pub best_sha: Option<String>,
    pub decision: Option<String>,
    pub agent_exit: Option<i32>,
    pub note: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum RecordKind {
    Baseline,
    Candidate,
    Summary,
}

impl IterationRecord {
    #[allow(clippy::too_many_arguments)]
    pub fn candidate(
        iteration: u32,
        agent_idx: u32,
        commit: Option<String>,
        worktree: Option<String>,
        samples: Vec<Option<f64>>,
        exits: Vec<Option<i32>>,
        failed: Vec<bool>,
        agg: Option<f64>,
        best: Option<f64>,
        best_sha: Option<String>,
        decision: &str,
        agent_exit: Option<i32>,
    ) -> Self {
        Self {
            iteration,
            agent_idx,
            kind: RecordKind::Candidate,
            commit,
            worktree,
            samples,
            exits,
            failed,
            agg,
            best,
            best_sha,
            decision: Some(decision.to_string()),
            agent_exit,
            note: None,
        }
    }

    /// Build the per-sample vectors from raw eval samples in one place so
    /// JSONL and CSV stay consistent.
    pub fn sample_columns(samples: &[crate::eval::EvalSample]) -> (
        Vec<Option<f64>>,
        Vec<Option<i32>>,
        Vec<bool>,
    ) {
        (
            samples.iter().map(|s| s.score).collect(),
            samples.iter().map(|s| s.exit_code).collect(),
            samples.iter().map(crate::eval::sample_failed).collect(),
        )
    }

    pub fn to_json_line(&self) -> anyhow::Result<String> {
        Ok(serde_json::to_string(self)?)
    }

    #[allow(dead_code)] // public API for future `resume`; exercised in tests
    pub fn from_json_line(line: &str) -> anyhow::Result<Self> {
        Ok(serde_json::from_str(line)?)
    }
}

pub async fn append_record(path: &str, rec: &IterationRecord) -> anyhow::Result<()> {
    let line = rec.to_json_line()?;
    let mut content = String::new();
    // Best-effort read existing; create parent dirs.
    if let Some(parent) = std::path::Path::new(path).parent() {
        if !parent.as_os_str().is_empty() {
            tokio::fs::create_dir_all(parent).await?;
        }
    }
    if tokio::fs::try_exists(path).await.unwrap_or(false) {
        content = tokio::fs::read_to_string(path).await.unwrap_or_default();
    }
    content.push_str(&line);
    content.push('\n');
    tokio::fs::write(path, content).await?;
    Ok(())
}
