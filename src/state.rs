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
            agg,
            best,
            best_sha,
            decision: Some(decision.to_string()),
            agent_exit,
            note: None,
        }
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn json_roundtrip() {
        let r = IterationRecord::candidate(
            2,
            1,
            Some("abc".into()),
            Some("/wt".into()),
            vec![Some(1.0), None],
            Some(1.0),
            Some(0.9),
            Some("best".into()),
            "revert",
            Some(0),
        );
        let line = r.to_json_line().unwrap();
        let back = IterationRecord::from_json_line(&line).unwrap();
        assert_eq!(r, back);
    }

    #[test]
    fn json_line_is_single_line() {
        let r =
            IterationRecord::candidate(0, 0, None, None, vec![], None, None, None, "keep", None);
        let line = r.to_json_line().unwrap();
        assert!(!line.contains('\n'));
        assert!(line.contains("\"candidate\""));
    }

    #[test]
    fn invalid_json_errors() {
        assert!(IterationRecord::from_json_line("not json").is_err());
    }

    #[tokio::test]
    async fn append_creates_file_and_appends() {
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path().join("sub").join("run.jsonl");
        let ps = p.to_str().unwrap();
        let r1 = IterationRecord::candidate(
            0,
            0,
            None,
            None,
            vec![Some(1.0)],
            Some(1.0),
            Some(1.0),
            None,
            "keep",
            Some(0),
        );
        let r2 = IterationRecord::candidate(
            1,
            0,
            None,
            None,
            vec![Some(0.9)],
            Some(0.9),
            Some(0.9),
            None,
            "keep",
            Some(0),
        );
        append_record(ps, &r1).await.unwrap();
        append_record(ps, &r2).await.unwrap();
        let content = std::fs::read_to_string(&p).unwrap();
        assert_eq!(content.lines().count(), 2);
        let back: IterationRecord = serde_json::from_str(content.lines().next().unwrap()).unwrap();
        assert_eq!(back.iteration, 0);
    }
}
