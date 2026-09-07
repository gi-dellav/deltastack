//! Save/resume support: reload a prior run's JSONL state and continue.
//!
//! Policy (as agreed):
//! - Explicit `--resume` flag; fresh runs refuse to append to a non-empty
//!   state file.
//! - The trailing partial iteration (missing rows or missing `decision`) is
//!   discarded and re-run; it is left in the file but ignored.
//! - Config drift warns and continues (via a `Summary` header row).
//! - On resume the checkout is reset to the saved `best_sha`.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

use crate::cli::Cli;
use crate::state::{IterationRecord, RecordKind};

/// Where a resumed run continues from.
#[derive(Debug, Clone, PartialEq)]
pub struct ResumedState {
    /// First iteration to execute (last complete + 1, or 1 if only baseline).
    pub next_iteration: u32,
    /// Global best score from the last complete iteration (or baseline).
    pub best: Option<f64>,
    /// Commit the checkout must be reset to. Always `Some` on success.
    pub best_sha: String,
    /// Winner score of the last complete iteration (baseline agg if none).
    pub last_score: Option<f64>,
    /// Consecutive non-improving iterations (recomputed with current thresholds).
    pub fails_since_best: u32,
    /// Consecutive trailing iterations whose winner hit `--target` (0 if none).
    pub target_streak: u32,
}

/// Load and parse every JSONL row in `path`.
pub async fn load_records(path: &str) -> anyhow::Result<Vec<IterationRecord>> {
    let content = tokio::fs::read_to_string(path)
        .await
        .map_err(|e| anyhow::anyhow!("cannot read state file {path:?}: {e:#}"))?;
    parse_records_content(&content)
}

/// Parse JSONL content; blank lines are skipped, malformed lines are errors.
pub fn parse_records_content(content: &str) -> anyhow::Result<Vec<IterationRecord>> {
    let mut out = Vec::new();
    for (idx, line) in content.lines().enumerate() {
        let t = line.trim();
        if t.is_empty() {
            continue;
        }
        let rec: IterationRecord = serde_json::from_str(t).map_err(|e| {
            anyhow::anyhow!("state file line {} is not a valid record: {e:#}", idx + 1)
        })?;
        out.push(rec);
    }
    Ok(out)
}

/// True if `path` exists and contains at least one non-blank line.
pub async fn state_file_has_content(path: &str) -> bool {
    let Ok(content) = tokio::fs::read_to_string(path).await else {
        return false;
    };
    content.lines().any(|l| !l.trim().is_empty())
}

// ---------------------------------------------------------------------------
// Run-config header (warn-and-continue on drift)
// ---------------------------------------------------------------------------

const CONFIG_NOTE_PREFIX: &str = "deltastack-config-v1 ";

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
struct RunConfig {
    eval: String,
    mode: String,
    aggregate: String,
    agents: u32,
    samples: u32,
    min_improvement: f64,
    min_improvement_rel: f64,
}

fn eval_descriptor(cli: &Cli) -> String {
    match cli.eval_kind() {
        Some(crate::cli::EvalKind::Custom(c)) => format!("eval:{c}"),
        Some(crate::cli::EvalKind::Speed(c)) => format!("speed:{c}"),
        Some(crate::cli::EvalKind::Memory(c)) => format!("memory:{c}"),
        None => "none".to_string(),
    }
}

fn config_snapshot(cli: &Cli) -> RunConfig {
    RunConfig {
        eval: eval_descriptor(cli),
        mode: format!("{:?}", cli.mode),
        aggregate: format!("{:?}", cli.aggregate),
        agents: cli.agents,
        samples: cli.samples,
        min_improvement: cli.min_improvement,
        min_improvement_rel: cli.min_improvement_rel,
    }
}

/// Build the header row written once on fresh runs (never re-written on resume).
pub fn make_config_record(cli: &Cli) -> IterationRecord {
    let note = format!(
        "{CONFIG_NOTE_PREFIX}{}",
        serde_json::to_string(&config_snapshot(cli)).unwrap_or_else(|_| "{}".to_string())
    );
    IterationRecord {
        iteration: 0,
        agent_idx: 0,
        kind: RecordKind::Summary,
        commit: None,
        worktree: None,
        samples: vec![],
        exits: vec![],
        failed: vec![],
        agg: None,
        best: None,
        best_sha: None,
        decision: None,
        agent_exit: None,
        note: Some(note),
    }
}

/// Latest config header in the file, if any (old files have none).
fn find_config_snapshot(records: &[IterationRecord]) -> Option<RunConfig> {
    records
        .iter()
        .rev()
        .filter(|r| r.kind == RecordKind::Summary)
        .filter_map(|r| r.note.as_deref())
        .filter_map(|n| n.strip_prefix(CONFIG_NOTE_PREFIX))
        .filter_map(|j| serde_json::from_str(j).ok())
        .next()
}

/// Human-readable drift warnings; empty means compatible (or no header).
pub fn config_warnings(records: &[IterationRecord], cli: &Cli) -> Vec<String> {
    let Some(saved) = find_config_snapshot(records) else {
        return vec![
            "state file has no config header (pre-resume version); assuming compatible config"
                .to_string(),
        ];
    };
    let cur = config_snapshot(cli);
    let mut w = Vec::new();
    if saved.eval != cur.eval {
        w.push(format!(
            "eval command differs (was {:?}, now {:?}); scores may not be comparable",
            saved.eval, cur.eval
        ));
    }
    if saved.mode != cur.mode {
        w.push(format!(
            "mode differs (was {}, now {}); best-score semantics changed",
            saved.mode, cur.mode
        ));
    }
    if saved.aggregate != cur.aggregate {
        w.push(format!(
            "aggregate differs (was {}, now {})",
            saved.aggregate, cur.aggregate
        ));
    }
    if saved.agents != cur.agents {
        w.push(format!(
            "agents differs (was {}, now {}); the trailing iteration will be re-run with {} agents",
            saved.agents, cur.agents, cur.agents
        ));
    }
    if saved.samples != cur.samples {
        w.push(format!(
            "samples differs (was {}, now {})",
            saved.samples, cur.samples
        ));
    }
    if saved.min_improvement != cur.min_improvement
        || saved.min_improvement_rel != cur.min_improvement_rel
    {
        w.push(format!(
            "improvement thresholds differ (was {}/{}, now {}/{}); keep/revert boundary changed",
            saved.min_improvement,
            saved.min_improvement_rel,
            cur.min_improvement,
            cur.min_improvement_rel
        ));
    }
    w
}

// ---------------------------------------------------------------------------
// Resume derivation
// ---------------------------------------------------------------------------

/// Derive where to continue from. Partial trailing iterations are discarded
/// (left in the file, ignored in memory). Non-trailing incomplete iterations
/// are treated as corruption and rejected.
pub fn derive_resume(records: &[IterationRecord], cli: &Cli) -> anyhow::Result<ResumedState> {
    let baseline = records
        .iter()
        .find(|r| r.kind == RecordKind::Baseline && r.iteration == 0);

    let mut by_iter: BTreeMap<u32, Vec<&IterationRecord>> = BTreeMap::new();
    for r in records {
        if r.kind == RecordKind::Candidate && r.iteration >= 1 {
            by_iter.entry(r.iteration).or_default().push(r);
        }
    }

    if baseline.is_none() && by_iter.is_empty() {
        anyhow::bail!("state file has no baseline or candidate records; nothing to resume");
    }

    // Split complete vs discarded-partial tail.
    let mut complete: Vec<u32> = Vec::new();
    let mut discarded_partial: Option<u32> = None;
    if let Some(&max_iter) = by_iter.keys().next_back() {
        for (&iter, rows) in &by_iter {
            let all_decided = !rows.is_empty() && rows.iter().all(|r| r.decision.is_some());
            if iter < max_iter {
                if !all_decided {
                    anyhow::bail!(
                        "iteration {iter} is incomplete (missing rows/decision); state file may be corrupt"
                    );
                }
                complete.push(iter);
            } else {
                // Tail: must match the current agent count to be reusable.
                let expected = cli.agents as usize;
                if all_decided && rows.len() == expected {
                    complete.push(iter);
                } else {
                    discarded_partial = Some(iter);
                }
            }
        }
    }
    if let Some(p) = discarded_partial {
        tracing::info!("discarding partial iteration {p} (will be re-run from scratch)");
    }

    // Walk complete history to recompute fails_since_best with current thresholds.
    let mut cur_best: Option<f64> = baseline.and_then(|b| b.best);
    let mut fails_since_best: u32 = 0;
    for &iter in &complete {
        let rows = &by_iter[&iter];
        let winner_agg = winner_agg_for_rows(rows, cli);
        match (winner_agg, cur_best) {
            (Some(s), None) => {
                cur_best = Some(s);
                fails_since_best = 0;
            }
            (Some(s), Some(b)) => {
                if crate::orchestrator::is_improvement_full(
                    s,
                    b,
                    cli.mode,
                    cli.min_improvement,
                    cli.min_improvement_rel,
                ) {
                    cur_best = Some(s);
                    fails_since_best = 0;
                } else {
                    fails_since_best += 1;
                }
            }
            (None, _) => fails_since_best += 1,
        }
    }

    // Trailing target streak (for sticky-target early stop).
    let mut target_streak: u32 = 0;
    if let Some(t) = cli.target {
        for &iter in complete.iter().rev() {
            let rows = &by_iter[&iter];
            match winner_agg_for_rows(rows, cli) {
                Some(s) if crate::orchestrator::reached_target(s, t, cli.mode) => {
                    target_streak += 1;
                }
                _ => break,
            }
        }
    }

    if complete.is_empty() {
        let b = baseline.ok_or_else(|| {
            anyhow::anyhow!("no complete iterations and no baseline; nothing to resume")
        })?;
        let sha = b
            .best_sha
            .clone()
            .ok_or_else(|| anyhow::anyhow!("baseline record is missing best_sha; cannot resume"))?;
        return Ok(ResumedState {
            next_iteration: 1,
            best: b.best,
            best_sha: sha,
            last_score: b.agg.or(b.best),
            fails_since_best: 0,
            target_streak: 0,
        });
    }

    let last = *complete.last().expect("non-empty");
    let rows = &by_iter[&last];
    let first = rows[0];
    let best = first.best;
    let best_sha = first
        .best_sha
        .clone()
        .ok_or_else(|| anyhow::anyhow!("iteration {last} is missing best_sha; cannot resume"))?;
    let last_score = winner_agg_for_rows(rows, cli);

    Ok(ResumedState {
        next_iteration: last + 1,
        best,
        best_sha,
        last_score,
        fails_since_best,
        target_streak,
    })
}

/// Winner's `agg` for one iteration's rows (mode-aware; failures sort last).
fn winner_agg_for_rows(rows: &[&IterationRecord], cli: &Cli) -> Option<f64> {
    let cands: Vec<crate::orchestrator::CandidateScore> = rows
        .iter()
        .map(|r| crate::orchestrator::CandidateScore {
            agent_idx: r.agent_idx,
            score: r.agg,
        })
        .collect();
    let w = crate::orchestrator::select_winner(&cands, cli.mode)?;
    rows.iter().find(|r| r.agent_idx == w)?.agg
}
