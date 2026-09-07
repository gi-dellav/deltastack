use crate::cli::Cli;
use crate::resume::*;
use crate::state::{IterationRecord, RecordKind};
use clap::Parser;

fn cli_for(extra: &[&str]) -> Cli {
    let mut base = vec!["deltastack", "--prompt", "x", "--eval", "echo 1"];
    base.extend_from_slice(extra);
    Cli::parse_from(base)
}

fn baseline(best: Option<f64>, sha: &str) -> IterationRecord {
    IterationRecord {
        iteration: 0,
        agent_idx: 0,
        kind: RecordKind::Baseline,
        commit: Some(sha.into()),
        worktree: None,
        samples: vec![best],
        exits: vec![Some(0)],
        failed: vec![best.is_none()],
        agg: best,
        best,
        best_sha: Some(sha.into()),
        decision: Some("baseline".into()),
        agent_exit: None,
        note: None,
    }
}

fn cand(
    iter: u32,
    idx: u32,
    agg: Option<f64>,
    best: Option<f64>,
    sha: &str,
    decision: Option<&str>,
) -> IterationRecord {
    IterationRecord {
        iteration: iter,
        agent_idx: idx,
        kind: RecordKind::Candidate,
        commit: Some(format!("c{iter}-{idx}")),
        worktree: None,
        samples: vec![agg],
        exits: vec![Some(0)],
        failed: vec![agg.is_none()],
        agg,
        best,
        best_sha: Some(sha.into()),
        decision: decision.map(str::to_string),
        agent_exit: Some(0),
        note: None,
    }
}

fn keep(iter: u32, idx: u32, agg: f64, best: f64, sha: &str) -> IterationRecord {
    cand(iter, idx, Some(agg), Some(best), sha, Some("keep"))
}

fn revert(iter: u32, idx: u32, agg: f64, best: f64, sha: &str) -> IterationRecord {
    cand(iter, idx, Some(agg), Some(best), sha, Some("revert"))
}

#[test]
fn resume_flag_defaults_off_and_parses() {
    let c = cli_for(&[]);
    assert!(!c.resume);
    let c = cli_for(&["--resume"]);
    assert!(c.resume);
    assert!(c.validate().is_ok());
}

#[test]
fn baseline_only_resumes_at_iteration_1() {
    let cli = cli_for(&[]);
    let recs = vec![baseline(Some(1.0), "aaa")];
    let r = derive_resume(&recs, &cli).unwrap();
    assert_eq!(r.next_iteration, 1);
    assert_eq!(r.best, Some(1.0));
    assert_eq!(r.best_sha, "aaa");
    assert_eq!(r.last_score, Some(1.0));
    assert_eq!(r.fails_since_best, 0);
}

#[test]
fn complete_iterations_advance_next_and_track_best() {
    let cli = cli_for(&[]);
    let recs = vec![
        baseline(Some(1.0), "aaa"),
        keep(1, 0, 0.9, 0.9, "bbb"),
        keep(2, 0, 0.8, 0.8, "ccc"),
    ];
    let r = derive_resume(&recs, &cli).unwrap();
    assert_eq!(r.next_iteration, 3);
    assert_eq!(r.best, Some(0.8));
    assert_eq!(r.best_sha, "ccc");
    assert_eq!(r.last_score, Some(0.8));
    assert_eq!(r.fails_since_best, 0);
}

#[test]
fn non_improving_tail_counts_fails() {
    let cli = cli_for(&[]);
    let recs = vec![
        baseline(Some(1.0), "aaa"),
        keep(1, 0, 0.9, 0.9, "bbb"),
        revert(2, 0, 0.95, 0.9, "bbb"),
    ];
    let r = derive_resume(&recs, &cli).unwrap();
    assert_eq!(r.next_iteration, 3);
    assert_eq!(r.best, Some(0.9));
    assert_eq!(r.best_sha, "bbb");
    assert_eq!(r.last_score, Some(0.95));
    assert_eq!(r.fails_since_best, 1);
}

#[test]
fn partial_tail_with_missing_agent_is_discarded() {
    let cli = cli_for(&["--agents", "2"]);
    let recs = vec![
        baseline(Some(1.0), "aaa"),
        keep(1, 0, 0.9, 0.9, "bbb"),
        revert(1, 1, 0.95, 0.9, "bbb"),
        // iter 2 has only one of two agents -> partial
        revert(2, 0, 0.85, 0.9, "bbb"),
    ];
    let r = derive_resume(&recs, &cli).unwrap();
    assert_eq!(r.next_iteration, 2);
    assert_eq!(r.best, Some(0.9));
    assert_eq!(r.best_sha, "bbb");
}

#[test]
fn tail_missing_decision_is_discarded() {
    let cli = cli_for(&[]);
    let recs = vec![
        baseline(Some(1.0), "aaa"),
        keep(1, 0, 0.9, 0.9, "bbb"),
        cand(2, 0, Some(0.7), Some(0.9), "bbb", None),
    ];
    let r = derive_resume(&recs, &cli).unwrap();
    assert_eq!(r.next_iteration, 2);
    assert_eq!(r.best, Some(0.9));
}

#[test]
fn incomplete_middle_iteration_is_corruption() {
    let cli = cli_for(&[]);
    let recs = vec![
        baseline(Some(1.0), "aaa"),
        cand(1, 0, Some(0.9), Some(0.9), "bbb", None),
        keep(2, 0, 0.8, 0.8, "ccc"),
    ];
    assert!(derive_resume(&recs, &cli).is_err());
}

#[test]
fn empty_state_has_nothing_to_resume() {
    let cli = cli_for(&[]);
    assert!(derive_resume(&[], &cli).is_err());
    assert!(parse_records_content("").unwrap().is_empty());
    assert!(parse_records_content("\n  \n").unwrap().is_empty());
}

#[test]
fn malformed_line_errors_with_line_number() {
    let good = keep(1, 0, 0.9, 0.9, "bbb").to_json_line().unwrap();
    let err = parse_records_content(&format!("{good}\nnot json\n")).unwrap_err();
    assert!(err.to_string().contains("line 2"), "{err:#}");
}

#[test]
fn blank_lines_are_skipped() {
    let line = keep(1, 0, 0.9, 0.9, "bbb").to_json_line().unwrap();
    let recs = parse_records_content(&format!("\n{line}\n\n")).unwrap();
    assert_eq!(recs.len(), 1);
}

#[test]
fn multi_agent_complete_iteration_advances() {
    let cli = cli_for(&["--agents", "2"]);
    let recs = vec![
        baseline(Some(1.0), "aaa"),
        // winner is agent 1 (lower is better)
        revert(1, 0, 0.95, 0.8, "ccc"),
        keep(1, 1, 0.8, 0.8, "ccc"),
    ];
    let r = derive_resume(&recs, &cli).unwrap();
    assert_eq!(r.next_iteration, 2);
    assert_eq!(r.best, Some(0.8));
    assert_eq!(r.last_score, Some(0.8));
    assert_eq!(r.fails_since_best, 0);
}

#[test]
fn first_score_wins_after_failed_baseline() {
    let cli = cli_for(&[]);
    let recs = vec![baseline(None, "aaa"), keep(1, 0, 2.5, 2.5, "bbb")];
    let r = derive_resume(&recs, &cli).unwrap();
    assert_eq!(r.best, Some(2.5));
    assert_eq!(r.best_sha, "bbb");
    assert_eq!(r.fails_since_best, 0);
}

#[test]
fn failed_iteration_increments_fails() {
    let cli = cli_for(&[]);
    let recs = vec![
        baseline(Some(1.0), "aaa"),
        cand(1, 0, None, Some(1.0), "aaa", Some("revert")),
    ];
    let r = derive_resume(&recs, &cli).unwrap();
    assert_eq!(r.next_iteration, 2);
    assert_eq!(r.fails_since_best, 1);
    assert_eq!(r.last_score, None);
}

#[test]
fn target_streak_counts_trailing_hits() {
    let cli = cli_for(&["--target", "0.85"]);
    let recs = vec![
        baseline(Some(1.0), "aaa"),
        revert(1, 0, 0.9, 1.0, "aaa"),
        keep(2, 0, 0.8, 0.8, "bbb"),
        keep(3, 0, 0.7, 0.7, "ccc"),
    ];
    let r = derive_resume(&recs, &cli).unwrap();
    assert_eq!(r.target_streak, 2);
}

#[test]
fn target_streak_zero_without_target() {
    let cli = cli_for(&[]);
    let recs = vec![baseline(Some(1.0), "aaa"), keep(1, 0, 0.5, 0.5, "bbb")];
    let r = derive_resume(&recs, &cli).unwrap();
    assert_eq!(r.target_streak, 0);
}

#[test]
fn config_header_roundtrips_and_matches() {
    let cli = cli_for(&[]);
    let header = make_config_record(&cli);
    assert_eq!(header.kind, RecordKind::Summary);
    let recs = vec![header, baseline(Some(1.0), "aaa")];
    assert!(config_warnings(&recs, &cli).is_empty());
}

#[test]
fn config_drift_warns_on_mode_and_agents() {
    let cli = cli_for(&[]);
    let header = make_config_record(&cli);
    let recs = vec![header];
    let changed = cli_for(&["--agents", "2", "--mode", "maximize"]);
    let warnings = config_warnings(&recs, &changed);
    assert!(
        warnings.iter().any(|w| w.contains("agents")),
        "{warnings:?}"
    );
    assert!(warnings.iter().any(|w| w.contains("mode")), "{warnings:?}");
}

#[test]
fn missing_header_warns_once() {
    let cli = cli_for(&[]);
    let recs = vec![baseline(Some(1.0), "aaa")];
    let warnings = config_warnings(&recs, &cli);
    assert_eq!(warnings.len(), 1);
    assert!(warnings[0].contains("no config header"));
}

#[tokio::test]
async fn state_file_content_detection() {
    let dir = tempfile::tempdir().unwrap();
    let p = dir.path().join("run.jsonl");
    let ps = p.to_str().unwrap();
    assert!(!state_file_has_content(ps).await);
    std::fs::write(&p, "\n  \n").unwrap();
    assert!(!state_file_has_content(ps).await);
    std::fs::write(&p, "{}\n").unwrap();
    assert!(state_file_has_content(ps).await);
}

#[tokio::test]
async fn load_records_reads_jsonl_file() {
    let dir = tempfile::tempdir().unwrap();
    let p = dir.path().join("run.jsonl");
    let ps = p.to_str().unwrap();
    let r = keep(1, 0, 0.9, 0.9, "bbb");
    std::fs::write(&p, format!("{}\n", r.to_json_line().unwrap())).unwrap();
    let recs = load_records(ps).await.unwrap();
    assert_eq!(recs.len(), 1);
    assert_eq!(recs[0].iteration, 1);
}
