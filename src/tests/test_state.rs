use crate::state::*;

#[test]
fn json_roundtrip() {
    let r = IterationRecord::candidate(
        2,
        1,
        Some("abc".into()),
        Some("/wt".into()),
        vec![Some(1.0), None],
        vec![Some(0), Some(1)],
        vec![false, true],
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
    let r = IterationRecord::candidate(
        0,
        0,
        None,
        None,
        vec![],
        vec![],
        vec![],
        None,
        None,
        None,
        "keep",
        None,
    );
    let line = r.to_json_line().unwrap();
    assert!(!line.contains('\n'));
    assert!(line.contains("\"candidate\""));
}

#[test]
fn invalid_json_errors() {
    assert!(IterationRecord::from_json_line("not json").is_err());
}

#[test]
fn old_json_without_diagnostics_still_reads() {
    // Back-compat: records written before exits/failed existed.
    let line = r#"{"iteration":1,"agent_idx":0,"kind":"candidate","commit":null,"worktree":null,"samples":[1.0],"agg":1.0,"best":1.0,"best_sha":null,"decision":"keep","agent_exit":0,"note":null}"#;
    let back = IterationRecord::from_json_line(line).unwrap();
    assert!(back.exits.is_empty());
    assert!(back.failed.is_empty());
}

#[test]
fn sample_columns_splits_score_exit_failed() {
    use crate::eval::EvalSample;
    let mk = |score: Option<f64>, exit_code: Option<i32>, timed_out: bool| EvalSample {
        score,
        stdout: String::new(),
        stderr: String::new(),
        exit_code,
        timed_out,
    };
    let samples = vec![
        mk(Some(1.0), Some(0), false),
        mk(None, Some(0), false), // parse failure, exit 0 -> failed
        mk(None, Some(2), false), // non-zero exit -> failed
        mk(None, None, true),     // timeout -> failed
    ];
    let (scores, exits, failed) = IterationRecord::sample_columns(&samples);
    assert_eq!(scores, vec![Some(1.0), None, None, None]);
    assert_eq!(exits, vec![Some(0), Some(0), Some(2), None]);
    assert_eq!(failed, vec![false, true, true, true]);
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
        vec![Some(0)],
        vec![false],
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
        vec![Some(0)],
        vec![false],
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
