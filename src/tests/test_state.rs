use crate::state::*;

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
    let r = IterationRecord::candidate(0, 0, None, None, vec![], None, None, None, "keep", None);
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
