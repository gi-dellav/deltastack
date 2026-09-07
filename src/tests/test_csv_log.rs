use crate::csv_log;
use crate::state::{IterationRecord, RecordKind};

fn cand() -> IterationRecord {
    IterationRecord {
        iteration: 1,
        agent_idx: 0,
        kind: RecordKind::Candidate,
        commit: Some("abc".into()),
        worktree: None,
        samples: vec![Some(1.5), None],
        exits: vec![Some(0), Some(3)],
        failed: vec![false, true],
        agg: Some(1.5),
        best: Some(1.5),
        best_sha: Some("abc".into()),
        decision: Some("keep".into()),
        agent_exit: Some(0),
        note: None,
    }
}

#[test]
fn header_has_sample_triples() {
    let h = csv_log::header(2);
    assert_eq!(
        h,
        vec![
            "iteration",
            "agent_idx",
            "kind",
            "commit",
            "worktree",
            "sample_0",
            "exit_0",
            "failed_0",
            "sample_1",
            "exit_1",
            "failed_1",
            "agg",
            "best",
            "best_sha",
            "decision",
            "agent_exit",
            "note",
        ]
    );
}

#[test]
fn row_encodes_failed_flags() {
    let r = csv_log::row(&cand(), 2);
    // sample_0 ok, sample_1 failed (non-zero exit, no score)
    assert_eq!(&r[5..11], ["1.5", "0", "0", "", "3", "1"]);
    assert_eq!(r[0], "1");
    assert_eq!(r[2], "candidate");
    assert_eq!(r[3], "abc");
    assert_eq!(r[11], "1.5"); // agg
    assert_eq!(r[14], "keep");
}

#[test]
fn row_pads_short_samples() {
    // Agent panic yields fewer samples than --samples; pad with empties.
    let r = csv_log::row(&cand(), 3);
    assert_eq!(&r[11..14], ["", "", ""]);
    assert_eq!(r.len(), csv_log::header(3).len());
}

#[test]
fn row_truncates_extra_samples() {
    let r = csv_log::row(&cand(), 1);
    assert_eq!(&r[5..8], ["1.5", "0", "0"]);
    assert_eq!(r.len(), csv_log::header(1).len());
}

#[tokio::test]
async fn append_writes_header_once_then_rows() {
    let dir = tempfile::tempdir().unwrap();
    let p = dir.path().join("sub").join("run.csv");
    let ps = p.to_str().unwrap();
    csv_log::append_csv_record(ps, &cand(), 2).await.unwrap();
    csv_log::append_csv_record(ps, &cand(), 2).await.unwrap();
    let content = std::fs::read_to_string(&p).unwrap();
    let lines: Vec<&str> = content.lines().collect();
    assert_eq!(lines.len(), 3, "{content}");
    assert!(lines[0].starts_with("iteration,agent_idx,kind"), "{content}");
    assert!(lines[1].contains(",candidate,abc,"), "{content}");
}

#[tokio::test]
async fn append_quotes_fields_with_commas() {
    let dir = tempfile::tempdir().unwrap();
    let p = dir.path().join("run.csv");
    let ps = p.to_str().unwrap();
    let mut r = cand();
    r.note = Some("hello, world".into());
    csv_log::append_csv_record(ps, &r, 2).await.unwrap();
    let content = std::fs::read_to_string(&p).unwrap();
    assert!(content.contains("\"hello, world\""), "{content}");
    // And it round-trips through a real CSV parser.
    let mut rdr = csv::Reader::from_reader(content.as_bytes());
    let headers: Vec<String> = rdr
        .headers()
        .unwrap()
        .iter()
        .map(str::to_string)
        .collect();
    assert_eq!(headers, csv_log::header(2));
    let rows: Vec<Vec<String>> = rdr
        .records()
        .map(|r| r.unwrap().iter().map(str::to_string).collect())
        .collect();
    assert_eq!(rows.len(), 1);
    assert_eq!(rows[0], csv_log::row(&r, 2));
}
