//! Opt-in CSV mirror of the JSONL run log.
//!
//! Wide format, one row per `IterationRecord` (baseline/candidate/summary).
//! Per-sample detail is a repeated triple per sample index `i`:
//! `sample_i` (float or empty), `exit_i` (int or empty), `failed_i` (`0`/`1`).
//! `failed` is true on non-zero exit, timeout, or unparsable/missing score
//! (see `crate::eval::sample_failed`). Live writes only, no backfill.

use crate::state::{IterationRecord, RecordKind};

fn kind_str(kind: &RecordKind) -> &'static str {
    match kind {
        RecordKind::Baseline => "baseline",
        RecordKind::Candidate => "candidate",
        RecordKind::Summary => "summary",
    }
}

fn opt_f64(v: Option<f64>) -> String {
    v.map(|x| x.to_string()).unwrap_or_default()
}

fn opt_i32(v: Option<i32>) -> String {
    v.map(|x| x.to_string()).unwrap_or_default()
}

/// Header for `samples_n` samples. Stable for a run: `samples_n` comes from
/// `--samples`, so short rows (agent panics) are padded with empties.
pub fn header(samples_n: usize) -> Vec<String> {
    let mut h = vec![
        "iteration".to_string(),
        "agent_idx".to_string(),
        "kind".to_string(),
        "commit".to_string(),
        "worktree".to_string(),
    ];
    for i in 0..samples_n {
        h.push(format!("sample_{i}"));
        h.push(format!("exit_{i}"));
        h.push(format!("failed_{i}"));
    }
    h.extend(
        [
            "agg",
            "best",
            "best_sha",
            "decision",
            "agent_exit",
            "note",
        ]
        .iter()
        .map(|s| s.to_string()),
    );
    h
}

/// One CSV row for `rec`, padded/truncated to `samples_n` sample triples.
pub fn row(rec: &IterationRecord, samples_n: usize) -> Vec<String> {
    let mut r = vec![
        rec.iteration.to_string(),
        rec.agent_idx.to_string(),
        kind_str(&rec.kind).to_string(),
        rec.commit.clone().unwrap_or_default(),
        rec.worktree.clone().unwrap_or_default(),
    ];
    for i in 0..samples_n {
        r.push(rec.samples.get(i).copied().flatten().map_or_else(String::new, |x| x.to_string()));
        r.push(rec.exits.get(i).copied().flatten().map_or_else(String::new, |x| x.to_string()));
        r.push(match rec.failed.get(i) {
            Some(true) => "1".to_string(),
            Some(false) => "0".to_string(),
            None => String::new(),
        });
    }
    r.push(opt_f64(rec.agg));
    r.push(opt_f64(rec.best));
    r.push(rec.best_sha.clone().unwrap_or_default());
    r.push(rec.decision.clone().unwrap_or_default());
    r.push(opt_i32(rec.agent_exit));
    r.push(rec.note.clone().unwrap_or_default());
    r
}

/// Append one record to the CSV log, writing the header when the file is
/// new or empty. Creates parent dirs like `state::append_record`.
pub async fn append_csv_record(
    path: &str,
    rec: &IterationRecord,
    samples_n: usize,
) -> anyhow::Result<()> {
    if let Some(parent) = std::path::Path::new(path).parent() {
        if !parent.as_os_str().is_empty() {
            tokio::fs::create_dir_all(parent).await?;
        }
    }
    let needs_header = match tokio::fs::metadata(path).await {
        Err(_) => true,
        Ok(m) => m.len() == 0,
    };
    // Same read-modify-write pattern as state::append_record: all CSV writes
    // happen sequentially from the main loop, never concurrently.
    let buf: Vec<u8> = {
        let mut wtr = csv::WriterBuilder::new()
            .has_headers(false)
            .from_writer(vec![]);
        if needs_header {
            wtr.write_record(header(samples_n))?;
        }
        wtr.write_record(row(rec, samples_n))?;
        wtr.into_inner()?
    };
    if needs_header {
        tokio::fs::write(path, buf).await?;
    } else {
        let mut existing = tokio::fs::read(path).await.unwrap_or_default();
        existing.extend_from_slice(&buf);
        tokio::fs::write(path, existing).await?;
    }
    Ok(())
}
