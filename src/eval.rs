use std::path::{Path, PathBuf};
use std::time::Duration;

use std::sync::Arc;
use tokio::process::Command;
use tokio::sync::Semaphore;

use crate::cli::Aggregate;

/// Parse an eval score from script stdout: must be a single float (surrounding whitespace ok).
pub fn parse_score(stdout: &str) -> anyhow::Result<f64> {
    let t = stdout.trim();
    if t.is_empty() {
        anyhow::bail!("eval produced empty stdout (expected single float)");
    }
    // If multiple lines, try last non-empty line (common when scripts echo progress).
    let candidate = t.lines().map(str::trim).rfind(|l| !l.is_empty()).unwrap();
    let v: f64 = candidate.parse().map_err(|_| {
        anyhow::anyhow!("eval stdout is not a float: {candidate:?} (full output: {t:?})")
    })?;
    if !v.is_finite() {
        anyhow::bail!("eval score is not finite: {v}");
    }
    Ok(v)
}

/// Aggregate multiple samples into one score.
pub fn aggregate_scores(samples: &[f64], how: Aggregate) -> Option<f64> {
    if samples.is_empty() {
        return None;
    }
    match how {
        Aggregate::First => Some(samples[0]),
        Aggregate::Min => Some(samples.iter().cloned().fold(f64::INFINITY, f64::min)),
        Aggregate::Max => Some(samples.iter().cloned().fold(f64::NEG_INFINITY, f64::max)),
        Aggregate::Mean => Some(samples.iter().sum::<f64>() / samples.len() as f64),
        Aggregate::Median => {
            let mut v = samples.to_vec();
            v.sort_by(|a, b| a.total_cmp(b));
            let n = v.len();
            if n % 2 == 1 {
                Some(v[n / 2])
            } else {
                Some((v[n / 2 - 1] + v[n / 2]) / 2.0)
            }
        }
    }
}

#[derive(Debug, Clone)]
#[allow(dead_code)] // stdout/stderr/exit_code/timed_out kept for log files + JSONL debugging
pub struct EvalSample {
    pub score: Option<f64>,
    pub stdout: String,
    pub stderr: String,
    pub exit_code: Option<i32>,
    pub timed_out: bool,
}

/// Run the eval command once in `workdir`.
pub async fn run_eval_once(
    eval_cmd: &str,
    workdir: &Path,
    timeout: Duration,
    log_file: Option<&Path>,
) -> EvalSample {
    let mut cmd = Command::new("sh");
    cmd.arg("-c").arg(eval_cmd).current_dir(workdir);
    cmd.stdout(std::process::Stdio::piped());
    cmd.stderr(std::process::Stdio::piped());

    let run = async {
        let out = cmd.output().await?;
        anyhow::Ok(out)
    };

    let timed = if timeout.is_zero() {
        run.await.map_err(|e| (e, false))
    } else {
        match tokio::time::timeout(timeout, run).await {
            Ok(r) => r.map_err(|e| (e, false)),
            Err(_) => Err((anyhow::anyhow!("eval timed out"), true)),
        }
    };

    let (stdout, stderr, exit_code, timed_out, score) = match timed {
        Err((e, to)) => (String::new(), e.to_string(), None, to, None),
        Ok(output) => {
            let stdout = String::from_utf8_lossy(&output.stdout).to_string();
            let stderr = String::from_utf8_lossy(&output.stderr).to_string();
            let code = output.status.code();
            let score = if output.status.success() {
                parse_score(&stdout).ok()
            } else {
                None
            };
            (stdout, stderr, code, false, score)
        }
    };

    if let Some(path) = log_file {
        let _ = write_eval_log(path, eval_cmd, workdir, &stdout, &stderr, exit_code, score).await;
    }

    EvalSample {
        score,
        stdout,
        stderr,
        exit_code,
        timed_out,
    }
}

async fn write_eval_log(
    path: &Path,
    eval_cmd: &str,
    workdir: &Path,
    stdout: &str,
    stderr: &str,
    exit_code: Option<i32>,
    score: Option<f64>,
) {
    if let Some(parent) = path.parent() {
        let _ = tokio::fs::create_dir_all(parent).await;
    }
    let body = format!(
        "$ cd {} && {eval_cmd}\nexit={exit_code:?} score={score:?}\n--- stdout ---\n{stdout}\n--- stderr ---\n{stderr}\n",
        workdir.display()
    );
    let _ = tokio::fs::write(path, body).await;
}

/// Run `samples` evals with bounded concurrency; returns per-sample results in order.
#[allow(clippy::too_many_arguments)]
pub async fn run_eval_samples(
    eval_cmd: &str,
    workdir: PathBuf,
    samples: u32,
    jobs: usize,
    timeout: Duration,
    log_dir: Option<PathBuf>,
    iter: u32,
    agent_idx: u32,
) -> Vec<EvalSample> {
    let jobs = jobs.max(1);
    let sem = Arc::new(Semaphore::new(jobs));
    let mut handles = Vec::new();
    for k in 0..samples {
        let permit_sem = sem.clone();
        let cmd = eval_cmd.to_string();
        let dir = workdir.clone();
        let log = log_dir
            .clone()
            .map(|d| d.join(format!("eval-{iter}-{agent_idx}-{k}.log")));
        handles.push(tokio::spawn(async move {
            let _permit = permit_sem.acquire_owned().await.unwrap();
            run_eval_once(&cmd, &dir, timeout, log.as_deref()).await
        }));
    }
    let mut out = Vec::new();
    for h in handles {
        out.push(h.await.unwrap_or(EvalSample {
            score: None,
            stdout: String::new(),
            stderr: "eval task panicked".into(),
            exit_code: None,
            timed_out: false,
        }));
    }
    out
}

/// Successful scores only, preserving order.
pub fn successful_scores(samples: &[EvalSample]) -> Vec<f64> {
    samples.iter().filter_map(|s| s.score).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    #[test]
    fn parse_plain_float() {
        assert_eq!(parse_score("0.9979\n").unwrap(), 0.9979);
    }

    #[test]
    fn parse_with_whitespace() {
        assert_eq!(parse_score("  1.5  \n").unwrap(), 1.5);
    }

    #[test]
    fn parse_negative_and_scientific() {
        assert_eq!(parse_score("-2.5").unwrap(), -2.5);
        assert_eq!(parse_score("1e-3\n").unwrap(), 0.001);
    }

    #[test]
    fn parse_empty_errors() {
        assert!(parse_score("").is_err());
        assert!(parse_score("   \n ").is_err());
    }

    #[test]
    fn parse_non_float_errors() {
        assert!(parse_score("val_bpb: 0.99").is_err());
        assert!(parse_score("hello").is_err());
    }

    #[test]
    fn parse_non_finite_errors() {
        assert!(parse_score("nan").is_err());
        assert!(parse_score("inf").is_err());
        assert!(parse_score("-inf").is_err());
    }

    #[test]
    fn parse_takes_last_line_for_noisy_scripts() {
        assert_eq!(parse_score("training...\ndone\n0.42\n").unwrap(), 0.42);
    }

    #[test]
    fn aggregate_empty_is_none() {
        assert_eq!(aggregate_scores(&[], Aggregate::Mean), None);
    }

    #[test]
    fn aggregate_first() {
        assert_eq!(
            aggregate_scores(&[3.0, 1.0, 2.0], Aggregate::First),
            Some(3.0)
        );
    }

    #[test]
    fn aggregate_min_max() {
        assert_eq!(
            aggregate_scores(&[3.0, 1.0, 2.0], Aggregate::Min),
            Some(1.0)
        );
        assert_eq!(
            aggregate_scores(&[3.0, 1.0, 2.0], Aggregate::Max),
            Some(3.0)
        );
    }

    #[test]
    fn aggregate_mean() {
        assert_eq!(
            aggregate_scores(&[1.0, 2.0, 3.0], Aggregate::Mean),
            Some(2.0)
        );
    }

    #[test]
    fn aggregate_median_odd() {
        assert_eq!(
            aggregate_scores(&[3.0, 1.0, 2.0], Aggregate::Median),
            Some(2.0)
        );
    }

    #[test]
    fn aggregate_median_even() {
        assert_eq!(
            aggregate_scores(&[4.0, 1.0, 3.0, 2.0], Aggregate::Median),
            Some(2.5)
        );
    }

    #[test]
    fn aggregate_median_single() {
        assert_eq!(aggregate_scores(&[7.0], Aggregate::Median), Some(7.0));
    }

    #[test]
    fn successful_scores_filters_failures() {
        let samples = vec![
            EvalSample {
                score: Some(1.0),
                stdout: String::new(),
                stderr: String::new(),
                exit_code: Some(0),
                timed_out: false,
            },
            EvalSample {
                score: None,
                stdout: String::new(),
                stderr: String::new(),
                exit_code: Some(1),
                timed_out: false,
            },
            EvalSample {
                score: Some(2.0),
                stdout: String::new(),
                stderr: String::new(),
                exit_code: Some(0),
                timed_out: false,
            },
        ];
        assert_eq!(successful_scores(&samples), vec![1.0, 2.0]);
    }

    #[tokio::test]
    async fn run_eval_once_echo_float() {
        let dir = tempfile::tempdir().unwrap();
        let s = run_eval_once("echo 0.5", dir.path(), Duration::from_secs(5), None).await;
        assert_eq!(s.score, Some(0.5));
        assert_eq!(s.exit_code, Some(0));
    }

    #[tokio::test]
    async fn run_eval_once_failure_has_no_score() {
        let dir = tempfile::tempdir().unwrap();
        let s = run_eval_once("exit 3", dir.path(), Duration::from_secs(5), None).await;
        assert_eq!(s.score, None);
        assert_eq!(s.exit_code, Some(3));
    }

    #[tokio::test]
    async fn run_eval_once_non_float_has_no_score_but_exit_zero() {
        let dir = tempfile::tempdir().unwrap();
        let s = run_eval_once("echo hello", dir.path(), Duration::from_secs(5), None).await;
        assert_eq!(s.score, None);
        assert_eq!(s.exit_code, Some(0));
    }

    #[tokio::test]
    async fn run_eval_once_timeout() {
        let dir = tempfile::tempdir().unwrap();
        let s = run_eval_once(
            "sleep 5; echo 1.0",
            dir.path(),
            Duration::from_millis(200),
            None,
        )
        .await;
        assert!(s.timed_out);
        assert_eq!(s.score, None);
    }

    #[tokio::test]
    async fn run_eval_samples_preserves_count_and_order() {
        let dir = tempfile::tempdir().unwrap();
        let out = run_eval_samples(
            "echo 1.0",
            dir.path().to_path_buf(),
            4,
            2,
            Duration::from_secs(5),
            None,
            0,
            0,
        )
        .await;
        assert_eq!(out.len(), 4);
        assert!(out.iter().all(|s| s.score == Some(1.0)));
    }

    #[tokio::test]
    async fn run_eval_samples_writes_logs() {
        let dir = tempfile::tempdir().unwrap();
        let logdir = tempfile::tempdir().unwrap();
        let out = run_eval_samples(
            "echo 2.0",
            dir.path().to_path_buf(),
            2,
            2,
            Duration::from_secs(5),
            Some(logdir.path().to_path_buf()),
            1,
            2,
        )
        .await;
        assert_eq!(out.len(), 2);
        assert!(logdir.path().join("eval-1-2-0.log").exists());
        assert!(logdir.path().join("eval-1-2-1.log").exists());
    }
}
