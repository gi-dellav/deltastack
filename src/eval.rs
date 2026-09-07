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

/// Which GNU time binary to use for `--optimize-memory`.
/// `GTIME_BIN` env overrides; otherwise `/usr/bin/time`, then `gtime` (macOS brew).
pub fn gnu_time_bin() -> String {
    if let Ok(b) = std::env::var("GTIME_BIN") {
        if !b.trim().is_empty() {
            return b;
        }
    }
    if Path::new("/usr/bin/time").exists() {
        return "/usr/bin/time".to_string();
    }
    "gtime".to_string()
}

/// Parse peak RSS (kB) from GNU `time -v` stderr.
/// Looks for `Maximum resident set size (kbytes): <N>`.
pub fn parse_peak_rss_kb(time_stderr: &str) -> anyhow::Result<f64> {
    for line in time_stderr.lines() {
        let t = line.trim();
        if let Some(rest) = t.strip_prefix("Maximum resident set size (kbytes):") {
            let v: f64 = rest.trim().parse().map_err(|_| {
                anyhow::anyhow!("cannot parse peak RSS from {t:?} (full output: {time_stderr:?})")
            })?;
            if !v.is_finite() || v < 0.0 {
                anyhow::bail!("peak RSS is not a finite non-negative value: {v}");
            }
            return Ok(v);
        }
    }
    anyhow::bail!("GNU time output missing 'Maximum resident set size (kbytes)' (output: {time_stderr:?})")
}

/// Run the target command once and score wall-clock seconds (lower is better).
/// Non-zero exit or unparsable run => score None (same EvalSample shape as custom eval).
pub async fn run_speed_once(
    target_cmd: &str,
    workdir: &Path,
    timeout: Duration,
    log_file: Option<&Path>,
) -> EvalSample {
    use std::time::Instant;
    let mut cmd = Command::new("sh");
    cmd.arg("-c").arg(target_cmd).current_dir(workdir);
    cmd.stdout(std::process::Stdio::piped());
    cmd.stderr(std::process::Stdio::piped());

    let start = Instant::now();
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
    let elapsed = start.elapsed().as_secs_f64();

    let (stdout, stderr, exit_code, timed_out, score) = match timed {
        Err((e, to)) => (String::new(), e.to_string(), None, to, None),
        Ok(output) => {
            let stdout = String::from_utf8_lossy(&output.stdout).to_string();
            let stderr = String::from_utf8_lossy(&output.stderr).to_string();
            let code = output.status.code();
            let score = if output.status.success() {
                Some(elapsed)
            } else {
                None
            };
            (stdout, stderr, code, false, score)
        }
    };

    if let Some(path) = log_file {
        let _ = write_eval_log(
            path,
            &format!("{target_cmd}  [optimize-speed: {elapsed:.6}s]"),
            workdir,
            &stdout,
            &stderr,
            exit_code,
            score,
        )
        .await;
    }

    EvalSample {
        score,
        stdout,
        stderr,
        exit_code,
        timed_out,
    }
}

/// Run the target command once under GNU `time -v` and score peak RSS kB (lower is better).
/// Score is Some(kB) only when the inner command exits 0 and the RSS line parses.
pub async fn run_memory_once(
    target_cmd: &str,
    workdir: &Path,
    timeout: Duration,
    log_file: Option<&Path>,
) -> EvalSample {
    let time_bin = gnu_time_bin();
    let mut cmd = Command::new(&time_bin);
    cmd.arg("-v")
        .arg("sh")
        .arg("-c")
        .arg(target_cmd)
        .current_dir(workdir);
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
            // GNU time writes its report to stderr, merged with the child's stderr.
            let time_stderr = String::from_utf8_lossy(&output.stderr).to_string();
            let code = output.status.code();
            let score = if output.status.success() {
                parse_peak_rss_kb(&time_stderr).ok()
            } else {
                None
            };
            (stdout, time_stderr, code, false, score)
        }
    };

    if let Some(path) = log_file {
        let _ = write_eval_log(
            path,
            &format!("{time_bin} -v sh -c {target_cmd:?}  [optimize-memory]"),
            workdir,
            &stdout,
            &stderr,
            exit_code,
            score,
        )
        .await;
    }

    EvalSample {
        score,
        stdout,
        stderr,
        exit_code,
        timed_out,
    }
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

/// Run the eval command once in `workdir`, retrying failures up to `retries` times.
/// A run is retried when it yields no score (non-zero exit, parse failure,
/// timeout, or spawn error). `delay_between` is slept between attempts.
/// Returns the last attempt's sample.
pub async fn run_eval_once_with_retries(
    eval_cmd: &str,
    workdir: &Path,
    timeout: Duration,
    log_file: Option<&Path>,
    retries: u32,
    delay_between: Duration,
) -> EvalSample {
    let mut sample = run_eval_once(eval_cmd, workdir, timeout, log_file).await;
    for _ in 0..retries {
        if sample.score.is_some() {
            break;
        }
        if !delay_between.is_zero() {
            tokio::time::sleep(delay_between).await;
        }
        sample = run_eval_once(eval_cmd, workdir, timeout, log_file).await;
    }
    sample
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

/// GNU time is required for --optimize-memory. Fail fast with install hint.
pub async fn check_gnu_time_available() -> anyhow::Result<String> {
    let bin = gnu_time_bin();
    let probe = Command::new(&bin)
        .arg("--version")
        .output()
        .await
        .map_err(|e| anyhow::anyhow!("GNU time not found at {bin:?} ({e}); install it (apt install time / brew install gnu-time) or set GTIME_BIN"))?;
    let out = String::from_utf8_lossy(&probe.stdout).to_string()
        + &String::from_utf8_lossy(&probe.stderr).to_string();
    if !out.contains("GNU time") {
        anyhow::bail!("{bin:?} does not look like GNU time (need `time -v` with 'Maximum resident set size'); install GNU time (apt install time / brew install gnu-time) or set GTIME_BIN");
    }
    Ok(bin)
}

/// Run `samples` speed/memory evals with bounded concurrency; returns per-sample results in order.
#[allow(clippy::too_many_arguments)]
#[allow(dead_code)] // legacy wrapper; full variant preferred in binary. Used in tests.
pub async fn run_optimize_samples(
    kind: &crate::cli::EvalKind,
    workdir: PathBuf,
    samples: u32,
    jobs: usize,
    timeout: Duration,
    log_dir: Option<PathBuf>,
    iter: u32,
    agent_idx: u32,
) -> Vec<EvalSample> {
    run_optimize_samples_full(
        kind, workdir, samples, jobs, timeout, log_dir, iter, agent_idx, 0, Duration::ZERO,
    )
    .await
}

/// Full variant with per-sample retries and stagger delay between sample starts.
#[allow(clippy::too_many_arguments)]
pub async fn run_optimize_samples_full(
    kind: &crate::cli::EvalKind,
    workdir: PathBuf,
    samples: u32,
    jobs: usize,
    timeout: Duration,
    log_dir: Option<PathBuf>,
    iter: u32,
    agent_idx: u32,
    retries: u32,
    delay_between: Duration,
) -> Vec<EvalSample> {
    use crate::cli::EvalKind;
    let target = match kind {
        EvalKind::Speed(c) | EvalKind::Memory(c) => c.clone(),
        EvalKind::Custom(c) => c.clone(),
    };
    let is_memory = matches!(kind, EvalKind::Memory(_));
    let jobs = jobs.max(1);
    let sem = Arc::new(Semaphore::new(jobs));
    let mut handles = Vec::new();
    for k in 0..samples {
        let permit_sem = sem.clone();
        let cmd = target.clone();
        let dir = workdir.clone();
        let log = log_dir
            .clone()
            .map(|d| d.join(format!("eval-{iter}-{agent_idx}-{k}.log")));
        handles.push(tokio::spawn(async move {
            let _permit = permit_sem.acquire_owned().await.unwrap();
            if !delay_between.is_zero() && k > 0 {
                tokio::time::sleep(delay_between * k).await;
            }
            run_optimize_once_with_retries(
                &cmd,
                is_memory,
                &dir,
                timeout,
                log.as_deref(),
                retries,
                delay_between,
            )
            .await
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

/// Retry wrapper shared by speed/memory evals.
pub async fn run_optimize_once_with_retries(
    target_cmd: &str,
    is_memory: bool,
    workdir: &Path,
    timeout: Duration,
    log_file: Option<&Path>,
    retries: u32,
    delay_between: Duration,
) -> EvalSample {
    let once = |cmd: String, dir: PathBuf, log: Option<PathBuf>| async move {
        if is_memory {
            run_memory_once(&cmd, &dir, timeout, log.as_deref()).await
        } else {
            run_speed_once(&cmd, &dir, timeout, log.as_deref()).await
        }
    };
    let mut sample = once(
        target_cmd.to_string(),
        workdir.to_path_buf(),
        log_file.map(Path::to_path_buf),
    )
    .await;
    for _ in 0..retries {
        if sample.score.is_some() {
            break;
        }
        if !delay_between.is_zero() {
            tokio::time::sleep(delay_between).await;
        }
        sample = once(
            target_cmd.to_string(),
            workdir.to_path_buf(),
            log_file.map(Path::to_path_buf),
        )
        .await;
    }
    sample
}
/// Run `samples` evals with bounded concurrency; returns per-sample results in order.
#[allow(clippy::too_many_arguments)]
#[allow(dead_code)] // legacy wrapper; full variant preferred in binary. Used in tests.
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
    run_eval_samples_full(
        eval_cmd,
        workdir,
        samples,
        jobs,
        timeout,
        log_dir,
        iter,
        agent_idx,
        0,
        Duration::ZERO,
    )
    .await
}

/// Full variant with per-sample retries and stagger delay between sample starts.
/// `delay_between` is slept between retry attempts and staggered (k * delay)
/// before starting sample k, giving rate-limited evals breathing room.
#[allow(clippy::too_many_arguments)]
pub async fn run_eval_samples_full(
    eval_cmd: &str,
    workdir: PathBuf,
    samples: u32,
    jobs: usize,
    timeout: Duration,
    log_dir: Option<PathBuf>,
    iter: u32,
    agent_idx: u32,
    retries: u32,
    delay_between: Duration,
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
            if !delay_between.is_zero() && k > 0 {
                tokio::time::sleep(delay_between * k).await;
            }
            run_eval_once_with_retries(&cmd, &dir, timeout, log.as_deref(), retries, delay_between)
                .await
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

    #[tokio::test]
    async fn retries_recover_from_flaky_eval() {
        // First attempt fails, retry-marker file makes second attempt succeed.
        let dir = tempfile::tempdir().unwrap();
        let marker = dir.path().join("flaky.marker");
        let marker_s = marker.to_str().unwrap().to_string();
        let cmd = format!(
            "if [ -f {marker_s} ]; then echo 1.0; else touch {marker_s}; exit 1; fi"
        );
        let s = run_eval_once_with_retries(
            &cmd,
            dir.path(),
            Duration::from_secs(5),
            None,
            1,
            Duration::ZERO,
        )
        .await;
        assert_eq!(s.score, Some(1.0));
    }

    #[tokio::test]
    async fn retries_exhausted_keep_last_failure() {
        let dir = tempfile::tempdir().unwrap();
        let s = run_eval_once_with_retries(
            "exit 1",
            dir.path(),
            Duration::from_secs(5),
            None,
            2,
            Duration::ZERO,
        )
        .await;
        assert_eq!(s.score, None);
    }

    #[tokio::test]
    async fn retries_zero_means_single_attempt() {
        let dir = tempfile::tempdir().unwrap();
        let s = run_eval_once_with_retries(
            "exit 1",
            dir.path(),
            Duration::from_secs(5),
            None,
            0,
            Duration::ZERO,
        )
        .await;
        assert_eq!(s.score, None);
        assert_eq!(s.exit_code, Some(1));
    }

    #[test]
    fn parse_peak_rss_kb_from_gnu_time_output() {
        let stderr = "\tMaximum resident set size (kbytes): 12345\n\tMinor faults: 1\n";
        assert_eq!(parse_peak_rss_kb(stderr).unwrap(), 12345.0);
    }

    #[test]
    fn parse_peak_rss_kb_missing_line_errors() {
        assert!(parse_peak_rss_kb("elapsed: 0.1\n").is_err());
    }

    #[tokio::test]
    async fn run_speed_once_measures_elapsed() {
        let dir = tempfile::tempdir().unwrap();
        let s = run_speed_once("true", dir.path(), Duration::from_secs(5), None).await;
        assert_eq!(s.exit_code, Some(0));
        let v = s.score.expect("speed score");
        assert!(v >= 0.0 && v < 5.0, "score={v}");
    }

    #[tokio::test]
    async fn run_speed_once_failure_has_no_score() {
        let dir = tempfile::tempdir().unwrap();
        let s = run_speed_once("exit 3", dir.path(), Duration::from_secs(5), None).await;
        assert_eq!(s.score, None);
        assert_eq!(s.exit_code, Some(3));
    }

    // Requires GNU time (`apt install time` / `brew install gnu-time`).
    #[tokio::test]
    async fn run_memory_once_reports_kb() {
        if check_gnu_time_available().await.is_err() {
            eprintln!("skipping: GNU time not installed");
            return;
        }
        let dir = tempfile::tempdir().unwrap();
        let s = run_memory_once("echo hi", dir.path(), Duration::from_secs(10), None).await;
        let v = s.score.expect("memory score");
        assert!(v > 0.0, "score={v}");
    }

    #[tokio::test]
    async fn run_memory_once_failure_has_no_score() {
        if check_gnu_time_available().await.is_err() {
            eprintln!("skipping: GNU time not installed");
            return;
        }
        let dir = tempfile::tempdir().unwrap();
        let s = run_memory_once("exit 3", dir.path(), Duration::from_secs(10), None).await;
        assert_eq!(s.score, None);
    }
}
