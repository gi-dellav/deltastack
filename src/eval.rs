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
    anyhow::bail!(
        "GNU time output missing 'Maximum resident set size (kbytes)' (output: {time_stderr:?})"
    )
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
        kind,
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

/// True if an eval sample counts as failed: non-zero (or missing) exit code,
/// a timeout, or an unparsable/missing score.
///
/// This is deliberately explicit rather than just `score.is_none()` so the
/// semantics stay correct if score-producing logic ever changes: exit 0 with
/// a valid score is the only success case.
pub fn sample_failed(s: &EvalSample) -> bool {
    if s.timed_out {
        return true;
    }
    if s.exit_code != Some(0) {
        return true;
    }
    s.score.is_none()
}
