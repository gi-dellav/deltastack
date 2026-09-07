use crate::cli::Aggregate;
use crate::eval::*;
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
    let cmd = format!("if [ -f {marker_s} ]; then echo 1.0; else touch {marker_s}; exit 1; fi");
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
