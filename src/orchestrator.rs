use crate::cli::{Aggregate, Mode};
use crate::eval::{aggregate_scores, successful_scores, EvalSample};

/// True if `candidate` is an improvement over `best` given mode + epsilon.
#[allow(dead_code)] // legacy wrapper; full variant preferred in binary. Used in tests.
pub fn is_improvement(candidate: f64, best: f64, mode: Mode, min_improvement: f64) -> bool {
    is_improvement_full(candidate, best, mode, min_improvement, 0.0)
}

/// True if `candidate` improves over `best` satisfying BOTH absolute and
/// relative thresholds (when non-zero).
/// Relative threshold is a fraction (e.g. 0.01 = 1%) of `|best|`:
/// required delta = min_rel * |best|. When `best == 0.0` the relative check
/// is skipped (falls back to absolute-only) to avoid divide-by-zero / dead-lock.
pub fn is_improvement_full(
    candidate: f64,
    best: f64,
    mode: Mode,
    min_abs: f64,
    min_rel: f64,
) -> bool {
    if !candidate.is_finite() || !best.is_finite() {
        return false;
    }
    let abs_ok = match mode {
        Mode::Maximize => candidate > best + min_abs,
        Mode::Minimize => candidate < best - min_abs,
    };
    if !abs_ok {
        return false;
    }
    if !min_rel.is_finite() || min_rel <= 0.0 {
        return true;
    }
    if best == 0.0 {
        return true;
    }
    let required = min_rel * best.abs();
    match mode {
        Mode::Maximize => candidate > best + required,
        Mode::Minimize => candidate < best - required,
    }
}

/// True if `score` reached the early-stop target (mode-aware, epsilon=0).
pub fn reached_target(score: f64, target: f64, mode: Mode) -> bool {
    match mode {
        Mode::Maximize => score >= target,
        Mode::Minimize => score <= target,
    }
}

/// Aggregate samples -> Option<score>. None when all samples failed.
pub fn score_candidate(samples: &[EvalSample], aggregate: Aggregate) -> Option<f64> {
    aggregate_scores(&successful_scores(samples), aggregate)
}

#[derive(Debug, Clone, PartialEq)]
pub struct CandidateScore {
    pub agent_idx: u32,
    pub score: Option<f64>,
}

/// Pick the best candidate by mode. None-valued candidates sort last.
/// Ties -> lowest agent_idx wins (deterministic).
pub fn select_winner(candidates: &[CandidateScore], mode: Mode) -> Option<u32> {
    if candidates.is_empty() {
        return None;
    }
    let mut best: Option<&CandidateScore> = None;
    for c in candidates {
        match best {
            None => best = Some(c),
            Some(b) => match (c.score, b.score) {
                (None, None) => {
                    if c.agent_idx < b.agent_idx {
                        best = Some(c);
                    }
                }
                (Some(_), None) => best = Some(c),
                (None, Some(_)) => {}
                (Some(s), Some(bs)) => {
                    let better = match mode {
                        Mode::Maximize => s > bs,
                        Mode::Minimize => s < bs,
                    };
                    if better || (s == bs && c.agent_idx < b.agent_idx) {
                        best = Some(c);
                    }
                }
            },
        }
    }
    best.map(|b| b.agent_idx)
}

/// Loop-stop decision after an iteration completes.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum StopReason {
    Continue,
    MaxIterations,
    PatienceExhausted,
    TargetReached,
    WallTimeExceeded,
}

#[allow(dead_code)] // legacy wrapper; full variant preferred in binary. Used in tests.
pub fn should_stop(
    iteration: u32, // 1-based completed iterations
    max_iterations: u32,
    fails_since_best: u32,
    patience: u32,
    winner_score: Option<f64>,
    target: Option<f64>,
    mode: Mode,
) -> StopReason {
    should_stop_full(
        iteration,
        max_iterations,
        fails_since_best,
        patience,
        winner_score,
        target,
        mode,
        1,
        0,
        None,
        None,
    )
}

/// Full stop check with sticky-target, wall-clock, and consecutive-hit tracking.
///
/// - `target_sticky`: target must be reached this many consecutive iterations
///   (tracked via `target_hits_so_far` + current winner score).
/// - `max_wall_time`: `Some(budget)` with `elapsed >= budget` stops (checked
///   after target/max-iterations so an on-target final iteration reports
///   `TargetReached` rather than `WallTimeExceeded`).
/// - `elapsed`: time since run start; `None` disables the wall-clock check.
#[allow(clippy::too_many_arguments)]
pub fn should_stop_full(
    iteration: u32, // 1-based completed iterations
    max_iterations: u32,
    fails_since_best: u32,
    patience: u32,
    winner_score: Option<f64>,
    target: Option<f64>,
    mode: Mode,
    target_sticky: u32,
    target_hits_so_far: u32,
    elapsed: Option<std::time::Duration>,
    max_wall_time: Option<std::time::Duration>,
) -> StopReason {
    let sticky = target_sticky.max(1);
    if let (Some(s), Some(t)) = (winner_score, target) {
        if reached_target(s, t, mode) && target_hits_so_far + 1 >= sticky {
            return StopReason::TargetReached;
        }
    }
    if iteration >= max_iterations {
        return StopReason::MaxIterations;
    }
    if patience > 0 && fails_since_best >= patience {
        return StopReason::PatienceExhausted;
    }
    if let (Some(budget), Some(el)) = (max_wall_time, elapsed) {
        if !budget.is_zero() && el >= budget {
            return StopReason::WallTimeExceeded;
        }
    }
    StopReason::Continue
}

/// Wall-clock helper: true when `elapsed >= budget` and budget is non-zero.
pub fn wall_time_exceeded(elapsed: std::time::Duration, max_wall_time_secs: u64) -> bool {
    max_wall_time_secs > 0 && elapsed.as_secs() >= max_wall_time_secs
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn minimize_improvement_needs_epsilon() {
        assert!(is_improvement(0.9, 1.0, Mode::Minimize, 0.0));
        assert!(!is_improvement(1.0, 1.0, Mode::Minimize, 0.0));
        assert!(!is_improvement(0.999, 1.0, Mode::Minimize, 0.01));
        assert!(is_improvement(0.98, 1.0, Mode::Minimize, 0.01));
    }

    #[test]
    fn maximize_improvement_needs_epsilon() {
        assert!(is_improvement(1.1, 1.0, Mode::Maximize, 0.0));
        assert!(!is_improvement(1.0, 1.0, Mode::Maximize, 0.0));
        assert!(!is_improvement(1.005, 1.0, Mode::Maximize, 0.01));
        assert!(is_improvement(1.02, 1.0, Mode::Maximize, 0.01));
    }

    #[test]
    fn non_finite_never_improves() {
        assert!(!is_improvement(f64::NAN, 1.0, Mode::Minimize, 0.0));
        assert!(!is_improvement(0.5, f64::INFINITY, Mode::Minimize, 0.0));
    }

    #[test]
    fn relative_improvement_scales_with_best() {
        // 1% of |100| = 1.0: need strictly better than 101.
        assert!(!is_improvement_full(
            100.5,
            100.0,
            Mode::Maximize,
            0.0,
            0.01
        ));
        assert!(is_improvement_full(101.5, 100.0, Mode::Maximize, 0.0, 0.01));
        // Minimize mirror.
        assert!(!is_improvement_full(99.5, 100.0, Mode::Minimize, 0.0, 0.01));
        assert!(is_improvement_full(98.5, 100.0, Mode::Minimize, 0.0, 0.01));
    }

    #[test]
    fn relative_and_absolute_both_required() {
        // Passes abs (0.5 > 0.1) but fails rel (needs > 1.0 on best=100).
        assert!(!is_improvement_full(
            100.5,
            100.0,
            Mode::Maximize,
            0.1,
            0.01
        ));
        assert!(is_improvement_full(101.5, 100.0, Mode::Maximize, 0.1, 0.01));
    }

    #[test]
    fn relative_skipped_when_best_is_zero() {
        assert!(is_improvement_full(0.5, 0.0, Mode::Maximize, 0.1, 0.5));
        assert!(!is_improvement_full(0.05, 0.0, Mode::Maximize, 0.1, 0.5));
    }

    #[test]
    fn sticky_target_needs_consecutive_hits() {
        use std::time::Duration;
        // First hit with sticky=2 -> continue.
        assert_eq!(
            should_stop_full(
                1,
                10,
                0,
                0,
                Some(0.5),
                Some(0.6),
                Mode::Minimize,
                2,
                0,
                None,
                None
            ),
            StopReason::Continue
        );
        // Second consecutive hit -> reached.
        assert_eq!(
            should_stop_full(
                2,
                10,
                0,
                0,
                Some(0.5),
                Some(0.6),
                Mode::Minimize,
                2,
                1,
                None,
                None
            ),
            StopReason::TargetReached
        );
    }

    #[test]
    fn wall_time_stop_and_helper() {
        use std::time::Duration;
        assert!(wall_time_exceeded(Duration::from_secs(10), 10));
        assert!(!wall_time_exceeded(Duration::from_secs(9), 10));
        assert!(!wall_time_exceeded(Duration::from_secs(100), 0));
        assert_eq!(
            should_stop_full(
                1,
                10,
                0,
                0,
                Some(1.0),
                None,
                Mode::Minimize,
                1,
                0,
                Some(Duration::from_secs(3700)),
                Some(Duration::from_secs(3600)),
            ),
            StopReason::WallTimeExceeded
        );
    }

    #[test]
    fn reached_target_modes() {
        assert!(reached_target(0.9, 1.0, Mode::Minimize));
        assert!(!reached_target(1.1, 1.0, Mode::Minimize));
        assert!(reached_target(1.1, 1.0, Mode::Maximize));
        assert!(!reached_target(0.9, 1.0, Mode::Maximize));
    }

    #[test]
    fn score_candidate_none_when_all_fail() {
        let fails = vec![EvalSample {
            score: None,
            stdout: String::new(),
            stderr: String::new(),
            exit_code: Some(1),
            timed_out: false,
        }];
        assert_eq!(score_candidate(&fails, Aggregate::Mean), None);
    }

    #[test]
    fn score_candidate_aggregates_successes() {
        let mk = |v: f64| EvalSample {
            score: Some(v),
            stdout: String::new(),
            stderr: String::new(),
            exit_code: Some(0),
            timed_out: false,
        };
        let s = vec![mk(1.0), mk(3.0)];
        assert_eq!(score_candidate(&s, Aggregate::Mean), Some(2.0));
        assert_eq!(score_candidate(&s, Aggregate::Max), Some(3.0));
    }

    #[test]
    fn select_winner_minimize() {
        let c = vec![
            CandidateScore {
                agent_idx: 0,
                score: Some(1.0),
            },
            CandidateScore {
                agent_idx: 1,
                score: Some(0.5),
            },
            CandidateScore {
                agent_idx: 2,
                score: Some(0.8),
            },
        ];
        assert_eq!(select_winner(&c, Mode::Minimize), Some(1));
    }

    #[test]
    fn select_winner_maximize() {
        let c = vec![
            CandidateScore {
                agent_idx: 0,
                score: Some(1.0),
            },
            CandidateScore {
                agent_idx: 1,
                score: Some(0.5),
            },
        ];
        assert_eq!(select_winner(&c, Mode::Maximize), Some(0));
    }

    #[test]
    fn select_winner_prefers_scored_over_failed() {
        let c = vec![
            CandidateScore {
                agent_idx: 0,
                score: None,
            },
            CandidateScore {
                agent_idx: 1,
                score: Some(9.9),
            },
        ];
        assert_eq!(select_winner(&c, Mode::Minimize), Some(1));
    }

    #[test]
    fn select_winner_all_failed_picks_lowest_idx() {
        let c = vec![
            CandidateScore {
                agent_idx: 2,
                score: None,
            },
            CandidateScore {
                agent_idx: 1,
                score: None,
            },
        ];
        assert_eq!(select_winner(&c, Mode::Minimize), Some(1));
    }

    #[test]
    fn select_winner_tie_lowest_idx() {
        let c = vec![
            CandidateScore {
                agent_idx: 1,
                score: Some(1.0),
            },
            CandidateScore {
                agent_idx: 0,
                score: Some(1.0),
            },
        ];
        assert_eq!(select_winner(&c, Mode::Minimize), Some(0));
    }

    #[test]
    fn select_winner_empty_is_none() {
        assert_eq!(select_winner(&[], Mode::Minimize), None);
    }

    #[test]
    fn stop_max_iterations() {
        assert_eq!(
            should_stop(10, 10, 0, 0, Some(1.0), None, Mode::Minimize),
            StopReason::MaxIterations
        );
        assert_eq!(
            should_stop(9, 10, 0, 0, Some(1.0), None, Mode::Minimize),
            StopReason::Continue
        );
    }

    #[test]
    fn stop_patience() {
        assert_eq!(
            should_stop(5, 10, 3, 3, Some(1.0), None, Mode::Minimize),
            StopReason::PatienceExhausted
        );
        assert_eq!(
            should_stop(5, 10, 2, 3, Some(1.0), None, Mode::Minimize),
            StopReason::Continue
        );
    }

    #[test]
    fn stop_target_takes_precedence() {
        assert_eq!(
            should_stop(1, 10, 0, 0, Some(0.5), Some(0.6), Mode::Minimize),
            StopReason::TargetReached
        );
        assert_eq!(
            should_stop(1, 10, 0, 0, Some(0.7), Some(0.6), Mode::Minimize),
            StopReason::Continue
        );
    }

    #[test]
    fn stop_no_winner_score_ignores_target() {
        assert_eq!(
            should_stop(1, 10, 0, 0, None, Some(0.6), Mode::Minimize),
            StopReason::Continue
        );
    }
}
