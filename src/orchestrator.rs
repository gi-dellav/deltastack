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
