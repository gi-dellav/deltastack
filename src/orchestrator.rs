use crate::cli::{Aggregate, Mode};
use crate::eval::{aggregate_scores, successful_scores, EvalSample};

/// True if `candidate` is an improvement over `best` given mode + epsilon.
pub fn is_improvement(candidate: f64, best: f64, mode: Mode, min_improvement: f64) -> bool {
    if !candidate.is_finite() || !best.is_finite() {
        return false;
    }
    match mode {
        Mode::Maximize => candidate > best + min_improvement,
        Mode::Minimize => candidate < best - min_improvement,
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
}

pub fn should_stop(
    iteration: u32, // 1-based completed iterations
    max_iterations: u32,
    fails_since_best: u32,
    patience: u32,
    winner_score: Option<f64>,
    target: Option<f64>,
    mode: Mode,
) -> StopReason {
    if let (Some(s), Some(t)) = (winner_score, target) {
        if reached_target(s, t, mode) {
            return StopReason::TargetReached;
        }
    }
    if iteration >= max_iterations {
        return StopReason::MaxIterations;
    }
    if patience > 0 && fails_since_best >= patience {
        return StopReason::PatienceExhausted;
    }
    StopReason::Continue
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
