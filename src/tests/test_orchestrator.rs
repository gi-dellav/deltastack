use crate::cli::{Aggregate, Mode};
use crate::eval::EvalSample;
use crate::orchestrator::*;
use std::time::Duration;

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
