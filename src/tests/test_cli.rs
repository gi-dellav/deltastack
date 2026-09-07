use crate::cli::*;
use clap::Parser;

fn base_cli() -> Cli {
    Cli::parse_from(["deltastack", "--prompt", "hi", "--eval", "echo 1.0"])
}

#[test]
fn requires_prompt_or_prompt_file_for_custom_eval() {
    let mut c = base_cli();
    c.prompt = None;
    c.prompt_file = None;
    assert!(c.validate().is_err());
}

#[test]
fn optimize_speed_needs_no_prompt() {
    let c = Cli::parse_from(["deltastack", "--optimize-speed", "make bench"]);
    assert_eq!(c.eval_kind(), Some(EvalKind::Speed("make bench".into())));
    assert!(c.is_optimize_mode());
    assert!(c.validate().is_ok());
}

#[test]
fn optimize_memory_needs_no_prompt() {
    let c = Cli::parse_from(["deltastack", "--optimize-memory", "make bench"]);
    assert_eq!(c.eval_kind(), Some(EvalKind::Memory("make bench".into())));
    assert!(c.validate().is_ok());
}

#[test]
fn missing_eval_source_is_rejected() {
    let mut c = base_cli();
    c.eval_cmd = None;
    assert!(c.validate().is_err());
}

#[test]
fn eval_and_optimize_conflict_at_clap_level() {
    let r = Cli::try_parse_from([
        "deltastack",
        "--prompt",
        "a",
        "--eval",
        "echo 1",
        "--optimize-speed",
        "make bench",
    ]);
    assert!(r.is_err());
}

#[test]
fn speed_and_memory_conflict_at_clap_level() {
    let r = Cli::try_parse_from([
        "deltastack",
        "--optimize-speed",
        "a",
        "--optimize-memory",
        "b",
    ]);
    assert!(r.is_err());
}

#[test]
fn rejects_empty_optimize_cmds() {
    let mut c = Cli::parse_from(["deltastack", "--optimize-speed", "x"]);
    c.optimize_speed = Some("  ".into());
    assert!(c.validate().is_err());
    let mut c = Cli::parse_from(["deltastack", "--optimize-memory", "x"]);
    c.optimize_memory = Some("".into());
    assert!(c.validate().is_err());
}

#[test]
fn prompt_and_prompt_file_conflict_at_clap_level() {
    let r = Cli::try_parse_from([
        "deltastack",
        "--prompt",
        "a",
        "--prompt-file",
        "b",
        "--eval",
        "echo 1",
    ]);
    assert!(r.is_err());
}

#[test]
fn defaults_are_sane() {
    let c = base_cli();
    assert_eq!(c.max_iterations, 10);
    assert_eq!(c.agents, 1);
    assert_eq!(c.samples, 1);
    assert_eq!(c.mode, Mode::Minimize);
    assert_eq!(c.aggregate, Aggregate::Mean);
    assert!(!c.uses_worktree_isolation());
}

#[test]
fn multi_agent_enables_worktree_isolation_by_default() {
    let c = Cli::parse_from([
        "deltastack",
        "--prompt",
        "x",
        "--eval",
        "echo 1",
        "--agents",
        "4",
    ]);
    assert!(c.uses_worktree_isolation());
}

#[test]
fn no_isolate_disables_worktrees() {
    let c = Cli::parse_from([
        "deltastack",
        "--prompt",
        "x",
        "--eval",
        "echo 1",
        "--agents",
        "2",
        "--no-isolate",
    ]);
    assert!(!c.uses_worktree_isolation());
}

#[test]
fn wt_base_dir_alone_does_not_enable_worktrees_for_single_agent() {
    let c = Cli::parse_from([
        "deltastack",
        "--prompt",
        "x",
        "--eval",
        "echo 1",
        "--wt-base-dir",
        "/tmp/wt",
    ]);
    assert_eq!(c.agents, 1);
    assert!(!c.uses_worktree_isolation());
    assert!(c.validate().is_ok());
}

#[test]
fn no_isolate_with_many_agents_is_rejected() {
    let mut c = base_cli();
    c.agents = 3;
    c.no_isolate = true;
    assert!(c.validate().is_err());
}

#[test]
fn session_ephemeral_by_default() {
    let c = base_cli();
    assert!(!c.keep_agent_session);
    assert!(c.validate().is_ok());
}

#[test]
fn rejects_zero_iterations_agents_samples() {
    for (field, set) in [("max_iterations", 0u32), ("agents", 0), ("samples", 0)] {
        let mut c = base_cli();
        match field {
            "max_iterations" => c.max_iterations = set,
            "agents" => c.agents = set,
            _ => c.samples = set,
        }
        assert!(c.validate().is_err(), "{field} should be rejected");
    }
}

#[test]
fn rejects_negative_or_nan_min_improvement() {
    let mut c = base_cli();
    c.min_improvement = -0.1;
    assert!(c.validate().is_err());
    c.min_improvement = f64::NAN;
    assert!(c.validate().is_err());
}

#[test]
fn rejects_negative_or_nan_min_improvement_rel() {
    let mut c = base_cli();
    c.min_improvement_rel = -0.01;
    assert!(c.validate().is_err());
    c.min_improvement_rel = f64::NAN;
    assert!(c.validate().is_err());
    c.min_improvement_rel = 0.01;
    assert!(c.validate().is_ok());
}

#[test]
fn rejects_zero_target_sticky() {
    let mut c = base_cli();
    c.target_sticky = 0;
    assert!(c.validate().is_err());
    c.target_sticky = 3;
    assert!(c.validate().is_ok());
}

#[test]
fn new_loop_flags_parse() {
    let c = Cli::parse_from([
        "deltastack",
        "--prompt",
        "x",
        "--eval",
        "echo 1",
        "--eval-retries",
        "2",
        "--max-wall-time",
        "3600",
        "--min-improvement-rel",
        "0.01",
        "--target",
        "0.5",
        "--target-sticky",
        "3",
        "--delay-between-iterations",
        "5",
        "--delay-between-eval",
        "1",
    ]);
    assert_eq!(c.eval_retries, 2);
    assert_eq!(c.max_wall_time, 3600);
    assert_eq!(c.min_improvement_rel, 0.01);
    assert_eq!(c.target, Some(0.5));
    assert_eq!(c.target_sticky, 3);
    assert_eq!(c.delay_between_iterations, 5);
    assert_eq!(c.delay_between_eval, 1);
    assert!(c.validate().is_ok());
}

#[test]
fn accepts_any_finite_temperature() {
    let mut c = base_cli();
    c.temperature = Some(2.5);
    assert!(c.validate().is_ok());
    c.temperature = Some(-1.5);
    assert!(c.validate().is_ok());
    c.temperature = Some(1.5);
    assert!(c.validate().is_ok());
    c.temperature = Some(f64::NAN);
    assert!(c.validate().is_err());
    c.temperature = Some(f64::INFINITY);
    assert!(c.validate().is_err());
}

#[test]
fn rejects_non_finite_temperature_range_and_step() {
    let mut c = base_cli();
    c.temperature_min = Some(0.0);
    c.temperature_max = Some(f64::NAN);
    assert!(c.validate().is_err());

    let mut c = base_cli();
    c.temperature_min = Some(f64::INFINITY);
    c.temperature_max = Some(1.0);
    assert!(c.validate().is_err());

    let mut c = base_cli();
    c.temperature = Some(0.5);
    c.temperature_step = Some(f64::INFINITY);
    assert!(c.validate().is_err());
}

#[test]
fn temperature_min_requires_max_and_vice_versa() {
    let mut c = base_cli();
    c.temperature_min = Some(0.1);
    assert!(c.validate().is_err());

    let mut c = base_cli();
    c.temperature_max = Some(0.9);
    assert!(c.validate().is_err());

    let mut c = base_cli();
    c.temperature_min = Some(0.1);
    c.temperature_max = Some(0.9);
    assert!(c.validate().is_ok());
}

#[test]
fn temperature_min_must_not_exceed_max() {
    let mut c = base_cli();
    c.temperature_min = Some(1.0);
    c.temperature_max = Some(0.5);
    assert!(c.validate().is_err());

    // Equal bounds = constant temperature, allowed.
    c.temperature_max = Some(1.0);
    assert!(c.validate().is_ok());
}

#[test]
fn temperature_step_requires_base_temperature() {
    let mut c = base_cli();
    c.temperature_step = Some(0.1);
    assert!(c.validate().is_err());

    c.temperature = Some(0.5);
    assert!(c.validate().is_ok());

    let mut c = base_cli();
    c.temperature_min = Some(0.1);
    c.temperature_max = Some(0.9);
    c.temperature_step = Some(-0.05);
    assert!(c.validate().is_ok());
}

#[test]
fn negative_temperatures_accepted() {
    let c = Cli::parse_from([
        "deltastack",
        "--prompt",
        "x",
        "--eval",
        "echo 1",
        "--temperature",
        "-0.5",
        "--temperature-step",
        "-0.1",
    ]);
    assert_eq!(c.temperature, Some(-0.5));
    assert_eq!(c.temperature_step, Some(-0.1));
    assert!(c.validate().is_ok());
}

#[test]
fn effective_eval_jobs_auto() {
    let mut c = base_cli();
    c.agents = 4;
    c.samples = 3;
    c.eval_jobs = 0;
    assert_eq!(c.effective_eval_jobs(), 3);
    c.eval_jobs = 2;
    assert_eq!(c.effective_eval_jobs(), 2);
}

#[test]
fn tools_repeatable_and_csv() {
    let c = Cli::parse_from([
        "deltastack",
        "--prompt",
        "x",
        "--eval",
        "echo 1",
        "--tools",
        "read,write",
        "--tools",
        "bash",
    ]);
    assert_eq!(c.tools, vec!["read,write", "bash"]);
}

#[test]
fn csv_file_opt_in_and_validated() {
    let c = base_cli();
    assert_eq!(c.csv_file, None);
    assert!(c.validate().is_ok());
    let c = Cli::parse_from([
        "deltastack",
        "--prompt",
        "x",
        "--eval",
        "echo 1",
        "--csv-file",
        "run.csv",
    ]);
    assert_eq!(c.csv_file.as_deref(), Some("run.csv"));
    assert!(c.validate().is_ok());
    let mut c = base_cli();
    c.csv_file = Some("  ".into());
    assert!(c.validate().is_err());
}
