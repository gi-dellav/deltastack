use crate::agent::*;
use crate::cli::Cli;
use clap::Parser;
use rand::SeedableRng;

fn cli(extra: &[&str]) -> Cli {
    let mut base = vec!["deltastack", "--prompt", "x", "--eval", "echo 1"];
    base.extend_from_slice(extra);
    Cli::parse_from(base)
}

#[test]
fn flatten_tools_splits_csv_and_trims() {
    assert_eq!(
        flatten_tools(&["read,write".into(), " bash ".into(), "".into()]),
        vec!["read", "write", "bash"]
    );
    assert!(flatten_tools(&[]).is_empty());
}

#[test]
fn single_agent_has_no_worktree_flags_by_default() {
    let c = cli(&[]);
    let argv = build_zerostack_argv(&c, 0, 0, "prompt");
    assert!(argv.contains(&"-p".to_string()));
    assert!(!argv.iter().any(|a| a == "--worktree"));
}

#[test]
fn multi_agent_uses_deterministic_worktree() {
    let c = cli(&["--agents", "4"]);
    let argv = build_zerostack_argv(&c, 3, 1, "prompt");
    let pos = argv
        .iter()
        .position(|a| a == "--worktree")
        .expect("should have --worktree");
    assert_eq!(argv[pos + 1], "deltastack-iter3-agent1");
}

#[test]
fn session_ephemeral_by_default() {
    let c = cli(&[]);
    let argv = build_zerostack_argv(&c, 0, 0, "p");
    assert!(argv.contains(&"--no-session".to_string()));
    assert!(!argv.iter().any(|a| a == "--name"));
}

#[test]
fn keep_session_flag() {
    let c = cli(&["--keep-agent-session"]);
    let argv = build_zerostack_argv(&c, 2, 3, "p");
    let pos = argv.iter().position(|a| a == "--name").unwrap();
    assert_eq!(argv[pos + 1], "deltastack-iter2-agent3");
    assert!(!argv.iter().any(|a| a == "--no-session"));
}

#[test]
fn wt_auto_merge_and_force_forwarded() {
    let c = cli(&[
        "--agents",
        "2",
        "--wt-auto-merge",
        "--wt-force",
        "--wt-base-dir",
        "/tmp/wt",
    ]);
    let argv = build_zerostack_argv(&c, 0, 0, "p");
    assert!(argv.contains(&"--wt-auto-merge".to_string()));
    assert!(argv.contains(&"--wt-force".to_string()));
    let pos = argv.iter().position(|a| a == "--wt-base-dir").unwrap();
    assert_eq!(argv[pos + 1], "/tmp/wt");
}

#[test]
fn wt_flags_absent_in_place_mode() {
    let c = cli(&["--agents", "2", "--no-isolate", "--wt-auto-merge"]);
    // isolation disabled -> no worktree machinery forwarded
    let argv = build_zerostack_argv(&c, 0, 0, "p");
    assert!(!argv
        .iter()
        .any(|a| a == "--worktree" || a == "--wt-auto-merge"));
}

#[test]
fn wt_base_dir_alone_does_not_trigger_worktree_for_single_agent() {
    let c = cli(&["--wt-base-dir", "/tmp/wt"]);
    let argv = build_zerostack_argv(&c, 0, 0, "p");
    assert!(!argv.iter().any(|a| a == "--worktree"));
    assert!(!argv.iter().any(|a| a == "--wt-base-dir"));
}

#[test]
fn session_naming_with_keep_flag() {
    let c = cli(&["--keep-agent-session"]);
    let argv = build_zerostack_argv(&c, 2, 3, "p");
    let pos = argv.iter().position(|a| a == "--name").unwrap();
    assert_eq!(argv[pos + 1], "deltastack-iter2-agent3");
    assert!(!argv.iter().any(|a| a == "--no-session"));
}

#[test]
fn no_session_by_default() {
    let c = cli(&[]);
    let argv = build_zerostack_argv(&c, 0, 0, "p");
    assert!(argv.contains(&"--no-session".to_string()));
    assert!(!argv.iter().any(|a| a == "--name"));
}

#[test]
fn model_passthrough() {
    let c = cli(&[
        "--model",
        "foo",
        "--max-agent-turns",
        "50",
        "--temperature",
        "0.7",
        "--yolo",
    ]);
    let argv = build_zerostack_argv(&c, 0, 0, "p");
    assert!(argv.contains(&"--model".to_string()));
    assert!(argv.contains(&"foo".to_string()));
    assert!(argv.contains(&"--yolo".to_string()));
}

#[test]
fn prompt_is_second_argv_element() {
    let c = cli(&[]);
    let argv = build_zerostack_argv(&c, 0, 0, "FULL PROMPT");
    assert_eq!(argv[0], "-p");
    assert_eq!(argv[1], "FULL PROMPT");
}

#[test]
fn sandbox_network_bool_rendering() {
    let c = cli(&["--sandbox-network", "true"]);
    // clap parses Option<bool> from string "true"
    let argv = build_zerostack_argv(&c, 0, 0, "p");
    let pos = argv.iter().position(|a| a == "--sandbox-network").unwrap();
    assert_eq!(argv[pos + 1], "true");
}

#[test]
fn no_temperature_by_default() {
    let c = cli(&[]);
    let mut rng = rand::rngs::StdRng::seed_from_u64(0);
    assert_eq!(resolve_temperature(&c, 1, &mut rng), None);
    let argv = build_zerostack_argv(&c, 1, 0, "p");
    assert!(!argv.iter().any(|a| a == "--temperature"));
}

#[test]
fn fixed_temperature_passthrough() {
    let c = cli(&["--temperature", "0.7"]);
    let mut rng = rand::rngs::StdRng::seed_from_u64(0);
    assert_eq!(resolve_temperature(&c, 1, &mut rng), Some(0.7));
    let argv = build_zerostack_argv(&c, 1, 0, "p");
    let pos = argv.iter().position(|a| a == "--temperature").unwrap();
    assert_eq!(argv[pos + 1], "0.7");
}

#[test]
fn random_temperature_within_range() {
    let c = cli(&["--temperature-min", "0.2", "--temperature-max", "0.8"]);
    for seed in 0..50 {
        let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
        let t = resolve_temperature(&c, 1, &mut rng).unwrap();
        assert!((0.2..=0.8).contains(&t), "out of range: {t}");
    }
}

#[test]
fn random_temperature_degenerate_range_is_constant() {
    let c = cli(&["--temperature-min", "0.5", "--temperature-max", "0.5"]);
    let mut rng = rand::rngs::StdRng::seed_from_u64(42);
    assert_eq!(resolve_temperature(&c, 3, &mut rng), Some(0.5));
}

#[test]
fn negative_random_range() {
    let c = cli(&["--temperature-min", "-1.0", "--temperature-max", "-0.2"]);
    let mut rng = rand::rngs::StdRng::seed_from_u64(7);
    let t = resolve_temperature(&c, 1, &mut rng).unwrap();
    assert!((-1.0..=-0.2).contains(&t), "out of range: {t}");
}

#[test]
fn temperature_step_applies_from_step_one() {
    let c = cli(&["--temperature", "1.0", "--temperature-step", "0.1"]);
    let mut rng = rand::rngs::StdRng::seed_from_u64(0);
    assert!((resolve_temperature(&c, 1, &mut rng).unwrap() - 1.1).abs() < 1e-12);
    assert!((resolve_temperature(&c, 2, &mut rng).unwrap() - 1.2).abs() < 1e-12);
    assert!((resolve_temperature(&c, 5, &mut rng).unwrap() - 1.5).abs() < 1e-12);
}

#[test]
fn temperature_step_with_negative_values() {
    let c = cli(&["--temperature", "-0.5", "--temperature-step", "-0.1"]);
    let mut rng = rand::rngs::StdRng::seed_from_u64(0);
    assert!((resolve_temperature(&c, 1, &mut rng).unwrap() - -0.6).abs() < 1e-12);
    assert!((resolve_temperature(&c, 3, &mut rng).unwrap() - -0.8).abs() < 1e-12);
}

#[test]
fn temperature_step_applies_on_random_base() {
    // Seeded draw must equal base; step shifts it by step * current_step.
    let c = cli(&[
        "--temperature-min",
        "0.0",
        "--temperature-max",
        "1.0",
        "--temperature-step",
        "0.5",
    ]);
    let c_no_step = cli(&["--temperature-min", "0.0", "--temperature-max", "1.0"]);
    for seed in [1u64, 2, 3] {
        let mut r1 = rand::rngs::StdRng::seed_from_u64(seed);
        let mut r2 = rand::rngs::StdRng::seed_from_u64(seed);
        let base = resolve_temperature(&c_no_step, 1, &mut r1).unwrap();
        let stepped = resolve_temperature(&c, 2, &mut r2).unwrap();
        assert!((stepped - (base + 0.5 * 2.0)).abs() < 1e-12);
    }
}
