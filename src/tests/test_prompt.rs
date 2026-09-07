use crate::prompt::*;

#[test]
fn load_inline_prompt() {
    assert_eq!(load_user_prompt(Some("hello"), None).unwrap(), "hello");
}

#[test]
fn load_missing_prompt_errors() {
    assert!(load_user_prompt(None, None).is_err());
}

#[test]
fn load_conflicting_prompts_errors() {
    assert!(load_user_prompt(Some("a"), Some("b")).is_err());
}

#[test]
fn load_prompt_file_roundtrip() {
    let dir = tempfile::tempdir().unwrap();
    let p = dir.path().join("task.md");
    std::fs::write(&p, "file prompt").unwrap();
    let s = p.to_str().unwrap().to_string();
    assert_eq!(load_user_prompt(None, Some(&s)).unwrap(), "file prompt");
}

#[test]
fn load_missing_prompt_file_errors() {
    assert!(load_user_prompt(None, Some("/nonexistent-xyz/task.md")).is_err());
}

#[test]
fn commit_suffix_contains_git_commit_and_ids() {
    let s = auto_commit_suffix(3, 1, "deltastack/");
    assert!(s.contains("git add -A"), "{s}");
    assert!(s.contains("git commit"), "{s}");
    assert!(s.contains("iter3"), "{s}");
    assert!(s.contains("agent1"), "{s}");
    assert!(s.contains("deltastack/"), "{s}");
}

#[test]
fn commit_suffix_custom_prefix() {
    let s = auto_commit_suffix(1, 0, "exp/");
    assert!(s.contains("exp/iter1"), "{s}");
}

#[test]
fn context_first_iteration_mentions_no_baseline() {
    let c = iteration_context(1, 10, 0, 1, None, None, None);
    assert!(c.contains("1/10"), "{c}");
    assert!(c.contains("No baseline"), "{c}");
}

#[test]
fn context_with_best_and_last() {
    let c = iteration_context(2, 5, 1, 4, Some(0.99), Some("abc123"), Some(1.01));
    assert!(c.contains("0.99"), "{c}");
    assert!(c.contains("abc123"), "{c}");
    assert!(c.contains("1.01"), "{c}");
    assert!(c.contains("distinct"), "{c}");
}

#[test]
fn context_single_agent_has_no_distinct_hint() {
    let c = iteration_context(1, 5, 0, 1, Some(1.0), None, None);
    assert!(!c.contains("distinct"), "{c}");
}

#[test]
fn compose_includes_user_prompt_and_suffix() {
    let p = compose_iteration_prompt("Do X", 1, 10, 0, 1, None, None, None, "deltastack/", true);
    assert!(p.starts_with("Do X"), "{p}");
    assert!(p.contains("git commit"), "{p}");
}

#[test]
fn compose_without_suffix_omits_git_commit() {
    let p = compose_iteration_prompt("Do X", 1, 10, 0, 1, None, None, None, "deltastack/", false);
    assert!(!p.contains("git commit"), "{p}");
}

#[test]
fn compose_trims_user_prompt() {
    let p = compose_iteration_prompt("  Do X\n\n", 1, 2, 0, 1, None, None, None, "x/", false);
    assert!(p.starts_with("Do X\n\n---"), "{p}");
}

#[test]
fn default_speed_prompt_mentions_seconds_and_cmd() {
    let p = default_optimize_speed_prompt("make bench");
    assert!(p.contains("make bench"), "{p}");
    assert!(p.contains("seconds"), "{p}");
    assert!(p.contains("lower is better"), "{p}");
}

#[test]
fn default_memory_prompt_mentions_kb_and_cmd() {
    let p = default_optimize_memory_prompt("make bench");
    assert!(p.contains("make bench"), "{p}");
    assert!(p.contains("kilobytes"), "{p}");
    assert!(p.contains("lower is better"), "{p}");
}

#[test]
fn resolve_prefers_explicit_prompt_over_default() {
    let (p, used_default) = resolve_user_prompt(Some("custom"), None, Some("default")).unwrap();
    assert_eq!(p, "custom");
    assert!(!used_default);
}

#[test]
fn resolve_falls_back_to_default() {
    let (p, used_default) = resolve_user_prompt(None, None, Some("default")).unwrap();
    assert_eq!(p, "default");
    assert!(used_default);
}

#[test]
fn resolve_errors_without_any_prompt() {
    assert!(resolve_user_prompt(None, None, None).is_err());
}
