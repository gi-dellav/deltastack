use std::fs;

/// Load the user task prompt from `--prompt` or `--prompt-file`.
pub fn load_user_prompt(prompt: Option<&str>, prompt_file: Option<&str>) -> anyhow::Result<String> {
    match (prompt, prompt_file) {
        (Some(p), None) => Ok(p.to_string()),
        (None, Some(f)) => Ok(fs::read_to_string(f)?),
        (None, None) => anyhow::bail!("one of --prompt or --prompt-file is required"),
        (Some(_), Some(_)) => anyhow::bail!("--prompt conflicts with --prompt-file"),
    }
}

/// Suffix appended to every agent prompt so the agent commits its attempt.
/// This makes the eval-gate + auto-revert logic work: each candidate is a commit.
pub fn auto_commit_suffix(iteration: u32, agent_idx: u32, branch_prefix: &str) -> String {
    format!(
        "\n\n---\nWhen finished, you MUST commit your attempt with git:\n\
         `git add -A && git commit -m \"{prefix}iter{iteration} agent{agent_idx}: <one-line summary of what you tried>\"`\n\
         Put details in the commit body if useful. Do not push. Do not modify files outside this repo/worktree."
        ,
        prefix = branch_prefix,
        iteration = iteration,
        agent_idx = agent_idx,
    )
}

/// Context block describing loop state, prepended/appended to the user prompt.
pub fn iteration_context(
    iteration: u32,
    max_iterations: u32,
    agent_idx: u32,
    agents: u32,
    best_score: Option<f64>,
    best_sha: Option<&str>,
    last_score: Option<f64>,
) -> String {
    let mut s = format!(
        "\n\n---\nIteration context: iteration {iteration}/{max_iterations}, agent {agent_idx}/{agents}."
    );
    match (best_score, best_sha) {
        (Some(b), Some(sha)) => s.push_str(&format!(" Global best so far: {b} (commit {sha}).")),
        (Some(b), None) => s.push_str(&format!(" Global best so far: {b}.")),
        _ => s.push_str(" No baseline yet: this may be the first attempt."),
    }
    if let Some(l) = last_score {
        s.push_str(&format!(" Last evaluated score: {l}."));
    }
    if agents > 1 {
        s.push_str(" Other agents are trying different ideas in parallel — be bold and distinct.");
    }
    s
}

/// Compose the full prompt sent to `zerostack -p` for one (iteration, agent).
#[allow(clippy::too_many_arguments)]
pub fn compose_iteration_prompt(
    user_prompt: &str,
    iteration: u32,
    max_iterations: u32,
    agent_idx: u32,
    agents: u32,
    best_score: Option<f64>,
    best_sha: Option<&str>,
    last_score: Option<f64>,
    branch_prefix: &str,
    include_commit_suffix: bool,
) -> String {
    let mut out = String::new();
    out.push_str(user_prompt.trim());
    out.push_str(&iteration_context(
        iteration,
        max_iterations,
        agent_idx,
        agents,
        best_score,
        best_sha,
        last_score,
    ));
    if include_commit_suffix {
        out.push_str(&auto_commit_suffix(iteration, agent_idx, branch_prefix));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

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
        let p =
            compose_iteration_prompt("Do X", 1, 10, 0, 1, None, None, None, "deltastack/", true);
        assert!(p.starts_with("Do X"), "{p}");
        assert!(p.contains("git commit"), "{p}");
    }

    #[test]
    fn compose_without_suffix_omits_git_commit() {
        let p =
            compose_iteration_prompt("Do X", 1, 10, 0, 1, None, None, None, "deltastack/", false);
        assert!(!p.contains("git commit"), "{p}");
    }

    #[test]
    fn compose_trims_user_prompt() {
        let p = compose_iteration_prompt("  Do X\n\n", 1, 2, 0, 1, None, None, None, "x/", false);
        assert!(p.starts_with("Do X\n\n---"), "{p}");
    }
}
