use std::fs;

/// Load the user task prompt from `--prompt` or `--prompt-file`.
/// Kept for the plain `--eval` path; optimize modes use [`resolve_user_prompt`].
#[allow(dead_code)]
pub fn load_user_prompt(prompt: Option<&str>, prompt_file: Option<&str>) -> anyhow::Result<String> {
    match (prompt, prompt_file) {
        (Some(p), None) => Ok(p.to_string()),
        (None, Some(f)) => Ok(fs::read_to_string(f)?),
        (None, None) => anyhow::bail!("one of --prompt or --prompt-file is required"),
        (Some(_), Some(_)) => anyhow::bail!("--prompt conflicts with --prompt-file"),
    }
}

/// Default task prompt for `--optimize-speed <cmd>`: minimize wall-clock time.
pub fn default_optimize_speed_prompt(cmd: &str) -> String {
    format!(
        "Optimize the wall-clock running time of this command:\n`{cmd}`\n\n\
         Edit the repository so the command runs faster. The score is the elapsed \
         time in seconds (lower is better), aggregated over --samples runs. \
         Keep the behavior/output of the command correct — do not break it, stub it out, \
         skip work, or suppress output just to look faster. Only genuine speedups count."
    )
}

/// Default task prompt for `--optimize-memory <cmd>`: minimize peak RSS.
pub fn default_optimize_memory_prompt(cmd: &str) -> String {
    format!(
        "Optimize the peak memory usage of this command:\n`{cmd}`\n\n\
         Edit the repository so the command uses less memory. The score is the peak \
         resident set size in kilobytes measured with GNU `time -v` (lower is better), \
         aggregated over --samples runs. Keep the behavior/output of the command correct — \
         do not break it, stub it out, skip work, or suppress output just to look leaner. \
         Only genuine memory reductions count."
    )
}

/// Resolve the effective task prompt: explicit --prompt/--prompt-file wins,
/// otherwise fall back to the built-in optimize default (if any).
/// Returns the prompt plus whether it came from the built-in default.
pub fn resolve_user_prompt(
    prompt: Option<&str>,
    prompt_file: Option<&str>,
    default_prompt: Option<&str>,
) -> anyhow::Result<(String, bool)> {
    match (prompt, prompt_file) {
        (Some(p), None) => Ok((p.to_string(), false)),
        (None, Some(f)) => Ok((fs::read_to_string(f)?, false)),
        (None, None) => match default_prompt {
            Some(d) => Ok((d.to_string(), true)),
            None => anyhow::bail!("one of --prompt or --prompt-file is required"),
        },
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
