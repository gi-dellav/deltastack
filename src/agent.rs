use std::path::Path;
use std::time::Duration;

use rand::RngExt;
use tokio::process::Command;

use crate::cli::Cli;
use crate::git::zerostack_worktree_name;

/// Split `--tools a,b --tools c` into individual tool names.
pub fn flatten_tools(tools: &[String]) -> Vec<String> {
    tools
        .iter()
        .flat_map(|s| s.split(','))
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(str::to_string)
        .collect()
}

/// Resolve the effective temperature for one (iteration, agent).
///
/// Rules:
/// - If `--temperature-min`/`--temperature-max` are set, draw a uniform
///   random base in `[min, max]` (per agent run; `rng` decides determinism).
/// - Otherwise the base is `--temperature` (None => no temperature flag).
/// - If `--temperature-step` is set, `real_temp = base + step * current_step`
///   with `current_step = iteration` (iterations start at 1).
///
/// All flags accept any finite f64, positive or negative.
pub fn resolve_temperature(cli: &Cli, iteration: u32, rng: &mut impl RngExt) -> Option<f64> {
    let base = match (cli.temperature_min, cli.temperature_max) {
        (Some(min), Some(max)) => Some(rng.random_range(min..=max)),
        _ => cli.temperature,
    }?;
    match cli.temperature_step {
        Some(step) => Some(base + step * f64::from(iteration.max(1))),
        None => Some(base),
    }
}

/// Build the `zerostack` argv for one (iteration, agent).
///
/// Multi-agent isolation uses zerostack's *integrated* workflow flag:
///
/// - with `--agents > 1`: `--worktree <deterministic-name>` (zerostack creates the worktree)
///
/// Single-agent (`--agents 1`) always runs in-place and passes no worktree flags.
#[allow(clippy::too_many_arguments)]
pub fn build_zerostack_argv(
    cli: &Cli,
    iteration: u32,
    agent_idx: u32,
    full_prompt: &str,
) -> Vec<String> {
    let mut argv: Vec<String> = vec!["-p".into(), full_prompt.to_string()];

    // Sessions are ephemeral by default (`--no-session`); opt into named
    // sessions with --keep-agent-session (`--name ...` for `zerostack --resume`).
    if cli.keep_agent_session {
        argv.push("--name".into());
        argv.push(format!(
            "{}-iter{}-agent{}",
            cli.agent_session_prefix, iteration, agent_idx
        ));
    } else {
        argv.push("--no-session".into());
    }

    // passthrough model/limits
    if let Some(v) = &cli.provider {
        argv.push("--provider".into());
        argv.push(v.clone());
    }
    if let Some(v) = &cli.model {
        argv.push("--model".into());
        argv.push(v.clone());
    }
    if let Some(v) = &cli.quick_model {
        argv.push("--quick-model".into());
        argv.push(v.clone());
    }
    if let Some(v) = cli.max_tokens {
        argv.push("--max-tokens".into());
        argv.push(v.to_string());
    }
    if let Some(v) = cli.max_agent_turns {
        argv.push("--max-agent-turns".into());
        argv.push(v.to_string());
    }
    if let Some(v) = resolve_temperature(cli, iteration, &mut rand::rng()) {
        argv.push("--temperature".into());
        argv.push(v.to_string());
    }
    let tools = flatten_tools(&cli.tools);
    for t in tools {
        argv.push("--tools".into());
        argv.push(t);
    }
    if cli.no_context_files {
        argv.push("--no-context-files".into());
    }
    if cli.accept_all {
        argv.push("--accept-all".into());
    }
    if cli.yolo {
        argv.push("--yolo".into());
    }
    if cli.dangerously_skip_permissions {
        argv.push("--dangerously-skip-permissions".into());
    }
    if cli.sandbox {
        argv.push("--sandbox".into());
    }
    if let Some(v) = cli.sandbox_network {
        argv.push("--sandbox-network".into());
        argv.push(if v { "true".into() } else { "false".into() });
    }
    if let Some(v) = &cli.shell {
        argv.push("--shell".into());
        argv.push(v.clone());
    }

    // ---- integrated workflow flags (isolation) ----
    if cli.uses_worktree_isolation() {
        argv.push("--worktree".into());
        argv.push(zerostack_worktree_name(
            &cli.branch_prefix,
            iteration,
            agent_idx,
        ));
        if cli.wt_auto_merge {
            argv.push("--wt-auto-merge".into());
        }
        if cli.wt_force {
            argv.push("--wt-force".into());
        }
        if let Some(d) = &cli.wt_base_dir {
            argv.push("--wt-base-dir".into());
            argv.push(d.clone());
        }
    }

    argv
}

#[derive(Debug, Clone)]
#[allow(dead_code)] // timed_out/stdout_tail kept for JSONL observability
pub struct AgentResult {
    pub exit_code: Option<i32>,
    pub timed_out: bool,
    pub stdout_tail: String,
}

/// Run one zerostack agent; stream output to `log_file`.
pub async fn run_agent(
    cli: &Cli,
    workdir: &Path,
    iteration: u32,
    agent_idx: u32,
    full_prompt: &str,
    log_file: &Path,
) -> anyhow::Result<AgentResult> {
    let argv = build_zerostack_argv(cli, iteration, agent_idx, full_prompt);

    if let Some(parent) = log_file.parent() {
        tokio::fs::create_dir_all(parent).await?;
    }

    let mut cmd = Command::new(&cli.zerostack_bin);
    cmd.args(&argv).current_dir(workdir);
    cmd.stdout(std::process::Stdio::piped());
    cmd.stderr(std::process::Stdio::piped());

    let timeout = if cli.agent_timeout == 0 {
        Duration::ZERO
    } else {
        Duration::from_secs(cli.agent_timeout)
    };

    let run = async {
        let out = cmd.output().await?;
        anyhow::Ok(out)
    };

    let (timed_out, output) = if timeout.is_zero() {
        (false, run.await?)
    } else {
        match tokio::time::timeout(timeout, run).await {
            Ok(Ok(o)) => (false, o),
            Ok(Err(e)) => return Err(e),
            Err(_) => {
                let body = format!(
                    "agent timed out after {}s\nprompt:\n{full_prompt}\n",
                    cli.agent_timeout
                );
                let _ = tokio::fs::write(log_file, body).await;
                return Ok(AgentResult {
                    exit_code: None,
                    timed_out: true,
                    stdout_tail: String::new(),
                });
            }
        }
    };

    let stdout = String::from_utf8_lossy(&output.stdout).to_string();
    let stderr = String::from_utf8_lossy(&output.stderr).to_string();
    let body = format!(
        "$ {} {}\n--- stdout ---\n{stdout}\n--- stderr ---\n{stderr}\n",
        cli.zerostack_bin,
        shlex::try_join(argv.iter().map(String::as_str)).unwrap_or_default()
    );
    tokio::fs::write(log_file, body).await?;

    Ok(AgentResult {
        exit_code: output.status.code(),
        timed_out,
        stdout_tail: stdout
            .chars()
            .rev()
            .take(2000)
            .collect::<String>()
            .chars()
            .rev()
            .collect(),
    })
}
