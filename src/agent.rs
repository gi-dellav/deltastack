use std::path::Path;
use std::time::Duration;

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

/// Build the `zerostack` argv for one (iteration, agent).
///
/// Multi-agent isolation uses zerostack's *integrated* workflow flag:
///
/// - default: `--worktree <deterministic-name>` (zerostack creates the worktree)
///
/// Single-agent in-place passes neither (unless `--wt-base-dir` forces isolation).
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
    if let Some(v) = cli.temperature {
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cli::Cli;
    use clap::Parser;

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
        let c = cli(&["--no-isolate", "--wt-auto-merge"]);
        let argv = build_zerostack_argv(&c, 0, 0, "p");
        // isolation disabled -> no worktree machinery forwarded
        assert!(!argv
            .iter()
            .any(|a| a == "--worktree" || a == "--wt-auto-merge"));
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
}
