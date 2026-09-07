mod agent;
mod cli;
mod eval;
mod git;
mod orchestrator;
mod prompt;
mod state;

use std::path::{Path, PathBuf};
use std::time::Duration;

use clap::Parser;
use tracing::{info, warn};

use cli::Cli;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();
    init_logging(&cli);
    cli.validate()?;

    let user_prompt = prompt::load_user_prompt(cli.prompt.as_deref(), cli.prompt_file.as_deref())?;
    let repo = std::env::current_dir()?;

    if cli.dry_run {
        return dry_run(&cli, &user_prompt);
    }

    if !git::is_git_repo(&repo).await {
        anyhow::bail!(
            "current directory is not a git repo (deltastack requires git for keep/revert)"
        );
    }
    if !cli.allow_dirty && !git::is_clean_filtered(&repo, &cli.state_file, &cli.log_dir).await? {
        anyhow::bail!("git tree is dirty; commit/stash first or pass --allow-dirty");
    }
    if cli.wt_auto_merge && cli.uses_worktree_isolation() {
        warn!(
            "--wt-auto-merge is ON: zerostack will merge worktrees on exit BEFORE eval gating. Prefer OFF."
        );
    }
    if !cli.uses_worktree_isolation()
        && (cli.wt_base_dir.is_some() || cli.wt_force || cli.wt_auto_merge)
    {
        warn!(
            "worktree flags (--wt-base-dir/--wt-force/--wt-auto-merge) are ignored with --agents 1 (single-agent runs execute in-place)"
        );
    }

    tokio::fs::create_dir_all(&cli.log_dir).await?;

    // ---- baseline ----
    let baseline_sha = git::current_sha(&repo).await?;
    info!("baseline commit {baseline_sha}");
    let baseline_samples = eval::run_eval_samples(
        &cli.eval_cmd,
        repo.clone(),
        cli.samples,
        cli.effective_eval_jobs(),
        eval_timeout(&cli),
        Some(PathBuf::from(&cli.log_dir)),
        0,
        99,
    )
    .await;
    let baseline_scores: Vec<Option<f64>> = baseline_samples.iter().map(|s| s.score).collect();
    let mut best = orchestrator::score_candidate(&baseline_samples, cli.aggregate);
    let mut best_sha = baseline_sha.clone();
    info!("baseline samples={baseline_scores:?} agg={best:?}");

    state::append_record(
        &cli.state_file,
        &state::IterationRecord {
            iteration: 0,
            agent_idx: 0,
            kind: state::RecordKind::Baseline,
            commit: Some(baseline_sha.clone()),
            worktree: None,
            samples: baseline_scores,
            agg: best,
            best,
            best_sha: Some(best_sha.clone()),
            decision: Some("baseline".into()),
            agent_exit: None,
            note: None,
        },
    )
    .await?;

    let mut fails_since_best: u32 = 0;
    let mut last_score = best;

    for iter in 1..=cli.max_iterations {
        info!(
            "=== iteration {iter}/{} (agents={}) ===",
            cli.max_iterations, cli.agents
        );

        // Spawn agents (concurrent). Each builds its own prompt; isolation via zerostack --worktree.
        let mut handles = Vec::new();
        for j in 0..cli.agents {
            let c = cli.clone();
            let up = user_prompt.clone();
            let log_dir = PathBuf::from(&cli.log_dir);
            let repo_c = repo.clone();
            let best_s = best;
            let best_sha_c = best_sha.clone();
            let last_s = last_score;
            handles.push(tokio::spawn(async move {
                run_single_agent(
                    &c,
                    &repo_c,
                    &up,
                    iter,
                    j,
                    best_s,
                    Some(best_sha_c),
                    last_s,
                    &log_dir,
                )
                .await
            }));
        }

        // Collect per-agent outcomes.
        let mut outcomes: Vec<AgentOutcome> = Vec::new();
        for (j, h) in handles.into_iter().enumerate() {
            match h.await {
                Err(e) => {
                    warn!("agent {j} panicked: {e:#}");
                    outcomes.push(AgentOutcome {
                        agent_idx: j as u32,
                        commit: None,
                        worktree_path: None,
                        branch: None,
                        samples: vec![],
                        agg: None,
                        agent_exit: None,
                    });
                }
                Ok(Ok(o)) => outcomes.push(o),
                Ok(Err(e)) => {
                    warn!("agent {j} failed: {e:#}");
                    outcomes.push(AgentOutcome {
                        agent_idx: j as u32,
                        commit: None,
                        worktree_path: None,
                        branch: None,
                        samples: vec![],
                        agg: None,
                        agent_exit: None,
                    });
                }
            }
        }

        // Winner selection (mode-aware; failures sort last).
        let cands: Vec<orchestrator::CandidateScore> = outcomes
            .iter()
            .map(|o| orchestrator::CandidateScore {
                agent_idx: o.agent_idx,
                score: o.agg,
            })
            .collect();
        let winner_idx = orchestrator::select_winner(&cands, cli.mode);
        let winner = winner_idx.and_then(|w| outcomes.iter().find(|o| o.agent_idx == w));

        let mut iter_best = best;
        let mut decision_global = "revert";
        if let Some(w) = winner {
            if let Some(score) = w.agg {
                let improves = match best {
                    Some(b) => {
                        orchestrator::is_improvement(score, b, cli.mode, cli.min_improvement)
                    }
                    None => true, // no baseline (all failed) -> first score wins
                };
                if improves {
                    // Promote: reset main checkout to winner commit (worktree or in-place).
                    if cli.uses_worktree_isolation() {
                        if let Some(commit) = &w.commit {
                            git::reset_hard(&repo, commit).await?;
                            best = Some(score);
                            best_sha = commit.clone();
                            iter_best = best;
                            decision_global = "keep";
                        }
                    } else {
                        // in-place single agent: HEAD already at candidate commit
                        best = Some(score);
                        best_sha = w.commit.clone().unwrap_or_else(|| best_sha.clone());
                        iter_best = best;
                        decision_global = "keep";
                    }
                    fails_since_best = 0;
                    last_score = Some(score);
                } else {
                    // Revert in-place; worktrees are simply dropped.
                    if !cli.uses_worktree_isolation() && !cli.no_revert {
                        git::reset_hard(&repo, &best_sha).await?;
                    }
                    fails_since_best += 1;
                    last_score = Some(score);
                }
            } else {
                // Winner has no score (all agents failed).
                if !cli.uses_worktree_isolation() && !cli.no_revert {
                    git::reset_hard(&repo, &best_sha).await?;
                }
                fails_since_best += 1;
            }
        }

        // Log + cleanup worktrees.
        for o in &outcomes {
            let is_winner = winner.map(|w| w.agent_idx == o.agent_idx).unwrap_or(false);
            let decision = if is_winner { decision_global } else { "revert" };
            state::append_record(
                &cli.state_file,
                &state::IterationRecord::candidate(
                    iter,
                    o.agent_idx,
                    o.commit.clone(),
                    o.worktree_path
                        .as_ref()
                        .map(|p| p.to_string_lossy().to_string()),
                    o.samples.iter().map(|s| s.score).collect(),
                    o.agg,
                    iter_best,
                    Some(best_sha.clone()),
                    decision,
                    o.agent_exit,
                ),
            )
            .await?;
            if cli.uses_worktree_isolation() && !cli.keep_worktrees {
                if let Some(p) = &o.worktree_path {
                    let _ = git::remove_worktree(&repo, p, cli.wt_force).await;
                }
                if let Some(b) = &o.branch {
                    // Winner branch now equals main HEAD; still safe to delete.
                    let _ = git::delete_branch(&repo, b, true).await;
                }
            }
        }

        info!("iter {iter}: best={best:?} ({best_sha}) fails_since_best={fails_since_best}");

        match orchestrator::should_stop(
            iter,
            cli.max_iterations,
            fails_since_best,
            cli.patience,
            winner.and_then(|w| w.agg),
            cli.target,
            cli.mode,
        ) {
            orchestrator::StopReason::Continue => {}
            r => {
                info!("stopping: {r:?}");
                break;
            }
        }
    }

    info!("done. best={best:?} sha={best_sha}");
    println!("best={best:?} sha={best_sha}");
    Ok(())
}

struct AgentOutcome {
    agent_idx: u32,
    commit: Option<String>,
    worktree_path: Option<PathBuf>,
    branch: Option<String>,
    samples: Vec<eval::EvalSample>,
    agg: Option<f64>,
    agent_exit: Option<i32>,
}

/// Run one agent end-to-end: prompt -> zerostack -> commit normalize -> eval samples.
#[allow(clippy::too_many_arguments)]
async fn run_single_agent(
    cli: &Cli,
    repo: &Path,
    user_prompt: &str,
    iteration: u32,
    agent_idx: u32,
    best: Option<f64>,
    best_sha: Option<String>,
    last_score: Option<f64>,
    log_dir: &Path,
) -> anyhow::Result<AgentOutcome> {
    let full_prompt = prompt::compose_iteration_prompt(
        user_prompt,
        iteration,
        cli.max_iterations,
        agent_idx,
        cli.agents,
        best,
        best_sha.as_deref(),
        last_score,
        &cli.branch_prefix,
        !cli.no_auto_commit_prompt,
    );

    // Isolation: let zerostack create the worktree via its integrated flag.
    // We resolve the path afterwards via `git worktree list`.
    let branch = if cli.uses_worktree_isolation() {
        Some(crate::git::worktree_branch_name(
            &cli.branch_prefix,
            iteration,
            agent_idx,
        ))
    } else {
        None
    };

    let agent_log = log_dir.join(format!("agent-{iteration}-{agent_idx}.log"));
    let res = agent::run_agent(cli, repo, iteration, agent_idx, &full_prompt, &agent_log).await?;

    // Resolve execution dir: worktree path if zerostack created one, else repo.
    let (exec_dir, worktree_path) = if cli.uses_worktree_isolation() {
        match resolve_agent_worktree(repo, cli, iteration, agent_idx).await {
            Some(p) => (p.clone(), Some(p)),
            None => {
                // Fallback: zerostack may have failed before creating the worktree.
                // Run eval in-place and continue.
                warn!("agent {agent_idx}: worktree not found after run; evaluating in-place");
                (repo.to_path_buf(), None)
            }
        }
    } else {
        (repo.to_path_buf(), None)
    };

    // Normalize commit: detect new HEAD commit, else fallback commit if dirty.
    let commit = normalize_commit(&exec_dir, cli, iteration, agent_idx).await?;

    // Eval samples in the candidate dir.
    let samples = eval::run_eval_samples(
        &cli.eval_cmd,
        exec_dir,
        cli.samples,
        cli.effective_eval_jobs(),
        eval_timeout(cli),
        Some(log_dir.to_path_buf()),
        iteration,
        agent_idx,
    )
    .await;
    let agg = orchestrator::score_candidate(&samples, cli.aggregate);

    Ok(AgentOutcome {
        agent_idx,
        commit,
        worktree_path,
        branch,
        samples,
        agg,
        agent_exit: res.exit_code,
    })
}

/// Find the worktree zerostack created for this agent.
async fn resolve_agent_worktree(
    repo: &Path,
    cli: &Cli,
    iteration: u32,
    agent_idx: u32,
) -> Option<PathBuf> {
    let branch = crate::git::worktree_branch_name(&cli.branch_prefix, iteration, agent_idx);
    let list = git::list_worktrees(repo).await.ok()?;
    git::find_worktree_for_branch(&list, &branch)
}

/// If the agent committed, return new HEAD; if dirty and fallback allowed, commit; else current HEAD/None.
/// Ignores deltastack's own state file + log dir so logs never trigger commits.
async fn normalize_commit(
    exec_dir: &Path,
    cli: &Cli,
    iteration: u32,
    agent_idx: u32,
) -> anyhow::Result<Option<String>> {
    let head = git::current_sha(exec_dir).await.ok();
    let clean = git::is_clean_filtered(exec_dir, &cli.state_file, &cli.log_dir)
        .await
        .unwrap_or(true);
    if !clean && !cli.no_auto_commit_fallback {
        let msg = format!(
            "{}iter{} agent{}: auto-commit (agent left uncommitted changes)",
            cli.branch_prefix, iteration, agent_idx
        );
        let sha =
            git::fallback_commit_filtered(exec_dir, &msg, &cli.state_file, &cli.log_dir).await?;
        return Ok(Some(sha));
    }
    Ok(head)
}

fn eval_timeout(cli: &Cli) -> Duration {
    if cli.eval_timeout == 0 {
        Duration::ZERO
    } else {
        Duration::from_secs(cli.eval_timeout)
    }
}

fn init_logging(cli: &Cli) {
    let level = if cli.quiet {
        "warn"
    } else if cli.verbose {
        "debug"
    } else {
        "info"
    };
    let _ = tracing_subscriber::fmt()
        .with_env_filter(format!("deltastack={level}"))
        .with_target(false)
        .try_init();
}

fn dry_run(cli: &Cli, user_prompt: &str) -> anyhow::Result<()> {
    let full = prompt::compose_iteration_prompt(
        user_prompt,
        1,
        cli.max_iterations,
        0,
        cli.agents,
        None,
        None,
        None,
        &cli.branch_prefix,
        !cli.no_auto_commit_prompt,
    );
    let argv = agent::build_zerostack_argv(cli, 1, 0, &full);
    println!(
        "zerostack argv:\n{} {}",
        cli.zerostack_bin,
        shlex::try_join(argv.iter().map(String::as_str)).unwrap_or_default()
    );
    println!("\neval:\nsh -c {:?}", cli.eval_cmd);
    println!(
        "\nmode={:?} aggregate={:?} agents={} samples={} isolation={}",
        cli.mode,
        cli.aggregate,
        cli.agents,
        cli.samples,
        cli.uses_worktree_isolation()
    );
    Ok(())
}
