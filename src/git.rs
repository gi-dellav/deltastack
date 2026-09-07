use std::path::{Path, PathBuf};
use std::process::Stdio;

use tokio::process::Command;

/// Deterministic worktree branch name for (iteration, agent).
pub fn worktree_branch_name(branch_prefix: &str, iteration: u32, agent_idx: u32) -> String {
    format!("{branch_prefix}iter{iteration}-agent{agent_idx}")
}

/// Deterministic zerostack `--worktree <NAME>` value.
/// zerostack creates the worktree itself; we only compute the name.
pub fn zerostack_worktree_name(branch_prefix: &str, iteration: u32, agent_idx: u32) -> String {
    // Reuse branch naming so `git branch --list` and `git worktree list` correlate.
    // Prefix may contain `/`; zerostack worktree *directory* names shouldn't.
    format!("{branch_prefix}iter{iteration}-agent{agent_idx}").replace('/', "-")
}

/// Parse `git worktree list --porcelain` into (path, branch?) entries.
/// Porcelain format repeats blocks: `worktree <path>`, optional `branch <ref>`, etc.
pub fn parse_worktree_porcelain(output: &str) -> Vec<(PathBuf, Option<String>)> {
    let mut out = Vec::new();
    let mut cur_path: Option<PathBuf> = None;
    let mut cur_branch: Option<String> = None;
    let flush = |p: &mut Option<PathBuf>,
                 b: &mut Option<String>,
                 dst: &mut Vec<(PathBuf, Option<String>)>| {
        if let Some(path) = p.take() {
            dst.push((path, b.take()));
        }
    };
    for line in output.lines() {
        if let Some(rest) = line.strip_prefix("worktree ") {
            flush(&mut cur_path, &mut cur_branch, &mut out);
            cur_path = Some(PathBuf::from(rest.trim()));
        } else if let Some(rest) = line.strip_prefix("branch ") {
            // refs/heads/<name> or (detached)
            let b = rest.trim();
            cur_branch = Some(b.strip_prefix("refs/heads/").unwrap_or(b).to_string());
        }
    }
    flush(&mut cur_path, &mut cur_branch, &mut out);
    out
}

/// Find the filesystem path of the worktree hosting `branch`.
pub fn find_worktree_for_branch(
    worktrees: &[(PathBuf, Option<String>)],
    branch: &str,
) -> Option<PathBuf> {
    let short = branch.strip_prefix("refs/heads/").unwrap_or(branch);
    worktrees
        .iter()
        .find(|(_, b)| b.as_deref() == Some(short))
        .map(|(p, _)| p.clone())
}

/// Compute repo-relative ignore prefixes for deltastack's own outputs
/// (state file + log dir) so they never pollute clean-checks or commits.
/// Absolute paths outside the repo yield no ignores.
pub fn output_ignores(repo: &Path, state_file: &str, log_dir: &str) -> Vec<String> {
    let mut out = Vec::new();
    for raw in [state_file, log_dir] {
        let p = Path::new(raw);
        let abs = if p.is_absolute() {
            p.to_path_buf()
        } else {
            repo.join(p)
        };
        if let Ok(rel) = abs.strip_prefix(repo) {
            let s = rel.to_string_lossy().replace('\\', "/");
            if !s.is_empty() {
                out.push(s);
            }
        }
    }
    out
}

/// True if a `git status --porcelain` line refers to an ignored output path.
/// Porcelain v1: `XY <path>[ -> <new>]`. Untracked dirs show as `<dir>/`.
pub fn status_line_is_ignored(line: &str, ignores: &[String]) -> bool {
    if line.len() < 4 {
        return false;
    }
    let mut path = line[3..].trim();
    // Handle renames: `R  old -> new` — check the new path.
    if let Some((_, new)) = path.split_once(" -> ") {
        path = new.trim().trim_matches('"');
    } else {
        path = path.trim_matches('"');
    }
    ignores.iter().any(|ig| {
        let ig = ig.trim_start_matches("./");
        let path = path.trim_start_matches("./");
        path == ig || path.starts_with(&format!("{ig}/"))
    })
}

/// Filter porcelain output, dropping deltastack's own state/log paths.
pub fn filter_porcelain(output: &str, ignores: &[String]) -> String {
    output
        .lines()
        .filter(|l| !status_line_is_ignored(l, ignores))
        .collect::<Vec<_>>()
        .join("\n")
}

async fn git(repo: &Path, args: &[&str]) -> anyhow::Result<std::process::Output> {
    let out = Command::new("git")
        .args(args)
        .current_dir(repo)
        .stdin(Stdio::null())
        .output()
        .await?;
    Ok(out)
}

pub async fn is_git_repo(dir: &Path) -> bool {
    git(dir, &["rev-parse", "--git-dir"])
        .await
        .map(|o| o.status.success())
        .unwrap_or(false)
}

pub async fn current_sha(repo: &Path) -> anyhow::Result<String> {
    let out = git(repo, &["rev-parse", "HEAD"]).await?;
    if !out.status.success() {
        anyhow::bail!(
            "git rev-parse HEAD failed: {}",
            String::from_utf8_lossy(&out.stderr)
        );
    }
    Ok(String::from_utf8_lossy(&out.stdout).trim().to_string())
}

pub async fn porcelain_status(repo: &Path) -> anyhow::Result<String> {
    let out = git(repo, &["status", "--porcelain"]).await?;
    if !out.status.success() {
        anyhow::bail!(
            "git status failed: {}",
            String::from_utf8_lossy(&out.stderr)
        );
    }
    Ok(String::from_utf8_lossy(&out.stdout).to_string())
}

#[allow(dead_code)] // legacy helper; filtered variant preferred in binary. Used in tests.
pub async fn is_clean(repo: &Path) -> anyhow::Result<bool> {
    Ok(porcelain_status(repo).await?.trim().is_empty())
}

/// Like `is_clean` but ignores deltastack's own state file + log dir.
pub async fn is_clean_filtered(
    repo: &Path,
    state_file: &str,
    log_dir: &str,
) -> anyhow::Result<bool> {
    let raw = porcelain_status(repo).await?;
    let ignores = output_ignores(repo, state_file, log_dir);
    Ok(filter_porcelain(&raw, &ignores).trim().is_empty())
}

/// `git worktree list --porcelain` from `repo`.
pub async fn list_worktrees(repo: &Path) -> anyhow::Result<Vec<(PathBuf, Option<String>)>> {
    let out = git(repo, &["worktree", "list", "--porcelain"]).await?;
    if !out.status.success() {
        anyhow::bail!(
            "git worktree list failed: {}",
            String::from_utf8_lossy(&out.stderr)
        );
    }
    Ok(parse_worktree_porcelain(&String::from_utf8_lossy(
        &out.stdout,
    )))
}

/// Remove a worktree created for a candidate. Best-effort; returns stderr on failure.
pub async fn remove_worktree(repo: &Path, path: &Path, force: bool) -> anyhow::Result<()> {
    let mut args = vec!["worktree", "remove"];
    if force {
        args.push("--force");
    }
    let path_s = path.to_string_lossy().to_string();
    let args2: Vec<&str> = args
        .iter()
        .copied()
        .chain(std::iter::once(path_s.as_str()))
        .collect();
    let out = git(repo, &args2).await?;
    if !out.status.success() {
        anyhow::bail!(
            "git worktree remove failed: {}",
            String::from_utf8_lossy(&out.stderr)
        );
    }
    Ok(())
}

/// Delete a branch (used to drop losing candidates).
pub async fn delete_branch(repo: &Path, branch: &str, force: bool) -> anyhow::Result<()> {
    let flag = if force { "-D" } else { "-d" };
    let out = git(repo, &["branch", flag, branch]).await?;
    if !out.status.success() {
        anyhow::bail!(
            "git branch delete failed: {}",
            String::from_utf8_lossy(&out.stderr)
        );
    }
    Ok(())
}

/// True if `sha` exists in the repo (used to validate a resume point).
pub async fn commit_exists(repo: &Path, sha: &str) -> bool {
    git(repo, &["cat-file", "-e", sha])
        .await
        .map(|o| o.status.success())
        .unwrap_or(false)
}

/// Promote winner: point current checkout at `sha` (used after eval gate passes).
/// NOTE: intentionally does NOT `git clean -fd` — untracked files (e.g. build
/// outputs, and deltastack's own logs) are left alone; only tracked state reverts.
pub async fn reset_hard(repo: &Path, sha: &str) -> anyhow::Result<()> {
    let out = git(repo, &["reset", "--hard", sha]).await?;
    if !out.status.success() {
        anyhow::bail!(
            "git reset --hard failed: {}",
            String::from_utf8_lossy(&out.stderr)
        );
    }
    Ok(())
}

/// Fallback commit when the agent left a dirty tree but no commit.
/// Stages everything EXCEPT deltastack's own state file + log dir.
pub async fn fallback_commit_filtered(
    repo: &Path,
    message: &str,
    state_file: &str,
    log_dir: &str,
) -> anyhow::Result<String> {
    let ignores = output_ignores(repo, state_file, log_dir);
    // Stage all, then unstage our own outputs (simpler than pathspec negation across git versions).
    let out = git(repo, &["add", "-A"]).await?;
    if !out.status.success() {
        anyhow::bail!(
            "git add -A failed: {}",
            String::from_utf8_lossy(&out.stderr)
        );
    }
    for ig in &ignores {
        let _ = git(repo, &["reset", "-q", "--", ig]).await?;
    }
    let out = Command::new("git")
        .args(["commit", "-m", message])
        .current_dir(repo)
        .output()
        .await?;
    if !out.status.success() {
        anyhow::bail!(
            "git commit failed: {}",
            String::from_utf8_lossy(&out.stderr)
        );
    }
    current_sha(repo).await
}

/// Legacy fallback commit (no ignores). Prefer `fallback_commit_filtered`.
#[allow(dead_code)] // kept for tests/back-compat
pub async fn fallback_commit(repo: &Path, message: &str) -> anyhow::Result<String> {
    let out = git(repo, &["add", "-A"]).await?;
    if !out.status.success() {
        anyhow::bail!(
            "git add -A failed: {}",
            String::from_utf8_lossy(&out.stderr)
        );
    }
    let out = Command::new("git")
        .args(["commit", "-m", message])
        .current_dir(repo)
        .output()
        .await?;
    if !out.status.success() {
        anyhow::bail!(
            "git commit failed: {}",
            String::from_utf8_lossy(&out.stderr)
        );
    }
    current_sha(repo).await
}
