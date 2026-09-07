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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn branch_name_format() {
        assert_eq!(
            worktree_branch_name("deltastack/", 3, 1),
            "deltastack/iter3-agent1"
        );
        assert_eq!(worktree_branch_name("exp-", 0, 0), "exp-iter0-agent0");
    }

    #[test]
    fn zerostack_worktree_name_sanitizes_slash() {
        // zerostack worktree *directory* names shouldn't contain `/`.
        assert_eq!(
            zerostack_worktree_name("deltastack/", 3, 1),
            "deltastack-iter3-agent1"
        );
    }

    #[test]
    fn parse_porcelain_two_worktrees() {
        let sample = "worktree /repo\nbranch refs/heads/main\n\nworktree /tmp/wt\nbranch refs/heads/deltastack/iter1-agent0\n\n";
        let w = parse_worktree_porcelain(sample);
        assert_eq!(w.len(), 2);
        assert_eq!(w[0].0, PathBuf::from("/repo"));
        assert_eq!(w[0].1.as_deref(), Some("main"));
        assert_eq!(w[1].1.as_deref(), Some("deltastack/iter1-agent0"));
    }

    #[test]
    fn parse_porcelain_detached_and_empty() {
        assert!(parse_worktree_porcelain("").is_empty());
        let s = "worktree /repo\nbranch (detached)\n\n";
        let w = parse_worktree_porcelain(s);
        assert_eq!(w.len(), 1);
        assert_eq!(w[0].1.as_deref(), Some("(detached)"));
    }

    #[test]
    fn find_branch_path() {
        let w = vec![
            (PathBuf::from("/repo"), Some("main".into())),
            (PathBuf::from("/wt"), Some("deltastack/iter2-agent1".into())),
        ];
        assert_eq!(
            find_worktree_for_branch(&w, "deltastack/iter2-agent1"),
            Some(PathBuf::from("/wt"))
        );
        assert_eq!(
            find_worktree_for_branch(&w, "refs/heads/deltastack/iter2-agent1"),
            Some(PathBuf::from("/wt"))
        );
        assert_eq!(find_worktree_for_branch(&w, "missing"), None);
    }

    #[test]
    fn output_ignores_relativizes_inside_repo() {
        let repo = Path::new("/repo");
        let ig = output_ignores(repo, "deltastack.jsonl", "deltastack-logs");
        assert_eq!(ig, vec!["deltastack.jsonl", "deltastack-logs"]);
    }

    #[test]
    fn output_ignores_drops_absolute_outside_repo() {
        let repo = Path::new("/repo");
        let ig = output_ignores(repo, "/tmp/run.jsonl", "logs");
        assert_eq!(ig, vec!["logs"]);
    }

    #[test]
    fn status_line_ignored_matches_file_and_dir_prefix() {
        let ig = vec![
            "deltastack.jsonl".to_string(),
            "deltastack-logs".to_string(),
        ];
        assert!(status_line_is_ignored("?? deltastack.jsonl", &ig));
        assert!(status_line_is_ignored("?? deltastack-logs/", &ig));
        assert!(status_line_is_ignored(
            " M deltastack-logs/agent-1-0.log",
            &ig
        ));
        assert!(!status_line_is_ignored(" M src/main.rs", &ig));
        assert!(!status_line_is_ignored("?? target/", &ig));
    }

    #[test]
    fn status_line_ignored_handles_rename() {
        let ig = vec!["run.jsonl".to_string()];
        assert!(status_line_is_ignored("R  old.jsonl -> run.jsonl", &ig));
        assert!(!status_line_is_ignored("R  old.jsonl -> other.rs", &ig));
    }

    #[test]
    fn filter_porcelain_keeps_real_changes() {
        let ig = vec![
            "deltastack.jsonl".to_string(),
            "deltastack-logs".to_string(),
        ];
        let raw = " M src/main.rs\n?? deltastack.jsonl\n?? deltastack-logs/\n";
        let f = filter_porcelain(raw, &ig);
        assert!(f.contains("src/main.rs"));
        assert!(!f.contains("deltastack"));
    }

    #[test]
    fn filter_porcelain_empty_when_only_outputs() {
        let ig = vec!["run.jsonl".to_string()];
        assert!(filter_porcelain("?? run.jsonl\n", &ig).trim().is_empty());
    }

    // --- integration tests against real temp git repos (no LLM) ---

    async fn init_repo() -> tempfile::TempDir {
        let dir = tempfile::tempdir().unwrap();
        for args in [
            vec!["init"],
            vec!["config", "user.email", "t@t.t"],
            vec!["config", "user.name", "t"],
        ] {
            let st = tokio::process::Command::new("git")
                .args(&args)
                .current_dir(dir.path())
                .status()
                .await
                .unwrap();
            assert!(st.success());
        }
        std::fs::write(dir.path().join("f.txt"), "v1").unwrap();
        let st = tokio::process::Command::new("git")
            .args(["add", "-A"])
            .current_dir(dir.path())
            .status()
            .await
            .unwrap();
        assert!(st.success());
        let st = tokio::process::Command::new("git")
            .args(["commit", "-m", "init"])
            .current_dir(dir.path())
            .status()
            .await
            .unwrap();
        assert!(st.success());
        dir
    }

    #[tokio::test]
    async fn git_helpers_on_temp_repo() {
        let dir = init_repo().await;
        assert!(is_git_repo(dir.path()).await);
        assert!(is_clean(dir.path()).await.unwrap());
        let sha = current_sha(dir.path()).await.unwrap();
        assert_eq!(sha.len(), 40);

        std::fs::write(dir.path().join("f.txt"), "v2").unwrap();
        assert!(!is_clean(dir.path()).await.unwrap());
        let sha2 = fallback_commit(dir.path(), "change").await.unwrap();
        assert_ne!(sha, sha2);
        assert!(is_clean(dir.path()).await.unwrap());
        reset_hard(dir.path(), &sha).await.unwrap();
        assert_eq!(current_sha(dir.path()).await.unwrap(), sha);
        assert_eq!(
            std::fs::read_to_string(dir.path().join("f.txt")).unwrap(),
            "v1"
        );
    }

    #[tokio::test]
    async fn filtered_clean_ignores_state_and_logs() {
        let dir = init_repo().await;
        std::fs::write(dir.path().join("run.jsonl"), "{}").unwrap();
        std::fs::create_dir_all(dir.path().join("logs")).unwrap();
        std::fs::write(dir.path().join("logs/a.log"), "x").unwrap();
        assert!(!is_clean(dir.path()).await.unwrap());
        assert!(is_clean_filtered(dir.path(), "run.jsonl", "logs")
            .await
            .unwrap());
        std::fs::write(dir.path().join("f.txt"), "dirty").unwrap();
        assert!(!is_clean_filtered(dir.path(), "run.jsonl", "logs")
            .await
            .unwrap());
    }

    #[tokio::test]
    async fn filtered_fallback_commit_excludes_outputs() {
        let dir = init_repo().await;
        std::fs::write(dir.path().join("f.txt"), "v2").unwrap();
        std::fs::write(dir.path().join("run.jsonl"), "{}").unwrap();
        fallback_commit_filtered(dir.path(), "msg", "run.jsonl", "logs")
            .await
            .unwrap();
        // run.jsonl must remain untracked
        let st = tokio::process::Command::new("git")
            .args(["status", "--porcelain"])
            .current_dir(dir.path())
            .output()
            .await
            .unwrap();
        let s = String::from_utf8_lossy(&st.stdout).to_string();
        assert!(s.contains("run.jsonl"), "{s}");
        assert!(!s.contains("f.txt"), "{s}");
    }

    #[tokio::test]
    async fn reset_hard_leaves_untracked_files_alone() {
        let dir = init_repo().await;
        let sha = current_sha(dir.path()).await.unwrap();
        std::fs::write(dir.path().join("f.txt"), "v2").unwrap();
        fallback_commit(dir.path(), "c2").await.unwrap();
        // untracked file created AFTER the commit: reset must preserve it
        std::fs::write(dir.path().join("notes.txt"), "untracked").unwrap();
        reset_hard(dir.path(), &sha).await.unwrap();
        // tracked file reverted; untracked file preserved (no `clean -fd`)
        assert_eq!(
            std::fs::read_to_string(dir.path().join("f.txt")).unwrap(),
            "v1"
        );
        assert!(dir.path().join("notes.txt").exists());
    }

    #[tokio::test]
    async fn worktree_add_remove_roundtrip() {
        let dir = init_repo().await;
        let wt = dir.path().join("wt0");
        let st = tokio::process::Command::new("git")
            .args([
                "worktree",
                "add",
                wt.to_str().unwrap(),
                "-b",
                "deltastack/iter0-agent0",
            ])
            .current_dir(dir.path())
            .status()
            .await
            .unwrap();
        assert!(st.success());
        let list = list_worktrees(dir.path()).await.unwrap();
        assert!(find_worktree_for_branch(&list, "deltastack/iter0-agent0").is_some());
        remove_worktree(dir.path(), &wt, true).await.unwrap();
        delete_branch(dir.path(), "deltastack/iter0-agent0", true)
            .await
            .unwrap();
        let list = list_worktrees(dir.path()).await.unwrap();
        assert!(find_worktree_for_branch(&list, "deltastack/iter0-agent0").is_none());
    }

    #[tokio::test]
    async fn is_git_repo_false_outside_repo() {
        let dir = tempfile::tempdir().unwrap();
        assert!(!is_git_repo(dir.path()).await);
    }
}
