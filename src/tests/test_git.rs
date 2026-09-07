use crate::git::*;
use std::path::{Path, PathBuf};

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
