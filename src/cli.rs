use clap::{Parser, ValueEnum};

/// deltastack: eval-gated autoresearch loop on top of zerostack.
///
/// Each iteration spawns fresh `zerostack -p` agents, scores candidates with
/// your eval script (stdout = single float), keeps the best and reverts the rest.
#[derive(Parser, Debug, Clone)]
#[command(name = "deltastack", version, about)]
pub struct Cli {
    // ---------- Task definition ----------
    /// Inline task prompt (conflicts with --prompt-file).
    #[arg(short, long, conflicts_with = "prompt_file")]
    pub prompt: Option<String>,

    /// Read task prompt from file.
    #[arg(long, conflicts_with = "prompt")]
    pub prompt_file: Option<String>,

    /// Eval shell command (run via `sh -c`). Stdout must be a single float.
    #[arg(short, long = "eval", conflicts_with_all = ["optimize_speed", "optimize_memory"])]
    pub eval_cmd: Option<String>,

    /// Optimize wall-clock time of CMD (run via `sh -c`). Score = seconds (lower is better).
    /// Replaces --eval; provides a default prompt (override with --prompt/--prompt-file).
    #[arg(long, value_name = "CMD", conflicts_with_all = ["eval_cmd", "optimize_memory"])]
    pub optimize_speed: Option<String>,

    /// Optimize peak memory of CMD (run via `sh -c` under GNU `time -v`).
    /// Score = peak RSS in kilobytes (lower is better). Replaces --eval;
    /// provides a default prompt (override with --prompt/--prompt-file).
    #[arg(long, value_name = "CMD", conflicts_with_all = ["eval_cmd", "optimize_speed"])]
    pub optimize_memory: Option<String>,

    /// zerostack binary path.
    #[arg(long, default_value = "zerostack")]
    pub zerostack_bin: String,

    // ---------- Loop / eval control ----------
    /// Max outer-loop iterations.
    #[arg(long, default_value_t = 10)]
    pub max_iterations: u32,

    /// Concurrent agents per iteration (each runs in its own worktree via zerostack's integrated --worktree flag).
    #[arg(long, default_value_t = 1)]
    pub agents: u32,

    /// Measurements (eval runs) per candidate.
    #[arg(long, default_value_t = 1)]
    pub samples: u32,

    /// Concurrent eval processes. 0 = auto (min(samples, agents*samples)).
    #[arg(long, default_value_t = 0)]
    pub eval_jobs: u32,

    /// How to aggregate multiple samples into one score.
    #[arg(long, value_enum, default_value_t = Aggregate::Mean)]
    pub aggregate: Aggregate,

    /// Whether higher or lower eval scores are better.
    #[arg(long, value_enum, default_value_t = Mode::Minimize)]
    pub mode: Mode,

    /// Minimum improvement over global best to count as `keep`.
    #[arg(long, default_value_t = 0.0)]
    pub min_improvement: f64,

    /// Minimum relative improvement over global best to count as `keep`
    /// (fraction, e.g. 0.01 = 1%). Combined with --min-improvement:
    /// candidate must satisfy BOTH when both are non-zero. Relative is
    /// computed against |best| (falls back to absolute-only when best == 0).
    #[arg(long, default_value_t = 0.0)]
    pub min_improvement_rel: f64,

    /// Stop after N consecutive iterations without improvement (0 = disabled).
    #[arg(long, default_value_t = 0)]
    pub patience: u32,

    /// Early-stop once this score is reached/exceeded (mode-aware).
    #[arg(long)]
    pub target: Option<f64>,

    /// Target must be reached N consecutive iterations before stopping (default 1).
    /// Only meaningful with --target; higher values guard against lucky samples.
    #[arg(long, default_value_t = 1)]
    pub target_sticky: u32,

    /// Timeout per agent run in seconds (0 = none).
    #[arg(long, default_value_t = 0)]
    pub agent_timeout: u64,

    /// Timeout per single eval run in seconds (0 = none).
    #[arg(long, default_value_t = 600)]
    pub eval_timeout: u64,

    /// Additional retries per failed eval (score parse or non-zero exit).
    /// Total attempts per sample = 1 + retries.
    #[arg(long, default_value_t = 0)]
    pub eval_retries: u32,

    /// Delay between eval invocations in seconds (0 = none).
    /// Applied between retry attempts and staggered between samples.
    #[arg(long, default_value_t = 0)]
    pub delay_between_eval: u64,

    /// Max total wall-clock time for the whole run in seconds (0 = none).
    /// Checked between iterations; the current iteration is allowed to finish.
    #[arg(long, default_value_t = 0)]
    pub max_wall_time: u64,

    /// Delay between outer-loop iterations in seconds (0 = none).
    #[arg(long, default_value_t = 0)]
    pub delay_between_iterations: u64,

    /// Disable git auto-revert on non-improving iterations.
    #[arg(long, default_value_t = false)]
    pub no_revert: bool,

    /// Allow starting with a dirty git tree (default: require clean).
    #[arg(long, default_value_t = false)]
    pub allow_dirty: bool,

    /// Branch prefix for per-agent worktree branches.
    #[arg(long, default_value = "deltastack/")]
    pub branch_prefix: String,

    /// Keep failed/candidate worktrees for debugging (default: remove).
    #[arg(long, default_value_t = false)]
    pub keep_worktrees: bool,

    // ---------- zerostack agent passthrough (explicit subset) ----------
    /// set-up model and providers used
    #[arg(long)]
    pub provider: Option<String>,
    /// set-up model and providers used
    #[arg(long)]
    pub model: Option<String>,
    /// set-up model from a set of configured provider-model combinations
    #[arg(long)]
    pub quick_model: Option<String>,
    /// max tokens for the model response
    #[arg(long)]
    pub max_tokens: Option<u32>,
    /// max turns for the coding agent
    #[arg(long)]
    pub max_agent_turns: Option<u32>,
    /// sampling temperature for the model
    #[arg(long, allow_hyphen_values = true)]
    pub temperature: Option<f64>,
    /// minimum sampling temperature for randomly generated temperatures
    /// (requires --temperature-max; per-agent uniform random in [min, max]).
    #[arg(long, allow_hyphen_values = true)]
    pub temperature_min: Option<f64>,
    /// maximum sampling temperature for randomly generated temperatures
    /// (requires --temperature-min; per-agent uniform random in [min, max]).
    #[arg(long, allow_hyphen_values = true)]
    pub temperature_max: Option<f64>,
    /// per-iteration temperature step: real_temp = base + step * current_step,
    /// with current_step = iteration starting at 1. Base is the random
    /// temperature when --temperature-min/--temperature-max are set,
    /// otherwise --temperature. Requires a base temperature.
    #[arg(long, allow_hyphen_values = true)]
    pub temperature_step: Option<f64>,
    /// Allowlisted tools, comma-separated; repeatable. Passed as `--tools <CSV>`.
    #[arg(short, long = "tools")]
    pub tools: Vec<String>,
    /// don't load AGENTS.md and similar files
    #[arg(long, default_value_t = false)]
    pub no_context_files: bool,
    /// auto-accept all agent actions
    #[arg(long, default_value_t = false)]
    pub accept_all: bool,
    /// run agent in yolo mode (auto-approve everything)
    #[arg(long, default_value_t = false)]
    pub yolo: bool,
    /// skip all permission checks (dangerous)
    #[arg(long, default_value_t = false)]
    pub dangerously_skip_permissions: bool,
    /// run the coding agent in a sandbox
    #[arg(long, default_value_t = false)]
    pub sandbox: bool,
    /// Passed as `--sandbox-network[=true|false]` when set.
    #[arg(long)]
    pub sandbox_network: Option<bool>,
    /// shell binary used by the coding agent
    #[arg(long)]
    pub shell: Option<String>,

    // ---------- zerostack integrated workflow flags (multi-agent isolation) ----------
    /// Base directory for worktrees (only used with --agents > 1; passed through as --wt-base-dir).
    #[arg(long)]
    pub wt_base_dir: Option<String>,
    /// Pass `--wt-force` to zerostack (force worktree remove/branch delete even if dirty).
    #[arg(long, default_value_t = false)]
    pub wt_force: bool,
    /// Pass `--wt-auto-merge` to zerostack. WARNING: merges worktree branch on exit before deltastack scores it, bypassing the eval gate. Keep OFF.
    #[arg(long, default_value_t = false)]
    pub wt_auto_merge: bool,
    /// Force in-place execution. Single-agent runs are always in-place, so this
    /// is a no-op there; refused with --agents > 1 (parallel agents would clash
    /// on the same checkout).
    #[arg(long, default_value_t = false)]
    pub no_isolate: bool,
    /// Prefix for zerostack `--name <prefix>-iter<i>-agent<j>` sessions (only with --keep-agent-session).
    #[arg(long, default_value = "deltastack")]
    pub agent_session_prefix: String,
    /// Keep zerostack agent sessions (pass `--name`); default is ephemeral (`--no-session`).
    #[arg(long, default_value_t = false)]
    pub keep_agent_session: bool,

    // ---------- Commit behavior ----------
    /// Don't append the "git commit when done" suffix to agent prompts.
    #[arg(long, default_value_t = false)]
    pub no_auto_commit_prompt: bool,
    /// Disable fallback `git add -A && git commit` when agent leaves dirty tree.
    #[arg(long, default_value_t = false)]
    pub no_auto_commit_fallback: bool,

    // ---------- Resume ----------
    /// Resume from an existing state file instead of starting fresh.
    /// Continues at the first incomplete iteration, resets the checkout to
    /// the saved best commit, and skips re-running the baseline eval.
    /// The trailing partial iteration (if any) is discarded and re-run.
    #[arg(long, default_value_t = false)]
    pub resume: bool,

    // ---------- Output ----------
    /// JSONL run log path.
    #[arg(long, default_value = "deltastack.jsonl")]
    pub state_file: String,
    /// CSV run log path (wide, one row per candidate). Unset = disabled.
    #[arg(long)]
    pub csv_file: Option<String>,
    /// Directory for per-iteration agent/eval logs.
    #[arg(long, default_value = "deltastack-logs")]
    pub log_dir: String,
    /// Print zerostack + eval commands without running anything.
    #[arg(long, default_value_t = false)]
    pub dry_run: bool,
    /// Verbose output.
    #[arg(short, long, default_value_t = false)]
    pub verbose: bool,
    /// Quiet output (warnings only).
    #[arg(short, long, default_value_t = false)]
    pub quiet: bool,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum EvalKind {
    /// Custom eval script: stdout must be a single float.
    Custom(String),
    /// Wall-clock optimization: score = seconds.
    Speed(String),
    /// Peak-memory optimization: score = kB via GNU `time -v`.
    Memory(String),
}
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq, ValueEnum)]
pub enum Mode {
    /// Higher score is better.
    Maximize,
    /// Lower score is better (e.g. val_bpb). Default.
    #[default]
    Minimize,
}

#[derive(Copy, Clone, Debug, Default, PartialEq, Eq, ValueEnum)]
pub enum Aggregate {
    #[default]
    Mean,
    Median,
    Min,
    Max,
    First,
}

impl Cli {
    /// Effective eval concurrency.
    pub fn effective_eval_jobs(&self) -> usize {
        if self.eval_jobs > 0 {
            return self.eval_jobs as usize;
        }
        let total = (self.agents.max(1) * self.samples.max(1)) as usize;
        total.min(self.samples.max(1) as usize).max(1)
    }

    /// Which eval source is configured. Exactly one of --eval / --optimize-speed / --optimize-memory.
    pub fn eval_kind(&self) -> Option<EvalKind> {
        match (
            self.eval_cmd.as_deref(),
            self.optimize_speed.as_deref(),
            self.optimize_memory.as_deref(),
        ) {
            (Some(c), None, None) => Some(EvalKind::Custom(c.to_string())),
            (None, Some(c), None) => Some(EvalKind::Speed(c.to_string())),
            (None, None, Some(c)) => Some(EvalKind::Memory(c.to_string())),
            _ => None,
        }
    }

    /// Whether a built-in optimize prompt applies (prompt optional in that case).
    pub fn is_optimize_mode(&self) -> bool {
        matches!(
            self.eval_kind(),
            Some(EvalKind::Speed(_)) | Some(EvalKind::Memory(_))
        )
    }

    /// Validate flag combinations. Called from main before running.
    pub fn validate(&self) -> anyhow::Result<()> {
        let n_eval = [&self.eval_cmd, &self.optimize_speed, &self.optimize_memory]
            .iter()
            .filter(|o| o.is_some())
            .count();
        if n_eval == 0 {
            anyhow::bail!(
                "one of --eval, --optimize-speed <CMD> or --optimize-memory <CMD> is required"
            );
        }
        if n_eval > 1 {
            anyhow::bail!("--eval, --optimize-speed and --optimize-memory are mutually exclusive");
        }
        match self.eval_kind() {
            Some(EvalKind::Custom(c)) if c.trim().is_empty() => {
                anyhow::bail!("--eval must not be empty");
            }
            Some(EvalKind::Speed(c)) if c.trim().is_empty() => {
                anyhow::bail!("--optimize-speed must not be empty");
            }
            Some(EvalKind::Memory(c)) if c.trim().is_empty() => {
                anyhow::bail!("--optimize-memory must not be empty");
            }
            None => {
                anyhow::bail!(
                    "one of --eval, --optimize-speed <CMD> or --optimize-memory <CMD> is required"
                );
            }
            _ => {}
        }
        // Prompt is required for custom eval; optimize modes supply a default
        // prompt but still allow --prompt/--prompt-file to override it.
        if !self.is_optimize_mode() && self.prompt.is_none() && self.prompt_file.is_none() {
            anyhow::bail!("one of --prompt or --prompt-file is required");
        }
        if self.max_iterations == 0 {
            anyhow::bail!("--max-iterations must be >= 1");
        }
        if self.agents == 0 {
            anyhow::bail!("--agents must be >= 1");
        }
        if self.samples == 0 {
            anyhow::bail!("--samples must be >= 1");
        }
        if !self.min_improvement.is_finite() || self.min_improvement < 0.0 {
            anyhow::bail!("--min-improvement must be a finite value >= 0");
        }
        if !self.min_improvement_rel.is_finite() || self.min_improvement_rel < 0.0 {
            anyhow::bail!("--min-improvement-rel must be a finite value >= 0");
        }
        if self.target_sticky == 0 {
            anyhow::bail!("--target-sticky must be >= 1");
        }
        // All temperature flags accept any finite f64 (positive or negative).
        if let Some(t) = self.temperature {
            if !t.is_finite() {
                anyhow::bail!("--temperature must be a finite floating point number");
            }
        }
        if let Some(t) = self.temperature_min {
            if !t.is_finite() {
                anyhow::bail!("--temperature-min must be a finite floating point number");
            }
        }
        if let Some(t) = self.temperature_max {
            if !t.is_finite() {
                anyhow::bail!("--temperature-max must be a finite floating point number");
            }
        }
        if let Some(t) = self.temperature_step {
            if !t.is_finite() {
                anyhow::bail!("--temperature-step must be a finite floating point number");
            }
        }
        match (self.temperature_min, self.temperature_max) {
            (Some(_), None) => {
                anyhow::bail!("--temperature-min requires --temperature-max");
            }
            (None, Some(_)) => {
                anyhow::bail!("--temperature-max requires --temperature-min");
            }
            (Some(min), Some(max)) => {
                if min > max {
                    anyhow::bail!("--temperature-min must be <= --temperature-max");
                }
            }
            (None, None) => {}
        }
        if self.temperature_step.is_some()
            && self.temperature.is_none()
            && (self.temperature_min.is_none() || self.temperature_max.is_none())
        {
            anyhow::bail!(
                "--temperature-step requires a base temperature (--temperature or --temperature-min/--temperature-max)"
            );
        }
        if self.no_isolate && self.agents > 1 {
            anyhow::bail!("--no-isolate cannot be used with --agents > 1 (parallel agents would clash on the same checkout)");
        }
        if let Some(p) = self.csv_file.as_deref() {
            if p.trim().is_empty() {
                anyhow::bail!("--csv-file must not be empty");
            }
        }
        Ok(())
    }

    /// Whether this iteration setup isolates agents in worktrees.
    /// Single-agent runs always execute in-place; multi-agent runs isolate
    /// via zerostack `--worktree`. `--wt-base-dir` alone never enables
    /// worktrees for a single agent.
    pub fn uses_worktree_isolation(&self) -> bool {
        self.agents > 1 && !self.no_isolate
    }

    /// All deltastack-owned output paths that must never pollute git
    /// clean-checks or fallback commits (state file + log dir + CSV log).
    pub fn output_paths(&self) -> Vec<&str> {
        let mut out = vec![self.state_file.as_str(), self.log_dir.as_str()];
        if let Some(csv) = self.csv_file.as_deref() {
            out.push(csv);
        }
        out
    }
}
