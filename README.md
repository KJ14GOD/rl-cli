# RLCLI

RLCLI is a local-first command line toolkit for managing reinforcement learning experiments end to end. The goal is to make RL workflows structured, reproducible, and easy to iterate on by standardizing how projects are initialized, trained, evaluated, visualized, compared, and later analyzed with RL-aware tooling.

> [!IMPORTANT]
> RLCLI is in active development. This repository is currently a project definition and roadmap for the intended system. The commands, layouts, and workflows below describe what RLCLI is meant to provide, not a released implementation.

## Why RLCLI Exists

Reinforcement learning projects tend to get messy fast. A typical setup ends up with scattered training scripts, inconsistent checkpoint folders, ad hoc evaluation code, and little discipline around what changed between runs.

That makes basic questions harder than they should be:

- Which config produced the best result?
- Where is the best checkpoint?
- Did the latest run actually improve, or did it just get lucky?
- What changed between two runs?
- Did training plateau, collapse, or become unstable?

RL libraries solve algorithm implementation. RLCLI is meant to solve the experiment workflow around those algorithms.

## Core Workflow

RLCLI is built around one loop:

`config -> train -> evaluate -> visualize -> compare -> analyze -> iterate`

The intent is to make every experiment follow the same structure:

- configs define the run
- training produces checkpoints and metrics
- evaluation measures policy quality consistently
- visualization makes learning behavior easier to inspect
- comparison makes run-to-run differences obvious
- later analysis tools help interpret outcomes and suggest next steps

## Phase 1: MVP

The first release is intentionally narrow. It should establish a clean, useful baseline before expanding into sweeps and AI-assisted features.

Planned commands:

```bash
rlx init <project_name>
rlx train <config_path>
rlx eval <checkpoint_path>
rlx video <checkpoint_path>
rlx compare <run_a> <run_b> [run_c ...]
```

Initial scope:

- PPO as the first supported algorithm
- Gymnasium-compatible environments
- YAML configs as the primary interface
- run folders for every training execution
- metrics logging and checkpointing
- deterministic evaluation
- video rendering and text-based run comparison

## Intended Project Layout

Running the planned `rlx init bossfight` command should create a standardized layout like this:

```text
bossfight/
  envs/
  configs/
  policies/
  runs/
  videos/
  logs/
  scripts/
  analysis/
```

The point of this layout is consistency. RL projects should not depend on whatever folder structure happened to emerge over time.

## Run-Based Storage Model

Every training execution should create a new run folder with the important artifacts stored together:

```text
runs/
  bossfight_001/
    config_snapshot.yaml
    metadata.json
    metrics.jsonl
    checkpoints/
      best.pt
      latest.pt
      step_50000.pt
    eval/
    videos/
    plots/
    logs/
```

Each run is intended to preserve:

- the exact config used
- seed and device
- environment name
- git commit when available
- metrics over time
- evaluation results
- artifacts such as checkpoints, videos, and plots

This run-based model is the foundation for reproducibility, comparison, and every later analysis feature.

## Config-Driven Training

YAML configs are intended to be the main interface for training. A config should fully describe the environment, algorithm settings, policy shape, checkpointing, and evaluation behavior.

Example PPO config:

```yaml
run_name: bossfight_ppo
seed: 42
device: cuda

env:
  id: BossFight-v0
  num_envs: 8

algo:
  name: ppo
  total_timesteps: 500000
  rollout_steps: 128
  batch_size: 256
  learning_rate: 0.0003
  gamma: 0.99
  gae_lambda: 0.95
  clip_range: 0.2
  entropy_coef: 0.01
  value_coef: 0.5
  update_epochs: 4

policy:
  type: mlp
  hidden_sizes: [256, 256]

checkpoint:
  save_every: 50000

eval:
  every: 25000
  episodes: 20
  deterministic: true
```

Configs matter because they make runs reproducible, comparable, and easy to mutate later in a controlled way.

## Roadmap

### Phase 2: Run Management and Experiment Discipline

Planned additions:

- `rlx ls`
- `rlx info <run_name>`
- `rlx resume <run_name>`
- `rlx plot <run_a> [run_b ...]`
- `rlx sweep <sweep_config>`
- `rlx tag <run_name> <tag>`

This phase adds run browsing, plotting, resuming, sweeps, and better organization.

### Phase 3: AI-Assisted Analysis

Planned additions:

- `rlx analyze <run_name>`
- `rlx diagnose <run_name>`
- `rlx suggest <run_name>`
- `rlx summarize <path_or_experiment>`
- `rlx explain-metrics <run_name>`

These features are meant to be grounded RL tools, not general chat. They should analyze actual run artifacts and help explain learning behavior, detect failure modes, and recommend what to try next.

### Phase 4: Advisor Mode

Planned addition:

- `rlx advisor run <run_name> --budget <N>`

Advisor mode is intended to run one bounded improvement loop: analyze a baseline run, generate a small set of candidate configs, train them under fixed budgets, evaluate them consistently, and report the best result.

### Phase 5: Research Mode

Planned addition:

- `rlx research <run_name> --rounds <R> --budget <N>`

Research mode extends advisor into a multi-round search loop. In early versions it should remain tightly constrained: fixed round counts, fixed training budgets, deterministic evaluation, full metadata logging, and config-only mutation rather than arbitrary code edits.

## Design Principles

- Local-first: runs, artifacts, and analysis should work without requiring a hosted service.
- Reproducible: every run should preserve the config, metadata, and outputs needed to understand what happened.
- CLI-first: the workflow should be scriptable, inspectable, and easy to integrate into existing research habits.
- Config-driven: configs should be the source of truth for training behavior.
- Bounded autonomy: future automated features should stay constrained, auditable, and grounded in experiment data.

## Recommended Build Order

1. Build the run system and folder structure.
2. Build config parsing and validation.
3. Build `train` with PPO, logging, and checkpoints.
4. Build `eval`.
5. Build `video`.
6. Build `compare`.
7. Add `ls`, `info`, `plot`, and `sweep`.
8. Add `analyze`, `diagnose`, `suggest`, `summarize`, and `explain-metrics`.
9. Add `advisor`.
10. Add `research`.

## Long-Term Direction

RLCLI is meant to become a disciplined operating layer for RL experiments:

- reproducible run structure
- config-driven training
- consistent evaluation
- visualization and comparison
- RL-aware analysis
- bounded automated iteration

The immediate priority is the core experiment workflow. If that foundation is solid, the later AI and research features have real data and structure to stand on.
