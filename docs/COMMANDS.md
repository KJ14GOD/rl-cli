# RLX Command Guide

This guide lists the public `rlx` commands and the normal way to use each one.
Every command also has built-in examples:

```bash
rlx <command> --help
```

## Project Setup

Create a new RLX project:

```bash
rlx init bossfight
cd bossfight
```

List bundled starter environments:

```bash
rlx envs
```

Set a persistent output style:

```bash
rlx styles
rlx --style minimal
rlx --style neon
```

## Training

Train a YAML config:

```bash
rlx train configs/ppo_cartpole.yaml
```

Each train command creates a new run under `runs/`:

```text
runs/cartpole_ppo_001/
```

The run contains the exact config snapshot, metrics JSONL, checkpoints, eval artifacts,
plots, videos, and metadata.

## Evaluation

Evaluate the latest checkpoint for a run:

```bash
rlx eval --run runs/cartpole_ppo_001
```

Evaluate a specific checkpoint:

```bash
rlx eval runs/cartpole_ppo_001/checkpoints/best.zip
rlx eval runs/cartpole_ppo_001/checkpoints/latest.zip
```

Evaluate canonical checkpoints:

```bash
rlx eval --run runs/cartpole_ppo_001 --all-checkpoints
```

Standalone evals write human-readable files:

```text
runs/cartpole_ppo_001/eval/manual_eval_001.json
```

## Video

Render GIF rollouts from a checkpoint:

```bash
rlx video runs/cartpole_ppo_001/checkpoints/best.zip --episodes 2
```

Videos are saved under:

```text
runs/cartpole_ppo_001/videos/manual_video_001/
```

## Inspection

List runs in the nearest RLX project:

```bash
rlx ls
```

Inspect one run:

```bash
rlx info cartpole_ppo_001
```

Tag a run:

```bash
rlx tag cartpole_ppo_001 baseline solved
```

Resume training from a previous run:

```bash
rlx resume cartpole_ppo_001
rlx resume cartpole_ppo_001 --checkpoint best --timesteps 50000
```

## Plotting And Compare

Generate plots for one or more runs:

```bash
rlx plot cartpole_ppo_001
rlx plot cartpole_ppo_001 cartpole_ppo_002
```

Compare runs:

```bash
rlx compare cartpole_ppo_001 cartpole_ppo_002
rlx compare cartpole_ppo_001 cartpole_ppo_002 cartpole_ppo_003
```

Compare uses tracked artifacts such as config snapshots, metrics, evals, checkpoints,
videos, and metadata.

## Web Reports And Dashboard

Generate a static HTML report for a run:

```bash
rlx report --preview
rlx report --preview --serve
rlx report --preview --preview-kind research --serve
rlx report cartpole_ppo_001
rlx report runs/cartpole_ppo_001
```

Generate a static HTML report for a research bundle:

```bash
rlx report analysis/research/cartpole_ppo_001_research_001
```

Start the connected local project dashboard. This serves the app at `/` and live
artifact data under `/api/...`:

```bash
rlx dashboard --demo
rlx dashboard
rlx dashboard --port 9000
rlx dashboard --open
```

Write an offline dashboard HTML snapshot without starting a server:

```bash
rlx dashboard --export
```

Report output goes under `analysis/reports/`. Dashboard export output goes to
`analysis/dashboard/index.html`. The served dashboard is the richer connected app.

## Analysis

Run-level analysis:

```bash
rlx analyze cartpole_ppo_001
```

PPO metric explanation:

```bash
rlx explain-metrics cartpole_ppo_001
```

Failure-mode diagnosis:

```bash
rlx diagnose cartpole_ppo_001
```

Concrete next actions:

```bash
rlx suggest cartpole_ppo_001
```

Compact summaries:

```bash
rlx summarize .
rlx summarize cartpole_ppo_001
rlx summarize analysis/sweeps/cartpole_seed_lr_entropy_001
```

## Sweeps

Run a grid from a sweep YAML:

```bash
rlx sweep configs/cartpole_sweep.yaml
```

Use sweeps when you know the exact parameter grid you want to test.

## Advisor

Create next-variant YAMLs from a completed run:

```bash
rlx advisor cartpole_ppo_001 --variants 4
```

Train the generated variants:

```bash
rlx advisor cartpole_ppo_001 --execute --variants 4
```

Use an LLM planner:

```bash
rlx advisor cartpole_ppo_001 --planner llm --llm-provider openai --llm-model gpt-5.4-mini
rlx advisor cartpole_ppo_001 --planner llm --llm-provider ollama --llm-model qwen3:8b
```

Advisor outputs:

```text
analysis/advisor/cartpole_ppo_001_advisor_001/
  manifest.json
  plan.md
  configs/
    variant_001.yaml
```

## Research

Run bounded advisor loops:

```bash
rlx research cartpole_ppo_001 --rounds 3 --variants 4
```

Train, eval, and promote champions:

```bash
rlx research cartpole_ppo_001 --execute --rounds 3 --variants 4
```

Use OpenAI:

```bash
rlx research cartpole_ppo_001 --planner llm --llm-provider openai --llm-model gpt-5.4-mini --execute --rounds 3 --variants 4
```

Use Ollama:

```bash
rlx research cartpole_ppo_001 --planner llm --llm-provider ollama --llm-model qwen3:8b --execute --rounds 3 --variants 4
```

Resume an existing research bundle:

```bash
rlx research --resume analysis/research/cartpole_ppo_001_research_001 --rounds 5
```

Research outputs:

```text
analysis/research/cartpole_ppo_001_research_001/
  manifest.json
  report.md
  scoreboard.png
  progress.png
```
