# RLCLI

RLCLI is a local-first command line toolkit for reinforcement learning experiments. It gives you one repeatable workflow for project setup, PPO training, evaluation, video rendering, comparison, plotting, sweeps, and artifact-grounded analysis.

The installed command is:

```bash
rlx
```

RLCLI is intentionally narrow today:

- PPO via Stable-Baselines3
- Gymnasium-compatible environments
- YAML configs as the main interface
- local run folders for all artifacts
- deterministic, rules-based analysis commands

RLCLI works out of the box with Gymnasium Classic Control env ids such as `CartPole-v1`, `Acrobot-v1`, `MountainCar-v0`, `MountainCarContinuous-v0`, and `Pendulum-v1`.

## Install

For development from this repo:

```bash
cd /Users/kj16/Desktop/RL-CLI
source .venv/bin/activate
python -m pip install -e .
```

For users after a PyPI release:

```bash
pipx install rlx-workbench
```

Then:

```bash
rlx --help
```

## Quickstart

Create a project:

```bash
rlx init bossfight
cd bossfight
```

Train the starter CartPole PPO config:

```bash
rlx train configs/ppo_cartpole.yaml
```

Inspect what was created:

```bash
rlx ls
rlx info cartpole_ppo_001
```

Evaluate and render the trained checkpoint:

```bash
rlx eval --run runs/cartpole_ppo_001
rlx video runs/cartpole_ppo_001/checkpoints/best.zip --episodes 2
```

Plot and analyze the run:

```bash
rlx plot cartpole_ppo_001
rlx analyze cartpole_ppo_001
rlx explain-metrics cartpole_ppo_001
rlx diagnose cartpole_ppo_001
rlx suggest cartpole_ppo_001
```

Compare multiple runs:

```bash
rlx compare cartpole_ppo_001 cartpole_ppo_002
```

## Project Layout

`rlx init bossfight` creates:

```text
bossfight/
  configs/
    ppo_acrobot.yaml
    ppo_cartpole.yaml
    ppo_mountain_car.yaml
    ppo_mountain_car_continuous.yaml
    ppo_pendulum.yaml
  envs/
  policies/
  runs/
  videos/
  logs/
  scripts/
  analysis/
```

The default starter config is the main editable entrypoint. You change training behavior by editing YAML values such as seed, device, environment id, timesteps, learning rate, rollout steps, checkpoint cadence, and eval cadence.

The scaffold includes one ready-to-run PPO config per Gymnasium Classic Control environment:

```bash
rlx train configs/ppo_cartpole.yaml
rlx train configs/ppo_acrobot.yaml
rlx train configs/ppo_mountain_car.yaml
rlx train configs/ppo_mountain_car_continuous.yaml
rlx train configs/ppo_pendulum.yaml
```

All of these use built-in Gymnasium ids directly. No custom env registration is required.

## Run Storage

Every successful `rlx train` creates a new run folder:

```text
runs/
  cartpole_ppo_001/
    config_snapshot.yaml
    metadata.json
    metrics.jsonl
    checkpoints/
      latest.zip
      best.zip
      step_*.zip
    eval/
      evaluations.npz
      manual_eval_*.json
    videos/
      manual_video_*/
    plots/
      manual_plot_*/
    logs/
```

Important files:

- `config_snapshot.yaml`: exact config used for that run
- `metadata.json`: run status, timestamps, device, env, artifact paths, tags, sweep/resume lineage
- `metrics.jsonl`: append-only training metrics
- `checkpoints/`: saved PPO models
- `eval/`: training-time eval and standalone eval JSON files
- `videos/`: rendered GIF bundles
- `plots/`: generated plot bundles

## Command Taxonomy

### Core Workflow

Use these to create and run experiments:

```bash
rlx init <project_name>
rlx train <config_path>
rlx eval <checkpoint_path>
rlx eval --run <run_path>
rlx eval --run <run_path> --all-checkpoints
rlx video <checkpoint_path>
rlx compare <run_a> <run_b> [run_c ...]
rlx plot <run_a> [run_b ...]
```

### Run Management

Use these to browse and organize runs:

```bash
rlx ls
rlx info <run>
rlx tag <run> <tag> [tag ...]
rlx resume <run>
rlx resume <run> --checkpoint best --timesteps 50000
rlx sweep <sweep_config>
```

### Analysis

These commands overlap in input, but not in purpose:

- `rlx info <run>` is factual: metadata, artifacts, config summary.
- `rlx analyze <run>` is interpretive: overall learning signal and artifact gaps.
- `rlx explain-metrics <run>` explains PPO metric columns and trends.
- `rlx diagnose <run>` focuses on likely problems and failure modes.
- `rlx suggest <run>` gives concrete next actions and config/sweep ideas.
- `rlx summarize <target>` gives a compact summary for a project, sweep, or run.

Examples:

```bash
rlx analyze cartpole_ppo_001
rlx explain-metrics cartpole_ppo_001
rlx diagnose cartpole_ppo_001
rlx suggest cartpole_ppo_001
rlx summarize .
rlx summarize analysis/sweeps/cartpole_seed_lr_entropy_001
```

These analysis commands are currently local and deterministic. They do not call an LLM or external API yet.

## Sweeps

`rlx sweep` runs multiple config variants from one sweep YAML.

Example `configs/cartpole_sweep.yaml`:

```yaml
name: cartpole_seed_lr_entropy
base_config: ppo_cartpole.yaml
tags: [cartpole, lr-study]

fixed:
  algo.total_timesteps: 50000

grid:
  seed: [42, 123]
  algo.learning_rate: [0.0003, 0.001]
  algo.entropy_coef: [0.0, 0.01]
```

Run it:

```bash
rlx sweep configs/cartpole_sweep.yaml
```

Then inspect outputs:

```bash
rlx ls
rlx compare cartpole_ppo_001 cartpole_ppo_002 cartpole_ppo_003
rlx summarize analysis/sweeps/cartpole_seed_lr_entropy_001
```

## Classic Control Configs

The generated starter configs currently target Gymnasium Classic Control:

- `configs/ppo_cartpole.yaml`: `CartPole-v1`
- `configs/ppo_acrobot.yaml`: `Acrobot-v1`
- `configs/ppo_mountain_car.yaml`: `MountainCar-v0`
- `configs/ppo_mountain_car_continuous.yaml`: `MountainCarContinuous-v0`
- `configs/ppo_pendulum.yaml`: `Pendulum-v1`

Each config uses the same PPO training flow:

```yaml
env:
  id: CartPole-v1
  num_envs: 4

policy:
  type: mlp
  hidden_sizes: [128, 128]
```

The only environment difference is the `env.id`. RLCLI validates the id with `gym.make(env.id)` before PPO starts.

## Styles

RLCLI has persistent output styles:

```bash
rlx styles
rlx --style minimal
rlx --style neon
rlx --style forest
```

The saved style is stored in:

```text
~/.config/rlx/config.toml
```

## Current Limitations

- PPO is the only supported algorithm.
- Custom envs must still register through Gymnasium before training starts.
- Custom policies must be Stable-Baselines3-compatible policy classes.
- Video output is GIF bundles, not mp4.
- Analysis commands are heuristic and local, not LLM-backed.

## Development Checks

Run targeted checks while developing:

```bash
python -m ruff check src tests
python -m compileall src tests
python -m pytest
```

Run the release packaging check before uploading to PyPI:

```bash
python scripts/check_release.py
```

That command builds into a temporary directory, runs `twine check`, and verifies the wheel includes the `rlx` entrypoint and bundled starter config.

If `pytest` is flaky in a local shell, the CLI can also be smoke-tested directly:

```bash
rlx init smokeproj
cd smokeproj
rlx train configs/ppo_cartpole.yaml
rlx eval --run runs/cartpole_ppo_001
rlx plot cartpole_ppo_001
rlx analyze cartpole_ppo_001
```

## Roadmap

The next major work is not adding more surface area. It is improving depth:

- command examples in each help screen
- stronger release packaging checks
- custom env and policy loading from scaffolded project code
- optional LLM-assisted analysis on top of current deterministic analysis
- advisor/research loops that stay bounded, auditable, and config-driven
