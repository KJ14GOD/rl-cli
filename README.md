# RLX Workbench

RLX Workbench is a local-first command line toolkit for reinforcement-learning
experiments. It gives you one workflow for scaffolding projects, training PPO
agents, evaluating checkpoints, rendering videos, comparing runs, plotting metrics,
running sweeps, and launching bounded advisor/research loops.

The package name is:

```bash
rlx-workbench
```

The installed command is:

```bash
rlx
```

RLX is intentionally focused today:

- PPO training via Stable-Baselines3.
- Gymnasium-compatible environments.
- YAML configs as the primary interface.
- Local run folders for all configs, metrics, checkpoints, evals, plots, videos, and research artifacts.
- Deterministic analysis commands plus optional LLM-assisted advisor/research planning.

## Install

Recommended user install after release:

```bash
pipx install rlx-workbench
rlx --help
```

Development install from this repository:

```bash
cd /Users/kj16/Desktop/RL-CLI
source .venv/bin/activate
python -m pip install -e .
rlx --help
```

If you use `pip` instead of `pipx`, install inside a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install rlx-workbench
```

## Quickstart

Create a project:

```bash
rlx init bossfight
cd bossfight
```

See bundled starter environments:

```bash
rlx envs
```

Train the starter CartPole PPO config:

```bash
rlx train configs/ppo_cartpole.yaml
```

Inspect the run:

```bash
rlx ls
rlx info cartpole_ppo_001
```

Evaluate and render a checkpoint:

```bash
rlx eval --run runs/cartpole_ppo_001
rlx eval runs/cartpole_ppo_001/checkpoints/best.zip
rlx video runs/cartpole_ppo_001/checkpoints/best.zip --episodes 2
```

Plot, compare, and analyze:

```bash
rlx plot cartpole_ppo_001
rlx compare cartpole_ppo_001 cartpole_ppo_002
rlx analyze cartpole_ppo_001
rlx explain-metrics cartpole_ppo_001
rlx diagnose cartpole_ppo_001
rlx suggest cartpole_ppo_001
```

Generate next experiment ideas:

```bash
rlx advisor cartpole_ppo_001 --variants 4
rlx research cartpole_ppo_001 --rounds 3 --variants 4
```

Train advisor/research variants:

```bash
rlx advisor cartpole_ppo_001 --execute --variants 4
rlx research cartpole_ppo_001 --execute --rounds 3 --variants 4
```

## Project Layout

`rlx init bossfight` creates:

```text
bossfight/
  .env.example
  .gitignore
  analysis/
  configs/
    ppo_acrobot.yaml
    ppo_cartpole.yaml
    ppo_frozen_lake.yaml
    ppo_mountain_car.yaml
    ppo_mountain_car_continuous.yaml
    ppo_pendulum.yaml
    ppo_taxi.yaml
  envs/
  logs/
  policies/
  runs/
  scripts/
  videos/
```

The YAML config is the main editable entrypoint. You can change values such as
`seed`, `device`, `env.id`, `env.num_envs`, `algo.total_timesteps`,
`algo.learning_rate`, `algo.rollout_steps`, `checkpoint.save_every`, and eval cadence.

Starter configs:

```bash
rlx train configs/ppo_cartpole.yaml
rlx train configs/ppo_acrobot.yaml
rlx train configs/ppo_mountain_car.yaml
rlx train configs/ppo_mountain_car_continuous.yaml
rlx train configs/ppo_pendulum.yaml
rlx train configs/ppo_frozen_lake.yaml
rlx train configs/ppo_taxi.yaml
```

Classic Control configs are the strongest PPO starters. FrozenLake and Taxi are included as
simple Toy Text examples, but they are not necessarily PPO-optimal tasks.

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
      step_*_steps.zip
    eval/
      evaluations.npz
      manual_eval_*.json
    videos/
      manual_video_*/
    plots/
      manual_plot_*/
    logs/
```

Important artifacts:

- `config_snapshot.yaml`: exact YAML used for the run.
- `metadata.json`: status, device, env, artifacts, tags, sweep/advisor/research/resume lineage.
- `metrics.jsonl`: append-only training metrics.
- `checkpoints/latest.zip`: final policy at the end of training.
- `checkpoints/best.zip`: best policy found by training-time eval.
- `checkpoints/step_*_steps.zip`: periodic checkpoint saves.
- `eval/evaluations.npz`: Stable-Baselines3 training-time eval history.
- `eval/manual_eval_*.json`: standalone `rlx eval` or research scoring results.
- `videos/manual_video_*`: rendered GIF bundles and manifest.
- `plots/manual_plot_*`: generated plots and manifest.

## Eval Semantics

Training-time eval is created by `rlx train` through Stable-Baselines3 `EvalCallback`.
It writes one `eval/evaluations.npz` file containing eval points over time. For example,
if the YAML says:

```yaml
eval:
  every: 10000
  episodes: 10
```

then a 50,000 timestep run records eval points around 10k, 20k, 30k, 40k, and 50k
inside the same `evaluations.npz` file.

Standalone eval is created by `rlx eval`:

```bash
rlx eval --run runs/cartpole_ppo_001
```

By default, `--run` evaluates `checkpoints/latest.zip` and writes
`eval/manual_eval_001.json`.

To evaluate a specific checkpoint:

```bash
rlx eval runs/cartpole_ppo_001/checkpoints/best.zip
rlx eval runs/cartpole_ppo_001/checkpoints/latest.zip
```

To evaluate canonical checkpoints:

```bash
rlx eval --run runs/cartpole_ppo_001 --all-checkpoints
```

Research currently scores candidates with standalone eval on `latest.zip` so every
candidate is compared as: train for the same budget, then evaluate the final policy.

## Command Reference

Every command has examples in its help screen:

```bash
rlx <command> --help
```

Core workflow:

```bash
rlx init <project_name>
rlx envs
rlx train <config_path>
rlx eval <checkpoint_path>
rlx eval --run <run_path>
rlx eval --run <run_path> --all-checkpoints
rlx video <checkpoint_path>
rlx plot <run_a> [run_b ...]
rlx compare <run_a> <run_b> [run_c ...]
```

Run management:

```bash
rlx ls
rlx info <run>
rlx tag <run> <tag> [tag ...]
rlx resume <run>
rlx resume <run> --checkpoint best --timesteps 50000
rlx sweep <sweep_config>
```

Analysis:

```bash
rlx analyze <run>
rlx explain-metrics <run>
rlx diagnose <run>
rlx suggest <run>
rlx summarize <target>
```

Advisor and research:

```bash
rlx advisor <run>
rlx advisor <run> --planner llm --llm-provider openai --llm-model gpt-5.4-mini
rlx advisor <run> --planner llm --llm-provider ollama --llm-model qwen3:8b
rlx research <run> --rounds 3 --variants 4
rlx research <run> --planner llm --execute --rounds 3 --variants 4
rlx research --resume analysis/research/cartpole_ppo_001_research_001 --rounds 5
```

Output styles:

```bash
rlx styles
rlx --style minimal
rlx --style neon
rlx --style forest
rlx --style ice
```

The saved style is stored at:

```text
~/.config/rlx/config.toml
```

## Advisor And Research

`rlx advisor` looks at one completed run and writes next-variant YAML configs:

```bash
rlx advisor cartpole_ppo_001 --variants 4
```

Outputs:

```text
analysis/advisor/cartpole_ppo_001_advisor_001/
  manifest.json
  plan.md
  configs/
    variant_001.yaml
    variant_002.yaml
```

Use `--execute` to train the generated variants:

```bash
rlx advisor cartpole_ppo_001 --execute --variants 4
```

`rlx research` runs repeated advisor rounds. In executed mode, it trains variants,
evaluates each latest checkpoint with standalone eval, promotes a candidate if it beats
the current champion, and then uses that champion as the next baseline.

```bash
rlx research cartpole_ppo_001 --execute --rounds 3 --variants 4
```

Outputs:

```text
analysis/research/cartpole_ppo_001_research_001/
  manifest.json
  report.md
  scoreboard.png
  progress.png
```

Research keeps these fields locked for comparability:

- `env.id`
- `algo.total_timesteps`
- `eval.every`
- `eval.episodes`
- `eval.deterministic`

Allowed YAML mutations for LLM planning include PPO and policy knobs such as:

- `seed`
- `env.num_envs`
- `algo.rollout_steps`
- `algo.batch_size`
- `algo.learning_rate`
- `algo.gamma`
- `algo.gae_lambda`
- `algo.clip_range`
- `algo.entropy_coef`
- `algo.value_coef`
- `algo.update_epochs`
- `policy.hidden_sizes`
- `checkpoint.save_every`

LLM proposal audits are stored in the advisor manifest:

```text
analysis/advisor/<bundle>/manifest.json
```

Read:

```json
protocol.planner_audit
```

That audit records accepted proposals, rejected proposals, repair attempts, and whether
rule fallback was used.

## LLM Planner Setup

LLMs are optional. Rules-based advisor/research works without API keys:

```bash
rlx advisor cartpole_ppo_001
rlx research cartpole_ppo_001 --rounds 3 --variants 4
```

To use OpenAI, create a project-local `.env`:

```bash
cd bossfight
cp .env.example .env
```

Set:

```bash
OPENAI_API_KEY=sk-...
RLX_LLM_PROVIDER=openai
RLX_LLM_MODEL=gpt-5.4-mini
RLX_LLM_TIMEOUT=300
```

Run:

```bash
rlx research cartpole_ppo_001 --planner llm --llm-provider openai --llm-model gpt-5.4-mini --execute --rounds 3 --variants 4
```

To use Ollama locally, install Ollama, pull a model, and keep the local server running:

```bash
ollama pull qwen3:8b
ollama list
```

Set:

```bash
RLX_LLM_PROVIDER=ollama
RLX_LLM_MODEL=qwen3:8b
OLLAMA_BASE_URL=http://localhost:11434
RLX_LLM_TIMEOUT=300
```

Run:

```bash
rlx advisor cartpole_ppo_001 --planner llm --llm-provider ollama --llm-model qwen3:8b --variants 2
rlx research cartpole_ppo_001 --planner llm --llm-provider ollama --llm-model qwen3:8b --execute --rounds 3 --variants 4
```

If the LLM returns invalid, duplicate, or forbidden changes, RLX asks the same provider
for repair proposals before falling back or stopping. If the proposal space is exhausted,
research stops cleanly and writes an exhausted advisor manifest.

## Sweeps

`rlx sweep` runs a fixed grid from one sweep YAML. Use it when you know the values you
want to test.

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

Run:

```bash
rlx sweep configs/cartpole_sweep.yaml
```

Inspect:

```bash
rlx ls
rlx compare cartpole_ppo_001 cartpole_ppo_002 cartpole_ppo_003
rlx summarize analysis/sweeps/cartpole_seed_lr_entropy_001
```

Difference from research:

- `sweep` executes a user-defined grid.
- `advisor` proposes variants from one run.
- `research` loops advisor, trains/evals variants, and promotes champions.

## Limitations

- PPO is the only implemented algorithm.
- Project-code editing is not implemented yet. Current advisor/research mutations are YAML-only.
- Custom envs must be Gymnasium-compatible and registered before training starts.
- Custom policy loading from scaffolded project code is planned but not public-release ready.
- Video output is GIF bundles, not mp4.
- LLM planner quality depends on the provider/model. Local Ollama models can be slower or weaker than hosted models.

## Development

Install development dependencies:

```bash
python -m pip install -e ".[dev]"
```

Run checks:

```bash
python -m ruff check src tests scripts/check_release.py
python -m pytest
python scripts/check_release.py
```

Smoke-test the CLI:

```bash
rlx init smokeproj
cd smokeproj
rlx train configs/ppo_cartpole.yaml
rlx eval --run runs/cartpole_ppo_001
rlx plot cartpole_ppo_001
rlx analyze cartpole_ppo_001
```

## Release

Before uploading to PyPI:

```bash
python -m ruff check src tests scripts/check_release.py
python -m pytest
python scripts/check_release.py
```

Build fresh artifacts:

```bash
python -m build
python -m twine check dist/*
```

Upload:

```bash
python -m twine upload dist/*
```

For the full checklist, see [docs/RELEASE.md](docs/RELEASE.md).

