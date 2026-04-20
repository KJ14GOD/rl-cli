# Advisor And Research

RLX has two experiment-generation commands:

- `rlx advisor`: inspect one run and create next experiment variants.
- `rlx research`: loop advisor rounds, train/evaluate variants, and promote the best run.

Both commands are local-first and artifact-grounded. They read run artifacts from `runs/`
and write plans, configs, manifests, reports, and plots under `analysis/`.

## Advisor

Create a dry-run advisor plan:

```bash
rlx advisor cartpole_ppo_001 --variants 4
```

Output:

```text
analysis/advisor/cartpole_ppo_001_advisor_001/
  manifest.json
  plan.md
  configs/
    variant_001.yaml
    variant_002.yaml
```

Train the variants:

```bash
rlx advisor cartpole_ppo_001 --execute --variants 4
```

The generated runs are tagged with advisor lineage in `metadata.json`.

## Research

Run dry-run research:

```bash
rlx research cartpole_ppo_001 --rounds 3 --variants 4
```

Run executed research:

```bash
rlx research cartpole_ppo_001 --execute --rounds 3 --variants 4
```

Executed research does this each round:

```text
current champion run
-> advisor proposes variants
-> RLX writes variant YAMLs
-> RLX trains each variant
-> RLX evaluates each latest checkpoint with standalone eval
-> RLX promotes the best candidate if it beats the champion
-> next round uses the champion as the new baseline
```

If a candidate improves once and later rounds do not improve, research keeps using the
promoted champion as the baseline until a better run appears.

## Scoring

Research scoring uses standalone eval on `checkpoints/latest.zip` for each candidate.
This means every candidate is compared as:

```text
train for the locked timestep budget
-> evaluate final policy
-> compare mean reward
```

The eval result is stored in:

```text
runs/<run_id>/eval/manual_eval_001.json
```

The score source in research reports usually looks like:

```text
standalone eval (manual_eval_001.json)
```

Training-time eval still creates `eval/evaluations.npz` and `checkpoints/best.zip`, but
research currently scores `latest.zip` for a simple apples-to-apples final-policy rule.

## Locked Fields

Research locks these values from the baseline config so variants are comparable:

- `env.id`
- `algo.total_timesteps`
- `eval.every`
- `eval.episodes`
- `eval.deterministic`

These are not allowed as LLM mutations during research.

## Allowed YAML Mutations

LLM and rules-based advisor planning can use safe YAML mutation keys such as:

- `seed`
- `device`
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

The LLM does not edit YAML directly. It returns JSON proposals. RLX validates those
proposals, converts them into mutation dictionaries, applies them to a copy of the
baseline config snapshot, and writes `variant_*.yaml`.

## LLM Planner

Rules planner:

```bash
rlx advisor cartpole_ppo_001 --planner rules
rlx research cartpole_ppo_001 --planner rules --rounds 3 --variants 4
```

OpenAI planner:

```bash
rlx advisor cartpole_ppo_001 --planner llm --llm-provider openai --llm-model gpt-5.4-mini
```

Ollama planner:

```bash
rlx advisor cartpole_ppo_001 --planner llm --llm-provider ollama --llm-model qwen3:8b
```

Project-local `.env` example:

```bash
RLX_LLM_PROVIDER=ollama
RLX_LLM_MODEL=qwen3:8b
OLLAMA_BASE_URL=http://localhost:11434
RLX_LLM_TIMEOUT=300
```

## Repair And Fallback

If an LLM proposes invalid or duplicate changes, RLX does not immediately accept them.
It validates and filters proposals, then asks the same LLM provider for repair proposals.

Repair context includes:

- accepted mutations
- rejected proposals
- rejection reasons
- already-tried mutation signatures
- allowed mutation keys
- locked fields

If repair still cannot produce enough variants, RLX can use rules-based fallback. If no
valid new proposals remain, research stops cleanly with a proposal-exhausted reason.

## Audit Files

Research round summary:

```text
analysis/research/<bundle>/manifest.json
```

Advisor proposal details:

```text
analysis/advisor/<bundle>/manifest.json
```

Useful fields:

```json
{
  "protocol": {
    "fallback": {},
    "planner_audit": {
      "llm_response": {},
      "repair_attempts": 1,
      "repairs": [],
      "proposals": []
    }
  },
  "variants": []
}
```

Generated YAMLs:

```text
analysis/advisor/<bundle>/configs/variant_001.yaml
```

Final full config used for a trained run:

```text
runs/<run_id>/config_snapshot.yaml
```

## Resume

Resume an existing research bundle:

```bash
rlx research --resume analysis/research/cartpole_ppo_001_research_001 --rounds 5
```

Resume keeps the original planner, provider, model, mode, score protocol, and timestep
budget. Start a new research bundle if you want to change those settings.

