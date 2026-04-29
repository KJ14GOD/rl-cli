# RLX Research Program

Objective:
- Maximize evaluation reward while keeping experiments comparable and auditable.

Rules:
- Treat `research.yaml` as the hard contract for budget, allowed changes, and locked fields.
- Prefer small, interpretable changes over broad churn.
- Do not change the environment identity or evaluation yardstick unless the protocol explicitly allows it.
- Use local artifacts first: run metrics, eval results, plots, diagnostics, and prior research rounds.
- Avoid repeating ideas that already failed without new evidence.

Quality bar:
- Keep hypotheses specific.
- Keep comparisons apples-to-apples.
- If a change cannot be explained from artifacts or bounded RL priors, do not make it.
