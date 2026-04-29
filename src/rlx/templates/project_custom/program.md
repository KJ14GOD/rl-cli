# RLX Research Program

Objective:
- Improve evaluation reward for this custom RL project while keeping changes bounded and reviewable.

Code priorities:
- Start with `policies/custom_policy.py` before touching environment code.
- Keep environment identity stable unless the protocol explicitly allows an environment edit.
- Favor compact architecture or PPO-shaping changes that can be justified from run artifacts.

Rules:
- Treat `research.yaml` as the hard contract.
- Do not edit files outside the declared workspace scope.
- Keep changes small enough that a single diff is easy to review.
