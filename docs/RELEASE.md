# Release Checklist

Use this checklist before publishing `rlx-workbench` to PyPI.

## 1. Confirm Version

Version must be bumped in both files:

```text
pyproject.toml
src/rlx/__init__.py
```

The release check verifies these match.

## 2. Run Tests

```bash
python -m ruff check src tests scripts/check_release.py
python -m pytest
python scripts/check_release.py
```

`scripts/check_release.py` builds into a temporary directory, runs `twine check`, and
verifies:

- package metadata exists
- runtime `__version__` matches `pyproject.toml`
- wheel exposes `rlx = rlx.cli:app`
- wheel and sdist include bundled starter configs
- wheel and sdist include `.env.example` and project `.gitignore`
- wheel does not include `__pycache__`

## 3. Build Fresh Artifacts

Remove old artifacts:

```bash
python - <<'PY'
from pathlib import Path
import shutil

for path in [Path("dist"), Path("build")]:
    if path.exists():
        shutil.rmtree(path)
for path in Path(".").glob("*.egg-info"):
    if path.is_dir():
        shutil.rmtree(path)
PY
```

Build:

```bash
python -m build
python -m twine check dist/*
```

## 4. Local Wheel Smoke Test

Use `pipx` to test the same artifact users will install:

```bash
pipx uninstall rlx-workbench
pipx install dist/rlx_workbench-*.whl
rlx --help
```

Create a temporary project:

```bash
mkdir -p /tmp/rlx-release-smoke
cd /tmp/rlx-release-smoke
rlx init smokeproj
cd smokeproj
rlx envs
rlx train configs/ppo_cartpole.yaml
rlx eval --run runs/cartpole_ppo_001
rlx advisor cartpole_ppo_001 --variants 2
```

Optional smoke checks:

```bash
rlx plot cartpole_ppo_001
rlx analyze cartpole_ppo_001
rlx report --preview
rlx report --preview --preview-kind research
rlx report cartpole_ppo_001
rlx dashboard --demo --export
rlx dashboard --export
rlx dashboard --demo --port 8765
rlx research cartpole_ppo_001 --rounds 1 --variants 2
```

## 5. Publish

Upload to PyPI:

```bash
python -m twine upload dist/*
```

If PyPI rejects the upload because the version already exists, bump the version and rebuild.

## 6. User Install Test

After PyPI finishes indexing:

```bash
pipx uninstall rlx-workbench
pipx install rlx-workbench
rlx --help
```

## 7. Known Beta Boundaries

Do not market these as finished yet:

- algorithms beyond PPO
- unrestricted project-code editing
- custom policy loading from scaffolded project code
- custom environment authoring inside the scaffold
- mp4 video output

Current public positioning:

```text
Local-first PPO experiment workbench with run storage, evals, plots, comparison,
sweeps, and bounded advisor/research loops.
```
