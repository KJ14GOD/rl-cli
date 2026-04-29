from pathlib import Path

PROJECT_DIRS = (
    "envs",
    "configs",
    "policies",
    "runs",
    "videos",
    "logs",
    "scripts",
    "analysis",
)

STARTER_CONFIG = Path("configs/ppo_cartpole.yaml")
CUSTOM_STARTER_CONFIG = Path("configs/custom_ppo.yaml")
RESEARCH_PROTOCOL = Path("research.yaml")
PROGRAM_DOC = Path("program.md")
RUNS_DIR = Path("runs")
RUN_ARTIFACT_DIRS = (
    "checkpoints",
    "eval",
    "videos",
    "plots",
    "logs",
)
CONFIG_SNAPSHOT_NAME = "config_snapshot.yaml"
METADATA_NAME = "metadata.json"
METRICS_NAME = "metrics.jsonl"
