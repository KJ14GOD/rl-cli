from pathlib import Path

from rlx.config.loader import load_config
from rlx.core.env_catalog import catalog_config_names, list_env_catalog


def test_catalog_entries_point_to_existing_templates() -> None:
    names = catalog_config_names()

    assert len(names) == len(set(names))

    for entry in list_env_catalog():
        config = load_config(Path("src/rlx/templates/project") / entry.config_path)

        assert config.env.id == entry.env_id
        assert config.policy.type == "mlp"
