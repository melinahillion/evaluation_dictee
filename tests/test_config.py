"""Test du chargement de configuration."""

from pathlib import Path

from evaluation_dictee.config import load_config


def test_charge_config_exemple() -> None:
    # On part de la racine du dépôt (pytest est lancé depuis là)
    config = load_config(Path("configs/dictee_qwen7b_zeroshot.yaml"))
    assert config.name == "dictee_qwen7b_zeroshot"
    assert config.model.kind == "vlm"
    assert config.grid.scheme == "simplifiee"
    assert config.prompt.method == "C"
    assert config.prompt.read_final_state is True
