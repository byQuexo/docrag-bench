"""ExperimentConfig: strict YAML, stable hash, every shipped config loads."""

from pathlib import Path

import pytest
from pydantic import ValidationError

from docrag.shared.config import ExperimentConfig

CONFIG_DIR = Path(__file__).resolve().parents[2] / "configs"


def test_unknown_keys_are_rejected() -> None:
    with pytest.raises(ValidationError):
        ExperimentConfig.model_validate({"name": "x", "top_kk": 5})
    with pytest.raises(ValidationError):
        ExperimentConfig.model_validate({"name": "x", "dataset": {"spilt": "dev"}})


def test_invalid_values_are_rejected() -> None:
    with pytest.raises(ValidationError):
        ExperimentConfig(name="x", retrieval="magic")  # type: ignore[arg-type]
    with pytest.raises(ValidationError):
        ExperimentConfig(name="x", top_k=0)


def test_hash_ignores_name_but_not_settings() -> None:
    a = ExperimentConfig(name="a")
    assert a.config_hash() == ExperimentConfig(name="b").config_hash()
    assert a.config_hash() != ExperimentConfig(name="a", top_k=5).config_hash()
    assert len(a.config_hash()) == 12


# LEARN: parametrize over files found at collection time: adding a new YAML to
# configs/ automatically adds a test case for it. `ids=` names each case.
@pytest.mark.parametrize(
    "path", sorted(CONFIG_DIR.glob("*.yaml")), ids=lambda p: p.name
)
def test_shipped_configs_load(path: Path) -> None:
    config = ExperimentConfig.from_yaml(path)
    assert config.name
    assert config.embedder.split(":")[0] in {"onnx", "fake"}


def test_configs_exist() -> None:
    assert (CONFIG_DIR / "baseline.yaml").exists()
