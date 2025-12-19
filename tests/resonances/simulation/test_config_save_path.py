import re
from pathlib import Path

import pytest

import resonances
from resonances.simulation import SimulationConfig


@pytest.fixture(autouse=True)
def setup_save_path_config(tmp_path):
    original_save_path = resonances.config.get('SAVE_PATH')
    resonances.config.set('SAVE_PATH', str(tmp_path))
    yield
    resonances.config.set('SAVE_PATH', original_save_path)


def test_save_path_explicit_is_used(tmp_path):
    explicit = tmp_path / "explicit_save"
    explicit.mkdir(parents=True, exist_ok=True)

    config = SimulationConfig(save_path=str(explicit))

    assert config.save_path == str(explicit)


def test_save_path_uses_name_when_missing():
    name = "test_sim"
    config = SimulationConfig(name=name)

    assert config.save_path == str(Path(resonances.config.get('SAVE_PATH')) / name)


def test_save_path_appends_datetime_when_exists(tmp_path):
    name = "existing_sim"
    base = tmp_path / name
    base.mkdir(parents=True, exist_ok=True)

    config = SimulationConfig(name=name)

    assert config.save_path.startswith(str(base) + "_")
    assert re.search(r"_\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}$", config.save_path)
