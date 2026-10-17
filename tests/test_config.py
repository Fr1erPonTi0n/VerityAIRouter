from pathlib import Path

from verity_ai import config


def test_env_returns_value(monkeypatch):
    monkeypatch.setenv("VERITY_TEST_VALUE", "custom")

    assert config._env("VERITY_TEST_VALUE", "default") == "custom"


def test_env_falls_back_for_missing_or_empty_value(monkeypatch):
    monkeypatch.delenv("VERITY_TEST_VALUE", raising=False)
    assert config._env("VERITY_TEST_VALUE", "default") == "default"

    monkeypatch.setenv("VERITY_TEST_VALUE", "")
    assert config._env("VERITY_TEST_VALUE", "default") == "default"


def test_env_path_resolves_relative_to_project_root(monkeypatch, tmp_path):
    monkeypatch.setenv("VERITY_TEST_PATH", "models/tts")

    assert config._env_path("VERITY_TEST_PATH", tmp_path) == (
        config.ROOT_DIR / "models" / "tts"
    )


def test_env_path_keeps_absolute_path(monkeypatch, tmp_path):
    target = tmp_path / "models"
    monkeypatch.setenv("VERITY_TEST_PATH", str(target))

    assert config._env_path("VERITY_TEST_PATH", target) == target


def test_env_path_expands_home(monkeypatch):
    monkeypatch.setenv("VERITY_TEST_PATH", "~/verity-models")

    assert config._env_path("VERITY_TEST_PATH", config.ROOT_DIR) == (
        Path.home() / "verity-models"
    )
