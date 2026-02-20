"""Tests for settings path resolution from PROJECT_ROOT."""

from pathlib import Path

import pytest

from txt2img.config import get_settings


@pytest.fixture(autouse=True)
def clear_settings_cache():
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


def test_default_paths_use_current_directory_as_project_root(monkeypatch, tmp_path: Path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("PROJECT_ROOT", raising=False)
    monkeypatch.delenv("OUTPUT_DIR", raising=False)
    monkeypatch.delenv("MODEL_CACHE_DIR", raising=False)
    monkeypatch.delenv("PRESETS_DIR", raising=False)
    monkeypatch.delenv("CLIENT_DIST_DIR", raising=False)

    settings = get_settings()

    assert settings.project_root == tmp_path.resolve()
    assert settings.output_dir == tmp_path / "outputs"
    assert settings.model_cache_dir == tmp_path / "models"
    assert settings.presets_dir == tmp_path / "server" / "presets"
    assert settings.client_dist_dir == tmp_path / "client" / "dist"


def test_relative_overrides_resolve_from_project_root(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("PROJECT_ROOT", str(tmp_path))
    monkeypatch.setenv("OUTPUT_DIR", "artifacts/outputs")
    monkeypatch.setenv("MODEL_CACHE_DIR", "cache/models")
    monkeypatch.setenv("PRESETS_DIR", "config/presets")
    monkeypatch.setenv("CLIENT_DIST_DIR", "web/dist")

    settings = get_settings()

    assert settings.project_root == tmp_path.resolve()
    assert settings.output_dir == tmp_path / "artifacts" / "outputs"
    assert settings.model_cache_dir == tmp_path / "cache" / "models"
    assert settings.presets_dir == tmp_path / "config" / "presets"
    assert settings.client_dist_dir == tmp_path / "web" / "dist"


def test_absolute_overrides_are_preserved(monkeypatch, tmp_path: Path):
    absolute_cache = tmp_path / "abs-cache"
    absolute_cache.mkdir(parents=True, exist_ok=True)

    monkeypatch.setenv("PROJECT_ROOT", str(tmp_path / "project-root"))
    monkeypatch.setenv("MODEL_CACHE_DIR", str(absolute_cache))

    settings = get_settings()

    assert settings.model_cache_dir == absolute_cache.resolve()
