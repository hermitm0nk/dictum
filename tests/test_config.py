"""Tests for the configuration loader."""

from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import patch

import pytest

from dictum.config import config_path, load_profile
from dictum.models import ResultTarget


@pytest.fixture
def config_dir(tmp_path: Path) -> Path:
    """Create a fake XDG_CONFIG_HOME and return the dictum config dir."""
    os.environ["XDG_CONFIG_HOME"] = str(tmp_path)
    d = tmp_path / "dictum"
    d.mkdir(parents=True)
    return d


def _write_config(config_dir: Path, content: str) -> None:
    (config_dir / "config.toml").write_text(content)


def test_load_profile_no_config_file(tmp_path: Path) -> None:
    """Returns defaults when config.toml does not exist."""
    with patch.dict(os.environ, {"XDG_CONFIG_HOME": str(tmp_path)}):
        profile = load_profile("default")
    assert profile.name == "default"
    assert profile.result == ResultTarget.PASTE
    assert "transcription editor" in profile.prompt


def test_load_profile_simple_prompt(config_dir: Path) -> None:
    """Loads a custom prompt from config.toml."""
    _write_config(
        config_dir,
        """
[profiles.default]
prompt = "Just fix commas."
result = "clipboard"
""",
    )
    profile = load_profile("default")
    assert profile.prompt == "Just fix commas."
    assert profile.result == ResultTarget.CLIPBOARD


def test_load_profile_not_found(config_dir: Path) -> None:
    """Returns defaults when the named profile is missing."""
    _write_config(
        config_dir,
        """
[profiles.default]
prompt = "Default prompt."
""",
    )
    profile = load_profile("nonexistent")
    assert profile.name == "nonexistent"
    assert "transcription editor" in profile.prompt  # falls back to default


def test_load_profile_with_asr(config_dir: Path) -> None:
    """Parses nested ASR config."""
    _write_config(
        config_dir,
        """
[profiles.default]
[profiles.default.asr]
backend = "parakeet"
model = "custom-model"
""",
    )
    profile = load_profile("default")
    assert profile.asr.model == "custom-model"


def test_load_profile_with_llm(config_dir: Path) -> None:
    """Parses nested LLM config."""
    _write_config(
        config_dir,
        """
[profiles.default]
[profiles.default.llm]
backend = "openai-compatible"
base_url = "http://localhost:9090"
model = "test-model"
temperature = 0.5
""",
    )
    profile = load_profile("default")
    assert profile.llm is not None
    assert profile.llm.backend == "openai-compatible"
    assert profile.llm.model == "test-model"
    assert profile.llm.temperature == 0.5


def test_load_profile_llm_disabled(config_dir: Path) -> None:
    """Supports backend = 'none' to disable LLM."""
    _write_config(
        config_dir,
        """
[profiles.default]
[profiles.default.llm]
backend = "none"
""",
    )
    profile = load_profile("default")
    assert profile.llm is not None
    assert profile.llm.backend == "none"


def test_load_profile_prompt_file(config_dir: Path) -> None:
    """Loads prompt from an external file via prompt_file."""
    prompt_file = config_dir / "custom_prompt.txt"
    prompt_file.write_text("Edit from file, nothing else.")

    _write_config(
        config_dir,
        """
[profiles.default]
prompt_file = "custom_prompt.txt"
""",
    )
    profile = load_profile("default")
    assert profile.prompt == "Edit from file, nothing else."


def test_load_profile_prompt_file_absolute(config_dir: Path) -> None:
    """prompt_file works with absolute paths too."""
    prompt_file = Path("/tmp/dictum_test_prompt.txt")
    prompt_file.write_text("Absolute path prompt.")

    _write_config(
        config_dir,
        f"""
[profiles.default]
prompt_file = "{prompt_file}"
""",
    )
    profile = load_profile("default")
    assert profile.prompt == "Absolute path prompt."
    prompt_file.unlink()


def test_load_profile_malformed_toml(config_dir: Path) -> None:
    """Falls back to defaults on malformed TOML."""
    _write_config(config_dir, "this is not valid toml {{{")
    profile = load_profile("default")
    assert "transcription editor" in profile.prompt


def test_config_path_respects_xdg() -> None:
    """config_path uses XDG_CONFIG_HOME when set."""
    with patch.dict(os.environ, {"XDG_CONFIG_HOME": "/tmp/fake_xdg"}):
        p = config_path()
    assert p == Path("/tmp/fake_xdg/dictum/config.toml")