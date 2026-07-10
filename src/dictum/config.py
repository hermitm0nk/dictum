"""Configuration loader — reads and parses ~/.config/dictum/config.toml into Profile models."""

from __future__ import annotations

import logging
import os
import tomllib
from pathlib import Path

from dictum.models import AsrConfig, LlmConfig, Profile, ResultTarget

log = logging.getLogger(__name__)


def config_dir() -> Path:
    """Return the dictum config directory ($XDG_CONFIG_HOME/dictum or ~/.config/dictum)."""
    xdg = os.environ.get("XDG_CONFIG_HOME")
    if xdg:
        return Path(xdg) / "dictum"
    return Path.home() / ".config" / "dictum"


def config_path() -> Path:
    """Return the path to config.toml."""
    return config_dir() / "config.toml"


def load_profile(profile_name: str = "default") -> Profile:
    """Load a Profile from config.toml by name.

    If config.toml does not exist, returns a Profile with defaults.
    If the named profile is not found, returns a Profile with defaults and logs a warning.
    """
    path = config_path()
    if not path.exists():
        log.debug("No config file at %s, using defaults", path)
        return Profile(name=profile_name)

    try:
        with open(path, "rb") as f:
            data = tomllib.load(f)
    except (tomllib.TOMLDecodeError, OSError) as exc:
        log.warning("Failed to read config %s: %s, using defaults", path, exc)
        return Profile(name=profile_name)

    profiles = data.get("profiles", {})
    section = profiles.get(profile_name)
    if section is None:
        log.warning("Profile '%s' not found in %s, using defaults", profile_name, path)
        return Profile(name=profile_name)

    return _profile_from_section(profile_name, section)


def _profile_from_section(name: str, section: dict) -> Profile:  # type: ignore[type-arg]
    """Build a Profile from a TOML profile section dict."""
    kwargs: dict = {"name": name}  # type: ignore[type-arg]

    # Simple scalar fields
    if "prompt" in section:
        kwargs["prompt"] = section["prompt"]

    if "result" in section:
        kwargs["result"] = ResultTarget(section["result"])

    # prompt_file: read the file contents into prompt, or pass the path
    if "prompt_file" in section:
        pfile = Path(section["prompt_file"])
        if not pfile.is_absolute():
            pfile = config_dir() / pfile
        try:
            kwargs["prompt"] = pfile.read_text(encoding="utf-8").strip()
            log.info("Loaded prompt from %s", pfile)
        except OSError as exc:
            log.warning("Could not read prompt_file %s: %s", pfile, exc)

    # Nested ASR config
    if "asr" in section:
        kwargs["asr"] = AsrConfig(**section["asr"])

    # Nested LLM config — explicitly allow None to disable LLM
    if "llm" in section:
        llm_section = section["llm"]
        if llm_section is None:
            kwargs["llm"] = None
        else:
            kwargs["llm"] = LlmConfig(**llm_section)

    return Profile(**kwargs)


