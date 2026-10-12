"""Shared pytest fixtures for the Gemini backend tests."""

from __future__ import annotations

from pathlib import Path

import pytest
from gemini_fakes import FakeFfmpeg, install_fake_ffmpeg


@pytest.fixture
def fake_ffmpeg(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> FakeFfmpeg:
    """A real executable standing in for ffmpeg that records its argv."""
    return install_fake_ffmpeg(tmp_path, monkeypatch)
