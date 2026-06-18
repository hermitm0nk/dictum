"""Regression tests for ParakeetASR binary-management behavior.

The daemon's `create_asr_backend` factory must produce an instance whose
`_binary_managed` flag is True when no DICTUM_PARAKEET_BIN override is set,
so that `ensure_native_binary("parakeet-main")` fires on first start and
downloads CrispASR. Passing an explicit `binary=` to the constructor
defeats this check, so the factory must pass `binary=None`.
"""

from __future__ import annotations

from pathlib import Path

from dictum.asr import ParakeetASR, create_asr_backend
from dictum.models import Profile


def test_create_asr_backend_marks_binary_as_managed(monkeypatch) -> None:
    """No env override → factory must produce a managed-binary instance."""
    monkeypatch.delenv("DICTUM_PARAKEET_BIN", raising=False)
    profile = Profile()
    asr = create_asr_backend(profile, use_server=True)
    assert isinstance(asr, ParakeetASR)
    assert asr._binary_managed is True


def test_create_asr_backend_marks_binary_as_unmanaged_when_env_set(
    monkeypatch, tmp_path: Path
) -> None:
    """DICTUM_PARAKEET_BIN set → factory must NOT mark binary as managed."""
    custom = tmp_path / "my-parakeet"
    monkeypatch.setenv("DICTUM_PARAKEET_BIN", str(custom))
    profile = Profile()
    asr = create_asr_backend(profile, use_server=True)
    assert isinstance(asr, ParakeetASR)
    assert asr._binary_managed is False
    assert asr.binary == custom


def test_parakeet_asr_default_construction_is_managed(monkeypatch) -> None:
    """Bare ParakeetASR() with no env override is managed."""
    monkeypatch.delenv("DICTUM_PARAKEET_BIN", raising=False)
    asr = ParakeetASR()
    assert asr._binary_managed is True


def test_parakeet_asr_explicit_binary_is_not_managed(monkeypatch, tmp_path: Path) -> None:
    """Passing binary=<path> opts out of the installer."""
    monkeypatch.delenv("DICTUM_PARAKEET_BIN", raising=False)
    custom = tmp_path / "explicit-parakeet"
    asr = ParakeetASR(binary=custom)
    assert asr._binary_managed is False
    assert asr.binary == custom


def test_parakeet_asr_env_override_is_not_managed(monkeypatch, tmp_path: Path) -> None:
    """DICTUM_PARAKEET_BIN env var opts out of the installer."""
    custom = tmp_path / "env-parakeet"
    monkeypatch.setenv("DICTUM_PARAKEET_BIN", str(custom))
    asr = ParakeetASR()
    assert asr._binary_managed is False
    assert asr.binary == custom


def test_create_asr_backend_unknown_backend_falls_back_to_managed(
    monkeypatch,
) -> None:
    """Unknown backend name falls back to ParakeetASR with managed binary."""
    monkeypatch.delenv("DICTUM_PARAKEET_BIN", raising=False)
    profile = Profile()
    profile.asr.backend = "does-not-exist"
    asr = create_asr_backend(profile, use_server=True)
    assert isinstance(asr, ParakeetASR)
    assert asr._binary_managed is True
