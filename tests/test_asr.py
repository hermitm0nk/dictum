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


def test_create_asr_backend_selects_gemini(monkeypatch) -> None:
    """`backend = "gemini"` builds a hosted backend, not a local Parakeet."""
    from dictum.asr_gemini import GeminiASR

    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    profile = Profile()
    profile.asr.backend = "gemini"
    asr = create_asr_backend(profile, use_server=True)

    assert isinstance(asr, GeminiASR)
    # The local Parakeet model name must never leak into the hosted backend.
    assert asr.model == "gemini-3.5-transcribe"
    assert asr.model != profile.asr.model
    assert asr.bitrate_kbps == 32
    assert asr.transcription_mode == "smart"
    assert asr.api_key_env == "GEMINI_API_KEY"
    assert asr.language_codes == []
    assert asr.ffmpeg_path == "ffmpeg"


def test_create_asr_backend_gemini_passes_profile_config(monkeypatch) -> None:
    """Gemini settings in `[asr]` are forwarded verbatim."""
    from dictum.asr_gemini import GeminiASR

    profile = Profile()
    profile.asr.backend = "gemini"
    profile.asr.gemini_model = "gemini-3.5-transcribe-preview"
    profile.asr.gemini_api_key_env = "MY_GEMINI_KEY"
    profile.asr.gemini_bitrate_kbps = 24
    profile.asr.gemini_transcription_mode = "verbatim"
    profile.asr.gemini_language_codes = ["en-US", "de-DE"]
    profile.asr.gemini_ffmpeg_path = "ffmpeg-pipewire"
    profile.asr.gemini_timeout_seconds = 45.0

    asr = create_asr_backend(profile, use_server=False)

    assert isinstance(asr, GeminiASR)
    assert asr.model == "gemini-3.5-transcribe-preview"
    assert asr.api_key_env == "MY_GEMINI_KEY"
    assert asr.bitrate_kbps == 24
    assert asr.transcription_mode == "verbatim"
    assert asr.language_codes == ["en-US", "de-DE"]
    assert asr.ffmpeg_path == "ffmpeg-pipewire"
    assert asr.timeout == 45.0


def test_create_asr_backend_gemini_needs_no_local_model(monkeypatch, tmp_path) -> None:
    """Selecting Gemini must not resolve or download a Parakeet model."""
    from dictum.asr_gemini import GeminiASR

    monkeypatch.setenv("DICTUM_PARAKEET_MODEL", str(tmp_path / "missing.gguf"))
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    profile = Profile()
    profile.asr.backend = "gemini"

    asr = create_asr_backend(profile, use_server=True)

    assert isinstance(asr, GeminiASR)
    assert not hasattr(asr, "model_path")
    assert not (tmp_path / "missing.gguf").exists()
