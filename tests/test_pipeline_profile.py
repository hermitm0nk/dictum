"""Profile-driven pipeline integration points.

Covers the two gaps that a Gemini profile hits:
  * `Daemon._polish` must skip the polishing state when `[llm] backend = "none"`;
  * `dictum once` must build the ASR backend from the profile (not a hardcoded
    Parakeet), own its start/transcribe/stop lifetime, and skip polishing when
    the LLM is disabled.
"""

from __future__ import annotations

import asyncio
import tomllib
from pathlib import Path

import pytest
from typer.testing import CliRunner

from dictum import cli, daemon
from dictum.models import LlmConfig, Profile, Transcript

runner = CliRunner()


# ─────────────────────────────────────────────────────────────────────────────
# daemon polishing gate
# ─────────────────────────────────────────────────────────────────────────────


class _FakeRecorder:
    """Recorder stand-in (the real one needs an audio device)."""

    def __init__(self, *args: object, **kwargs: object) -> None:
        pass


def _daemon(profile: Profile, monkeypatch: pytest.MonkeyPatch) -> daemon.Daemon:
    monkeypatch.setattr(daemon, "Recorder", _FakeRecorder)
    d = daemon.Daemon(profile=profile)

    async def _notify(*args: object, **kwargs: object) -> None:
        return None

    monkeypatch.setattr(d.notifier, "notify", _notify)
    monkeypatch.setattr(d.notifier, "close", _notify)
    return d


def _spy_polish(d: daemon.Daemon, text: str, monkeypatch: pytest.MonkeyPatch) -> list[Transcript]:
    calls: list[Transcript] = []

    async def _polish(transcript: Transcript, profile: Profile) -> str:
        calls.append(transcript)
        return text

    monkeypatch.setattr(d.llm, "polish", _polish)
    return calls


def test_polish_skipped_for_llm_backend_none(monkeypatch: pytest.MonkeyPatch) -> None:
    """backend = "none" must skip polishing entirely, not just be a no-op."""
    profile = Profile(name="gemini", llm=LlmConfig(backend="none"))
    d = _daemon(profile, monkeypatch)
    calls = _spy_polish(d, "should not happen", monkeypatch)

    polished = asyncio.run(d._polish(Transcript(text="raw text"), profile))

    assert polished is None
    assert calls == []
    assert d.state == daemon.DictumState.IDLE  # never entered POLISHING


def test_polish_skipped_for_llm_backend_none_uppercase(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    profile = Profile(name="gemini", llm=LlmConfig(backend="NONE"))
    d = _daemon(profile, monkeypatch)
    calls = _spy_polish(d, "should not happen", monkeypatch)

    polished = asyncio.run(d._polish(Transcript(text="raw text"), profile))

    assert polished is None
    assert calls == []


def test_polish_skipped_when_llm_section_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    profile = Profile(name="gemini", llm=None)
    d = _daemon(profile, monkeypatch)

    polished = asyncio.run(d._polish(Transcript(text="raw text"), profile))

    assert polished is None


def test_polish_runs_for_enabled_llm(monkeypatch: pytest.MonkeyPatch) -> None:
    """A real LLM backend still polishes and reports the state transition."""
    profile = Profile(
        name="default",
        llm=LlmConfig(backend="openai-compatible", base_url="http://127.0.0.1:1"),
    )
    d = _daemon(profile, monkeypatch)
    calls = _spy_polish(d, "polished text", monkeypatch)

    polished = asyncio.run(d._polish(Transcript(text="raw text"), profile))

    assert polished == "polished text"
    assert len(calls) == 1
    assert d.state == daemon.DictumState.POLISHING


def test_notifications_persist_until_idle_and_skip_stale_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    d = _daemon(Profile(name="gemini", llm=LlmConfig(backend="none")), monkeypatch)
    entered = asyncio.Event()
    release = asyncio.Event()
    events: list[tuple] = []

    async def notify(title, body, *, icon=None, timeout=None, **kwargs):
        events.append(("notify", title, body, icon, timeout))
        if title == "📝 Transcribing":
            entered.set()
            await release.wait()

    async def close():
        events.append(("close",))

    monkeypatch.setattr(d.notifier, "notify", notify)
    monkeypatch.setattr(d.notifier, "close", close)

    async def run() -> None:
        d._set(daemon.DictumState.TRANSCRIBING)
        await entered.wait()
        assert events[0] == (
            "notify",
            "📝 Transcribing",
            "Converting speech to text…",
            "accessories-text-editor",
            0,
        )
        d._set(daemon.DictumState.PASTING)
        d._set(daemon.DictumState.IDLE)
        release.set()
        await asyncio.sleep(0)
        await asyncio.sleep(0)

    asyncio.run(run())
    assert events == [
        ("notify", "📝 Transcribing", "Converting speech to text…", "accessories-text-editor", 0),
        ("close",),
    ]


def test_example_gemini_config_selects_gemini_backend(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """examples/config-gemini.toml parses and wires the Gemini backend."""
    from dictum.asr import create_asr_backend
    from dictum.asr_gemini import GeminiASR
    from dictum.config import load_profile
    from dictum.llm_local import polish_enabled
    from dictum.models import AsrConfig

    example = Path(__file__).resolve().parents[1] / "examples" / "config-gemini.toml"
    config_dir = tmp_path / "dictum"
    config_dir.mkdir()
    (config_dir / "config.toml").write_text(example.read_text(encoding="utf-8"))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)

    profile = load_profile("gemini")

    assert profile.asr.backend == "gemini"
    assert profile.asr.gemini_transcription_mode == "smart"
    assert profile.asr.gemini_bitrate_kbps == 32
    assert profile.llm is not None
    assert profile.llm.backend == "none"
    assert not polish_enabled(profile)

    asr = create_asr_backend(profile, use_server=False)
    assert isinstance(asr, GeminiASR)
    assert asr.transcription_mode == "smart"
    assert asr.bitrate_kbps == 32
    assert asr.api_key_env == "GEMINI_API_KEY"

    # Every key documented in the example maps onto a real AsrConfig field.
    section = tomllib.loads(example.read_text(encoding="utf-8"))["profiles"]["gemini"]["asr"]
    fields = set(AsrConfig.model_fields)
    unknown = {key for key in section if key not in fields}
    assert unknown == set()


# ─────────────────────────────────────────────────────────────────────────────
# `dictum once`
# ─────────────────────────────────────────────────────────────────────────────


class _FakeOnceAsr:
    """ASR backend recording its lifecycle, like the Gemini backend does."""

    def __init__(self, events: list[str]) -> None:
        self.events = events

    async def start(self) -> None:
        self.events.append("asr-start")

    async def transcribe(self, audio_path: Path, profile: Profile | None) -> Transcript:
        self.events.append("asr-transcribe")
        assert Path(audio_path).exists()
        return Transcript(text="hello from gemini")

    async def stop(self) -> None:
        self.events.append("asr-stop")


def _run_once(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    profile: Profile,
    args: list[str],
    llm_factory=None,  # type: ignore[no-untyped-def]
) -> tuple[object, list[str]]:
    events: list[str] = []

    class _Recorder:
        def __init__(self, *a: object, **k: object) -> None:
            pass

        def start(self) -> None:
            events.append("record-start")

        @property
        def is_recording(self) -> bool:
            return False

        def stop(self) -> Path:
            events.append("record-stop")
            audio = tmp_path / "rec.wav"
            audio.write_bytes(b"RIFF")
            return audio

    class _Sink:
        async def deliver(self, result, target, profile) -> None:  # type: ignore[no-untyped-def]
            events.append(f"deliver:{target.value}:{result.final_text}")

    def _factory(profile: Profile, use_server: bool = True):
        events.append(f"factory:use_server={use_server}")
        return _FakeOnceAsr(events)

    def _no_llm(profile: Profile):
        raise AssertionError("LLM backend must not be built when polishing is disabled")

    monkeypatch.setattr("dictum.recorder.Recorder", _Recorder)
    monkeypatch.setattr("dictum.asr.create_asr_backend", _factory)
    monkeypatch.setattr("dictum.config.load_profile", lambda name: profile)
    monkeypatch.setattr("dictum.output.OutputSink", _Sink)
    monkeypatch.setattr("dictum.llm_local.create_llm_backend", llm_factory or _no_llm)

    result = runner.invoke(cli.app, ["once", *args])
    return result, events


def test_once_uses_profile_asr_backend_and_skips_polish(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    profile = Profile(name="gemini", llm=LlmConfig(backend="none"))
    profile.asr.backend = "gemini"

    result, events = _run_once(monkeypatch, tmp_path, profile, ["--result", "none"])

    assert result.exit_code == 0, result.stderr
    assert "hello from gemini" in result.stderr
    # Select the backend before capture so streaming capabilities are available.
    assert events == [
        "factory:use_server=False",
        "record-start",
        "record-stop",
        "asr-start",
        "asr-transcribe",
        "asr-stop",
        "deliver:none:hello from gemini",
    ]


def test_once_loads_profile_before_transcribing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The profile drives the backend choice, so it must be loaded first."""
    profile = Profile(name="default", llm=LlmConfig(backend="none"))

    result, events = _run_once(monkeypatch, tmp_path, profile, ["--result", "none"])

    assert result.exit_code == 0, result.stderr
    assert events.index("factory:use_server=False") < events.index("asr-transcribe")


def test_once_polishes_when_llm_enabled(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    profile = Profile(
        name="default",
        llm=LlmConfig(backend="openai-compatible", base_url="http://127.0.0.1:1"),
    )

    async def _polish(transcript: Transcript, profile: Profile) -> str:
        return "polished text"

    result, events = _run_once(
        monkeypatch,
        tmp_path,
        profile,
        ["--result", "none"],
        llm_factory=lambda p: _FakeLlm(_polish),
    )

    assert result.exit_code == 0, result.stderr
    assert "polished text" in result.stderr
    assert events[-1] == "deliver:none:polished text"


class _FakeLlm:
    def __init__(self, polish) -> None:  # type: ignore[no-untyped-def]
        self._polish = polish

    async def polish(self, transcript: Transcript, profile: Profile) -> str:
        return await self._polish(transcript, profile)


def test_once_end_to_end_with_gemini_profile(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """`dictum once` drives the real Gemini backend (mocked HTTP + ffmpeg)."""
    from gemini_fakes import FakeGeminiApi, install_fake_ffmpeg, patch_client

    api = FakeGeminiApi()
    ffmpeg = install_fake_ffmpeg(tmp_path, monkeypatch)
    monkeypatch.setenv("GEMINI_API_KEY", "test-key-abc123")

    profile = Profile(name="gemini", llm=LlmConfig(backend="none"))
    profile.asr.backend = "gemini"
    profile.asr.gemini_ffmpeg_path = str(ffmpeg.path)

    delivered: list[str] = []

    class _Recorder:
        def __init__(self, *a: object, **k: object) -> None:
            pass

        def start(self) -> None:
            return None

        @property
        def is_recording(self) -> bool:
            return False

        def stop(self) -> Path:
            audio = tmp_path / "rec.wav"
            audio.write_bytes(b"RIFF")
            return audio

    class _Sink:
        async def deliver(self, result, target, profile) -> None:  # type: ignore[no-untyped-def]
            delivered.append(result.final_text)

    monkeypatch.setattr("dictum.recorder.Recorder", _Recorder)
    monkeypatch.setattr("dictum.config.load_profile", lambda name: profile)
    monkeypatch.setattr("dictum.output.OutputSink", _Sink)
    patch_client(monkeypatch, api.transport)

    result = runner.invoke(cli.app, ["once", "--result", "none"])

    assert result.exit_code == 0, result.stderr
    assert "hello from gemini" in result.stderr
    assert delivered == ["hello from gemini"]
    # ffmpeg encoded and sent the short clip inline in one request.
    assert ffmpeg.argv()
    assert not api.calls("upload")
    assert len(api.calls("interactions")) == 1
    assert not api.calls("delete")


def test_once_reports_asr_failure(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """A hosted backend failure surfaces as a CLI error, not a silent fallback."""
    profile = Profile(name="gemini", llm=LlmConfig(backend="none"))
    events: list[str] = []

    class _Failing:
        async def start(self) -> None:
            return None

        async def transcribe(self, audio_path: Path, profile: Profile | None) -> Transcript:
            raise RuntimeError("Gemini transcription failed (HTTP 429): quota")

        async def stop(self) -> None:
            events.append("asr-stop")
            return None

    monkeypatch.setattr(
        "dictum.recorder.Recorder",
        lambda *a, **k: _NoopRecorder(tmp_path / "rec.wav"),
    )
    monkeypatch.setattr("dictum.asr.create_asr_backend", lambda p, use_server=True: _Failing())
    monkeypatch.setattr("dictum.config.load_profile", lambda name: profile)

    result = runner.invoke(cli.app, ["once", "--result", "none"])

    assert result.exit_code != 0
    assert isinstance(result.exception, RuntimeError)
    assert "quota" in str(result.exception)
    # The backend lifetime is torn down even when transcription fails.
    assert events == ["asr-stop"]


class _NoopRecorder:
    def __init__(self, path: Path) -> None:
        self.path = path

    def start(self) -> None:
        return None

    @property
    def is_recording(self) -> bool:
        return False

    def stop(self) -> Path:
        self.path.write_bytes(b"RIFF")
        return self.path
