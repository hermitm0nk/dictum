from __future__ import annotations

import asyncio
import sys
import types
from pathlib import Path

import pytest

from dictum.asr import create_asr_backend
from dictum.asr_gemini_live import GeminiLiveASR
from dictum.models import Profile


class FakeSession:
    def __init__(
        self, receive_error: Exception | None = None, *, hold_after_transcription: bool = False
    ) -> None:
        self.sent: list[dict] = []
        self.receive_error = receive_error
        self.hold_after_transcription = hold_after_transcription

    async def __aenter__(self) -> FakeSession:
        return self

    async def __aexit__(self, *args: object) -> None:
        pass

    async def send_realtime_input(self, **kwargs: object) -> None:
        self.sent.append(kwargs)

    async def receive(self):
        if self.receive_error is not None:
            raise self.receive_error
        yield types.SimpleNamespace(
            server_content=types.SimpleNamespace(
                input_transcription=types.SimpleNamespace(text="live transcript"),
                turn_complete=not self.hold_after_transcription,
            )
        )
        if self.hold_after_transcription:
            await asyncio.Future()


class FakeLive:
    def __init__(self, session: FakeSession) -> None:
        self.session = session
        self.kwargs: dict = {}
        self.closed = False

    def connect(self, **kwargs: object) -> FakeSession:
        self.kwargs = kwargs
        return self.session


def install_fake_genai(monkeypatch: pytest.MonkeyPatch, session: FakeSession) -> FakeLive:
    class Config:
        def __init__(self, **kwargs: object) -> None:
            self.values = kwargs

    class Client:
        def __init__(self, *, api_key: str) -> None:
            self.api_key = api_key
            self.aio = types.SimpleNamespace(live=live, aclose=self.aclose)

        async def aclose(self) -> None:
            live.closed = True

        def close(self) -> None:
            pass

    live = FakeLive(session)
    genai = types.ModuleType("google.genai")
    genai.Client = Client
    genai_types = types.ModuleType("google.genai.types")
    genai_types.LiveConnectConfig = Config
    genai_types.AudioTranscriptionConfig = Config
    genai_types.Blob = Config
    genai_types.RealtimeInputConfig = Config
    genai_types.AutomaticActivityDetection = Config
    genai_types.ActivityStart = Config
    genai_types.ActivityEnd = Config
    genai_types.Modality = types.SimpleNamespace(TEXT="TEXT")
    genai_types.AudioTranscriptionConfigMode = types.SimpleNamespace(
        SMART="SMART", VERBATIM="VERBATIM"
    )
    google = types.ModuleType("google")
    google.__path__ = []
    google.genai = genai
    monkeypatch.setitem(sys.modules, "google", google)
    monkeypatch.setitem(sys.modules, "google.genai", genai)
    monkeypatch.setitem(sys.modules, "google.genai.types", genai_types)
    return live


def test_gemini_live_backend_config_selects_live_model() -> None:
    profile = Profile()
    profile.asr.backend = "gemini-live"
    profile.asr.gemini_live_model = "gemini-live-test"
    profile.asr.gemini_api_key_env = "DICTUM_LIVE_TEST_KEY"
    profile.asr.gemini_language_codes = ["en-US"]
    profile.asr.gemini_transcription_mode = "verbatim"

    backend = create_asr_backend(profile)

    assert isinstance(backend, GeminiLiveASR)
    assert backend.model == "gemini-live-test"
    assert backend.api_key_env == "DICTUM_LIVE_TEST_KEY"
    assert backend.language_codes == ["en-US"]
    assert backend.transcription_mode == "verbatim"


def test_live_backend_streams_pcm_and_returns_final_text(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("DICTUM_LIVE_TEST_KEY", "do-not-log-this")
    monkeypatch.setattr("dictum.asr_gemini_live.shutil.which", lambda _: "/usr/bin/ffmpeg")
    audio = tmp_path / "recording.wav"
    audio.write_bytes(b"wav")
    session = FakeSession()
    live = install_fake_genai(monkeypatch, session)
    backend = GeminiLiveASR(
        api_key_env="DICTUM_LIVE_TEST_KEY",
        model="gemini-3.5-transcribe-live",
        transcription_mode="smart",
        language_codes=["en-US"],
    )
    monkeypatch.setattr(backend, "_to_pcm", lambda _: asyncio.sleep(0, result=b"a" * 6402))

    transcript = asyncio.run(backend.transcribe(audio))

    assert transcript.text == "live transcript"
    assert len(session.sent) == 5  # explicit start, three PCM chunks, explicit end
    assert live.closed
    assert "activity_start" in session.sent[0]
    assert [len(item["audio"].values["data"]) for item in session.sent[1:-1]] == [3200, 3200, 2]
    assert session.sent[1]["audio"].values["mime_type"] == "audio/pcm;rate=16000"
    assert "activity_end" in session.sent[-1]
    assert live.kwargs["model"] == "gemini-3.5-transcribe-live"
    config = live.kwargs["config"].values
    assert config["response_modalities"] == ["TEXT"]
    assert config["realtime_input_config"].values["automatic_activity_detection"].values == {
        "disabled": True
    }
    assert config["input_audio_transcription"].values == {
        "language_codes": ["en-US"],
        "mode": "SMART",
    }


def test_live_backend_returns_on_final_transcript_without_turn_complete(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("DICTUM_LIVE_TEST_KEY", "test-key")
    monkeypatch.setattr("dictum.asr_gemini_live.shutil.which", lambda _: "/usr/bin/ffmpeg")
    audio = tmp_path / "recording.wav"
    audio.write_bytes(b"wav")
    session = FakeSession(hold_after_transcription=True)
    install_fake_genai(monkeypatch, session)
    backend = GeminiLiveASR(api_key_env="DICTUM_LIVE_TEST_KEY", timeout=0.05)
    monkeypatch.setattr(backend, "_to_pcm", lambda _: asyncio.sleep(0, result=b"pcm"))

    transcript = asyncio.run(backend.transcribe(audio))

    assert transcript.text == "live transcript"


def test_live_preflight_does_not_make_remote_calls_without_key(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.delenv("DICTUM_LIVE_MISSING_KEY", raising=False)
    audio = tmp_path / "recording.wav"
    audio.write_bytes(b"wav")
    backend = GeminiLiveASR(api_key_env="DICTUM_LIVE_MISSING_KEY")

    with pytest.raises(RuntimeError, match="DICTUM_LIVE_MISSING_KEY"):
        asyncio.run(backend.transcribe(audio))


@pytest.mark.parametrize("attribute, code", [("status_code", 503), ("code", 1008)])
def test_live_api_error_diagnostic_excludes_remote_message_and_secrets(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
    attribute: str,
    code: int,
) -> None:
    class RemoteError(Exception):
        pass

    setattr(RemoteError, attribute, code)

    secret = "private-test-key"
    monkeypatch.setenv("DICTUM_LIVE_TEST_KEY", secret)
    monkeypatch.setattr("dictum.asr_gemini_live.shutil.which", lambda _: "/usr/bin/ffmpeg")
    audio = tmp_path / "recording.wav"
    audio.write_bytes(b"wav")
    session = FakeSession(RemoteError(f"remote body contains {secret}"))
    install_fake_genai(monkeypatch, session)
    backend = GeminiLiveASR(api_key_env="DICTUM_LIVE_TEST_KEY")
    monkeypatch.setattr(backend, "_to_pcm", lambda _: asyncio.sleep(0, result=b"pcm"))

    with pytest.raises(RuntimeError, match="request failed"):
        asyncio.run(backend.transcribe(audio))

    messages = " ".join(record.getMessage() for record in caplog.records)
    assert "RemoteError" in messages
    assert str(code) in messages
    assert secret not in messages
    assert "remote body" not in messages


def test_live_missing_final_transcript_has_short_deadline(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    class NoSpeechSession(FakeSession):
        async def receive(self):
            await asyncio.Future()
            yield  # pragma: no cover

    monkeypatch.setenv("DICTUM_LIVE_TEST_KEY", "test-key")
    monkeypatch.setattr("dictum.asr_gemini_live.shutil.which", lambda _: "/usr/bin/ffmpeg")
    monkeypatch.setattr("dictum.asr_gemini_live.FINALIZATION_TIMEOUT", 0.01, raising=False)
    audio = tmp_path / "recording.wav"
    audio.write_bytes(b"wav")
    install_fake_genai(monkeypatch, NoSpeechSession())
    backend = GeminiLiveASR(api_key_env="DICTUM_LIVE_TEST_KEY", timeout=120)
    monkeypatch.setattr(backend, "_to_pcm", lambda _: asyncio.sleep(0, result=b"pcm"))

    async def run():
        return await asyncio.wait_for(backend.transcribe(audio), timeout=0.2)

    with pytest.raises(RuntimeError, match="no final transcript"):
        asyncio.run(run())


def test_existing_gemini_backend_remains_selectable() -> None:
    from dictum.asr_gemini import GeminiASR

    profile = Profile()
    profile.asr.backend = "gemini"
    assert isinstance(create_asr_backend(profile), GeminiASR)
