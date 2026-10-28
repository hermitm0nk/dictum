"""Live capture integration without a physical microphone or Google credentials."""

from __future__ import annotations

import asyncio
import json
from unittest.mock import AsyncMock, Mock

import numpy as np
import pytest

from dictum import daemon, recorder
from dictum.backends import StreamingAsrBackend
from dictum.models import LlmConfig, Profile, ResultTarget, Transcript
from dictum.state import DictumState


class LiveAsr(StreamingAsrBackend):
    def __init__(self):
        self.audio_seen = asyncio.Event()
        self.closed = asyncio.Event()
        self.fail = False

    async def transcribe(self, audio_path, profile):
        raise AssertionError("Live capture must not transcribe a file")

    async def transcribe_stream(self, audio, profile=None, on_update=None):
        try:
            async for chunk in audio:
                assert chunk == b"pcm"
                if on_update:
                    on_update("speaking")
                self.audio_seen.set()
            if self.fail:
                raise RuntimeError("stream failed")
            return Transcript(text="final text")
        finally:
            self.closed.set()


class LiveRecorder:
    def __init__(self, path):
        self.path = path
        self.is_recording = False
        self.ended = asyncio.Event()

    def start(self, *, streaming=False):
        assert streaming
        self.is_recording = True

    def stop(self):
        self.is_recording = False
        self.ended.set()
        self.path.write_bytes(b"wav")
        return self.path

    async def audio_chunks(self):
        yield b"pcm"
        await self.ended.wait()


async def ipc(d, cmd, **kwargs):
    reader = Mock(readline=AsyncMock(return_value=json.dumps({"cmd": cmd, **kwargs}).encode()))
    writer = Mock(drain=AsyncMock(), wait_closed=AsyncMock())
    await d.handle_connection(reader, writer)
    return json.loads(writer.write.call_args.args[0])


@pytest.mark.parametrize("end", ["stop", "toggle", "cancel", "error"])
def test_daemon_streams_during_capture_and_finishes_once(monkeypatch, tmp_path, end):
    async def run():
        rec = LiveRecorder(tmp_path / "audio.wav")
        asr = LiveAsr()
        monkeypatch.setattr(daemon, "Recorder", lambda **kw: rec)
        monkeypatch.setattr(daemon, "create_asr_backend", lambda *a, **kw: asr)
        profile = Profile(llm=LlmConfig(backend="none"))
        d = daemon.Daemon(profile)
        monkeypatch.setattr(d, "_notify_state", AsyncMock())
        deliver = AsyncMock()
        monkeypatch.setattr(d.output, "deliver", deliver)
        assert (await ipc(d, "start"))["ok"]
        await asyncio.wait_for(asr.audio_seen.wait(), 1)
        assert rec.is_recording
        assert (await ipc(d, "status"))["live_transcript"] == "speaking"
        deliver.assert_not_called()
        if end == "error":
            asr.fail = True
            rec.ended.set()
            with pytest.raises(RuntimeError, match="stream failed"):
                await asyncio.wait_for(d._live_task, 1)
            assert not rec.is_recording
            assert d.state == DictumState.FAILED
            deliver.assert_not_called()
            return
        else:
            assert (await ipc(d, end, target="none"))["ok"]
        if end == "cancel":
            assert asr.closed.is_set()
            assert not rec.is_recording
            assert d.state == DictumState.IDLE
            assert d.live_transcript == ""
            deliver.assert_not_called()
        else:
            result = await asyncio.wait_for(d._live_task, 1)
            assert result.final_text == "final text"
            assert result.target == ResultTarget.NONE
            assert deliver.await_count == 1
            assert d.state == DictumState.IDLE

    asyncio.run(run())


def test_daemon_recorder_survives_silence_until_stopped(monkeypatch, tmp_path):
    import time

    monkeypatch.setattr(recorder, "_native_rate", lambda: 16000)
    monkeypatch.setattr(recorder.sd, "InputStream", lambda **kw: Mock())
    monkeypatch.setattr(recorder.Recorder, "_write_wav", lambda *a: tmp_path / "audio.wav")
    rec = recorder.Recorder(silence_timeout=None)
    rec.start(streaming=True)
    try:
        rec._audio_callback(np.zeros((160, 1), dtype=np.int16), 160, None, None)
        time.sleep(1.8)
        assert rec.is_recording
    finally:
        rec.stop()
    assert not rec.is_recording


def test_daemon_configures_manual_stop(monkeypatch):
    factory = Mock()
    monkeypatch.setattr(daemon, "Recorder", factory)
    monkeypatch.setattr(daemon, "create_asr_backend", Mock())
    daemon.Daemon()
    factory.assert_called_once_with(silence_timeout=None)


def test_live_backend_return_before_stop_does_not_paste(monkeypatch, tmp_path):
    async def run():
        rec = LiveRecorder(tmp_path / "audio.wav")
        asr = LiveAsr()
        asr.transcribe_stream = AsyncMock(return_value=Transcript(text="partial"))
        monkeypatch.setattr(daemon, "Recorder", lambda **kw: rec)
        monkeypatch.setattr(daemon, "create_asr_backend", lambda *a, **kw: asr)
        d = daemon.Daemon(Profile(llm=LlmConfig(backend="none")))
        monkeypatch.setattr(d, "_notify_state", AsyncMock())
        deliver = AsyncMock()
        monkeypatch.setattr(d.output, "deliver", deliver)
        await ipc(d, "start")
        with pytest.raises(RuntimeError, match="before recording was stopped"):
            await d._live_task
        assert d.state == DictumState.FAILED
        assert not rec.is_recording
        deliver.assert_not_called()

    asyncio.run(run())


@pytest.mark.parametrize("rate", [16000, 44100, 48000])
def test_recorder_stream_resampling_keeps_phase_and_pcm_format(monkeypatch, tmp_path, rate):
    monkeypatch.setattr(recorder, "_native_rate", lambda: rate)
    stream = Mock()
    monkeypatch.setattr(recorder.sd, "InputStream", lambda **kw: stream)
    monkeypatch.setattr(recorder.Recorder, "_write_wav", lambda *a: tmp_path / "unused.wav")
    rec = recorder.Recorder()
    rec.start(streaming=True)
    samples = np.arange(10_003, dtype=np.int16).reshape(-1, 1)
    for chunk in np.array_split(samples, 7):
        rec._audio_callback(chunk, len(chunk), None, None)
    rec.stop()

    async def read():
        return b"".join([chunk async for chunk in rec.audio_chunks()])

    output = np.frombuffer(asyncio.run(read()), dtype="<i2")
    indices = np.arange((len(samples) * 16000 + rate - 1) // rate) * rate // 16000
    np.testing.assert_array_equal(output, samples[:, 0][indices])
    stream.close.assert_called_once()


def test_recorder_queue_overflow_fails_instead_of_losing_speech(monkeypatch, tmp_path):
    monkeypatch.setattr(recorder, "_native_rate", lambda: 16000)
    monkeypatch.setattr(recorder.sd, "InputStream", lambda **kw: Mock())
    monkeypatch.setattr(recorder.Recorder, "_write_wav", lambda *a: tmp_path / "unused.wav")
    rec = recorder.Recorder()
    rec.start(streaming=True)
    for _ in range(101):
        rec._audio_callback(np.ones((16, 1), dtype=np.int16), 16, None, None)
    rec.stop()

    async def read():
        with pytest.raises(RuntimeError, match="overflow"):
            async for _ in rec.audio_chunks():
                pass

    asyncio.run(read())


def test_once_streams_before_capture_ends(monkeypatch, tmp_path):
    from typer.testing import CliRunner

    from dictum import cli

    events = []
    asr = LiveAsr()

    class OnceRecorder(LiveRecorder):
        async def audio_chunks(self):
            assert self.is_recording
            events.append("audio-during-recording")
            yield b"pcm"
            # Stand in for the silence watchdog, after audio has been consumed.
            self.stop()

    class OnceAsr(LiveAsr):
        async def stop(self):
            events.append("asr-closed")

    asr = OnceAsr()
    rec = OnceRecorder(tmp_path / "once.wav")
    profile = Profile(llm=LlmConfig(backend="none"))
    profile.asr.backend = "gemini-live"
    monkeypatch.setattr(recorder, "Recorder", lambda **kw: rec)
    monkeypatch.setattr("dictum.asr.create_asr_backend", lambda *a, **kw: asr)
    monkeypatch.setattr("dictum.config.load_profile", lambda *a: profile)
    deliver = AsyncMock()
    monkeypatch.setattr("dictum.output.OutputSink", lambda: Mock(deliver=deliver))
    result = CliRunner().invoke(cli.app, ["once", "--result", "none"])
    assert result.exit_code == 0, result.exception
    assert events == ["audio-during-recording", "asr-closed"]
    assert deliver.await_count == 1
    assert deliver.call_args.args[0].final_text == "final text"
    assert not rec.is_recording
