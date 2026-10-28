"""IPC recording recovery without microphone or model dependencies."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from unittest.mock import AsyncMock, Mock

import pytest

from dictum import daemon
from dictum.models import Profile, ResultTarget
from dictum.state import DictumState


@pytest.mark.parametrize("command", ["start", "toggle"])
@pytest.mark.parametrize("initial", [DictumState.IDLE, DictumState.FAILED])
def test_recording_can_start_from_idle_or_failed(monkeypatch, command, initial):
    recorder = Mock()
    monkeypatch.setattr(daemon, "Recorder", lambda **kw: recorder)
    monkeypatch.setattr(daemon, "create_asr_backend", lambda *a, **kw: Mock())
    monkeypatch.setattr(daemon, "create_llm_backend", lambda *a, **kw: Mock())
    d = daemon.Daemon(Profile())
    d.state = initial
    d.last_error = "previous transcription failed"
    monkeypatch.setattr(d, "_notify_state", AsyncMock())
    reader = Mock(readline=AsyncMock(return_value=json.dumps({"cmd": command}).encode() + b"\n"))
    writer = Mock(drain=AsyncMock(), wait_closed=AsyncMock())

    async def run():
        if initial == DictumState.FAILED:
            d.asr.transcribe = AsyncMock(side_effect=RuntimeError("transcription failed"))
            with pytest.raises(RuntimeError, match="transcription failed"):
                await d._process_audio(Path("unused.wav"), d.active_profile, ResultTarget.NONE)
            assert d.state == DictumState.FAILED
        await d.handle_connection(reader, writer)

    asyncio.run(run())

    response = json.loads(writer.write.call_args.args[0])
    assert response["ok"] is True
    recorder.start.assert_called_once_with()
    assert d.state == DictumState.RECORDING
    assert d.last_error is None


@pytest.mark.parametrize("command", ["start", "toggle"])
@pytest.mark.parametrize(
    "initial", [DictumState.TRANSCRIBING, DictumState.POLISHING, DictumState.PASTING]
)
def test_recording_does_not_start_while_processing(monkeypatch, command, initial):
    recorder = Mock()
    monkeypatch.setattr(daemon, "Recorder", lambda **kw: recorder)
    monkeypatch.setattr(daemon, "create_asr_backend", lambda *a, **kw: Mock())
    monkeypatch.setattr(daemon, "create_llm_backend", lambda *a, **kw: Mock())
    d = daemon.Daemon(Profile())
    d.state = initial
    reader = Mock(readline=AsyncMock(return_value=json.dumps({"cmd": command}).encode() + b"\n"))
    writer = Mock(drain=AsyncMock(), wait_closed=AsyncMock())

    asyncio.run(d.handle_connection(reader, writer))

    response = json.loads(writer.write.call_args.args[0])
    assert response["ok"] is False
    assert response["error"] == f"Busy ({initial.value})"
    recorder.start.assert_not_called()
    assert d.state == initial
