"""Mocked-HTTP and real-subprocess tests for the Gemini ASR backend.

Nothing here needs an API key or network access — see `tests/gemini_fakes.py`
for the fakes (Files/Interactions API over `httpx.MockTransport`, plus a real
ffmpeg stand-in that records its argv).
"""

from __future__ import annotations

import asyncio
import json
import logging
import tempfile
from pathlib import Path

import httpx
import pytest
from gemini_fakes import (
    KEY,
    OPUS_BYTES,
    FakeGeminiApi,
    audio_file,
    flag,
    make_backend,
    patch_client,
    run_transcribe,
)

from dictum import asr_gemini
from dictum.asr_gemini import DEFAULT_BASE_URL, GeminiASR, GeminiError, _safe_detail
from dictum.models import Profile


@pytest.fixture
def api() -> FakeGeminiApi:
    return FakeGeminiApi()


# ─────────────────────────────────────────────────────────────────────────────
# configuration / preflight
# ─────────────────────────────────────────────────────────────────────────────


def test_missing_api_key_fails_before_any_request(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    fake_ffmpeg,
    api: FakeGeminiApi,
) -> None:
    patch_client(monkeypatch, api.transport)
    asr = GeminiASR(api_key_env="DICTUM_TEST_NO_KEY", ffmpeg_path=str(fake_ffmpeg.path))

    with pytest.raises(GeminiError, match="DICTUM_TEST_NO_KEY"):
        asyncio.run(asr.start())
    with pytest.raises(GeminiError, match="DICTUM_TEST_NO_KEY"):
        run_transcribe(asr, audio_file(tmp_path))

    assert api.seen == {}


def test_missing_ffmpeg_fails_clearly(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    api: FakeGeminiApi,
) -> None:
    patch_client(monkeypatch, api.transport)
    asr = GeminiASR(api_key="k", ffmpeg_path=str(tmp_path / "no-such-ffmpeg"))

    with pytest.raises(GeminiError, match="ffmpeg"):
        asyncio.run(asr.start())
    with pytest.raises(GeminiError, match="ffmpeg"):
        run_transcribe(asr, audio_file(tmp_path))

    assert api.seen == {}


def test_invalid_transcription_mode_raises(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    fake_ffmpeg,
    api: FakeGeminiApi,
) -> None:
    patch_client(monkeypatch, api.transport)
    asr = make_backend(fake_ffmpeg, transcription_mode="bogus")

    with pytest.raises(GeminiError, match="transcription mode"):
        run_transcribe(asr, audio_file(tmp_path))

    assert api.seen == {}


def test_invalid_bitrate_raises(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    fake_ffmpeg,
    api: FakeGeminiApi,
) -> None:
    patch_client(monkeypatch, api.transport)
    asr = make_backend(fake_ffmpeg, bitrate_kbps=0)

    with pytest.raises(GeminiError, match="bitrate"):
        run_transcribe(asr, audio_file(tmp_path))

    assert api.seen == {}


def test_missing_audio_file_raises(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    fake_ffmpeg,
    api: FakeGeminiApi,
) -> None:
    patch_client(monkeypatch, api.transport)
    asr = make_backend(fake_ffmpeg)

    with pytest.raises(GeminiError, match="not found"):
        run_transcribe(asr, tmp_path / "gone.wav")

    assert api.seen == {}


def test_api_key_read_from_environment(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    fake_ffmpeg,
    api: FakeGeminiApi,
) -> None:
    monkeypatch.setenv("DICTUM_TEST_GEMINI_KEY", "env-key")
    patch_client(monkeypatch, api.transport)
    asr = GeminiASR(
        api_key_env="DICTUM_TEST_GEMINI_KEY",
        ffmpeg_path=str(fake_ffmpeg.path),
        inline_max_bytes=0,
    )

    transcript = run_transcribe(asr, audio_file(tmp_path))

    assert transcript.text == "hello from gemini"
    assert api.last("upload-start").headers["x-goog-api-key"] == "env-key"


# ─────────────────────────────────────────────────────────────────────────────
# happy path: encode → upload → transcribe → cleanup
# ─────────────────────────────────────────────────────────────────────────────


def test_full_pipeline_payloads_and_cleanup(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    fake_ffmpeg,
    api: FakeGeminiApi,
) -> None:
    patch_client(monkeypatch, api.transport)
    asr = make_backend(fake_ffmpeg, bitrate_kbps=24, language_codes=["en-US"])
    audio = audio_file(tmp_path)

    transcript = run_transcribe(asr, audio, Profile(name="gemini"))

    assert transcript.text == "hello from gemini"

    # 1. ffmpeg gets a mono, low-bitrate, voip-tuned Ogg Opus command.
    argv = fake_ffmpeg.argv()
    assert len(argv) == 1
    assert argv[0][0] == "-nostdin"  # stdin closed: ffmpeg never blocks on a tty
    assert flag(argv[0], "-i") == str(audio)
    assert flag(argv[0], "-c:a") == "libopus"
    assert flag(argv[0], "-b:a") == "24k"
    assert flag(argv[0], "-application") == "voip"
    assert flag(argv[0], "-ac") == "1"
    assert flag(argv[0], "-ar") == "48000"
    assert argv[0][-3:-1] == ["-f", "ogg"]
    assert argv[0][-1] == str(fake_ffmpeg.encoded_path())

    # 2. Resumable upload start (key in a header, never in the request URL).
    start = api.last("upload-start")
    assert start.headers["x-goog-api-key"] == KEY
    assert start.headers["x-goog-upload-protocol"] == "resumable"
    assert start.headers["x-goog-upload-command"] == "start"
    assert start.headers["x-goog-upload-header-content-type"] == "audio/ogg"
    assert start.headers["x-goog-upload-header-content-length"] == str(len(OPUS_BYTES))
    assert start.url.query.decode() == ""
    assert json.loads(start.content.decode()) == {"file": {"display_name": "dictum-audio"}}

    # 3. The audio bytes are posted to the session URL with offset 0.
    up = api.last("upload")
    assert up.content == OPUS_BYTES
    assert up.headers["content-length"] == str(len(OPUS_BYTES))
    assert up.headers["x-goog-upload-offset"] == "0"
    assert up.headers["x-goog-upload-command"] == "upload, finalize"
    assert up.headers["x-goog-api-key"] == KEY
    assert "upload_id=test-upload" in str(up.url)

    # 4. Interactions payload: smart mode, language hints, nothing stored.
    assert api.payload("interactions") == {
        "model": "gemini-3.5-transcribe",
        "input": [
            {
                "type": "audio",
                "uri": "https://generativelanguage.googleapis.com/v1beta/files/test-file-123",
                "mime_type": "audio/ogg",
            }
        ],
        "generation_config": {
            "transcription_config": {"mode": "smart", "language_codes": ["en-US"]}
        },
        "store": False,
    }
    assert api.last("interactions").headers["x-goog-api-key"] == KEY

    # 5. Remote file deleted; local encoded copy and temp directory removed.
    assert len(api.calls("delete")) == 1
    assert api.last("delete").url.path == "/v1beta/files/test-file-123"
    encoded_path = fake_ffmpeg.encoded_path()
    assert not encoded_path.exists()
    assert not encoded_path.parent.exists()


def test_verbatim_mode_sends_object_payload(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    fake_ffmpeg,
    api: FakeGeminiApi,
) -> None:
    patch_client(monkeypatch, api.transport)
    asr = make_backend(fake_ffmpeg, transcription_mode="VERBATIM")

    run_transcribe(asr, audio_file(tmp_path))

    mode = api.payload("interactions")["generation_config"]["transcription_config"]["mode"]
    assert mode == {"type": "verbatim"}
    assert len(api.calls("delete")) == 1


def test_default_mode_and_no_language_codes(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    fake_ffmpeg,
    api: FakeGeminiApi,
) -> None:
    patch_client(monkeypatch, api.transport)
    asr = make_backend(fake_ffmpeg)

    run_transcribe(asr, audio_file(tmp_path))

    transcription_config = api.payload("interactions")["generation_config"]["transcription_config"]
    assert transcription_config == {"mode": "smart", "language_codes": []}


def test_small_recording_uses_single_inline_interaction(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    fake_ffmpeg,
    api: FakeGeminiApi,
) -> None:
    patch_client(monkeypatch, api.transport)
    asr = GeminiASR(api_key=KEY, ffmpeg_path=str(fake_ffmpeg.path), language_codes=["en-US"])

    assert run_transcribe(asr, audio_file(tmp_path)).text == "hello from gemini"
    payload = api.payload("interactions")
    import base64

    assert payload["input"] == [
        {"type": "audio", "mime_type": "audio/ogg", "data": base64.b64encode(OPUS_BYTES).decode()}
    ]
    assert payload["generation_config"]["transcription_config"] == {
        "mode": "smart",
        "language_codes": ["en-US"],
    }
    assert payload["store"] is False
    assert not api.calls("upload-start")
    assert not api.calls("delete")


@pytest.mark.parametrize(("limit", "file_route"), [(8, False), (7, True), (0, True)])
def test_inline_threshold_routes_at_boundary(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    fake_ffmpeg,
    limit: int,
    file_route: bool,
) -> None:
    api = FakeGeminiApi()
    patch_client(monkeypatch, api.transport)
    asr = GeminiASR(api_key=KEY, ffmpeg_path=str(fake_ffmpeg.path), inline_max_bytes=limit)
    run_transcribe(asr, audio_file(tmp_path))
    assert bool(api.calls("upload-start")) is file_route
    assert bool(api.calls("interactions"))


def test_timing_log_reports_phases_without_audio_or_key(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, fake_ffmpeg, api: FakeGeminiApi, caplog
) -> None:
    patch_client(monkeypatch, api.transport)
    audio = audio_file(tmp_path)
    with caplog.at_level(logging.INFO, logger="dictum.asr_gemini"):
        run_transcribe(GeminiASR(api_key=KEY, ffmpeg_path=str(fake_ffmpeg.path)), audio)
    timing = next(
        record.getMessage() for record in caplog.records if "Gemini timing" in record.getMessage()
    )
    assert "path=inline" in timing and "encode=" in timing and "transcription=" in timing
    assert KEY not in timing and "OPUSDATA" not in timing and str(audio) not in timing


@pytest.mark.parametrize("failure", [httpx.ConnectError, 429])
def test_inline_failure_does_not_fallback_and_cleans_scratch(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    fake_ffmpeg,
    failure,
) -> None:
    runtime = tmp_path / "runtime"
    monkeypatch.setenv("XDG_RUNTIME_DIR", str(runtime))
    if failure == 429:
        api = FakeGeminiApi(interactions_status=429)
        transport = api.transport
    else:
        api = FakeGeminiApi()

        def fail(request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError("secret-bearing url", request=request)

        transport = httpx.MockTransport(fail)
    patch_client(monkeypatch, transport)
    asr = GeminiASR(api_key=KEY, ffmpeg_path=str(fake_ffmpeg.path))
    with pytest.raises(GeminiError):
        run_transcribe(asr, audio_file(tmp_path))
    assert not api.calls("upload-start")
    assert not api.calls("delete")
    assert list((runtime / "dictum" / "scratch").glob("dictum-gemini-*")) == []


def test_started_client_is_reused_and_closed(monkeypatch, tmp_path, fake_ffmpeg, api):
    patch_client(monkeypatch, api.transport)
    real_client = asr_gemini._client
    made = []

    def tracked(timeout):
        client = real_client(timeout)
        made.append(client)
        return client

    monkeypatch.setattr(asr_gemini, "_client", tracked)
    asr = make_backend(fake_ffmpeg)

    async def run():
        await asr.start()
        await asr.transcribe(audio_file(tmp_path))
        await asr.transcribe(audio_file(tmp_path))
        assert len(made) == 1
        await asr.stop()

    asyncio.run(run())
    assert made[0].is_closed
    assert len(api.calls("delete")) == 2
    assert all(req.extensions["timeout"]["read"] == 10.0 for req in api.calls("delete"))


# ─────────────────────────────────────────────────────────────────────────────
# uploaded file lifecycle
# ─────────────────────────────────────────────────────────────────────────────


def test_polls_processing_file_until_active(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    fake_ffmpeg,
) -> None:
    api = FakeGeminiApi(upload_state="PROCESSING", pending_polls=1)
    patch_client(monkeypatch, api.transport)
    asr = make_backend(fake_ffmpeg)

    transcript = run_transcribe(asr, audio_file(tmp_path))

    assert transcript.text == "hello from gemini"
    statuses = api.calls("status")
    assert len(statuses) == 2  # one PROCESSING poll, then ACTIVE
    assert all(request.headers["x-goog-api-key"] == KEY for request in statuses)
    assert statuses[0].url.path == "/v1beta/files/test-file-123"
    assert len(api.calls("interactions")) == 1
    assert len(api.calls("delete")) == 1


def test_active_file_is_not_polled(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    fake_ffmpeg,
    api: FakeGeminiApi,
) -> None:
    patch_client(monkeypatch, api.transport)
    asr = make_backend(fake_ffmpeg)

    run_transcribe(asr, audio_file(tmp_path))

    assert api.calls("status") == []


def test_processing_timeout_raises_and_still_deletes(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    fake_ffmpeg,
) -> None:
    monkeypatch.setattr(asr_gemini, "FILE_ACTIVE_TIMEOUT", 0.05)
    monkeypatch.setattr(asr_gemini, "FILE_POLL_INTERVAL", 0.01)
    api = FakeGeminiApi(upload_state="PROCESSING", pending_polls=999)
    patch_client(monkeypatch, api.transport)
    asr = make_backend(fake_ffmpeg)

    with pytest.raises(GeminiError, match="PROCESSING"):
        run_transcribe(asr, audio_file(tmp_path))

    assert api.calls("interactions") == []
    assert len(api.calls("delete")) == 1
    assert not fake_ffmpeg.encoded_path().exists()


def test_failed_file_state_raises(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    fake_ffmpeg,
    api: FakeGeminiApi,
) -> None:
    api.upload_state = "PROCESSING"
    api.poll_body = {"name": "files/test-file-123", "state": "FAILED"}
    patch_client(monkeypatch, api.transport)
    asr = make_backend(fake_ffmpeg)

    with pytest.raises(GeminiError, match="FAILED"):
        run_transcribe(asr, audio_file(tmp_path))

    assert api.calls("interactions") == []
    assert len(api.calls("delete")) == 1


def test_failed_upload_state_rejected_upfront(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    fake_ffmpeg,
) -> None:
    """A terminal FAILED state on the upload response fails fast, unpolled."""
    api = FakeGeminiApi(upload_state="FAILED")
    patch_client(monkeypatch, api.transport)
    asr = make_backend(fake_ffmpeg)

    with pytest.raises(GeminiError, match="FAILED"):
        run_transcribe(asr, audio_file(tmp_path))

    assert api.calls("status") == []
    assert api.calls("interactions") == []
    assert len(api.calls("delete")) == 1


# ─────────────────────────────────────────────────────────────────────────────
# failure paths: no silent fallback, remote cleanup, secret safety
# ─────────────────────────────────────────────────────────────────────────────


def test_rate_limited_transcription_raises_and_deletes(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    fake_ffmpeg,
) -> None:
    api = FakeGeminiApi(
        interactions_status=429,
        interactions_body={
            "error": {
                "code": 429,
                "message": "Quota exceeded for the day",
                "status": "RESOURCE_EXHAUSTED",
            }
        },
    )
    patch_client(monkeypatch, api.transport)
    asr = make_backend(fake_ffmpeg)

    with pytest.raises(GeminiError) as excinfo:
        run_transcribe(asr, audio_file(tmp_path))

    message = str(excinfo.value)
    assert "HTTP 429" in message
    assert "quota" in message.lower()
    assert "Quota exceeded" in message
    # No silent switch to a local model: the error propagates, cleanup happens.
    assert len(api.calls("delete")) == 1
    assert not fake_ffmpeg.encoded_path().exists()


def test_auth_failure_does_not_leak_key(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    fake_ffmpeg,
) -> None:
    api = FakeGeminiApi(upload_start_status=403)
    patch_client(monkeypatch, api.transport)
    asr = make_backend(fake_ffmpeg)

    with pytest.raises(GeminiError) as excinfo:
        run_transcribe(asr, audio_file(tmp_path))

    message = str(excinfo.value)
    assert "HTTP 403" in message
    assert "GEMINI_API_KEY" in message  # actionable hint
    assert KEY not in message
    # Nothing was uploaded, so there is no remote file to delete.
    assert api.calls("delete") == []
    assert api.calls("interactions") == []


def test_error_body_containing_key_is_redacted(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    fake_ffmpeg,
) -> None:
    api = FakeGeminiApi(
        interactions_status=400,
        interactions_body={
            "error": {
                "message": "API key not valid: test-key-abc123",
                "status": "INVALID_ARGUMENT",
            }
        },
    )
    patch_client(monkeypatch, api.transport)
    asr = make_backend(fake_ffmpeg)

    with pytest.raises(GeminiError) as excinfo:
        run_transcribe(asr, audio_file(tmp_path))

    assert KEY not in str(excinfo.value)
    assert "[REDACTED]" in str(excinfo.value)


def test_redaction_happens_before_truncation(tmp_path: Path) -> None:
    """A cutoff that would bisect the key must not leak a fragment of it."""
    # Position the key so a 400-char truncation would cut it in the middle;
    # only redacting *before* truncating keeps the whole key out of the message.
    message = "x" * 390 + KEY + "trailing detail"
    detail = _safe_detail(message, KEY)
    assert KEY not in detail
    assert "test-key-a" not in detail  # the fragment a naive truncate would leave
    assert "[REDACTED]" in detail


def test_error_body_redacted_before_truncation_end_to_end(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    fake_ffmpeg,
) -> None:
    api = FakeGeminiApi(
        interactions_status=400,
        interactions_body={"error": {"message": "y" * 390 + KEY}},
    )
    patch_client(monkeypatch, api.transport)
    asr = make_backend(fake_ffmpeg)

    with pytest.raises(GeminiError) as excinfo:
        run_transcribe(asr, audio_file(tmp_path))

    text = str(excinfo.value)
    assert KEY not in text
    assert "test-key-a" not in text
    assert "[REDACTED]" in text


def test_http_200_error_object_is_redacted(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    fake_ffmpeg,
) -> None:
    """An error object in a 200 body echoes its message — keep it secret-safe."""
    api = FakeGeminiApi(
        interactions_status=200,
        interactions_body={"error": {"message": f"invalid key {KEY} supplied"}},
    )
    patch_client(monkeypatch, api.transport)
    asr = make_backend(fake_ffmpeg)

    with pytest.raises(GeminiError) as excinfo:
        run_transcribe(asr, audio_file(tmp_path))

    assert KEY not in str(excinfo.value)
    assert "[REDACTED]" in str(excinfo.value)
    assert len(api.calls("delete")) == 1


def test_non_json_error_body_is_not_echoed(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    fake_ffmpeg,
) -> None:
    """A non-JSON error body may hold private audio or a URL, so it is generic."""
    private = "SECRET-AUDIO transcript fragment / signed url"

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, text=private)

    patch_client(monkeypatch, httpx.MockTransport(handler))
    asr = make_backend(fake_ffmpeg)

    with pytest.raises(GeminiError) as excinfo:
        run_transcribe(asr, audio_file(tmp_path))

    text = str(excinfo.value)
    assert private not in text
    assert "non-JSON" in text


def test_server_error_suggests_retry(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    fake_ffmpeg,
) -> None:
    api = FakeGeminiApi(interactions_status=503)
    patch_client(monkeypatch, api.transport)
    asr = make_backend(fake_ffmpeg)

    with pytest.raises(GeminiError) as excinfo:
        run_transcribe(asr, audio_file(tmp_path))

    assert "HTTP 503" in str(excinfo.value)
    assert "retry later" in str(excinfo.value)
    assert len(api.calls("delete")) == 1


def test_upload_start_without_session_header_raises(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    fake_ffmpeg,
    api: FakeGeminiApi,
) -> None:
    def transport_handler(request: httpx.Request) -> httpx.Response:
        api._record("upload-start", request)
        return httpx.Response(200, json={})

    patch_client(monkeypatch, httpx.MockTransport(transport_handler))
    asr = make_backend(fake_ffmpeg)

    with pytest.raises(GeminiError, match="x-goog-upload-url"):
        run_transcribe(asr, audio_file(tmp_path))

    assert api.calls("upload") == []
    assert api.calls("interactions") == []
    assert api.calls("delete") == []


def test_unexpected_response_shape_raises(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    fake_ffmpeg,
    api: FakeGeminiApi,
) -> None:
    api.interactions_body = {"unexpected": "shape"}
    patch_client(monkeypatch, api.transport)
    asr = make_backend(fake_ffmpeg)

    with pytest.raises(GeminiError, match="steps"):
        run_transcribe(asr, audio_file(tmp_path))

    assert len(api.calls("delete")) == 1


def test_empty_transcript_is_not_an_error(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    fake_ffmpeg,
    api: FakeGeminiApi,
) -> None:
    api.interactions_body = {
        "steps": [{"type": "model_output", "content": []}],
        "status": "completed",
    }
    patch_client(monkeypatch, api.transport)
    asr = make_backend(fake_ffmpeg)

    transcript = run_transcribe(asr, audio_file(tmp_path))

    assert transcript.text == ""
    assert len(api.calls("delete")) == 1


def test_error_object_in_response_raises(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    fake_ffmpeg,
    api: FakeGeminiApi,
) -> None:
    api.interactions_body = {"error": {"message": "audio too long", "status": "INVALID_ARGUMENT"}}
    patch_client(monkeypatch, api.transport)
    asr = make_backend(fake_ffmpeg)

    with pytest.raises(GeminiError, match="audio too long"):
        run_transcribe(asr, audio_file(tmp_path))

    assert len(api.calls("delete")) == 1


def test_failed_interaction_status_rejected(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    fake_ffmpeg,
    api: FakeGeminiApi,
) -> None:
    """A non-completed status must not return a partial transcript as final."""
    api.interactions_body = {
        "status": "failed",
        "steps": [{"type": "model_output", "content": [{"type": "text", "text": "partial"}]}],
    }
    patch_client(monkeypatch, api.transport)
    asr = make_backend(fake_ffmpeg)

    with pytest.raises(GeminiError, match="did not complete"):
        run_transcribe(asr, audio_file(tmp_path))

    assert len(api.calls("delete")) == 1


def test_in_progress_interaction_status_rejected(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    fake_ffmpeg,
    api: FakeGeminiApi,
) -> None:
    api.interactions_body = {
        "status": "in_progress",
        "steps": [{"type": "model_output", "content": [{"type": "text", "text": "partial"}]}],
    }
    patch_client(monkeypatch, api.transport)
    asr = make_backend(fake_ffmpeg)

    with pytest.raises(GeminiError, match="did not complete"):
        run_transcribe(asr, audio_file(tmp_path))

    assert len(api.calls("delete")) == 1


def test_delete_failure_is_only_warned_about(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    fake_ffmpeg,
    api: FakeGeminiApi,
    caplog: pytest.LogCaptureFixture,
) -> None:
    api.delete_status = 500
    patch_client(monkeypatch, api.transport)
    asr = make_backend(fake_ffmpeg)

    with caplog.at_level("WARNING", logger="dictum.asr_gemini"):
        transcript = run_transcribe(asr, audio_file(tmp_path))

    assert transcript.text == "hello from gemini"
    assert "Could not delete uploaded Gemini file" in caplog.text


def test_delete_network_failure_warns_without_exposing_url(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    fake_ffmpeg,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Cleanup failures are logged by type/status, never the raw exception."""
    api = FakeGeminiApi()
    base_transport = api.transport
    secret_url = f"{DEFAULT_BASE_URL}/v1beta/files/test-file-123?key={KEY}"

    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "DELETE":
            raise httpx.ConnectError(f"cannot reach {secret_url}", request=request)
        return base_transport.handle_request(request)

    patch_client(monkeypatch, httpx.MockTransport(handler))
    asr = make_backend(fake_ffmpeg)

    with caplog.at_level("WARNING", logger="dictum.asr_gemini"):
        transcript = run_transcribe(asr, audio_file(tmp_path))

    assert transcript.text == "hello from gemini"
    assert "Could not delete uploaded Gemini file" in caplog.text
    assert secret_url not in caplog.text
    assert KEY not in caplog.text


# ─────────────────────────────────────────────────────────────────────────────
# encoder failures (real subprocess)
# ─────────────────────────────────────────────────────────────────────────────


def test_ffmpeg_failure_raises_and_never_uploads(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    fake_ffmpeg,
    api: FakeGeminiApi,
) -> None:
    monkeypatch.setenv("DICTUM_TEST_FFMPEG_MODE", "fail")
    patch_client(monkeypatch, api.transport)
    asr = make_backend(fake_ffmpeg)

    with pytest.raises(GeminiError, match="libopus"):
        run_transcribe(asr, audio_file(tmp_path))

    assert api.seen == {}


def test_ffmpeg_timeout_raises(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    fake_ffmpeg,
    api: FakeGeminiApi,
) -> None:
    monkeypatch.setenv("DICTUM_TEST_FFMPEG_MODE", "hang")
    patch_client(monkeypatch, api.transport)
    asr = make_backend(fake_ffmpeg, encode_timeout=0.3)

    with pytest.raises(GeminiError, match="did not finish encoding"):
        run_transcribe(asr, audio_file(tmp_path))

    assert api.seen == {}


def test_ffmpeg_absolute_path_is_required(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    api: FakeGeminiApi,
) -> None:
    """A path with a separator must not be looked up on PATH."""
    patch_client(monkeypatch, api.transport)
    missing = tmp_path / "not-on-path" / "ffmpeg"
    asr = GeminiASR(api_key="k", ffmpeg_path=str(missing))

    with pytest.raises(GeminiError, match=str(missing)):
        asyncio.run(asr.start())

    assert api.seen == {}


# ─────────────────────────────────────────────────────────────────────────────
# connectivity errors become safe, actionable errors (no credential-bearing URL)
# ─────────────────────────────────────────────────────────────────────────────


def test_network_failure_after_upload_is_wrapped_and_cleaned_up(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    fake_ffmpeg,
    api: FakeGeminiApi,
) -> None:
    secret = f"{DEFAULT_BASE_URL}/v1beta/interactions?key={KEY}&session=SESSION_TOKEN"
    api.interactions_raise = httpx.ConnectError(f"cannot reach {secret}", request=None)
    patch_client(monkeypatch, api.transport)
    asr = make_backend(fake_ffmpeg)

    with pytest.raises(GeminiError) as excinfo:
        run_transcribe(asr, audio_file(tmp_path))

    message = str(excinfo.value)
    assert "connect" in message.lower()
    assert "transcription" in message.lower()  # stage is named
    # The raw exception (with its credential-bearing URL) is never surfaced.
    assert secret not in message
    assert KEY not in message
    assert "SESSION_TOKEN" not in message
    # The already-uploaded remote file is still cleaned up.
    assert len(api.calls("delete")) == 1


def test_network_failure_before_upload_needs_no_cleanup(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    fake_ffmpeg,
) -> None:
    secret = f"{DEFAULT_BASE_URL}/upload/v1beta/files?key={KEY}&session=SESSION_TOKEN"

    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError(f"cannot reach {secret}", request=request)

    patch_client(monkeypatch, httpx.MockTransport(handler))
    asr = make_backend(fake_ffmpeg)

    with pytest.raises(GeminiError) as excinfo:
        run_transcribe(asr, audio_file(tmp_path))

    message = str(excinfo.value)
    assert "upload" in message.lower()  # stage is named
    assert secret not in message
    assert KEY not in message


def test_cancellation_is_preserved(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    fake_ffmpeg,
    api: FakeGeminiApi,
) -> None:
    """A cancellation during the upload is not mistaken for a network error."""

    async def _cancel(*_args: object, **_kwargs: object) -> None:
        raise asyncio.CancelledError

    async def _no_cancel(*_args: object, **_kwargs: object) -> None:
        return None

    monkeypatch.setattr(api, "_handle", _cancel)
    patch_client(monkeypatch, httpx.MockTransport(api._handle))
    asr = make_backend(fake_ffmpeg)

    with pytest.raises(asyncio.CancelledError):
        run_transcribe(asr, audio_file(tmp_path))


# ─────────────────────────────────────────────────────────────────────────────
# httpx request logs never leak the credential-bearing upload URL
# ─────────────────────────────────────────────────────────────────────────────


def test_credential_in_upload_url_is_scrubbed_from_logs(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    fake_ffmpeg,
    caplog: pytest.LogCaptureFixture,
) -> None:
    token = "upload-session-token-9f8e7d6c5b4a"
    session = f"{DEFAULT_BASE_URL}/upload/v1beta/files?key={KEY}&upload_id={token}"
    api = FakeGeminiApi(upload_session=session)
    patch_client(monkeypatch, api.transport)
    asr = make_backend(fake_ffmpeg)

    with caplog.at_level(logging.INFO):
        run_transcribe(asr, audio_file(tmp_path))

    # httpx still logs the request line (logging is not globally disabled) …
    assert "HTTP Request:" in caplog.text
    # … but neither the upload query token nor the API key reaches the logs.
    assert token not in caplog.text
    assert "upload_id" not in caplog.text
    assert KEY not in caplog.text


def test_scrub_filter_leaves_unrelated_urls_untouched() -> None:
    """Only upload URLs have their query dropped; other URLs are logged as-is."""
    record = logging.LogRecord(
        name="httpx",
        level=logging.INFO,
        pathname=__file__,
        lineno=1,
        msg='HTTP Request: %s %s "%s %d %s"',
        args=(
            "POST",
            httpx.URL(f"{DEFAULT_BASE_URL}/v1beta/interactions?model=x&signoff=keep-me"),
            "HTTP/1.1",
            200,
            "OK",
        ),
        exc_info=None,
    )

    filter_ = asr_gemini._ScrubUploadQuery()
    assert filter_.filter(record) is True
    assert "keep-me" in str(record.args[1])
    assert "model=x" in str(record.args[1])


# ─────────────────────────────────────────────────────────────────────────────
# encoded upload scratch lives under $XDG_RUNTIME_DIR/dictum when available
# ─────────────────────────────────────────────────────────────────────────────


def test_runtime_base_uses_xdg_runtime_dir(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("XDG_RUNTIME_DIR", str(tmp_path / "rt"))
    assert asr_gemini._runtime_base() == tmp_path / "rt" / "dictum" / "scratch"


def test_runtime_base_none_without_xdg(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("XDG_RUNTIME_DIR", raising=False)
    assert asr_gemini._runtime_base() is None


def test_scratch_dir_prefers_runtime_dir_and_cleans_up(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("XDG_RUNTIME_DIR", str(tmp_path / "rt"))
    base = asr_gemini._runtime_base()
    assert base is not None
    with asr_gemini._scratch_dir() as scratch:
        assert scratch.parent == base
        (scratch / "encoded.opus").write_bytes(b"opus")
    assert not scratch.exists()  # the per-run dir is removed
    assert base.is_dir()


def test_scratch_dir_falls_back_to_tempdir(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("XDG_RUNTIME_DIR", raising=False)
    with asr_gemini._scratch_dir() as scratch:
        assert scratch.parent == Path(tempfile.gettempdir())
        (scratch / "encoded.opus").write_bytes(b"opus")
    assert not scratch.exists()


def test_encoded_upload_lives_in_runtime_dir_end_to_end(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    fake_ffmpeg,
    api: FakeGeminiApi,
) -> None:
    runtime = tmp_path / "runtime"
    monkeypatch.setenv("XDG_RUNTIME_DIR", str(runtime))
    patch_client(monkeypatch, api.transport)
    asr = make_backend(fake_ffmpeg)

    run_transcribe(asr, audio_file(tmp_path))

    base = runtime / "dictum" / "scratch"
    assert base.is_dir()
    # The encoded upload was written under the runtime dir and cleaned up.
    assert base in fake_ffmpeg.encoded_path().parents
    assert list(base.glob("dictum-gemini-*")) == []
