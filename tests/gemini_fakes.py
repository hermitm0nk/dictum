"""Test doubles shared by the Gemini backend tests.

These keep the Gemini tests hermetic: no API key, no network, and no real
ffmpeg — the Files API and Interactions API are served by an
`httpx.MockTransport`, and ffmpeg is a small real executable that records its
argv so the Opus encoder command line is asserted end to end.
"""

from __future__ import annotations

import asyncio
import json
import stat
from pathlib import Path

import httpx
import pytest

from dictum import asr_gemini
from dictum.asr_gemini import DEFAULT_BASE_URL, GeminiASR
from dictum.models import Profile, Transcript

KEY = "test-key-abc123"
FILE_ID = "files/test-file-123"
OPUS_BYTES = b"OPUSDATA"


# ─────────────────────────────────────────────────────────────────────────────
# fake ffmpeg (real subprocess)
# ─────────────────────────────────────────────────────────────────────────────

FAKE_FFMPEG = """#!/usr/bin/env python3
import json
import os
import sys
import time

args = sys.argv[1:]
with open(os.environ["DICTUM_TEST_FFMPEG_LOG"], "a") as fh:
    fh.write(json.dumps(args) + "\\n")

mode = os.environ.get("DICTUM_TEST_FFMPEG_MODE", "ok")
if mode == "fail":
    sys.stderr.write("fake ffmpeg: unknown encoder 'libopus'\\n")
    raise SystemExit(1)
if mode == "hang":
    time.sleep(300)

with open(args[-1], "wb") as fh:
    fh.write(b"OPUSDATA")
"""


class FakeFfmpeg:
    """A real executable standing in for ffmpeg that records its argv."""

    def __init__(self, path: Path, log: Path) -> None:
        self.path = path
        self.log = log

    def argv(self) -> list[list[str]]:
        """One entry per invocation."""
        if not self.log.exists():
            return []
        return [json.loads(line) for line in self.log.read_text().splitlines() if line.strip()]

    def encoded_path(self) -> Path:
        argv = self.argv()
        assert argv, "ffmpeg was not invoked"
        return Path(argv[0][-1])


def install_fake_ffmpeg(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> FakeFfmpeg:
    """Install a real executable named `ffmpeg` that records its argv.

    The key/value logging happens through environment variables the script
    reads, so the tests can assert on the exact command line.
    """
    bindir = tmp_path / "bin"
    bindir.mkdir(exist_ok=True)
    script = bindir / "ffmpeg"
    if not script.exists():
        script.write_text(FAKE_FFMPEG)
        script.chmod(script.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    log = tmp_path / "ffmpeg-args.log"
    monkeypatch.setenv("DICTUM_TEST_FFMPEG_LOG", str(log))
    monkeypatch.delenv("DICTUM_TEST_FFMPEG_MODE", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    return FakeFfmpeg(script, log)


# ─────────────────────────────────────────────────────────────────────────────
# fake Gemini REST API (httpx.MockTransport)
# ─────────────────────────────────────────────────────────────────────────────


class FakeGeminiApi:
    """Serves the resumable Files API and the Interactions API."""

    def __init__(
        self,
        *,
        upload_state: str = "ACTIVE",
        pending_polls: int = 0,
        interactions_status: int = 200,
        interactions_body: dict | None = None,
        upload_start_status: int = 200,
        upload_session: str | None = None,
        interactions_raise: Exception | None = None,
        delete_status: int = 200,
        poll_status: int = 200,
        poll_body: dict | None = None,
    ) -> None:
        self.upload_state = upload_state
        self.pending_polls = pending_polls
        self.interactions_status = interactions_status
        self.interactions_body = interactions_body
        self.upload_start_status = upload_start_status
        self.upload_session = upload_session
        self.interactions_raise = interactions_raise
        self.delete_status = delete_status
        self.poll_status = poll_status
        self.poll_body = poll_body
        self.seen: dict[str, list[httpx.Request]] = {}
        self.transport = httpx.MockTransport(self._handle)

    # -- helpers ------------------------------------------------------------

    def _record(self, label: str, request: httpx.Request) -> None:
        self.seen.setdefault(label, []).append(request)

    def calls(self, label: str) -> list[httpx.Request]:
        return self.seen.get(label, [])

    def last(self, label: str) -> httpx.Request:
        calls = self.calls(label)
        assert calls, f"no {label} request was made"
        return calls[-1]

    def payload(self, label: str) -> dict:
        """Decode a JSON request body recorded under `label`."""
        body = json.loads(self.last(label).content.decode())
        assert isinstance(body, dict)
        return body

    # -- routes -------------------------------------------------------------

    def _handle(self, request: httpx.Request) -> httpx.Response:
        path = request.url.path

        if request.method == "POST" and path.startswith("/upload/v1beta/files"):
            # The resumable protocol separates the session-start POST from the
            # data POST with the X-Goog-Upload-Command header.
            if request.headers.get("x-goog-upload-command", "").startswith("start"):
                self._record("upload-start", request)
                if self.upload_start_status != 200:
                    return httpx.Response(
                        self.upload_start_status,
                        json={
                            "error": {
                                "code": 401,
                                "message": "API key not valid",
                                "status": "UNAUTHENTICATED",
                            }
                        },
                    )
                return httpx.Response(
                    200,
                    headers={
                        "x-goog-upload-url": (
                            self.upload_session
                            or f"{DEFAULT_BASE_URL}/upload/v1beta/files?upload_id=test-upload"
                        )
                    },
                    json={},
                )

            self._record("upload", request)
            return httpx.Response(
                200,
                json={
                    "file": {
                        "name": FILE_ID,
                        "uri": f"{DEFAULT_BASE_URL}/v1beta/files/test-file-123",
                        "mimeType": "audio/ogg",
                        "state": self.upload_state,
                    }
                },
            )

        if request.method == "GET" and path.startswith("/v1beta/files/"):
            self._record("status", request)
            if self.poll_body is not None:
                return httpx.Response(self.poll_status, json=self.poll_body)
            done = len(self.calls("status")) > self.pending_polls
            return httpx.Response(
                200, json={"name": FILE_ID, "state": "ACTIVE" if done else "PROCESSING"}
            )

        if request.method == "POST" and path == "/v1beta/interactions":
            self._record("interactions", request)
            if self.interactions_raise is not None:
                raise self.interactions_raise
            body = self.interactions_body or {
                "steps": [
                    {
                        "type": "model_output",
                        "content": [{"type": "text", "text": "hello from gemini"}],
                    }
                ],
                "status": "completed",
            }
            return httpx.Response(self.interactions_status, json=body)

        if request.method == "DELETE" and path.startswith("/v1beta/files/"):
            self._record("delete", request)
            return httpx.Response(self.delete_status, json={})

        return httpx.Response(
            404,
            json={"error": {"code": 404, "message": f"unrouted {request.method} {path}"}},
        )


# ─────────────────────────────────────────────────────────────────────────────
# helpers
# ─────────────────────────────────────────────────────────────────────────────


def patch_client(monkeypatch: pytest.MonkeyPatch, transport: httpx.MockTransport) -> None:
    """Route GeminiASR's HTTP calls into a mock transport."""
    monkeypatch.setattr(
        asr_gemini,
        "_client",
        lambda timeout: httpx.AsyncClient(transport=transport, timeout=timeout),
    )


def make_backend(ffmpeg: FakeFfmpeg, **kwargs: object) -> GeminiASR:
    # Existing Files API tests exercise upload/poll/delete behavior explicitly.
    kwargs.setdefault("inline_max_bytes", 0)
    return GeminiASR(api_key=KEY, ffmpeg_path=str(ffmpeg.path), **kwargs)


def audio_file(tmp_path: Path) -> Path:
    audio = tmp_path / "rec_1700000000000.wav"
    audio.write_bytes(b"RIFFfake-wav-bytes")
    return audio


def flag(argv: list[str], name: str) -> str:
    """Value that follows `name` in an ffmpeg argv."""
    assert name in argv, f"{name} missing from {argv}"
    return argv[argv.index(name) + 1]


def run_transcribe(asr: GeminiASR, audio: Path, profile: Profile | None = None) -> Transcript:
    return asyncio.run(asr.transcribe(audio, profile))
