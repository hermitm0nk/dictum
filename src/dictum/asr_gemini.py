"""Gemini hosted transcription backend — non-live.

The recording is Opus-compressed locally with ffmpeg, then sent inline for
small clips or uploaded through the resumable Files API for larger clips. Plain
REST over httpx is used (no Google SDK), and the API key is read from the
environment and never written to logs or error messages.

References:
  https://ai.google.dev/gemini-api/docs/transcribe
  https://ai.google.dev/api/interactions-api
  https://ai.google.dev/api/files
"""

from __future__ import annotations

import asyncio
import base64
import contextlib
import logging
import os
import shutil
import tempfile
import time
from collections.abc import AsyncIterator, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx

from dictum.backends import AsrBackend
from dictum.models import Profile, Transcript

log = logging.getLogger(__name__)

DEFAULT_MODEL = "gemini-3.5-transcribe"
DEFAULT_BASE_URL = "https://generativelanguage.googleapis.com"
DEFAULT_API_KEY_ENV = "GEMINI_API_KEY"
DEFAULT_BITRATE_KBPS = 32
DEFAULT_MODE = "smart"
DEFAULT_INLINE_MAX_BYTES = 128 * 1024
MODES = ("smart", "verbatim")

# Ogg-wrapped Opus is what the Files API is told we are uploading.
AUDIO_MIME = "audio/ogg"
UPLOAD_DISPLAY_NAME = "dictum-audio"

# Bounded waits: encoding a short dictation clip is instant, uploads may crawl
# on a slow link, and uploaded files finish server-side processing quickly.
ENCODE_TIMEOUT = 60.0
UPLOAD_TIMEOUT = 120.0
FILE_ACTIVE_TIMEOUT = 60.0
FILE_POLL_INTERVAL = 0.5
# Remote cleanup is best-effort; a short, distinct budget keeps a successful
# dictation from blocking for the full upload timeout if deletion stalls.
DELETE_TIMEOUT = 10.0

# Upload/session URLs and error bodies can carry credentials, so the API key is
# redacted from errors and the upload token is kept out of httpx's request logs
# (see the `_ScrubUploadQuery` filter below).
_MAX_ERROR_CHARS = 400
_REDACTED = "[REDACTED]"


class GeminiError(RuntimeError):
    """Actionable Gemini failure (config, encoder, upload, auth, quota, shape)."""


class _ScrubUploadQuery(logging.Filter):
    """Drop the query string from httpx INFO request logs for upload URLs.

    httpx logs ``HTTP Request: <METHOD> <url> "<...>"`` at INFO on its "httpx"
    logger. The resumable-upload session URL — and any future signed upload
    URL — can carry a credential-bearing token in its query string, so this
    narrow filter strips just that query for ``/upload/`` requests. Every other
    log record passes through untouched.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        if record.name != "httpx" or not record.args:
            return True
        args = record.args
        if not isinstance(args, tuple) or len(args) < 2:
            return True
        url = args[1]
        if not isinstance(url, httpx.URL):
            return True
        if "/upload/" in url.path and url.query:
            scrubbed = list(args)
            scrubbed[1] = url.copy_with(query=b"")
            record.args = tuple(scrubbed)
        return True


# Installed only when this module is imported (i.e. only when the Gemini
# backend is actually used), so default Parakeet profiles leave httpx alone.
if not any(isinstance(f, _ScrubUploadQuery) for f in logging.getLogger("httpx").filters):
    logging.getLogger("httpx").addFilter(_ScrubUploadQuery())


@dataclass(frozen=True)
class UploadedFile:
    """A file registered with the Gemini Files API."""

    name: str  # resource name, e.g. "files/abc123"
    uri: str
    mime_type: str
    state: str

    @property
    def file_id(self) -> str:
        """Resource id without the ``files/`` prefix."""
        return self.name.split("/", 1)[-1]


def _client(timeout: float) -> httpx.AsyncClient:
    """Build an HTTP client for the Gemini API (patched in tests)."""
    return httpx.AsyncClient(timeout=timeout, limits=httpx.Limits(keepalive_expiry=60))


@contextlib.contextmanager
def _measure(timings: dict[str, float], phase: str) -> Iterator[None]:
    started = time.perf_counter()
    try:
        yield
    finally:
        timings[phase] = time.perf_counter() - started


def _runtime_base() -> Path | None:
    """Private scratch base under ``$XDG_RUNTIME_DIR/dictum`` when available."""
    xdg = os.environ.get("XDG_RUNTIME_DIR")
    if xdg:
        return Path(xdg) / "dictum" / "scratch"
    return None


@contextlib.contextmanager
def _scratch_dir() -> Iterator[Path]:
    """Yield a private temp dir for the encoded upload, removed on exit.

    Scratch lives under ``$XDG_RUNTIME_DIR/dictum`` (tmpfs, auto-cleared, and
    private to the session) per AGENTS.md when that runtime dir is set, and
    falls back to the standard ``tempfile`` location (honoring ``$TMPDIR``)
    otherwise. The directory is always removed on the way out — success,
    failure, or cancellation.
    """
    base = _runtime_base()
    if base is not None:
        base.mkdir(parents=True, exist_ok=True)
        path = Path(tempfile.mkdtemp(prefix="dictum-gemini-", dir=base))
        try:
            yield path
        finally:
            shutil.rmtree(path, ignore_errors=True)
    else:
        with tempfile.TemporaryDirectory(prefix="dictum-gemini-") as tmp_name:
            yield Path(tmp_name)


def _truncate(text: str) -> str:
    return text[:_MAX_ERROR_CHARS]


def _safe_detail(body: str, secret: str | None) -> str:
    """Redact the API key from an error body, then truncate.

    Redaction happens *before* truncation so a cutoff that would otherwise cut
    the key in half cannot leak a fragment of it.
    """
    detail = body
    if secret and secret in detail:
        detail = detail.replace(secret, _REDACTED)
    return _truncate(detail)


def _error_detail(response: httpx.Response, secret: str | None) -> str:
    """Extract a secret-safe, human-readable error message from a response.

    A non-JSON body is never echoed back: it can carry the private audio, a
    transcript fragment, or a credential-bearing URL, so only a generic note is
    returned for it.
    """
    try:
        body: Any = response.json()
    except ValueError:
        return "the server returned a non-JSON error body"
    if isinstance(body, dict):
        err = body.get("error")
        if isinstance(err, dict):
            message = err.get("message")
            if isinstance(message, str):
                return _safe_detail(message, secret)
    return ""


def _connectivity_message(exc: httpx.RequestError, stage: str) -> str:
    """A safe, actionable message for a network failure during ``stage``.

    The raw exception is intentionally dropped: httpx connection/timeout errors
    embed the request URL, which can carry an upload token or the API key in
    its query string.
    """
    reason = "timed out" if isinstance(exc, httpx.TimeoutException) else "could not connect"
    return f"Gemini {stage} {reason}; check the network connection and retry"


def _parse_uploaded_file(body: dict[str, Any]) -> UploadedFile:
    """Parse the ``file`` object of a Files API upload response."""
    file_obj = body.get("file")
    if not isinstance(file_obj, dict):
        raise GeminiError("Gemini upload response did not contain a 'file' object")
    name = file_obj.get("name")
    uri = file_obj.get("uri")
    if not isinstance(name, str) or not isinstance(uri, str):
        raise GeminiError("Gemini upload response is missing the file name or uri")
    mime_type = file_obj.get("mimeType")
    state = file_obj.get("state")
    return UploadedFile(
        name=name,
        uri=uri,
        mime_type=mime_type if isinstance(mime_type, str) else AUDIO_MIME,
        state=state if isinstance(state, str) else "ACTIVE",
    )


def _extract_transcript(body: dict[str, Any], secret: str | None) -> str:
    """Pull the transcript text out of an Interactions API response.

    The REST shape is ``steps: [{type: "model_output", content: [{type: "text",
    text: "..."}]}]``; ``output_text`` is an SDK-side helper, not a REST field.
    A present-but-non-``completed`` status (e.g. ``failed``/``in_progress``) is
    rejected so a partial transcript is never returned as final text.
    """
    error = body.get("error")
    if isinstance(error, dict):
        message = error.get("message")
        detail = _safe_detail(message, secret) if isinstance(message, str) else "unknown error"
        raise GeminiError(f"Gemini interaction failed: {detail}")

    status = body.get("status")
    if isinstance(status, str) and status.strip().lower() != "completed":
        raise GeminiError(f"Gemini interaction did not complete (status: {status.strip()})")

    steps = body.get("steps")
    if not isinstance(steps, list):
        raise GeminiError("Gemini interaction response did not contain 'steps'")

    chunks: list[str] = []
    for step in steps:
        if not isinstance(step, dict) or step.get("type") != "model_output":
            continue
        content = step.get("content")
        if not isinstance(content, list):
            continue
        for part in content:
            if isinstance(part, dict) and part.get("type") == "text":
                text = part.get("text")
                if isinstance(text, str):
                    chunks.append(text)

    return "\n".join(chunk.strip() for chunk in chunks if chunk.strip())


class GeminiASR(AsrBackend):
    """Transcribe audio with Gemini's hosted (non-live) transcription model.

    Local model weights and the CrispASR binary are never touched: this
    backend only needs an API key and ffmpeg with an Opus encoder.
    """

    def __init__(
        self,
        api_key: str | None = None,
        api_key_env: str = DEFAULT_API_KEY_ENV,
        model: str = DEFAULT_MODEL,
        base_url: str = DEFAULT_BASE_URL,
        bitrate_kbps: int = DEFAULT_BITRATE_KBPS,
        transcription_mode: str = DEFAULT_MODE,
        language_codes: list[str] | None = None,
        ffmpeg_path: str = "ffmpeg",
        timeout: float = UPLOAD_TIMEOUT,
        encode_timeout: float = ENCODE_TIMEOUT,
        inline_max_bytes: int = DEFAULT_INLINE_MAX_BYTES,
    ) -> None:
        self.api_key_env = api_key_env
        self.api_key = (api_key or os.environ.get(api_key_env) or "").strip() or None
        self.model = model or DEFAULT_MODEL
        self.base_url = (base_url or DEFAULT_BASE_URL).rstrip("/")
        self.bitrate_kbps = bitrate_kbps
        self.transcription_mode = (transcription_mode or DEFAULT_MODE).strip().lower()
        self.language_codes = [code.strip() for code in (language_codes or []) if code.strip()]
        self.ffmpeg_path = ffmpeg_path or "ffmpeg"
        self.timeout = timeout
        self.encode_timeout = encode_timeout
        self.inline_max_bytes = inline_max_bytes
        self._http_client: httpx.AsyncClient | None = None

    # ------------------------------------------------------------------
    # lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        """Verify the prerequisites up front so hotkeys fail fast and clearly."""
        self._validate()
        if self._http_client is None:
            self._http_client = _client(self.timeout)

    async def stop(self) -> None:
        """Close the daemon's reusable HTTP connection pool."""
        if self._http_client is not None:
            await self._http_client.aclose()
            self._http_client = None

    @contextlib.asynccontextmanager
    async def _session(self) -> AsyncIterator[httpx.AsyncClient]:
        """Reuse the daemon client, or scope a client to a standalone request."""
        if self._http_client is not None:
            yield self._http_client
        else:
            async with _client(self.timeout) as client:
                yield client

    # ------------------------------------------------------------------
    # transcription pipeline
    # ------------------------------------------------------------------

    async def transcribe(self, audio_path: Path, profile: Profile | None = None) -> Transcript:
        """Encode, upload, transcribe, and always clean up both sides.

        The remote file is deleted even when transcription fails, and the
        temporary encoded audio lives in a directory that is removed here.
        """
        self._validate()
        source = Path(audio_path)
        if not source.exists():
            raise GeminiError(f"Audio file not found: {source}")

        timings: dict[str, float] = {}
        started = time.perf_counter()
        route = "encoding"
        size = 0
        try:
            with _scratch_dir() as tmp:
                with _measure(timings, "encode"):
                    encoded = await self._encode_opus(source, tmp)
                size = encoded.stat().st_size
                headers = self._headers(self.api_key or "")
                stage = "transcription"
                uploaded: UploadedFile | None = None
                try:
                    if self.inline_max_bytes > 0 and size <= self.inline_max_bytes:
                        route = "inline"
                        with _measure(timings, "transcription"):
                            text = await self._request_transcript(
                                None, headers, data=encoded.read_bytes()
                            )
                    else:
                        route = "files"
                        stage = "upload"
                        uploaded = await self._upload(encoded, headers, timings)
                        stage = "file processing"
                        with _measure(timings, "processing"):
                            await self._await_active(uploaded, headers)
                        stage = "transcription"
                        with _measure(timings, "transcription"):
                            text = await self._request_transcript(uploaded.uri, headers)
                except httpx.RequestError as exc:
                    raise GeminiError(_connectivity_message(exc, stage)) from None
                finally:
                    if uploaded is not None:
                        with _measure(timings, "cleanup"):
                            await self._delete_file(uploaded, headers)
        finally:
            log.info(
                "Gemini timing path=%s bytes=%d encode=%.3fs upload_start=%.3fs "
                "upload_bytes=%.3fs processing=%.3fs transcription=%.3fs cleanup=%.3fs total=%.3fs",
                route,
                size,
                *(
                    timings.get(phase, 0.0)
                    for phase in (
                        "encode",
                        "upload_start",
                        "upload_bytes",
                        "processing",
                        "transcription",
                        "cleanup",
                    )
                ),
                time.perf_counter() - started,
            )

        text = text.strip()
        log.info("Gemini transcript: %s", text[:200])
        return Transcript(text=text)

    async def _encode_opus(self, audio_path: Path, out_dir: Path) -> Path:
        """Compress the recording to mono Ogg Opus before uploading it."""
        ffmpeg = self._require_ffmpeg()
        kbps = int(self.bitrate_kbps)

        out_path = out_dir / f"{audio_path.stem}-{kbps}k.opus"
        cmd = [
            ffmpeg,
            "-nostdin",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(audio_path),
            "-map_metadata",
            "-1",
            "-vn",
            "-ac",
            "1",
            "-ar",
            "48000",
            "-c:a",
            "libopus",
            "-b:a",
            f"{kbps}k",
            "-application",
            "voip",
            "-f",
            "ogg",
            str(out_path),
        ]
        log.info("Encoding audio with ffmpeg (Opus/Ogg, mono, %d kbps)", kbps)

        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            _stdout, stderr = await asyncio.wait_for(proc.communicate(), self.encode_timeout)
        except TimeoutError:
            await _kill(proc)
            raise GeminiError(
                f"ffmpeg did not finish encoding within {self.encode_timeout:g}s"
            ) from None
        except asyncio.CancelledError:
            await _kill(proc)
            raise

        if proc.returncode != 0:
            err = stderr.decode(errors="replace").strip()
            raise GeminiError(_truncate(f"ffmpeg Opus encoding failed: {err}") or "ffmpeg failed")
        if not out_path.exists():
            raise GeminiError("ffmpeg reported success but produced no Opus output")
        return out_path

    async def _upload(
        self, encoded: Path, headers: dict[str, str], timings: dict[str, float] | None = None
    ) -> UploadedFile:
        """Resumable two-step upload through the Files API."""
        data = encoded.read_bytes()
        start_headers = {
            **headers,
            "Content-Type": "application/json",
            "X-Goog-Upload-Protocol": "resumable",
            "X-Goog-Upload-Command": "start",
            "X-Goog-Upload-Header-Content-Length": str(len(data)),
            "X-Goog-Upload-Header-Content-Type": AUDIO_MIME,
        }
        payload: dict[str, Any] = {"file": {"display_name": UPLOAD_DISPLAY_NAME}}

        timings = timings if timings is not None else {}
        async with self._session() as client:
            with _measure(timings, "upload_start"):
                r = await client.post(
                    f"{self.base_url}/upload/v1beta/files",
                    headers=start_headers,
                    json=payload,
                )
            self._raise_for_status(r, "upload start", headers)
            upload_url = r.headers.get("x-goog-upload-url")
            if not upload_url:
                raise GeminiError(
                    "Gemini upload start response did not return an x-goog-upload-url header"
                )

            with _measure(timings, "upload_bytes"):
                up = await client.post(
                    upload_url,
                    headers={
                        **headers,
                        "Content-Length": str(len(data)),
                        "X-Goog-Upload-Offset": "0",
                        "X-Goog-Upload-Command": "upload, finalize",
                    },
                    content=data,
                )
        self._raise_for_status(up, "audio upload", headers)

        try:
            body: Any = up.json()
        except ValueError as exc:
            raise GeminiError("Gemini upload response was not valid JSON") from exc
        if not isinstance(body, dict):
            raise GeminiError("Gemini upload response was not a JSON object")
        uploaded = _parse_uploaded_file(body)
        log.info("Uploaded audio to Gemini (state=%s)", uploaded.state)
        return uploaded

    async def _await_active(self, uploaded: UploadedFile, headers: dict[str, str]) -> None:
        """Poll the Files API until the uploaded file is ACTIVE (bounded).

        A terminal non-ACTIVE state on the upload response (e.g. FAILED) is
        rejected up front. PROCESSING is polled under a single overall
        ``asyncio.timeout`` so the whole wait stays bounded even though each
        status GET may itself allow up to ``self.timeout``.
        """
        state = uploaded.state.upper()
        if state == "ACTIVE":
            return
        if state != "PROCESSING":
            raise GeminiError(
                f"Gemini file {uploaded.file_id} is {uploaded.state}; "
                "upload did not reach a usable state"
            )

        last = uploaded.state
        try:
            async with asyncio.timeout(FILE_ACTIVE_TIMEOUT):
                async with self._session() as client:
                    while True:
                        await asyncio.sleep(FILE_POLL_INTERVAL)
                        r = await client.get(
                            f"{self.base_url}/v1beta/files/{uploaded.file_id}", headers=headers
                        )
                        self._raise_for_status(r, "file status", headers)
                        try:
                            body: Any = r.json()
                        except ValueError as exc:
                            raise GeminiError(
                                "Gemini file status response was not valid JSON"
                            ) from exc
                        if not isinstance(body, dict):
                            raise GeminiError("Gemini file status response was not a JSON object")
                        polled = body.get("state")
                        if not isinstance(polled, str):
                            raise GeminiError("Gemini file status response is missing 'state'")
                        last = polled
                        if polled.upper() == "ACTIVE":
                            log.info("Gemini file %s is ACTIVE", uploaded.file_id)
                            return
                        if polled.upper() != "PROCESSING":
                            raise GeminiError(f"Gemini file {uploaded.file_id} is {polled}")
        except TimeoutError:
            raise GeminiError(
                f"Gemini file {uploaded.file_id} stayed {last} "
                f"for {FILE_ACTIVE_TIMEOUT:g}s; try again later"
            ) from None

    async def _request_transcript(
        self, uri: str | None, headers: dict[str, str], *, data: bytes | None = None
    ) -> str:
        """Transcribe inline Opus or an uploaded file using the same model settings."""
        audio: dict[str, str] = {"type": "audio", "mime_type": AUDIO_MIME}
        if data is not None:
            audio["data"] = base64.b64encode(data).decode("ascii")
        elif uri is not None:
            audio["uri"] = uri
        else:
            raise GeminiError("No audio data or file URI supplied for transcription")
        payload: dict[str, Any] = {
            "model": self.model,
            "input": [audio],
            "generation_config": {
                "transcription_config": {
                    "mode": self._mode_payload(),
                    "language_codes": self.language_codes,
                }
            },
            "store": False,
        }
        log.info(
            "Requesting Gemini transcription (model=%s, mode=%s)",
            self.model,
            self.transcription_mode,
        )

        async with self._session() as client:
            r = await client.post(
                f"{self.base_url}/v1beta/interactions",
                headers=headers,
                json=payload,
            )
        self._raise_for_status(r, "transcription", headers)

        try:
            body: Any = r.json()
        except ValueError as exc:
            raise GeminiError("Gemini transcription response was not valid JSON") from exc
        if not isinstance(body, dict):
            raise GeminiError("Gemini transcription response was not a JSON object")
        return _extract_transcript(body, headers.get("x-goog-api-key"))

    async def _delete_file(self, uploaded: UploadedFile, headers: dict[str, str]) -> None:
        """Best-effort remote cleanup; never masks the original failure.

        Uses a short, distinct timeout so deletion can never stall a successful
        dictation for the full upload budget. Raw exception text is not logged
        because it can embed the credential-bearing request URL.
        """
        try:
            async with self._session() as client:
                r = await client.delete(
                    f"{self.base_url}/v1beta/files/{uploaded.file_id}",
                    headers=headers,
                    timeout=DELETE_TIMEOUT,
                )
            if r.is_success:
                log.info("Deleted uploaded Gemini file %s", uploaded.file_id)
            else:
                log.warning(
                    "Could not delete uploaded Gemini file %s (HTTP %d)",
                    uploaded.file_id,
                    r.status_code,
                )
        except (httpx.HTTPError, OSError) as exc:
            log.warning(
                "Could not delete uploaded Gemini file %s (%s)",
                uploaded.file_id,
                type(exc).__name__,
            )

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------

    def _validate(self) -> None:
        """Check API key, encoder, mode and bitrate before doing any work."""
        self._require_api_key()
        self._require_ffmpeg()
        self._mode_payload()
        if self.inline_max_bytes < 0:
            raise GeminiError("Gemini inline_max_bytes must be nonnegative")
        if not 0 < int(self.bitrate_kbps) <= 510:
            raise GeminiError(
                f"Invalid Gemini Opus bitrate {self.bitrate_kbps}; use 1..510 kbps "
                "(24-32 kbps mono is a good range for speech)."
            )

    def _mode_payload(self) -> object:
        """Interactions API accepts ``"smart"`` or ``{"type": "verbatim"}``."""
        mode = (self.transcription_mode or DEFAULT_MODE).strip().lower()
        if mode == "smart":
            return "smart"
        if mode == "verbatim":
            return {"type": "verbatim"}
        raise GeminiError(
            f"Invalid Gemini transcription mode {self.transcription_mode!r}; "
            f"choose one of {', '.join(MODES)}"
        )

    def _headers(self, key: str) -> dict[str, str]:
        """Auth + content headers. The key is never logged."""
        return {
            "x-goog-api-key": key,
            "Content-Type": "application/json",
        }

    def _require_api_key(self) -> str:
        if not self.api_key:
            raise GeminiError(
                f"Gemini API key not found: set the {self.api_key_env} environment "
                "variable (e.g. in ~/.config/dictum/.env)."
            )
        return self.api_key

    def _require_ffmpeg(self) -> str:
        """Resolve ffmpeg, preferring an explicit path over PATH lookup."""
        candidate = self.ffmpeg_path
        if "/" in candidate:
            path = Path(candidate).expanduser()
            if not path.is_file() or not os.access(path, os.X_OK):
                raise GeminiError(
                    f"ffmpeg not found or not executable at {path}. Install FFmpeg "
                    "(e.g. `sudo apt install ffmpeg`); it needs the libopus encoder."
                )
            return str(path)
        found = shutil.which(candidate)
        if found is None:
            raise GeminiError(
                "ffmpeg not found on PATH. Install FFmpeg (e.g. `sudo apt install "
                "ffmpeg`); it needs the libopus encoder for Opus compression."
            )
        return found

    def _raise_for_status(
        self, response: httpx.Response, what: str, headers: dict[str, str]
    ) -> None:
        """Turn an HTTP error into an actionable message without leaking secrets."""
        if response.is_success:
            return
        secret = headers.get("x-goog-api-key")
        detail = _error_detail(response, secret)
        status = response.status_code
        if status in (401, 403):
            hint = f"check the {self.api_key_env} API key and its Gemini API access"
        elif status == 429:
            hint = "quota or rate limit exceeded; wait and retry, or check billing"
        elif status >= 500:
            hint = "Gemini service error; retry later"
        else:
            hint = "review the Gemini backend configuration"
        message = f"Gemini {what} failed (HTTP {status}): {hint}"
        if detail:
            message = f"{message} — {detail}"
        raise GeminiError(message)


async def _kill(proc: asyncio.subprocess.Process) -> None:
    """Kill a subprocess and reap it (timeout or cancellation cleanup)."""
    if proc.returncode is None:
        proc.kill()
        try:
            await proc.wait()
        except asyncio.CancelledError:
            pass
