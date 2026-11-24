"""Gemini Live API speech transcription for live microphone capture and files.

Audio is converted locally to the Live API's 16 kHz mono PCM format and
streamed over the authenticated Google GenAI SDK connection. Credentials are
read from the configured environment variable and are never logged.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import os
import shutil
import time
from collections.abc import AsyncIterator, Callable
from pathlib import Path

from dictum.backends import AudioStreamError, StreamingAsrBackend
from dictum.models import Profile, Transcript

log = logging.getLogger(__name__)

DEFAULT_MODEL = "gemini-3.5-transcribe-live"
DEFAULT_API_VERSION = "v1beta"
DEFAULT_API_KEY_ENV = "GEMINI_API_KEY"
DEFAULT_FFMPEG = "ffmpeg"
DEFAULT_TIMEOUT = 120.0
FINALIZATION_TIMEOUT = 10.0
PCM_SAMPLE_RATE = 16_000
PCM_CHUNK_BYTES = 3_200  # 100 ms, mono 16-bit PCM at 16 kHz
MODES = {"smart", "verbatim"}


class GeminiLiveError(RuntimeError):
    """Actionable, credential-safe Gemini Live API failure."""


class GeminiLiveASR(StreamingAsrBackend):
    """Stream microphone audio through Gemini's Live transcription API."""

    def __init__(
        self,
        *,
        api_key_env: str = DEFAULT_API_KEY_ENV,
        model: str = DEFAULT_MODEL,
        transcription_mode: str = "smart",
        language_codes: list[str] | None = None,
        ffmpeg_path: str = DEFAULT_FFMPEG,
        timeout: float = DEFAULT_TIMEOUT,
    ) -> None:
        self.api_key_env = api_key_env
        self.model = model
        self.transcription_mode = transcription_mode.lower()
        self.language_codes = language_codes or []
        self.ffmpeg_path = ffmpeg_path
        self.timeout = timeout

    def _preflight(self, audio_path: Path | None = None) -> str:
        if audio_path is not None and not audio_path.is_file():
            raise GeminiLiveError(f"Audio file not found: {audio_path}")
        if self.transcription_mode not in MODES:
            raise GeminiLiveError("Gemini transcription mode must be 'smart' or 'verbatim'")
        if self.timeout <= 0:
            raise GeminiLiveError("Gemini Live timeout must be greater than zero")
        if audio_path is not None and shutil.which(self.ffmpeg_path) is None:
            raise GeminiLiveError(
                f"ffmpeg executable not found: {self.ffmpeg_path}; "
                "install ffmpeg or configure its path"
            )
        api_key = os.environ.get(self.api_key_env, "").strip()
        if not api_key:
            raise GeminiLiveError(
                f"Gemini API key not found; set {self.api_key_env} in the service environment"
            )
        return api_key

    async def _to_pcm(self, audio_path: Path) -> bytes:
        """Convert the recording locally to mono 16 kHz signed 16-bit PCM."""
        try:
            proc = await asyncio.create_subprocess_exec(
                self.ffmpeg_path,
                "-nostdin",
                "-v",
                "error",
                "-i",
                str(audio_path),
                "-f",
                "s16le",
                "-acodec",
                "pcm_s16le",
                "-ac",
                "1",
                "-ar",
                str(PCM_SAMPLE_RATE),
                "pipe:1",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
        except OSError as exc:
            raise GeminiLiveError(
                "Could not start ffmpeg for Gemini Live audio conversion"
            ) from exc

        try:
            pcm, stderr = await asyncio.wait_for(proc.communicate(), timeout=60)
        except TimeoutError as exc:
            proc.kill()
            await proc.wait()
            raise GeminiLiveError("ffmpeg timed out converting audio for Gemini Live") from exc
        if proc.returncode != 0:
            # ffmpeg output can include the input's local path; do not echo it.
            detail = " (check the recording format and ffmpeg installation)" if stderr else ""
            raise GeminiLiveError(f"ffmpeg failed converting audio for Gemini Live{detail}")
        if not pcm:
            raise GeminiLiveError("ffmpeg produced no audio for Gemini Live transcription")
        return pcm

    async def transcribe(self, audio_path: Path, profile: Profile | None = None) -> Transcript:
        """Convert and stream a completed recording, returning finalized text."""
        del profile  # Live API settings are entirely controlled by AsrConfig.
        self._preflight(audio_path)
        pcm = await self._to_pcm(audio_path)

        async def chunks() -> AsyncIterator[bytes]:
            for offset in range(0, len(pcm), PCM_CHUNK_BYTES):
                yield pcm[offset : offset + PCM_CHUNK_BYTES]

        return await self.transcribe_stream(chunks())

    async def transcribe_stream(
        self,
        audio: AsyncIterator[bytes],
        profile: Profile | None = None,
        on_update: Callable[[str], None] | None = None,
    ) -> Transcript:
        """Send capture and receive interim/final text concurrently."""
        del profile
        api_key = self._preflight()
        try:
            from google import genai
            from google.genai import types
        except ImportError as exc:
            raise GeminiLiveError(
                "Gemini Live requires google-genai; reinstall Dictum with its dependencies"
            ) from exc

        config = types.LiveConnectConfig(
            response_modalities=[types.Modality.TEXT],
            realtime_input_config=types.RealtimeInputConfig(
                automatic_activity_detection=types.AutomaticActivityDetection(disabled=True)
            ),
            input_audio_transcription=types.AudioTranscriptionConfig(
                language_codes=self.language_codes,
                mode=(
                    types.AudioTranscriptionConfigMode.SMART
                    if self.transcription_mode == "smart"
                    else types.AudioTranscriptionConfigMode.VERBATIM
                ),
            ),
        )
        client = genai.Client(
            api_key=api_key,
            http_options=types.HttpOptions(api_version=DEFAULT_API_VERSION),
        )
        text_parts: list[str] = []
        phase = "connect"
        started_at = time.monotonic()
        chunks_sent = 0
        bytes_sent = 0
        responses_received = 0
        transcription_events = 0
        interim_events = 0
        turn_complete_events = 0
        last_event = "none"
        ended = asyncio.Event()
        final = asyncio.Event()

        log.info(
            "Gemini Live starting (model=%s, mode=%s, languages=%s, timeout=%.1fs, "
            "finalize_timeout=%.1fs)",
            self.model,
            self.transcription_mode,
            self.language_codes or ["auto"],
            self.timeout,
            min(self.timeout, FINALIZATION_TIMEOUT),
        )
        try:
            async with contextlib.AsyncExitStack() as stack:
                async with asyncio.timeout(self.timeout):
                    session = await stack.enter_async_context(
                        client.aio.live.connect(model=self.model, config=config)
                    )
                connect_s = time.monotonic() - started_at
                phase = "stream"
                log.info("Gemini Live connected (model=%s, connect_s=%.3f)", self.model, connect_s)
                async with asyncio.timeout(self.timeout):
                    await session.send_realtime_input(activity_start=types.ActivityStart())
                log.debug("Gemini Live sent activity_start")

                async def send() -> None:
                    nonlocal phase, chunks_sent, bytes_sent
                    stream_started = time.monotonic()
                    async for chunk in audio:
                        async with asyncio.timeout(self.timeout):
                            await session.send_realtime_input(
                                audio=types.Blob(
                                    data=chunk, mime_type=f"audio/pcm;rate={PCM_SAMPLE_RATE}"
                                )
                            )
                        chunks_sent += 1
                        bytes_sent += len(chunk)
                    # Mark before the await so an immediate server response
                    # cannot race the end marker on the receiving task.
                    phase = "finalize"
                    ended.set()
                    log.info(
                        "Gemini Live audio ended; sending activity_end "
                        "(chunks=%d, bytes=%d, stream_s=%.3f, parts_so_far=%d, chars_so_far=%d, "
                        "interims=%d, turn_completes=%d)",
                        chunks_sent,
                        bytes_sent,
                        time.monotonic() - stream_started,
                        len(text_parts),
                        sum(len(p) for p in text_parts),
                        interim_events,
                        turn_complete_events,
                    )
                    async with asyncio.timeout(min(self.timeout, FINALIZATION_TIMEOUT)):
                        await session.send_realtime_input(activity_end=types.ActivityEnd())
                        await final.wait()

                async def receive() -> None:
                    nonlocal responses_received, transcription_events, interim_events
                    nonlocal turn_complete_events, last_event
                    # SDK receive() ends at a server turn boundary. Continue
                    # listening across any boundaries received during capture.
                    while True:
                        got_response = False
                        async for response in session.receive():
                            got_response = True
                            responses_received += 1
                            voice_activity = getattr(response, "voice_activity", None)
                            va_type = getattr(voice_activity, "voice_activity_type", None)
                            va_name = (
                                getattr(va_type, "value", str(va_type))
                                if va_type is not None
                                else None
                            )
                            is_activity_end = va_name is not None and va_name.endswith(
                                "ACTIVITY_END"
                            )
                            content = getattr(response, "server_content", None)
                            if content is None:
                                top_fields = [
                                    name
                                    for name in (
                                        "setup_complete",
                                        "go_away",
                                        "tool_call",
                                        "tool_call_cancellation",
                                        "session_resumption_update",
                                        "usage_metadata",
                                    )
                                    if getattr(response, name, None) is not None
                                ]
                                if va_name is not None:
                                    top_fields.append(f"voice_activity({va_name})")
                                last_event = f"non_content({','.join(top_fields) or 'empty'})"
                                if is_activity_end and ended.is_set():
                                    final.set()
                                log.info(
                                    "Gemini Live event #%d: %s (ended=%s, final=%s)",
                                    responses_received,
                                    last_event,
                                    ended.is_set(),
                                    final.is_set(),
                                )
                                continue
                            interim = getattr(content, "interim_input_transcription", None)
                            transcription = getattr(content, "input_transcription", None)
                            turn_complete = bool(getattr(content, "turn_complete", False))
                            generation_complete = bool(
                                getattr(content, "generation_complete", False)
                            )
                            interrupted = bool(getattr(content, "interrupted", False))
                            event_tags: list[str] = []
                            if va_name is not None:
                                event_tags.append(f"voice_activity({va_name})")
                            if transcription is not None:
                                transcription_events += 1
                                t_text = transcription.text or ""
                                finished = getattr(transcription, "finished", None)
                                event_tags.append(
                                    f"input_transcription(chars={len(t_text)},finished={finished})"
                                )
                                if t_text:
                                    text_parts.append(t_text)
                                    if on_update is not None:
                                        on_update("".join(text_parts))
                                # Dedicated Transcribe Live emits finalized
                                # segments; SDK versions may also mark streamed
                                # pieces explicitly unfinished. Never stop on one
                                # of those pieces, even after activity_end.
                                if ended.is_set() and (
                                    finished is True or (finished is None and t_text)
                                ):
                                    final.set()
                            if interim is not None:
                                interim_events += 1
                                i_text = interim.text or ""
                                event_tags.append(f"interim(chars={len(i_text)})")
                                if i_text and on_update is not None:
                                    on_update("".join(text_parts) + i_text)
                            if turn_complete:
                                turn_complete_events += 1
                                event_tags.append("turn_complete")
                            if generation_complete:
                                event_tags.append("generation_complete")
                            if interrupted:
                                event_tags.append("interrupted")
                            if getattr(content, "model_turn", None) is not None:
                                event_tags.append("model_turn")
                            if ended.is_set() and (
                                turn_complete
                                or is_activity_end
                                or (generation_complete and bool(text_parts))
                            ):
                                final.set()
                            last_event = ",".join(event_tags) or "empty_server_content"
                            if transcription is not None or turn_complete or not interim:
                                log.info(
                                    "Gemini Live event #%d: %s (ended=%s, final=%s)",
                                    responses_received,
                                    last_event,
                                    ended.is_set(),
                                    final.is_set(),
                                )
                            else:
                                log.debug(
                                    "Gemini Live event #%d: %s (ended=%s)",
                                    responses_received,
                                    last_event,
                                    ended.is_set(),
                                )
                        log.info(
                            "Gemini Live receive() turn loop exited "
                            "(got_response=%s, ended=%s, final=%s, responses=%d)",
                            got_response,
                            ended.is_set(),
                            final.is_set(),
                            responses_received,
                        )
                        if final.is_set():
                            return
                        if not got_response:
                            raise GeminiLiveError(
                                "Gemini Live connection closed before transcription completed"
                            )
                        await asyncio.sleep(0)

                sender = asyncio.create_task(send())
                receiver = asyncio.create_task(receive())
                try:
                    # Propagate either side's failure immediately; always
                    # cancel and join its sibling before closing the session.
                    done, _ = await asyncio.wait(
                        (sender, receiver), return_when=asyncio.FIRST_COMPLETED
                    )
                    for task in done:
                        task.result()
                    # A receiver can finish just before send() observes final.
                    await sender
                finally:
                    for task in (sender, receiver):
                        task.cancel()
                    await asyncio.gather(sender, receiver, return_exceptions=True)
        except TimeoutError as exc:
            log.error(
                "Gemini Live timed out (phase=%s, elapsed_s=%.3f, chunks_sent=%d, bytes_sent=%d, "
                "responses=%d, transcriptions=%d, interims=%d, turn_completes=%d, "
                "accumulated_chars=%d, ended=%s, final=%s, last_event=%s)",
                phase,
                time.monotonic() - started_at,
                chunks_sent,
                bytes_sent,
                responses_received,
                transcription_events,
                interim_events,
                turn_complete_events,
                sum(len(p) for p in text_parts),
                ended.is_set(),
                final.is_set(),
                last_event,
            )
            if phase == "finalize":
                raise GeminiLiveError(
                    "Gemini Live returned no final transcript within the finalization deadline; "
                    "check microphone input and retry"
                ) from None
            raise GeminiLiveError(
                "Gemini Live transcription timed out; retry the recording"
            ) from exc
        except (GeminiLiveError, AudioStreamError) as exc:
            log.error(
                "Gemini Live aborted (%s: %s; phase=%s, elapsed_s=%.3f, chunks_sent=%d, "
                "bytes_sent=%d, responses=%d, transcriptions=%d, accumulated_chars=%d, "
                "last_event=%s)",
                type(exc).__name__,
                exc,
                phase,
                time.monotonic() - started_at,
                chunks_sent,
                bytes_sent,
                responses_received,
                transcription_events,
                sum(len(p) for p in text_parts),
                last_event,
            )
            raise
        except Exception as exc:
            # SDK exceptions may contain credential-bearing request URLs.
            status_code = getattr(exc, "code", getattr(exc, "status_code", None))
            if not isinstance(status_code, int):
                response_obj = getattr(exc, "response", None)
                status_code = getattr(response_obj, "status_code", None)
            if not isinstance(status_code, int):
                status_code = "unknown"
            log.error(
                "Gemini Live request failed (exception=%s, code=%s, phase=%s, elapsed_s=%.3f, "
                "chunks_sent=%d, bytes_sent=%d, responses=%d, accumulated_chars=%d, last_event=%s)",
                type(exc).__name__,
                status_code,
                phase,
                time.monotonic() - started_at,
                chunks_sent,
                bytes_sent,
                responses_received,
                sum(len(p) for p in text_parts),
                last_event,
            )
            raise GeminiLiveError(
                "Gemini Live transcription request failed; check service logs and retry"
            ) from None
        finally:
            await client.aio.aclose()
            client.close()

        text = "".join(text_parts).strip()
        log.info(
            "Gemini Live transcription completed (chunks_sent=%d, bytes_sent=%d, responses=%d, "
            "transcriptions=%d, interims=%d, chars=%d, elapsed_s=%.3f)",
            chunks_sent,
            bytes_sent,
            responses_received,
            transcription_events,
            interim_events,
            len(text),
            time.monotonic() - started_at,
        )
        return Transcript(text=text)
