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
from collections.abc import AsyncIterator, Callable
from pathlib import Path

from dictum.backends import AudioStreamError, StreamingAsrBackend
from dictum.models import Profile, Transcript

log = logging.getLogger(__name__)

DEFAULT_MODEL = "gemini-3.5-transcribe-live"
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
        client = genai.Client(api_key=api_key)
        text_parts: list[str] = []
        phase = "connect"
        try:
            async with contextlib.AsyncExitStack() as stack:
                async with asyncio.timeout(self.timeout):
                    session = await stack.enter_async_context(
                        client.aio.live.connect(model=self.model, config=config)
                    )
                phase = "stream"
                ended = asyncio.Event()
                final = asyncio.Event()
                async with asyncio.timeout(self.timeout):
                    await session.send_realtime_input(activity_start=types.ActivityStart())

                async def send() -> None:
                    nonlocal phase
                    async for chunk in audio:
                        async with asyncio.timeout(self.timeout):
                            await session.send_realtime_input(
                                audio=types.Blob(
                                    data=chunk, mime_type=f"audio/pcm;rate={PCM_SAMPLE_RATE}"
                                )
                            )
                    # Mark before the await so an immediate server response
                    # cannot race the end marker on the receiving task.
                    phase = "finalize"
                    ended.set()
                    async with asyncio.timeout(min(self.timeout, FINALIZATION_TIMEOUT)):
                        await session.send_realtime_input(activity_end=types.ActivityEnd())
                        await final.wait()

                async def receive() -> None:
                    # SDK receive() ends at a server turn boundary. Continue
                    # listening across any boundaries received during capture.
                    while True:
                        got_response = False
                        async for response in session.receive():
                            got_response = True
                            content = response.server_content
                            if content is None:
                                continue
                            interim = getattr(content, "interim_input_transcription", None)
                            transcription = content.input_transcription
                            if transcription is not None:
                                if transcription.text:
                                    text_parts.append(transcription.text)
                                    if on_update is not None:
                                        on_update("".join(text_parts))
                                # Dedicated Transcribe Live emits finalized
                                # segments; SDK versions may also mark streamed
                                # pieces explicitly unfinished. Never stop on one
                                # of those pieces, even after activity_end.
                                finished = getattr(transcription, "finished", None)
                                if ended.is_set() and (
                                    finished is True or (finished is None and transcription.text)
                                ):
                                    final.set()
                            if interim is not None and interim.text and on_update is not None:
                                on_update("".join(text_parts) + interim.text)
                            if content.turn_complete and ended.is_set():
                                final.set()
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
            log.error("Gemini Live timed out (phase=%s)", phase)
            if phase == "finalize":
                raise GeminiLiveError(
                    "Gemini Live returned no final transcript within the finalization deadline; "
                    "check microphone input and retry"
                ) from None
            raise GeminiLiveError(
                "Gemini Live transcription timed out; retry the recording"
            ) from exc
        except (GeminiLiveError, AudioStreamError):
            raise
        except Exception as exc:
            # SDK exceptions may contain credential-bearing request URLs.
            status_code = getattr(exc, "code", getattr(exc, "status_code", None))
            if not isinstance(status_code, int):
                status_code = "unknown"
            log.error(
                "Gemini Live request failed (exception=%s, code=%s, phase=%s)",
                type(exc).__name__,
                status_code,
                phase,
            )
            raise GeminiLiveError(
                "Gemini Live transcription request failed; check service logs and retry"
            ) from None
        finally:
            await client.aio.aclose()
            client.close()

        text = "".join(text_parts).strip()
        log.info("Gemini Live transcription completed (chars=%d)", len(text))
        return Transcript(text=text)
