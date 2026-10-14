"""Gemini Live API speech transcription for completed Dictum recordings.

Audio is converted locally to the Live API's 16 kHz mono PCM format and
streamed over the authenticated Google GenAI SDK connection. Credentials are
read from the configured environment variable and are never logged.
"""

from __future__ import annotations

import asyncio
import logging
import os
import shutil
from pathlib import Path

from dictum.backends import AsrBackend
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


class GeminiLiveASR(AsrBackend):
    """Transcribe a recorded audio file through Gemini's Live transcription API."""

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

    def _preflight(self, audio_path: Path) -> str:
        if not audio_path.is_file():
            raise GeminiLiveError(f"Audio file not found: {audio_path}")
        if self.transcription_mode not in MODES:
            raise GeminiLiveError("Gemini transcription mode must be 'smart' or 'verbatim'")
        if self.timeout <= 0:
            raise GeminiLiveError("Gemini Live timeout must be greater than zero")
        if shutil.which(self.ffmpeg_path) is None:
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
        api_key = self._preflight(audio_path)
        try:
            from google import genai
            from google.genai import types
        except ImportError as exc:
            raise GeminiLiveError(
                "Gemini Live requires google-genai; reinstall Dictum with its dependencies"
            ) from exc

        pcm = await self._to_pcm(audio_path)
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
        chunks = 0
        text_parts: list[str] = []
        phase = "connect"
        try:
            async with asyncio.timeout(self.timeout):
                async with client.aio.live.connect(model=self.model, config=config) as session:
                    phase = "send"
                    log.info("Gemini Live connected (pcm_bytes=%d)", len(pcm))
                    # A completed push-to-talk recording is one explicit activity,
                    # not a microphone stream whose boundaries require server VAD.
                    await session.send_realtime_input(activity_start=types.ActivityStart())
                    for offset in range(0, len(pcm), PCM_CHUNK_BYTES):
                        chunk = pcm[offset : offset + PCM_CHUNK_BYTES]
                        await session.send_realtime_input(
                            audio=types.Blob(
                                data=chunk, mime_type=f"audio/pcm;rate={PCM_SAMPLE_RATE}"
                            )
                        )
                        chunks += 1
                    await session.send_realtime_input(activity_end=types.ActivityEnd())
                    phase = "finalize"
                    log.info("Gemini Live recording sent (audio_chunks=%d)", chunks)
                    # Non-speech input can produce no final event at all. Do not
                    # wait for the server's minute-long idle connection timeout.
                    async with asyncio.timeout(min(self.timeout, FINALIZATION_TIMEOUT)):
                        async for response in session.receive():
                            content = response.server_content
                            if content is None:
                                continue
                            transcription = content.input_transcription
                            if transcription is not None:
                                if transcription.text:
                                    text_parts.append(transcription.text)
                                break
                            if content.turn_complete:
                                break
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
        except GeminiLiveError:
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
        log.info(
            "Gemini Live transcription completed (audio_chunks=%d, chars=%d)", chunks, len(text)
        )
        return Transcript(text=text)
