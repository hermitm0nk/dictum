"""Audio recording using sounddevice (PipeWire/PulseAudio via PortAudio)."""

from __future__ import annotations

import asyncio
import queue
import threading
import time
import wave
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import numpy as np
import sounddevice as sd  # type: ignore[import-untyped]

from dictum.backends import AudioStreamError


def _native_rate() -> int:
    """Return the default input device's native sample rate."""
    dev = sd.query_devices(kind="input")
    rate = int(dev["default_samplerate"])
    return rate if rate > 0 else 48_000


class Recorder:
    """Records audio from the default input device.

    Opens the stream at the device's native sample rate to avoid PortAudio
    EINVAL_SAMPLE_RATE errors when launched from non-terminal contexts (e.g.
    Hyprland exec).  The saved WAV is always resampled to 16 kHz mono int16,
    which is what Parakeet expects.
    """

    def __init__(
        self,
        sample_rate: int = 16_000,
        channels: int = 1,
        dtype: str = "int16",
        silence_threshold: float = 0.01,
        silence_timeout: float | None = 1.5,
    ) -> None:
        self.target_rate = sample_rate
        self.channels = channels
        self.dtype = dtype
        self.silence_threshold = silence_threshold
        self.silence_timeout = silence_timeout

        self._native_rate = _native_rate()
        self._stream: sd.InputStream | None = None
        self._frames: list[np.ndarray] = []
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._stopped = True
        self._last_path: Path | None = None
        self._audio_queue: queue.Queue[np.ndarray] | None = None
        self._stream_overflow = False

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------

    def start(self, *, streaming: bool = False) -> None:
        """Begin capturing audio into memory."""
        if self._thread is not None and self._thread.is_alive():
            return

        self._frames = []
        # Bound the microphone/network backlog to about eight seconds at 48 kHz.
        self._audio_queue = queue.Queue(maxsize=100) if streaming else None
        self._stream_overflow = False
        self._stopped = False
        self._stop_event.clear()

        self._stream = sd.InputStream(
            samplerate=self._native_rate,
            channels=self.channels,
            dtype=self.dtype,
            blocksize=4096,
            callback=self._audio_callback,
        )
        self._stream.start()
        self._thread = threading.Thread(target=self._silence_watchdog, daemon=True)
        self._thread.start()

    def stop(self) -> Path:
        """Stop recording and return the path to the WAV file (idempotent)."""
        if self._stopped:
            if self._last_path is not None and self._last_path.exists():
                return self._last_path
            return self._write_wav(np.zeros(self.target_rate, dtype=self.dtype))

        self._stopped = True
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=3)
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None

        with self._lock:
            audio = (
                np.concatenate(self._frames)
                if self._frames
                else np.zeros(self._native_rate, dtype=self.dtype)
            )

        # Resample from native rate to target rate (16 kHz)
        if self._native_rate != self.target_rate:
            audio = self._resample(audio)

        path = self._write_wav(audio)
        self._frames = []
        return path

    async def audio_chunks(self) -> AsyncIterator[bytes]:
        """Yield capture as 16 kHz PCM without blocking the PortAudio callback.

        Resampling uses a continuous sample clock across callback boundaries.
        A slow consumer fails explicitly instead of silently dropping speech.
        """
        audio_queue = self._audio_queue
        if audio_queue is None:
            raise RuntimeError("Recorder was not started in streaming mode")
        input_offset = 0
        output_offset = 0
        while True:
            if not self.is_recording and not self._stopped:
                # The watchdog has ended capture. Close PortAudio before draining
                # so it cannot keep adding chunks behind the end of speech.
                self.stop()
            if self._stream_overflow:
                raise AudioStreamError("Live audio queue overflow; check the network and retry")
            try:
                chunk = audio_queue.get_nowait()
            except queue.Empty:
                if not self.is_recording:
                    break
                await asyncio.sleep(0.01)
                continue
            mono = chunk.reshape(-1, self.channels).mean(axis=1)
            end = input_offset + len(mono)
            # Preserve the fractional resampling phase between chunks, including
            # devices whose native rate is not a multiple of 16 kHz (44.1 kHz).
            output_end = (end * 16_000 + self._native_rate - 1) // self._native_rate
            positions = (
                np.arange(output_offset, output_end) * self._native_rate // 16_000 - input_offset
            )
            pcm = mono[positions].astype("<i2").tobytes()
            input_offset = end
            output_offset = output_end
            for offset in range(0, len(pcm), 3200):
                yield pcm[offset : offset + 3200]

    @property
    def is_recording(self) -> bool:
        return not self._stopped and self._thread is not None and self._thread.is_alive()

    # ------------------------------------------------------------------
    # internals
    # ------------------------------------------------------------------

    def _resample(self, audio: np.ndarray) -> np.ndarray:
        """Simple linear resampling from native rate to target rate."""
        ratio = self.target_rate / self._native_rate
        new_len = int(len(audio) * ratio)
        indices = np.linspace(0, len(audio) - 1, new_len)
        return audio[indices.astype(int)]  # type: ignore[no-any-return]

    def _write_wav(self, audio: np.ndarray) -> Path:
        out_dir = Path.home() / ".cache" / "dictum" / "audio"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"rec_{int(time.time() * 1000)}.wav"
        with wave.open(str(out_path), "wb") as wf:
            wf.setnchannels(self.channels)
            wf.setsampwidth(2)
            wf.setframerate(self.target_rate)
            wf.writeframes(audio.tobytes())
        self._last_path = out_path
        return out_path

    def _audio_callback(
        self,
        indata: np.ndarray,
        frames: int,
        time_info: Any,
        status: sd.CallbackFlags,
    ) -> None:  # noqa: ANN401
        chunk = indata.copy()
        with self._lock:
            self._frames.append(chunk)
        if self._audio_queue is not None:
            try:
                self._audio_queue.put_nowait(chunk)
            except queue.Full:
                self._stream_overflow = True

    def _silence_watchdog(self) -> None:
        """Stop automatically after prolonged silence."""
        if self.silence_timeout is None:
            self._stop_event.wait()
            return
        silent_since: float | None = None
        while not self._stop_event.is_set():
            self._stop_event.wait(timeout=0.1)
            with self._lock:
                if not self._frames:
                    continue
                chunk = self._frames[-1].flatten().astype(np.float32) / 32768.0
            rms = float(np.sqrt(np.mean(chunk**2)))
            if rms < self.silence_threshold:
                if silent_since is None:
                    silent_since = time.monotonic()
                elif time.monotonic() - silent_since >= self.silence_timeout:
                    break
            else:
                silent_since = None
        self._stop_event.set()
