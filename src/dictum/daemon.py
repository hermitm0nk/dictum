"""Dictum daemon — state machine, pipeline, Unix socket IPC server."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import signal
import sys
from pathlib import Path
from typing import Any

from dictum.asr import create_asr_backend
from dictum.backends import AsrBackend, LlmBackend
from dictum.ipc import SOCKET_PATH, ensure_socket_dir, json_response
from dictum.llm_local import create_llm_backend
from dictum.models import (
    DictationResult,
    Profile,
    ResultTarget,
    Transcript,
)
from dictum.notify import Notifier
from dictum.output import OutputSink
from dictum.recorder import Recorder
from dictum.state import DictumState

log = logging.getLogger("dictum.daemon")


class Daemon:
    """Single-shot serial daemon: one recording pipeline at a time."""

    def __init__(self, profile: Profile | None = None) -> None:
        self.state = DictumState.IDLE
        self.active_profile = profile or Profile()
        self.last_error: str | None = None
        self.last_result: DictationResult | None = None

        # Backends
        self.recorder = Recorder()
        self.asr: AsrBackend = create_asr_backend(self.active_profile, use_server=True)
        self.llm: LlmBackend = create_llm_backend(self.active_profile)
        self.output = OutputSink()
        self.notifier = Notifier()

    # ------------------------------------------------------------------
    # state helpers
    # ------------------------------------------------------------------

    def _set(self, s: DictumState, err: str | None = None) -> None:
        """Update state and send notification."""
        self.state = s
        self.last_error = err
        log.info("State -> %s%s", s.value, f" ({err})" if err else "")

        # Schedule notification in event loop
        asyncio.create_task(self._notify_state(s, err))

    async def _notify_state(self, s: DictumState, err: str | None = None) -> None:
        """Send desktop notification — one at a time, dismissed on idle."""
        if s == DictumState.IDLE:
            await self.notifier.close()
        elif s == DictumState.RECORDING:
            await self.notifier.notify(
                "🎤 Recording", "Speak now…", icon="audio-input-microphone", timeout=0
            )
        elif s == DictumState.TRANSCRIBING:
            await self.notifier.notify(
                "📝 Transcribing",
                "Converting speech to text…",
                icon="accessories-text-editor",
                timeout=3000,
            )
        elif s == DictumState.POLISHING:
            await self.notifier.notify(
                "✨ Polishing", "Improving with LLM…", icon="edit-find-replace", timeout=3000
            )
        elif s == DictumState.PASTING:
            await self.notifier.notify(
                "📋 Pasting", "Sending to clipboard…", icon="edit-paste", timeout=2000
            )
        elif s == DictumState.FAILED:
            await self.notifier.notify(
                "❌ Failed",
                err or "Unknown error",
                icon="dialog-error",
                urgency="critical",
                timeout=5000,
            )

    def status_dict(self) -> dict[str, Any]:
        return {
            "state": self.state.value,
            "active_profile": self.active_profile.name,
            "last_error": self.last_error,
        }

    # ------------------------------------------------------------------
    # pipeline steps
    # ------------------------------------------------------------------

    def _start_recording(self) -> None:
        """Start capturing audio."""
        self.recorder.start()
        self._set(DictumState.RECORDING)
        log.info("Recording started")

    def _stop_recording(self) -> Path:
        """Stop capturing and return the WAV path."""
        path = self.recorder.stop()
        log.info("Recording saved: %s (%d bytes)", path, path.stat().st_size)
        return path

    async def _transcribe(self, audio_path: Path) -> Transcript:
        """Transcribe audio file, return Transcript."""
        self._set(DictumState.TRANSCRIBING)
        return await self.asr.transcribe(audio_path, self.active_profile)

    async def _polish(self, transcript: Transcript, profile: Profile) -> str | None:
        """Polish transcript with LLM. Returns None on failure."""
        if not profile.llm:
            return None
        try:
            self._set(DictumState.POLISHING)
            return await self.llm.polish(transcript, profile)
        except Exception as exc:
            log.warning("LLM polish failed, using raw transcript: %s", exc)
            return None

    async def _deliver(self, result: DictationResult) -> None:
        """Output the result."""
        self._set(DictumState.PASTING)
        await self.output.deliver(result, result.target)

    async def _process_audio(
        self, audio_path: Path, profile: Profile, target: ResultTarget
    ) -> DictationResult:
        """Transcribe → polish → deliver. Used by toggle-stop and start/stop."""
        try:
            transcript = await self._transcribe(audio_path)
            polished = await self._polish(transcript, profile)
            result = DictationResult(
                transcript=transcript,
                polished_text=polished,
                target=target,
            )
            await self._deliver(result)
            self._set(DictumState.IDLE)
            self.last_result = result
            return result
        except Exception as exc:
            self._set(DictumState.FAILED, str(exc))
            raise

    async def _record_and_process(self, profile: Profile, target: ResultTarget) -> DictationResult:
        """Record → transcribe → polish → deliver. Used by once and full pipeline."""
        self._start_recording()
        audio_path = await self._wait_for_recording()
        return await self._process_audio(audio_path, profile, target)

    async def _wait_for_recording(self) -> Path:
        """Block until the recorder's silence watchdog fires or it's stopped externally."""
        while self.recorder.is_recording:
            await asyncio.sleep(0.1)
        return self._stop_recording()

    # ------------------------------------------------------------------
    # IPC server
    # ------------------------------------------------------------------

    async def handle_connection(
        self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ) -> None:
        try:
            line = await asyncio.wait_for(reader.readline(), timeout=10)
            if not line:
                return
            req = json.loads(line)
            cmd = req.get("cmd", "")
            log.debug("IPC request: %s", cmd)

            if cmd == "ping":
                resp = {"ok": True, "msg": "pong"}

            elif cmd == "status":
                resp = self.status_dict()
                resp["ok"] = True

            elif cmd == "toggle":
                target = ResultTarget(req.get("target", "paste"))
                if self.state == DictumState.IDLE:
                    # Start recording; respond immediately
                    self._start_recording()
                    resp = {"ok": True, "started": True}
                elif self.state == DictumState.RECORDING:
                    # Stop recording, process in background
                    audio_path = self._stop_recording()
                    asyncio.create_task(
                        self._process_audio_bg(audio_path, self.active_profile, target)
                    )
                    resp = {"ok": True, "stopped_recording": True}
                else:
                    resp = {"ok": False, "error": f"Busy ({self.state.value})"}

            elif cmd == "start":
                if self.state != DictumState.IDLE:
                    resp = {"ok": False, "error": f"Busy ({self.state.value})"}
                else:
                    self._start_recording()
                    resp = {"ok": True, "recording": True}

            elif cmd == "stop":
                if self.state == DictumState.RECORDING:
                    audio_path = self._stop_recording()
                    target = ResultTarget(req.get("target", "paste"))
                    asyncio.create_task(
                        self._process_audio_bg(audio_path, self.active_profile, target)
                    )
                    resp = {"ok": True, "processing": True}
                else:
                    resp = {"ok": False, "error": f"Not recording ({self.state.value})"}

            elif cmd == "cancel":
                if self.recorder.is_recording:
                    self.recorder.stop()
                self._set(DictumState.IDLE)
                resp = {"ok": True, "cancelled": True}

            else:
                resp = {"ok": False, "error": f"Unknown command: {cmd}"}

            writer.write(json_response(resp))
            await writer.drain()
        except Exception as exc:
            log.error("IPC error: %s", exc)
            try:
                writer.write(json_response({"ok": False, "error": str(exc)}))
                await writer.drain()
            except Exception:
                pass
        finally:
            writer.close()
            try:
                await writer.wait_closed()
            except Exception:
                pass

    async def _process_audio_bg(
        self, audio_path: Path, profile: Profile, target: ResultTarget
    ) -> None:
        try:
            await self._process_audio(audio_path, profile, target)
        except Exception as exc:
            log.error("Pipeline failed: %s", exc)

    async def serve(self) -> None:
        """Start the Unix socket server."""
        ensure_socket_dir()
        if SOCKET_PATH.exists():
            SOCKET_PATH.unlink()

        # Start ASR backend (server mode for warm model)
        try:
            await self.asr.start()
            log.info("ASR backend started")
        except Exception as exc:
            log.error("Failed to start ASR backend: %s", exc)
            self._set(DictumState.FAILED, f"ASR backend failed to start: {exc}")

        # Start LLM backend
        try:
            await self.llm.start()
            log.info("LLM backend started")
        except Exception as exc:
            log.error("Failed to start LLM backend: %s", exc)
            self._set(DictumState.FAILED, f"LLM backend failed to start: {exc}")

        self._shutdown_event = asyncio.Event()

        server = await asyncio.start_unix_server(self.handle_connection, str(SOCKET_PATH))
        log.info("Daemon listening on %s (pid=%d)", SOCKET_PATH, os.getpid())

        pid_path = SOCKET_PATH.parent / "daemon.pid"
        pid_path.write_text(str(os.getpid()))

        loop = asyncio.get_running_loop()
        for sig in (signal.SIGTERM, signal.SIGINT):
            loop.add_signal_handler(sig, lambda: asyncio.create_task(self._shutdown(server)))

        try:
            async with server:
                await self._shutdown_event.wait()
        finally:
            self._cleanup()

    def _cleanup(self) -> None:
        if SOCKET_PATH.exists():
            SOCKET_PATH.unlink()
        pid_path = SOCKET_PATH.parent / "daemon.pid"
        if pid_path.exists():
            pid_path.unlink()

    async def _shutdown(self, server: asyncio.AbstractServer) -> None:
        log.info("Shutting down daemon")
        server.close()
        await server.wait_closed()

        # Stop ASR backend
        try:
            await self.asr.stop()
            log.info("ASR backend stopped")
        except Exception as exc:
            log.error("Error stopping ASR backend: %s", exc)

        # Stop LLM backend
        try:
            await self.llm.stop()
            log.info("LLM backend stopped")
        except Exception as exc:
            log.error("Error stopping LLM backend: %s", exc)

        self._shutdown_event.set()


def run_daemon(profile_name: str = "default") -> None:
    """Entry point for `dictum daemon`."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s %(message)s",
        stream=sys.stderr,
    )
    from dictum.config import load_profile

    profile = load_profile(profile_name)
    daemon = Daemon(profile=profile)
    asyncio.run(daemon.serve())
