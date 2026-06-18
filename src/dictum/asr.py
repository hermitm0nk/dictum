"""ASR backend — Parakeet TDT v0.6b via CrispASR (CLI or HTTP server)."""

from __future__ import annotations

import asyncio
import logging
import os
from pathlib import Path

import httpx

from dictum.accel import ensure_native_binary, native_binary, native_env_for
from dictum.backends import AsrBackend
from dictum.models import Profile, Transcript
from dictum.models_loader import asr_model_path, ensure_asr_model

log = logging.getLogger(__name__)

DEFAULT_PORT = 8081


def _default_binary() -> Path:
    return Path(os.environ.get("DICTUM_PARAKEET_BIN", str(native_binary("parakeet-main"))))


def _is_managed_binary(binary: Path | None) -> bool:
    """True if `binary` is the installer-managed default (no env override)."""
    if binary is not None:
        return False
    return os.environ.get("DICTUM_PARAKEET_BIN", "").strip() == ""


def _default_model() -> Path:
    return asr_model_path()


class ParakeetASR(AsrBackend):
    """Transcribe audio using CrispASR.

    Supports two modes:
    - CLI mode: invoke parakeet-main binary per-request (cold start each time)
    - Server mode: run parakeet-main --server (persistent, warm model, HTTP API)

    Server mode is used by default when running under the daemon for low latency.
    """

    def __init__(
        self,
        binary: Path | None = None,
        model: Path | None = None,
        port: int = DEFAULT_PORT,
        threads: int = 4,
        use_server: bool = True,
    ) -> None:
        self.binary = binary or _default_binary()
        self._binary_managed = _is_managed_binary(binary)
        self.model = model or _default_model()
        self.port = port
        self.threads = threads
        self.use_server = use_server

        self._proc: asyncio.subprocess.Process | None = None
        self._base_url = f"http://127.0.0.1:{port}"
        self._health_url = f"{self._base_url}/health"
        self._infer_url = f"{self._base_url}/inference"

    async def start(self) -> None:
        """Start the CrispASR server if using server mode."""
        if not self.use_server:
            return

        if await self._is_healthy():
            log.info("CrispASR server already running on port %d", self.port)
            return

        if self._binary_managed:
            log.info("Ensuring CrispASR binary is installed …")
            self.binary = ensure_native_binary("parakeet-main")
        elif not self.binary.exists():
            raise FileNotFoundError(
                f"parakeet-main binary not found at {self.binary}. "
                "Set DICTUM_PARAKEET_BIN or run `dictum native install`."
            )
        if not self.model.exists():
            log.info("Parakeet GGUF not found, downloading...")
            self.model = ensure_asr_model()

        log.info("Starting CrispASR server on port %d...", self.port)

        cmd = [
            str(self.binary),
            "-m",
            str(self.model),
            "-t",
            str(self.threads),
            "--server",
            "--port",
            str(self.port),
            "--host",
            "127.0.0.1",
        ]

        env = native_env_for("parakeet-main") if self._binary_managed else None
        self._proc = await asyncio.create_subprocess_exec(
            *cmd,
            env=env,
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.DEVNULL,
        )

        # Wait for health endpoint
        for i in range(200):  # up to 20 seconds (model loading takes longer)
            if await self._is_healthy():
                log.info("CrispASR server healthy after %d attempts", i + 1)
                return
            await asyncio.sleep(0.1)

        raise RuntimeError("CrispASR server failed to start within 20 seconds")

    async def stop(self) -> None:
        """Stop the CrispASR server."""
        if self._proc is not None:
            log.info("Stopping CrispASR server (pid=%d)...", self._proc.pid)
            self._proc.terminate()
            try:
                await asyncio.wait_for(self._proc.wait(), timeout=10)
            except TimeoutError:
                self._proc.kill()
                await self._proc.wait()
            self._proc = None

    async def _is_healthy(self) -> bool:
        try:
            async with httpx.AsyncClient(timeout=1) as client:
                r = await client.get(self._health_url)
                return r.status_code == 200
        except Exception:
            return False

    async def transcribe(self, audio_path: Path, profile: Profile | None = None) -> Transcript:
        """Transcribe audio file.

        Uses server mode if enabled (persistent warm model), otherwise CLI mode.
        """
        if self.use_server:
            return await self._transcribe_server(audio_path)
        return await self._transcribe_cli(audio_path)

    async def _transcribe_server(self, audio_path: Path) -> Transcript:
        """Transcribe via CrispASR HTTP server."""
        if not await self._is_healthy():
            raise RuntimeError("CrispASR server not healthy")

        # CrispASR server expects multipart form with audio file
        with open(audio_path, "rb") as f:
            files = {"file": (audio_path.name, f, "audio/wav")}
            async with httpx.AsyncClient(timeout=120) as client:
                r = await client.post(self._infer_url, files=files)
                r.raise_for_status()
                data = r.json()

        # Server returns JSON with text field
        text = data.get("text", "").strip()
        log.info("ASR output: %s", text[:200])
        return Transcript(text=text)

    async def _transcribe_cli(self, audio_path: Path) -> Transcript:
        """Transcribe via CrispASR CLI (cold start each time)."""
        if self._binary_managed:
            self.binary = ensure_native_binary("parakeet-main")
        elif not self.binary.exists():
            raise FileNotFoundError(
                f"parakeet-main binary not found at {self.binary}. "
                "Set DICTUM_PARAKEET_BIN or run `dictum native install`."
            )
        if not self.model.exists():
            log.info("Parakeet GGUF not found, downloading...")
            self.model = ensure_asr_model()

        cmd = [
            str(self.binary),
            "-m",
            str(self.model),
            "-t",
            str(self.threads),
            str(audio_path),
        ]

        log.info("ASR running: %s", " ".join(cmd))
        env = native_env_for("parakeet-main") if self._binary_managed else None
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            env=env,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=120)

        if proc.returncode != 0:
            err = stderr.decode(errors="replace").strip()
            raise RuntimeError(f"crispasr failed (rc={proc.returncode}): {err}")

        text = stdout.decode(errors="replace").strip()
        log.info("ASR output: %s", text[:200])
        return Transcript(text=text)


def create_asr_backend(profile: Profile, use_server: bool = True) -> AsrBackend:
    """Factory to create ASR backend from profile config."""
    asr_cfg = profile.asr
    backend_name = asr_cfg.backend.lower() if asr_cfg else "parakeet"

    if backend_name == "parakeet":
        # Pass binary=None so ParakeetASR treats it as installer-managed and
        # triggers ensure_native_binary() on first start. Passing
        # _default_binary() explicitly would defeat the managed-binary check.
        return ParakeetASR(
            binary=None,
            port=DEFAULT_PORT,
            threads=4,
            use_server=use_server,
        )

    log.warning("Unknown ASR backend: %s, falling back to parakeet", backend_name)
    return ParakeetASR(use_server=use_server)
