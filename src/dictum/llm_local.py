"""Managed local llama.cpp server backend."""

from __future__ import annotations

import asyncio
import logging
import os
from importlib.resources import files
from pathlib import Path

import httpx

from dictum.backends import LlmBackend
from dictum.models import Profile, Transcript
from dictum.models_loader import ensure_llm_model

log = logging.getLogger(__name__)


def _bundled_binary(name: str) -> str:
    """Resolve a bundled native binary from the installed package."""
    return str(files("dictum").joinpath("bin", name))


def _default_model() -> Path:
    return Path(os.environ.get("DICTUM_LLM_MODEL", str(Path.home() / ".cache" / "dictum" / "models" / "Qwen3.5-4B-Q3_K_M.gguf")))


def _default_binary() -> Path:
    return Path(os.environ.get("DICTUM_LLM_BIN", str(_bundled_binary("llama-server"))))


class ManagedLocalLlm(LlmBackend):
    """Manages a local llama-server process lifecycle.

    Binary is built with rpath ($ORIGIN/../lib/llama) so it finds its
    libraries without LD_LIBRARY_PATH.
    """

    def __init__(
        self,
        model_path: Path | None = None,
        binary_path: Path | None = None,
        port: int = 8080,
        ctx_size: int = 4096,
        n_gpu_layers: int = -1,
        temperature: float = 0.2,
        timeout: float = 20.0,
        max_tokens: int = 512,
    ) -> None:
        self.model_path = model_path or _default_model()
        self.binary_path = binary_path or _default_binary()
        self.port = port
        self.ctx_size = ctx_size
        self.n_gpu_layers = n_gpu_layers
        self.temperature = temperature
        self.timeout = timeout
        self.max_tokens = max_tokens

        self._proc: asyncio.subprocess.Process | None = None
        self._base_url = f"http://127.0.0.1:{port}"
        self._health_url = f"{self._base_url}/health"
        self._chat_url = f"{self._base_url}/v1/chat/completions"

    @property
    def base_url(self) -> str:
        return self._base_url

    async def start(self) -> None:
        """Start the llama-server if not already running."""
        if await self._is_healthy():
            log.info("llama-server already running on port %d", self.port)
            return

        if not self.binary_path.exists():
            raise FileNotFoundError(f"llama-server binary not found at {self.binary_path}")
        if not self.model_path.exists():
            log.info("LLM model not found, downloading...")
            self.model_path = ensure_llm_model()

        log.info("Starting llama-server on port %d...", self.port)

        cmd = [
            str(self.binary_path),
            "-m", str(self.model_path),
            "--port", str(self.port),
            "--ctx-size", str(self.ctx_size),
            "-ngl", str(self.n_gpu_layers),
        ]

        self._proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.DEVNULL,
        )

        # Wait for health endpoint
        for i in range(150):  # up to 15 seconds
            if await self._is_healthy():
                log.info("llama-server healthy after %d attempts", i + 1)
                return
            await asyncio.sleep(0.1)

        raise RuntimeError("llama-server failed to start within 15 seconds")

    async def stop(self) -> None:
        """Stop the llama-server process."""
        if self._proc is not None:
            log.info("Stopping llama-server (pid=%d)...", self._proc.pid)
            self._proc.terminate()
            try:
                await asyncio.wait_for(self._proc.wait(), timeout=5)
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

    async def check_health(self) -> bool:
        return await self._is_healthy()

    async def polish(self, transcript: Transcript, profile: Profile) -> str:
        raw = transcript.text.strip()
        if not raw:
            return raw

        system_msg = profile.prompt or (
            "Fix punctuation, capitalization, and remove filler words "
            "(um, uh, like, you know, I mean) from voice transcription. "
            "Break run-on sentences. Output only the corrected text, "
            "no explanations or notes."
        )

        payload: dict[str, object] = {
            "model": profile.llm.model if profile.llm else "qwen3.5-4b-3bit",
            "messages": [
                {"role": "system", "content": system_msg},
                {"role": "user", "content": raw},
            ],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "chat_template_kwargs": {"enable_thinking": False},
        }

        log.info("LLM request to %s", self._chat_url)
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            r = await client.post(self._chat_url, json=payload)
            r.raise_for_status()
            data = r.json()

        text: str = data["choices"][0]["message"]["content"].strip()
        log.info("LLM output: %s", text[:200])
        return text


def create_llm_backend(profile: Profile) -> LlmBackend | None:
    """Factory to create the appropriate LLM backend from profile config."""
    if not profile.llm:
        return None

    backend = profile.llm.backend.lower()

    if backend == "managed-local":
        return ManagedLocalLlm(
            model_path=profile.llm.model_path,
            binary_path=profile.llm.binary_path,
            port=profile.llm.port,
            ctx_size=profile.llm.ctx_size,
            n_gpu_layers=profile.llm.n_gpu_layers,
            temperature=profile.llm.temperature,
            timeout=profile.llm.timeout_seconds,
        )

    if backend == "openai-compatible":
        from dictum.llm import OpenAILLM
        return OpenAILLM(
            base_url=str(profile.llm.base_url) if profile.llm.base_url else "http://127.0.0.1:8080",
            model=profile.llm.model,
            temperature=profile.llm.temperature,
            timeout=profile.llm.timeout_seconds,
        )

    if backend == "none":
        return None

    log.warning("Unknown LLM backend: %s, falling back to openai-compatible", backend)
    from dictum.llm import OpenAILLM
    return OpenAILLM(
        base_url=str(profile.llm.base_url) if profile.llm.base_url else "http://127.0.0.1:8080",
        model=profile.llm.model,
        temperature=profile.llm.temperature,
        timeout=profile.llm.timeout_seconds,
    )