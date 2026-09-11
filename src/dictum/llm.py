"""LLM backend — OpenAI-compatible API (llama.cpp server, vLLM, cloud)."""

from __future__ import annotations

import logging
import os
from typing import Any

import httpx

from dictum.backends import LlmBackend
from dictum.models import Profile, Transcript

log = logging.getLogger(__name__)

_DEFAULT_BASE_URL = "http://127.0.0.1:8080"


class OpenAILLM(LlmBackend):
    """Polish transcripts through an OpenAI-compatible chat endpoint."""

    def __init__(
        self,
        base_url: str | None = None,
        model: str | None = None,
        temperature: float = 0.2,
        top_p: float | None = None,
        timeout: float = 120.0,
        max_tokens: int = 512,
        enable_thinking: bool = False,
        api_key: str | None = None,
        api_key_env: str | None = None,
        headers: dict[str, str] | None = None,
    ) -> None:
        self.base_url = (base_url or _DEFAULT_BASE_URL).rstrip("/")
        self.model = model or "qwen3.5-4b-q3_k_m"
        self.temperature = temperature
        self.top_p = top_p
        self.timeout = timeout
        self.max_tokens = max_tokens
        self.enable_thinking = enable_thinking
        self.api_key = api_key or (os.environ.get(api_key_env) if api_key_env else None)
        self.extra_headers = dict(headers or {})

    def _url(self) -> str:
        return f"{self.base_url}/v1/chat/completions"

    def _health_url(self) -> str:
        return f"{self.base_url}/health"

    def _headers(self) -> dict[str, str]:
        """Auth + extra headers. Never log the key itself."""
        headers = {"Content-Type": "application/json", **self.extra_headers}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    async def check_health(self) -> bool:
        """Return True if the LLM server is reachable."""
        try:
            async with httpx.AsyncClient(timeout=3) as client:
                r = await client.get(self._health_url(), headers=self._headers())
                return r.status_code == 200
        except (httpx.HTTPError, OSError):
            return False

    async def polish(self, transcript: Transcript, profile: Profile) -> str:
        """Send the transcript through the LLM with the profile's prompt.

        Returns the raw transcript unchanged if it's empty or the LLM is
        unreachable.
        """
        raw = transcript.text.strip()
        if not raw:
            return raw

        system_msg = profile.prompt or Profile.model_fields["prompt"].default

        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_msg},
                {"role": "user", "content": raw},
            ],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "chat_template_kwargs": {"enable_thinking": self.enable_thinking},
        }
        if self.top_p is not None:
            payload["top_p"] = self.top_p

        log.info("LLM request to %s (model=%s)", self._url(), self.model)
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            r = await client.post(self._url(), json=payload, headers=self._headers())
            r.raise_for_status()
            data = r.json()

        text: str = data["choices"][0]["message"]["content"].strip()
        log.info("LLM output: %s", text[:200])
        return text

    async def start(self) -> None:
        """No-op for remote OpenAI-compatible endpoints."""
        pass

    async def stop(self) -> None:
        """No-op for remote OpenAI-compatible endpoints."""
        pass
