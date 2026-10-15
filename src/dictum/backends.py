from abc import ABC, abstractmethod
from collections.abc import AsyncIterator, Callable
from pathlib import Path

from dictum.models import DictationResult, Profile, ResultTarget, Transcript


class Recorder(ABC):
    @abstractmethod
    async def start(self) -> None:
        raise NotImplementedError

    @abstractmethod
    async def stop(self) -> Path:
        raise NotImplementedError


class AsrBackend(ABC):
    @abstractmethod
    async def transcribe(self, audio_path: Path, profile: Profile) -> Transcript:
        raise NotImplementedError

    async def start(self) -> None:
        """Optional: start the backend (e.g., launch local server)."""
        pass  # noqa: B027

    async def stop(self) -> None:
        """Optional: stop the backend (e.g., terminate local server)."""
        pass  # noqa: B027


class AudioStreamError(RuntimeError):
    """Actionable local capture failure safe to show without remote error details."""


class StreamingAsrBackend(AsrBackend):
    """Optional PCM streaming capability; file-only backends remain unchanged."""

    @abstractmethod
    async def transcribe_stream(
        self,
        audio: AsyncIterator[bytes],
        profile: Profile | None = None,
        on_update: Callable[[str], None] | None = None,
    ) -> Transcript:
        """Consume mono 16 kHz little-endian PCM and report transcript previews."""
        raise NotImplementedError


class LlmBackend(ABC):
    @abstractmethod
    async def polish(self, transcript: Transcript, profile: Profile) -> str:
        raise NotImplementedError

    async def start(self) -> None:
        """Optional: start the backend (e.g., launch local server)."""
        pass  # noqa: B027

    async def stop(self) -> None:
        """Optional: stop the backend (e.g., terminate local server)."""
        pass  # noqa: B027


class OutputSink(ABC):
    @abstractmethod
    async def deliver(self, result: DictationResult, target: ResultTarget) -> None:
        raise NotImplementedError
