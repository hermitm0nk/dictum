from abc import ABC, abstractmethod
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
