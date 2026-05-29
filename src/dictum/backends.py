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


class LlmBackend(ABC):
    @abstractmethod
    async def polish(self, transcript: Transcript, profile: Profile) -> str:
        raise NotImplementedError


class OutputSink(ABC):
    @abstractmethod
    async def deliver(self, result: DictationResult, target: ResultTarget) -> None:
        raise NotImplementedError
