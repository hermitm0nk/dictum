from enum import StrEnum
from pathlib import Path

from pydantic import BaseModel, Field, HttpUrl

from dictum.state import DictumState


class ResultTarget(StrEnum):
    PASTE = "paste"
    CLIPBOARD = "clipboard"
    STDOUT = "stdout"
    FILE = "file"
    NONE = "none"


class Transcript(BaseModel):
    text: str
    language: str | None = None
    duration_seconds: float | None = None


class DictationResult(BaseModel):
    transcript: Transcript
    polished_text: str | None = None
    target: ResultTarget
    output_path: Path | None = None

    @property
    def final_text(self) -> str:
        return self.polished_text or self.transcript.text


class DaemonStatus(BaseModel):
    state: DictumState
    active_profile: str | None = None
    last_error: str | None = None


class AsrConfig(BaseModel):
    backend: str = "parakeet"
    model: str = "parakeet-tdt-v3-0.6b"
    device: str = "cuda"
    precision: str = "float16"


class LlmConfig(BaseModel):
    backend: str = "openai-compatible"
    base_url: HttpUrl | None = None
    model: str = "qwen3.5-4b-3bit"
    temperature: float = 0.2
    timeout_seconds: float = 20.0


class Profile(BaseModel):
    name: str = "default"
    prompt: str = Field(default="Polish the transcript without changing its meaning.")
    prompt_file: Path | None = None
    result: ResultTarget = ResultTarget.PASTE
    asr: AsrConfig = Field(default_factory=AsrConfig)
    llm: LlmConfig | None = Field(default_factory=LlmConfig)
