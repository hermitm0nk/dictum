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
    model: str = "parakeet-tdt-0.6b-v3-q4_k"
    # CrispASR runs on CPU with GGUF quantization; device/precision are not used
    # Kept for future GPU backends (ctranslate2, etc.)
    device: str = "cpu"
    precision: str = "q4_k"


class LlmConfig(BaseModel):
    backend: str = "managed-local"  # managed-local, openai-compatible, none
    
    # For managed-local backend
    model_path: Path | None = None
    binary_path: Path | None = None
    port: int = 8080
    ctx_size: int = 4096
    n_gpu_layers: int = -1  # -1 = all layers on GPU
    
    # For openai-compatible backend (remote or externally managed)
    base_url: HttpUrl | None = None
    model: str = "qwen3.5-4b-3bit"
    
    # Common
    temperature: float = 0.2
    timeout_seconds: float = 20.0


class Profile(BaseModel):
    name: str = "default"
    prompt: str = Field(
        default=(
            "Fix punctuation, capitalization, and remove filler words "
            "(um, uh, like, you know, I mean) from voice transcription. "
            "Break run-on sentences. Output only the corrected text, "
            "no explanations or notes."
        )
    )
    prompt_file: Path | None = None
    result: ResultTarget = ResultTarget.PASTE
    asr: AsrConfig = Field(default_factory=AsrConfig)
    llm: LlmConfig | None = Field(default_factory=LlmConfig)
