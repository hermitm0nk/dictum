"""Download models on first use with progress bar."""

from __future__ import annotations

import os
from pathlib import Path

from huggingface_hub import hf_hub_download

ASR_MODEL_FILENAME = "parakeet-tdt-0.6b-v3-q4_k.gguf"
LLM_MODEL_FILENAME = "qwen2.5-4b-instruct-q3_k_m.gguf"


def xdg_cache_home() -> Path:
    """Return the base XDG cache directory."""
    cache_home = os.environ.get("XDG_CACHE_HOME")
    if cache_home:
        return Path(cache_home).expanduser()
    return Path.home() / ".cache"


def model_dir() -> Path:
    """Return Dictum's model cache directory."""
    override = os.environ.get("DICTUM_MODEL_DIR")
    if override:
        return Path(override).expanduser()
    return xdg_cache_home() / "dictum" / "models"


def asr_model_path() -> Path:
    """Return the default local Parakeet model path."""
    override = os.environ.get("DICTUM_PARAKEET_MODEL")
    if override:
        return Path(override).expanduser()
    return model_dir() / ASR_MODEL_FILENAME


def llm_model_path() -> Path:
    """Return the default local Qwen model path."""
    override = os.environ.get("DICTUM_LLM_MODEL")
    if override:
        return Path(override).expanduser()
    return model_dir() / LLM_MODEL_FILENAME


ASR_MODEL = {
    "repo_id": "cstr/parakeet-tdt-0.6b-v3-GGUF",
    "filename": ASR_MODEL_FILENAME,
}

LLM_MODEL = {
    "repo_id": "Qwen/Qwen2.5-4B-Instruct-GGUF",
    "filename": LLM_MODEL_FILENAME,
}


def _download_with_progress(repo_id: str, filename: str, local_dir: Path) -> Path:
    """Download a file from HF Hub with tqdm progress bar."""
    local_dir_str = str(local_dir)
    local_dir.mkdir(parents=True, exist_ok=True)
    target_path = local_dir / filename
    if target_path.exists():
        return target_path

    print(f"Downloading {filename} from {repo_id} to {local_dir}...")

    # Use huggingface_hub's built-in progress (it uses tqdm internally)
    path = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        local_dir=local_dir_str,
    )
    return Path(path)


def ensure_asr_model() -> Path:
    """Ensure Parakeet ASR model is downloaded, return local path."""
    return _download_with_progress(
        ASR_MODEL["repo_id"],
        ASR_MODEL["filename"],
        model_dir(),
    )


def ensure_llm_model() -> Path:
    """Ensure Qwen LLM model is downloaded, return local path."""
    return _download_with_progress(
        LLM_MODEL["repo_id"],
        LLM_MODEL["filename"],
        model_dir(),
    )
