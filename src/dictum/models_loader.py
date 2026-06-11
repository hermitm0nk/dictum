"""Download models on first use with progress bar."""
from pathlib import Path
from huggingface_hub import hf_hub_download
from tqdm import tqdm

ASR_MODEL = {
    "repo_id": "cstr/parakeet-tdt-0.6b-v3-GGUF",
    "filename": "parakeet-tdt-0.6b-v3-q4_k.gguf",
    "local_dir": Path.home() / ".cache" / "dictum" / "models",
}

LLM_MODEL = {
    "repo_id": "Qwen/Qwen2.5-4B-Instruct-GGUF",
    "filename": "qwen2.5-4b-instruct-q3_k_m.gguf",
    "local_dir": Path.home() / ".cache" / "dictum" / "models",
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
        local_dir_use_symlinks=False,
        resume_download=True,
    )
    return Path(path)


def ensure_asr_model() -> Path:
    """Ensure Parakeet ASR model is downloaded, return local path."""
    return _download_with_progress(
        ASR_MODEL["repo_id"],
        ASR_MODEL["filename"],
        ASR_MODEL["local_dir"],
    )


def ensure_llm_model() -> Path:
    """Ensure Qwen LLM model is downloaded, return local path."""
    return _download_with_progress(
        LLM_MODEL["repo_id"],
        LLM_MODEL["filename"],
        LLM_MODEL["local_dir"],
    )