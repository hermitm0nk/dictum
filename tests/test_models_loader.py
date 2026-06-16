from pathlib import Path

from dictum.models_loader import asr_model_path, llm_model_path, model_dir


def test_model_dir_uses_xdg_cache_home(monkeypatch) -> None:
    monkeypatch.delenv("DICTUM_MODEL_DIR", raising=False)
    monkeypatch.setenv("XDG_CACHE_HOME", "/tmp/dictum-cache")

    assert model_dir() == Path("/tmp/dictum-cache/dictum/models")


def test_model_dir_can_be_overridden(monkeypatch) -> None:
    monkeypatch.setenv("DICTUM_MODEL_DIR", "/tmp/custom-dictum-models")
    monkeypatch.setenv("XDG_CACHE_HOME", "/tmp/dictum-cache")

    assert model_dir() == Path("/tmp/custom-dictum-models")


def test_default_model_paths_share_model_dir(monkeypatch) -> None:
    monkeypatch.delenv("DICTUM_LLM_MODEL", raising=False)
    monkeypatch.delenv("DICTUM_PARAKEET_MODEL", raising=False)
    monkeypatch.setenv("DICTUM_MODEL_DIR", "/tmp/dictum-models")

    assert asr_model_path() == Path("/tmp/dictum-models/parakeet-tdt-0.6b-v3-q4_k.gguf")
    assert llm_model_path() == Path("/tmp/dictum-models/Qwen3.5-4B-Q3_K_M.gguf")


def test_model_paths_can_be_overridden_individually(monkeypatch) -> None:
    monkeypatch.setenv("DICTUM_PARAKEET_MODEL", "/tmp/asr.gguf")
    monkeypatch.setenv("DICTUM_LLM_MODEL", "/tmp/llm.gguf")

    assert asr_model_path() == Path("/tmp/asr.gguf")
    assert llm_model_path() == Path("/tmp/llm.gguf")
