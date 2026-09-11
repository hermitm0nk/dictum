"""Auth handling for the OpenAI-compatible LLM backend."""

from dictum.llm import OpenAILLM


def test_api_key_from_env(monkeypatch) -> None:
    monkeypatch.setenv("DICTUM_TEST_KEY", "sk-env")
    llm = OpenAILLM(api_key_env="DICTUM_TEST_KEY")
    assert llm._headers()["Authorization"] == "Bearer sk-env"


def test_explicit_key_wins_over_env(monkeypatch) -> None:
    monkeypatch.setenv("DICTUM_TEST_KEY", "sk-env")
    llm = OpenAILLM(api_key="sk-direct", api_key_env="DICTUM_TEST_KEY")
    assert llm._headers()["Authorization"] == "Bearer sk-direct"


def test_no_key_no_auth_header() -> None:
    llm = OpenAILLM()
    assert "Authorization" not in llm._headers()


def test_extra_headers_merged(monkeypatch) -> None:
    monkeypatch.setenv("DICTUM_TEST_KEY", "sk-env")
    llm = OpenAILLM(
        api_key_env="DICTUM_TEST_KEY",
        headers={"HTTP-Referer": "https://example.com", "X-Title": "dictum"},
    )
    headers = llm._headers()
    assert headers["HTTP-Referer"] == "https://example.com"
    assert headers["X-Title"] == "dictum"
    assert headers["Authorization"] == "Bearer sk-env"
