"""Managed-local llama.cpp sampling controls and example messages."""

import asyncio
import json

from dictum import llm_local
from dictum.models import LlmConfig, Profile, Transcript


def test_managed_local_sampling_payload(monkeypatch) -> None:
    requests = []

    class FakeResponse:
        def raise_for_status(self) -> None:
            pass

        def json(self) -> dict:
            return {"choices": [{"message": {"content": "Hello."}}]}

    class FakeClient:
        def __init__(self, **kwargs) -> None:
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args) -> None:
            pass

        async def post(self, url, json):
            requests.append(json)
            return FakeResponse()

    monkeypatch.setattr(llm_local.httpx, "AsyncClient", FakeClient)
    monkeypatch.setattr(llm_local, "vulkan_available", lambda: True)
    profile = Profile(
        llm=LlmConfig(
            top_k=20,
            min_p=0.0,
            presence_penalty=2.0,
            repeat_penalty=1.0,
        )
    )
    backend = llm_local.create_llm_backend(profile)
    result = asyncio.run(backend.polish(Transcript(text="hello"), profile))
    assert result == "Hello."
    sampling = {k: requests[0][k] for k in ("top_k", "min_p", "presence_penalty", "repeat_penalty")}
    assert sampling == {
        "top_k": 20,
        "min_p": 0.0,
        "presence_penalty": 2.0,
        "repeat_penalty": 1.0,
    }


def test_few_shot_messages(monkeypatch, tmp_path) -> None:
    sent = []

    class FakeClient:
        def __init__(self, **kwargs) -> None:
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args) -> None:
            pass

        async def post(self, url, json):
            sent.append(json)
            return self

        def raise_for_status(self) -> None:
            pass

        def json(self) -> dict:
            return {"choices": [{"message": {"content": "Hello."}}]}

    monkeypatch.setattr(llm_local.httpx, "AsyncClient", FakeClient)
    monkeypatch.setattr(llm_local, "vulkan_available", lambda: True)
    examples = tmp_path / "examples.json"
    examples.write_text(json.dumps([{"input": "uh hello", "output": "Hello."}]))
    profile = Profile(llm=LlmConfig(few_shot_file=examples))
    backend = llm_local.create_llm_backend(profile)
    asyncio.run(backend.polish(Transcript(text="uh goodbye"), profile))
    assert sent[0]["messages"] == [
        {"role": "system", "content": profile.prompt},
        {"role": "user", "content": "Transcript: uh hello"},
        {"role": "assistant", "content": "Hello."},
        {"role": "user", "content": "Transcript: uh goodbye"},
    ]


def test_sampling_params_omitted_by_default(monkeypatch) -> None:
    sent = []

    class FakeClient:
        def __init__(self, **kwargs) -> None:
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args) -> None:
            pass

        async def post(self, url, json):
            sent.append(json)
            return self

        def raise_for_status(self) -> None:
            pass

        def json(self) -> dict:
            return {"choices": [{"message": {"content": "Hello."}}]}

    monkeypatch.setattr(llm_local.httpx, "AsyncClient", FakeClient)
    backend = llm_local.ManagedLocalLlm()
    asyncio.run(backend.polish(Transcript(text="hello"), Profile()))
    for key in ("top_k", "min_p", "presence_penalty", "repeat_penalty"):
        assert key not in sent[0]
