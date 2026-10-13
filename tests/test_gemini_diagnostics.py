"""Private response-shape diagnostics for Gemini transcription."""

import asyncio

import httpx
import pytest

from dictum.asr_gemini import GeminiASR, GeminiError


@pytest.mark.parametrize(
    ("body", "diagnostic"),
    [
        ({}, "empty JSON object"),
        ({"status": "completed", "outputs": [{"text": "PRIVATE"}]}, "missing 'steps'"),
        ({"status": "completed", "steps": None}, "invalid 'steps' type: NoneType"),
        ({"steps": [None]}, "invalid step object"),
        ({"steps": [{"type": "model_output", "content": {}}]}, "invalid model_output content"),
        (
            {"steps": [{"type": "model_output", "content": [{"type": "text", "text": 1}]}]},
            "invalid text value",
        ),
    ],
)
def test_response_shape_diagnostics_are_private(body, diagnostic, caplog):
    asr = GeminiASR(api_key="TEST-SECRET")

    async def run():
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(lambda request: httpx.Response(200, json=body))
        ) as client:
            asr._http_client = client
            return await asr._request_transcript(None, asr._headers("TEST-SECRET"), data=b"audio")

    with pytest.raises(GeminiError) as excinfo:
        asyncio.run(run())
    message = str(excinfo.value)
    assert diagnostic in message
    if "outputs" in body:
        assert "outputs=list" in message
        assert "status=completed" in message
    assert "PRIVATE" not in message + caplog.text
    assert "TEST-SECRET" not in message + caplog.text


@pytest.mark.parametrize("steps", [[], [{"type": "model_output", "content": []}]])
def test_valid_empty_transcript_has_distinct_diagnostic(steps, caplog):
    asr = GeminiASR(api_key="TEST-SECRET")

    async def run():
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(
                lambda request: httpx.Response(200, json={"status": "completed", "steps": steps})
            )
        ) as client:
            asr._http_client = client
            return await asr._request_transcript(None, asr._headers("TEST-SECRET"), data=b"audio")

    assert asyncio.run(run()) == ""
    assert "empty transcript" in caplog.text
    assert f"steps=list({len(steps)})" in caplog.text


def test_empty_http_body_has_distinct_diagnostic():
    asr = GeminiASR(api_key="TEST-SECRET")

    async def run():
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(lambda request: httpx.Response(200, content=b""))
        ) as client:
            asr._http_client = client
            return await asr._request_transcript(None, asr._headers("TEST-SECRET"), data=b"audio")

    with pytest.raises(GeminiError, match="empty HTTP body"):
        asyncio.run(run())
