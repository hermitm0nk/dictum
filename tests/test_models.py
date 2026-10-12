from __future__ import annotations

import pytest
from pydantic import ValidationError

from dictum.models import AsrConfig, DictationResult, ResultTarget, Transcript


def test_dictation_result_prefers_polished_text() -> None:
    result = DictationResult(
        transcript=Transcript(text="raw text"),
        polished_text="polished text",
        target=ResultTarget.STDOUT,
    )

    assert result.final_text == "polished text"


def test_dictation_result_falls_back_to_transcript() -> None:
    result = DictationResult(transcript=Transcript(text="raw text"), target=ResultTarget.STDOUT)

    assert result.final_text == "raw text"


def test_asr_gemini_defaults_are_valid() -> None:
    cfg = AsrConfig()
    assert cfg.gemini_bitrate_kbps == 32
    assert cfg.gemini_timeout_seconds == 120.0


def test_asr_gemini_bitrate_must_be_positive() -> None:
    with pytest.raises(ValidationError):
        AsrConfig(gemini_bitrate_kbps=0)


def test_asr_gemini_bitrate_is_bounded() -> None:
    # Opus audio bitrate tops out at 510 kbps.
    with pytest.raises(ValidationError):
        AsrConfig(gemini_bitrate_kbps=600)


def test_asr_gemini_timeout_must_be_positive() -> None:
    with pytest.raises(ValidationError):
        AsrConfig(gemini_timeout_seconds=0)
