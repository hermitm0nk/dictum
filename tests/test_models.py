from dictum.models import DictationResult, ResultTarget, Transcript


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
