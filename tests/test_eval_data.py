"""Keep the synthetic eval reproducible and separate from training examples."""

import json
import runpy
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_eval_cases_are_unique_and_split() -> None:
    cases = json.loads((ROOT / "evals/transcripts.json").read_text(encoding="utf-8"))
    assert len(cases) >= 40
    assert len({c["id"] for c in cases}) == len(cases)
    assert Counter(c["split"] for c in cases) == {"dev": 32, "holdout": 13}
    for case in cases:
        assert case["id"] and case["input"] and case["category"]
        assert case["expected"] and all(isinstance(s, str) and s for s in case["expected"])

    examples = json.loads((ROOT / "evals/few_shot.json").read_text(encoding="utf-8"))
    assert all(e["input"] not in {c["input"] for c in cases} for e in examples)


def test_eval_normalization_is_strict() -> None:
    normalized = runpy.run_path(str(ROOT / "scripts/eval_llm.py"))["normalized"]
    assert normalized("  Let’s   go.  ") == normalized("Let's go.")
    assert normalized("Let's go.") != normalized("Let's go?")
    assert normalized("Tuesday") != normalized("Wednesday")
