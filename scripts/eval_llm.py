"""Reusable synthetic dictation-polishing eval against a running llama-server.

Run with: .venv/bin/python scripts/eval_llm.py --params '{"temperature":0.7,"top_p":0.8}'
The score is strict (allowlisted outputs only); inspect failures for harmless variants.
No real transcripts are collected. The prompt defaults to the active Dictum profile.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import httpx

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from dictum.config import load_profile  # noqa: E402

DATA = ROOT / "evals/transcripts.json"
PARAMS = ("temperature", "top_p", "top_k", "min_p", "presence_penalty", "repeat_penalty")


def normalized(text: str) -> str:
    """Ignore only whitespace and typographical quote differences, not changed meaning."""
    return re.sub(r"\s+", " ", text.strip()).replace("’", "'").replace("“", '"').replace("”", '"')


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8080")
    parser.add_argument("--model", help="Override model name in the request")
    parser.add_argument("--allow-model-mismatch", action="store_true", help="Allow other model")
    parser.add_argument("--prompt-file", type=Path, help="Override active Dictum system prompt")
    parser.add_argument("--few-shot-file", type=Path, help="Override active profile's examples")
    parser.add_argument("--no-few-shot", action="store_true", help="Disable example messages")
    parser.add_argument("--data", type=Path, default=DATA)
    parser.add_argument("--split", choices=("all", "dev", "holdout"), default="all")
    parser.add_argument("--params", default="{}", help="JSON sampling overrides")
    parser.add_argument("--repeat", type=int, default=1, help="Attempts per case")
    parser.add_argument("--out", type=Path, help="Write machine-readable JSON results")
    args = parser.parse_args()
    if args.repeat < 1:
        parser.error("--repeat must be positive")

    profile = load_profile()
    if profile.llm is None:
        parser.error("Active profile has no LLM")
    prompt = (
        args.prompt_file.read_text(encoding="utf-8").strip() if args.prompt_file else profile.prompt
    )
    sampling = {key: getattr(profile.llm, key) for key in PARAMS}
    try:
        overrides = json.loads(args.params)
        if not isinstance(overrides, dict) or overrides.keys() - set(PARAMS):
            raise ValueError(f"Only {', '.join(PARAMS)} are allowed in --params")
        sampling.update(overrides)
    except (ValueError, TypeError) as exc:
        parser.error(f"Invalid --params: {exc}")
    sampling = {key: val for key, val in sampling.items() if val is not None}
    cases = json.loads(args.data.read_text(encoding="utf-8"))
    cases = [c for c in cases if args.split == "all" or c["split"] == args.split]
    examples_file = None if args.no_few_shot else args.few_shot_file or profile.llm.few_shot_file
    examples = json.loads(examples_file.read_text(encoding="utf-8")) if examples_file else []
    prefix = "Transcript: " if examples else ""
    if not cases:
        parser.error("No cases selected")

    results: list[dict[str, Any]] = []
    with httpx.Client(timeout=max(30, profile.llm.timeout_seconds)) as client:
        props = client.get(f"{args.base_url.rstrip('/')}/props")
        props.raise_for_status()
        loaded_model_path = props.json().get("model_path")
        if loaded_model_path != str(profile.llm.model_path) and not args.allow_model_mismatch:
            parser.error(
                f"Server loaded {loaded_model_path!r}, but profile selects "
                f"{str(profile.llm.model_path)!r}. Use --allow-model-mismatch if intentional."
            )
        print(f"Loaded model: {loaded_model_path}; sampling: {sampling}; cases: {len(cases)}")
        for case in cases:
            for attempt in range(args.repeat):
                payload = {
                    "model": args.model or profile.llm.model,
                    "messages": [
                        {"role": "system", "content": prompt},
                        *[
                            message
                            for example in examples
                            for message in (
                                {"role": "user", "content": prefix + example["input"]},
                                {"role": "assistant", "content": example["output"]},
                            )
                        ],
                        {"role": "user", "content": prefix + case["input"]},
                    ],
                    "max_tokens": profile.llm.max_tokens,
                    "chat_template_kwargs": {"enable_thinking": profile.llm.enable_thinking},
                    **sampling,
                }
                try:
                    response = client.post(
                        f"{args.base_url.rstrip('/')}/v1/chat/completions", json=payload
                    )
                    response.raise_for_status()
                    output = response.json()["choices"][0]["message"]["content"].strip()
                    error = None
                except (httpx.HTTPError, KeyError, IndexError, ValueError) as exc:
                    output = ""
                    error = str(exc)
                passed = error is None and normalized(output) in map(normalized, case["expected"])
                results.append(
                    {
                        "id": case["id"],
                        "category": case["category"],
                        "split": case["split"],
                        "attempt": attempt + 1,
                        "passed": passed,
                        "input": case["input"],
                        "expected": case["expected"],
                        "output": output,
                        "error": error,
                    }
                )
                print(f"{'PASS' if passed else 'FAIL'} {case['id']} #{attempt + 1}")
                if not passed:
                    print(f"  input:    {case['input']}")
                    print(f"  actual:   {output or error}")
                    print(f"  expected: {case['expected']}")

    counts = Counter((r["category"], r["passed"]) for r in results)
    passed = sum(r["passed"] for r in results)
    print(f"\nExact/allowlisted: {passed}/{len(results)} ({passed / len(results):.1%})")
    for category in sorted({r["category"] for r in results}):
        good, bad = counts[category, True], counts[category, False]
        print(f"  {category}: {good}/{good + bad}")
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(
            json.dumps(
                {
                    "model": args.model or profile.llm.model,
                    "model_path": loaded_model_path,
                    "sampling": sampling,
                    "prompt_file": str(args.prompt_file) if args.prompt_file else "active profile",
                    "few_shot_file": str(examples_file) if examples_file else None,
                    "passed": passed,
                    "total": len(results),
                    "results": results,
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
    if any(r["error"] for r in results):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
