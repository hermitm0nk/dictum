# Synthetic dictation-polishing eval

45 invented transcripts (32 development, 13 holdout) in `transcripts.json` test questions that must not be answered, instructions that must not be followed, self-corrections, fillers, technical terms, Russian and mixed-language input. `few_shot.json` contains separate demonstrations; none are eval inputs. No private recordings/transcripts are used.

From the repo root with Dictum running locally:

```sh
.venv/bin/python scripts/eval_llm.py --out /tmp/dictum-eval.json
.venv/bin/python scripts/eval_llm.py --split holdout --repeat 3 --out /tmp/dictum-holdout.json
# A/B a candidate without changing the live service:
.venv/bin/python scripts/eval_llm.py --split dev \
  --prompt-file evals/prompt-simple.md --few-shot-file evals/few_shot.json \
  --params '{"temperature":0.2,"top_p":0.8,"top_k":null,"min_p":null,"presence_penalty":null,"repeat_penalty":null}'
```

By default, the evaluator uses the active Dictum profile's prompt, model name, sampling settings, and example file, against the llama-server at `127.0.0.1:8080`. `--no-few-shot` disables active examples for an A/B run. `--params` can override/clear sampling fields. The eval does **not** switch the loaded model: change the service model or provide a different `--base-url` first. Outputs are printed and optionally saved as JSON. Each score counts an exact (whitespace/typographical-quotes-normalized) match against an allowlist. A mismatch can be minor punctuation, or a serious meaning-changing error; read the actual outputs. Repeated runs reveal stochastic variance.

## Results (Qwen3.5-0.8B, llama.cpp b9699)

| GGUF | Prompt / examples | Sampling | Dev | Holdout |
| --- | --- | --- | ---: | ---: |
| Q4_K_M | Original long prompt | Qwen non-thinking text recommendation: temp 1, top-p 1, top-k 20, min-p 0, presence 2, repeat 1 | 7/32 | — |
| Q4_K_M | Original long prompt | Previous Dictum temp 0.7, top-p 0.8, remaining fields unset | 7/32 | — |
| Q8_0 | Original long prompt | Qwen recommendation above | 7/32 | 4/13 |
| Q8_0 | Short prompt, 8 chat examples | temp 0.2, top-p 0.8, other controls unset | 12/32 | 7/13; **23/39** across 3 repeats |
| Q8_0 | Same as preceding, deployed profile | Same | 12/32 | 8/13 (20/45 total); another run 6/13 |
| Q8_0 | Same prompt/examples | temp 0, top-p 0.8, other controls unset | 13/32 | 8/13; 24/39 across 3 repeats |

### Matched Q4 vs Q8 comparison (same deployed setup)

With the deployed short prompt, 8 chat examples, temp 0.2 and top-p 0.8, both models were evaluated on the **same 45 cases, three times each**. Q4 ran on port 8082 and Q8 on the live server (port 8080); both used llama.cpp b9699. Strict allowlisted matches: **Q4: 58/135 (43.0%)**, **Q8: 63/135 (46.7%)**. On the holdout split both were **22/39 (56.4%)**; on development Q4 was 36/96 and Q8 41/96. Q8 passed 9 attempts Q4 missed; Q4 passed 4 Q8 missed. Because attempts repeat the same cases and generation is stochastic, the 3.7-point overall difference is *not* a reliable estimate of general accuracy. GGUF disk sizes: Q4 508 MiB, Q8 775 MiB. Neither eliminates serious meaning-changing failures. Full per-case comparison is in `/tmp/dictum-comparison-{q4,q8}.json` on the test machine.

### Qwen3.5-2B Q3/Q4 comparison

Against the same deployed prompt, 8 chat examples, temp 0.2 and top-p 0.8, with all 45 cases repeated three times, temporary llama.cpp b9699 servers (port 8082) yielded:

| Quantization | Dev | Holdout | Total |
| --- | ---: | ---: | ---: |
| 0.8B Q8 (matched reference above) | 41/96 | 22/39 | 63/135 (46.7%) |
| 2B Q3_K_M | 65/96 | 24/39 | 89/135 (65.9%) |
| 2B Q4_K_M | 60/96 | 23/39 | 83/135 (61.5%) |

Q3 exceeded Q4 in this stochastic run (12 Q3-only passes vs 6 Q4-only), but this does not establish that Q3 is inherently better; holdout scores are nearly identical. Both 2B quants handled the `um ... uh` filler example in 3/3 attempts, vs 0/3 on 0.8B Q8. However, both still failed on self-corrections, preserving mixed-language inputs, and some embedded instructions. GGUF sizes: 2B Q3 ~1.1 GiB and Q4 ~1.2 GiB vs 0.8B Q8 ~775 MiB. The live 0.8B service was **not** changed. Full per-case outputs: `/tmp/dictum-2b-{Q3_K_M,Q4_K_M}-comparison.json` on the test machine.

### Filler-prompt experiment (Q8, deployed examples/sampling)

Tested two prompt-only changes on all 45 cases, 3 attempts each. Original prompt: **63/135** overall; `prompt-filler-inline.md` (names `uh`/`um` within the short instructions): **59/135**; `prompt-filler-explicit.md` (adds a separate instruction listing fillers): **54/135**. The inline prompt did fix the test `um i uh think we should ship it tomorrow` in 3/3 attempts (vs 0/3 originally), but also regressed `send it to John no actually send it to James` from 3/3 correct to 0/3, with unrelated Russian replies. Neither candidate was deployed. These tests illustrate that stronger prompt wording is not a reliable substitute for deterministic filler removal on this model.

The Qwen [model card](https://huggingface.co/Qwen/Qwen3.5-0.8B) recommends temp 1, top-p 1, top-k 20, min-p 0, presence penalty 2, repeat penalty 1 for non-thinking text generation. That did **not** work well for this constrained editing task. Increasing quantization quality alone made little difference. A short prompt with separate chat examples and conservative sampling improved this test, but still failed to preserve meaning on some corrections and mixed-language inputs. Greedy temp 0 was slightly more consistent but also produced serious errors (e.g., translated an English command, answered a question), so it was **not** selected. Do not treat these scores as a production accuracy guarantee. In particular, this 0.8B model sometimes answers a question, translates text, or retains the wrong corrected detail. The previous 4B model may still be preferable if accuracy is critical.
