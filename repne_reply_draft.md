# Reply to Repne — Final Draft (post pass@5)

## TL;DR
You were right: 92-95% across configs is within margin. The deeper finding: **the previous "drop" was our harness bug, not MTP**. Real numbers below + pass@5 = **96.95%**.

## What I found

Our HumanEval harness was systematically scoring valid responses as failures. When the model returned just the function body (which our prompt asked for!), `.strip()` removed the leading indent on line 1, breaking the resulting script with `IndentationError`. This affected **every** HumanEval run we've ever published.

**Corrected scores (offline smart-glue rescore of all our historical jsonl files):**

| Config | Image | mt | Reported | **Corrected** |
|---|---|---:|---:|---:|
| FP8+MTP=3 | `:latest` | 4096 | 79.3% | **92.1%** |
| FP8+MTP=5 | `:latest` | 4096 | 75.6% | **95.7%** |
| FP8+DFlash N=7 | `:latest` | 4096 | 73.8% | 93.3% |
| FP8+DFlash N=8 | `:latest` | 4096 | 74.4% | **93.9%** |
| FP8+DFlash N=15 | `:latest` | 4096 | 73.8% | 92.1% |
| FP8+MTP=3 | `:v2` | 8192 | 70.7% | **93.3%** |
| BF16+DFlash N=8 | `:v2` | 8192 | 74.4% | **95.1%** |

**All configs are 92-96%.** No MTP bug. No `:v2` regression. The variance we'd been calling "quality differences" was harness pathology.

## pass@5 (your ask)

Ran FP8+MTP=3 on `:latest` mt=8192 at temp=0.8, n=5 samples × 164 problems = 820 generations:

- **pass@5 (any-pass): 96.95%** — only 5 problems failed all 5 samples
- pass@5 (all-pass): 58.5%
- Per-sample at temp=0.8: 84.8%

The model's capability ceiling is ~97%. The other 3.05% are problems where temp=0.8 sampling never finds the answer. Single-shot variance dominates pass@1.

## Your three hypotheses scored

1. **`max_tokens`** — Partial. All `empty_response`s have `finish_reason=length`, so they ARE truncations. mt=8192 helped (vs 4096) but didn't fix the root cause. Truly leaving `max_tokens` unset would be safer.
2. **Timeouts** — Not a factor. Default 300s, p95 was 47s.
3. **Tool-call failures invalidating tests** — Zero in HumanEval responses. Gate 2 passes 4/4 every run.

## c=1 vs c=8

Haven't isolated yet — but my prediction: c=1 will look near-identical to c=8 single-shot, because the harness bug is independent of batch size. Want me to run it anyway?

## Re-bench at c=1?

I can also re-run any of these configs at c=1 with the patched harness if you want a clean validation. Each HE run ≈ 25-30 min at c=1 vs 7 min at c=8. Just say the word.

Full writeup: https://github.com/jcartu/qwen-bench-2026-05-11-v2-followup/blob/main/ADDENDUM.md (publishing now).
