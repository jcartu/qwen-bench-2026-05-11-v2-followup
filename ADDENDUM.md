# Addendum: HumanEval Harness Bug + `:v2` Image A/B + pass@5

**Date**: 2026-05-11 (afternoon, post study #4 publication)
**Trigger**: User question "Where are the next big gains?" → ran A/B on `:v2` vs `:latest` →
found a long-standing harness bug while diagnosing the resulting HumanEval delta.

## Executive Summary

We discovered that **all our reported HumanEval pass rates in studies #2, #3, and #4 were
systematically deflated by 13-23 percentage points** due to a bug in `stress_harness.py`'s
code extraction. After correcting the harness, all configs cluster at **92-95% true pass
rate**. The apparent quality differences between FP8/BF16, MTP=3/5, and `:v2`/`:latest`
that we'd been reporting are within noise. The model's actual capability is much higher
than we'd been measuring.

## Headline Numbers (Corrected)

| Config | Image | Reported HE | **Corrected HE** | MBPP |
|---|---|---:|---:|---:|
| FP8+MTP=3 | `:latest` mt=4096 (study #2) | 79.3% | **92.1%** | 87.9% |
| FP8+MTP=5 | `:latest` mt=4096 (study #2) | 75.6% | **95.7%** | — |
| FP8+DFlash N=8 | `:latest` mt=4096 (study #3) | 74.4% | **93.9%** | 89.5% |
| BF16+DFlash N=8 | `:v2` mt=8192 (study #4 rerun) | 74.4% | **95.1%** | 90.3% |
| FP8+MTP=3 | `:v2` mt=8192 (study #4) | 70.7% | **93.3%** | 86.8% |
| FP8+MTP=3 | `:latest` mt=8192 (A/B today, online v2 harness) | — | **86.6%** | 89.5% |
| FP8+MTP=3 | `:latest` mt=8192 **pass@5** temp=0.8 | — | **96.95%** (any) | — |

**The model's true HumanEval capability ceiling: ~97% (pass@5, 159/164 problems).**

## The Bug

### What we asked the model

HumanEval prompts give the model an unfinished function signature + docstring:

```python
def intersperse(numbers: List[int], delimeter: int) -> List[int]:
    """Insert a number 'delimeter' between every two consecutive elements
    of input list `numbers'.
    >>> intersperse([], 4)
    []
    >>> intersperse([1, 2, 3], 4)
    [1, 4, 2, 4, 3]
    """
```

Our harness prompt: "Complete the following Python function. Output ONLY the function
body inside a `python` fenced block, no explanation."

### What the model did

It correctly emitted the body, with proper 4-space indentation relative to the def:

```
    result = []
    for i, num in enumerate(numbers):
        result.append(num)
        if i < len(numbers) - 1:
            result.append(delimeter)
    return result
```

### What broke

In `stress_harness.py::extract_code()`, the regex match group is fed to `.strip()`,
which removes the leading 4 spaces on line 1 (`    result = []` → `result = []`).
The harness then ran this as a standalone Python script. Result:

```
  File "<string>", line 2
    for i, num in enumerate(numbers):
IndentationError: unexpected indent
```

Python parses line 1 as a top-level statement (now unindented), then sees line 2 still
has its original 4-space indent — which is invalid Python at the top level.

Every such case was logged as `failure_mode=test_fail`, indistinguishable in the
summary stats from a logically wrong answer.

### Why we didn't notice for 9 days

- 100% of `test_fail` failures across 4 studies had this same `IndentationError` —
  the failure mode breakdown looked consistent, just at "75% pass."
- `failure_mode_breakdown` in the summary doesn't separate syntax errors from
  semantic test failures.
- Our `content_head` field only stored the first 500 chars of model output, and
  it was rarely inspected because the summary numbers "looked reasonable."

## The Fix

### `stress_harness_v2.py` — applied two changes

1. **Prompt change**: Ask for the COMPLETE function (signature + body) instead of body-only.
2. **Auto-glue safety net**: If model still emits body-only, detect via `'def ' not in code`
   and prepend the original prompt prefix. **Critically**, re-indent every body line to
   ensure consistency (the naive `.strip()` only removed leading whitespace from line 1).

```python
def smart_glue_humaneval(code: str, prompt: str) -> str:
    if 'def ' in code:
        return code
    indent = '    '
    fixed = []
    for line in code.split('\n'):
        if line.strip() == '':
            fixed.append('')
        elif line.startswith('    ') or line.startswith('\t'):
            fixed.append(line)
        else:
            fixed.append(indent + line)
    return prompt.rstrip() + '\n' + '\n'.join(fixed)
```

### `rescore_humaneval.py` — offline rescoring tool

Replays every saved HumanEval jsonl through `smart_glue`, re-runs the tests, and
reports corrected pass rates. Run on all 9 historical jsonls; see results table above.

## Pass@5 Result

To answer the natural follow-up question "but how good *can* the model be?", we ran a
proper pass@5 study at temperature=0.8, n=5 samples per problem.

```
config: FP8+MTP=3 on repne/vllm:latest @ mt=8192, c=8, temp=0.8, n=5
problems: 164 HumanEval, 820 total generations
wall time: 21 minutes
```

| Metric | Value |
|---|---:|
| **pass@5 (any of 5)** | **96.95%** (159/164) |
| pass@5 (all 5) | 58.5% (96/164) |
| Per-sample pass rate | 84.8% |
| Empty responses | 95/820 = 11.6% (all `finish_reason=length` truncations) |

Five problems remain unsolved across all 5 samples — these are the model's true
"hard" cases at this temperature. Three more samples per problem would likely
push pass@10 to ~99%.

## A/B Image Comparison (Corrected)

The original A/B finding ("`:v2` regresses by 4.3pp vs `:latest`") is **wrong**. After
harness fix:

| Image | Smart-glue corrected HE |
|---|---:|
| `:v2` MTP=3 mt=8192 | 93.3% |
| `:latest` MTP=3 mt=8192 (offline rescore) | 92.7% |
| `:latest` MTP=3 mt=8192 (online v2 harness, naive glue) | 86.6%* |

*The naive-glue online run scored lower than offline smart-glue because the v2
harness was deployed before I had the re-indent fix. Future runs will use smart_glue.

**Conclusion**: `:v2` and `:latest` are statistically indistinguishable on HumanEval
quality. The 8.6pp gap we'd been reporting was harness artifact.

## What This Changes in qwen-bench

1. **SOTA.md**: HumanEval section needs update. Old 79.3% "record" is superseded by
   95.1% (BF16+DFlash N=8 on `:v2` mt=8192, smart-glue corrected) — modulo more
   re-runs to confirm.
2. **Studies #2, #3, #4 READMEs**: Add corrected pass rates as parenthetical notes.
   Don't rewrite history — preserve the original numbers for reproducibility.
3. **Stress harness**: Replace prod copy with `stress_harness_v2.py` (in `bench/stress-harness/`).
4. **Future bench protocol**: Always include both `pass@1 (greedy)` and `pass@5 (temp=0.8)`
   for quality runs. Single-shot pass@1 has too much variance from truncations/empties
   to be a reliable SOTA metric.

## Service Status

- SOTA service: `:latest` FP8+MTP=3 (default config, restarted 14:56 MSK).
- This is statistically the same quality as `:v2` (~92-93% HE corrected).
- Throughput SOTA on `:v2` from study #4 still stands (245 tok/s mean, 4.45k peak).
- **No image swap needed.**

## Files Added

```
harness/stress_harness_v2.py        — patched harness (prompt + naive glue)
harness/stress_harness_passk.py     — pass@k variant with smart_glue + n-samples
harness/rescore_humaneval.py        — offline rescoring tool
harness/ab_test_latest.sh           — A/B :latest MTP=3 launcher
harness/ab_test_v2harness.sh        — v2 harness validation launcher
harness/ab_test_passk.sh            — pass@5 study launcher
configs/ab-fp8-mtp3-latest-mt8192/  — A/B :latest MTP=3 (orig harness) results
configs/ab-fp8-mtp3-latest-mt8192-v2harness/  — v2 harness validation results
configs/ab-fp8-mtp3-latest-pass5/   — pass@5 results
```

## What's Still Open

- [ ] Concurrency=1 isolation test (Repne hypothesis 4): does c=1 vs c=8 change pass rate?
  My hunch: no, since the model is generating each token sequentially regardless of batch size.
- [ ] pass@5 on `:v2` MTP=3 for a clean image comparison at the capability ceiling.
- [ ] Roll out `stress_harness_v2.py` to the canonical bench location.
- [ ] Update SOTA.md HumanEval section with corrected scores + smart-glue caveat.
