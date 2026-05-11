# Discord — Tier 2 Update (qwen R&D + Repne DM)

---

## qwen R&D chat (general)

**Tier 2 results — mt=16384 HumanEval rerun (`:v2`, patched harness)**

Followup to the corrected-HE thread earlier. Doubled the response budget (mt 8192 → 16384) to test whether Tier 1's 87.2 % / 83.5 % were budget-bottlenecked or capability-bottlenecked.

| Config | Tier 1 (mt=8192) | Tier 2 (mt=16384) | Δ |
|---|---:|---:|---:|
| BF16+DFlash N=8 | 87.2 % | **90.9 %** | **+3.7 pp** |
| FP8+MTP=3 | 83.5 % | 84.8 % | +1.3 pp |

- `length_truncated` went **13 → 0** on both configs ✅
- BF16 recovered those as passes; FP8 turned them into `empty_response` (7 → 15) — model giving up at long budget, not budget-bound
- The **offline-rescore 95.1 % is NOT fully reachable online**. Even at mt=16384 with the patched harness, BF16+DFlash tops at 90.9 %. Remaining ~4 pp is `empty_response` + `test_fail` — neither fixable with more tokens
- **Production unchanged**: FP8+MTP=3 on `:latest` stays SOTA on throughput × quality × stability. BF16+DFlash N=8 = +6.1 pp pass@1 quality at ~22 % throughput cost; pass@5 on FP8 already covers that gap at 1.3× cost

Hub: https://github.com/jcartu/qwen-bench/blob/main/SOTA.md#21-humaneval-164-problems--corrected-2026-05-11
Study: https://github.com/jcartu/qwen-bench-2026-05-11-v2-followup/blob/main/ADDENDUM.md#tier-2-max_tokens16384-ceiling-test-2026-05-11-200320-32-msk

---

## Repne DM

Tier 2 done. tl;dr: doubling `max_tokens` recovered all 13 truncations on both configs.

- **BF16+DFlash N=8 mt=16384 → 90.9 %** (was 87.2). Online single-shot SOTA.
- **FP8+MTP=3 mt=16384 → 84.8 %** (was 83.5). Truncations became `empty_response` (7→15) instead of passes — model gives up at long budgets.

So your `:v2` image is doing exactly what we expected: under DFlash it gets +3.7 pp once it has room to finish; under MTP=3 it hits a different ceiling that isn't budget-bound. Offline rescore 95.1 % was a soft ceiling, not reproducible online.

Numbers committed to the v2-followup repo, SOTA.md updated on the hub. Production still on `:latest` FP8+MTP=3, no image swap.

Concurrency=1 isolation test is still pending — I'll fire that when I'm back at the box.
