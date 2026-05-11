#!/usr/bin/env python3
"""Generate study #4 (v2-followup) headline plots.

Three plots tell the story:
  1. docs/images/speed_head_to_head.png
       Decode throughput bars: BF16+DFlash n=8 (study #3 winner) vs
       FP8+MTP=3 vs FP8+MTP=5, all on repne/vllm:v2.
  2. docs/images/quality_max_tokens.png
       HumanEval+MBPP pass@1 at max_tokens=4096 (study #3) vs 8192 (study #4),
       for both BF16+DFlash n=8 and FP8+MTP=3, with empty_response delta annotated.
  3. docs/images/empty_response_recovery.png
       Stacked bar of correct / wrong / empty_response counts, 4096 vs 8192.
"""
import json
import statistics
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path("/tmp/qwen-bench-2026-05-11-v2-followup")
PRIOR = Path("/tmp/qwen-bench-2026-05-dflash-v2-sweep")  # study #3
OUT = ROOT / "docs/images"
OUT.mkdir(parents=True, exist_ok=True)


def load_cells(p: Path) -> list[dict]:
    return [
        r for r in json.loads(p.read_text())["results"]
        if r.get("benchmark_mode") != "prefill" and r.get("aggregate_tps", 0) > 0
    ]


def mean_tps(p: Path) -> tuple[float, float, float]:
    cells = load_cells(p)
    ys = [r["aggregate_tps"] for r in cells]
    return statistics.mean(ys), min(ys), max(ys)


def accept_rate(p: Path) -> float:
    cells = load_cells(p)
    rs = [r.get("server_spec_accept_rate", 0) for r in cells]
    return statistics.mean(rs) if rs else 0.0


def load_quality(p: Path) -> dict:
    """Read both humaneval.json and mbpp.json summaries."""
    d = {}
    for bench in ("humaneval", "mbpp"):
        f = p / f"{bench}.json"
        if not f.exists():
            d[bench] = None
            continue
        data = json.loads(f.read_text())
        d[bench] = {
            "pass_rate": data.get("pass_rate", 0) * 100,
            "n_correct": data.get("n_correct", 0),
            "n_total": data.get("n_total", 0),
            "n_empty": data.get("n_empty_response", 0),
        }
    return d


# ──────────────────────────────────────────────────────────────────────────────
# PLOT 1: Speed head-to-head (BF16+DFlash n=8  vs  FP8+MTP=3  vs  FP8+MTP=5)
# ──────────────────────────────────────────────────────────────────────────────
prior_winner = PRIOR / "configs/stage-b/b32768_c256_n8/throughput.json"
fp8_mtp3 = ROOT / "configs/speed-fp8-mtp3/throughput.json"
fp8_mtp5 = ROOT / "configs/speed-fp8-mtp5/throughput.json"

bars = []
for label, path in [
    ("BF16+DFlash\nn=8\n(study #3)", prior_winner),
    ("FP8+MTP=3\n(study #4)", fp8_mtp3),
    ("FP8+MTP=5\n(study #4)", fp8_mtp5),
]:
    if not path.exists():
        print(f"  [skip] {path} not yet present")
        bars.append((label, 0, 0, 0, 0))
        continue
    mean, lo, hi = mean_tps(path)
    ar = accept_rate(path) * 100
    bars.append((label, mean, lo, hi, ar))

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5))

xs = np.arange(len(bars))
ys = [b[1] for b in bars]
los = [b[1] - b[2] for b in bars]
his = [b[3] - b[1] for b in bars]
colors = ["#1f77b4", "#2ca02c", "#ff7f0e"]
ax1.bar(xs, ys, yerr=[los, his], capsize=6, color=colors, alpha=0.85,
        error_kw={"elinewidth": 1.5, "ecolor": "#333"})
for x, (label, y, lo, hi, ar) in zip(xs, bars):
    if y > 0:
        ax1.text(x, y + 4, f"{y:.1f}", ha="center", fontsize=11, fontweight="bold")
ax1.set_xticks(xs, [b[0] for b in bars], fontsize=10)
ax1.set_ylabel("Aggregate decode tok/s (mean of 15 cells)", fontsize=11)
ax1.set_title("Decode throughput head-to-head\nrepne/vllm:v2 · GPU 0+1 · b=32768 c=256",
              fontsize=12)
ax1.grid(True, alpha=0.3, axis="y")
ax1.set_ylim(0, max(ys + [200]) * 1.18)

# Acceptance rate panel
ars = [b[4] for b in bars]
ax2.bar(xs, ars, color=colors, alpha=0.85)
for x, ar in zip(xs, ars):
    if ar > 0:
        ax2.text(x, ar + 0.5, f"{ar:.1f}%", ha="center", fontsize=11, fontweight="bold")
ax2.set_xticks(xs, [b[0] for b in bars], fontsize=10)
ax2.set_ylabel("Server spec acceptance rate (%)", fontsize=11)
ax2.set_title("Drafter acceptance rate", fontsize=12)
ax2.grid(True, alpha=0.3, axis="y")
ax2.set_ylim(0, max(ars + [30]) * 1.25)

plt.tight_layout()
plt.savefig(OUT / "speed_head_to_head.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"✓ {OUT/'speed_head_to_head.png'}")


# ──────────────────────────────────────────────────────────────────────────────
# PLOT 2: Quality at max_tokens=4096 (prior) vs max_tokens=8192 (this study)
# ──────────────────────────────────────────────────────────────────────────────
# Hardcoded study #3 reference (winner re-run at max_tokens=4096)
PRIOR_QUALITY = {
    "BF16+DFlash n=8": {
        "humaneval": {"pass_rate": 58.5, "n_correct": 96, "n_total": 164, "n_empty": 19},
        "mbpp":      {"pass_rate": 82.1, "n_correct": 211, "n_total": 257, "n_empty": 37},
    },
}
# Hardcoded study #2 reference (FP8+MTP=3 on repne/vllm:latest, 4096)
PRIOR_FP8_QUALITY = {
    "FP8+MTP=3 (study #2, latest)": {
        "humaneval": {"pass_rate": 79.3, "n_correct": 130, "n_total": 164, "n_empty": 0},
        "mbpp":      {"pass_rate": 85.6, "n_correct": 220, "n_total": 257, "n_empty": 0},
    },
}

q_bf16 = load_quality(ROOT / "configs/quality-bf16-dflash-n8-mt8192")
q_fp8 = load_quality(ROOT / "configs/quality-fp8-mtp3-mt8192")

def safe(q, key, field):
    return q.get(key, {}).get(field, 0) if q.get(key) else 0

groups = [
    ("BF16+DFlash n=8\n@ mt=4096 (study #3)",
        PRIOR_QUALITY["BF16+DFlash n=8"]["humaneval"]["pass_rate"],
        PRIOR_QUALITY["BF16+DFlash n=8"]["mbpp"]["pass_rate"]),
    ("BF16+DFlash n=8\n@ mt=8192 (study #4)",
        safe(q_bf16, "humaneval", "pass_rate"),
        safe(q_bf16, "mbpp", "pass_rate")),
    ("FP8+MTP=3\n@ mt=4096 (study #2)",
        PRIOR_FP8_QUALITY["FP8+MTP=3 (study #2, latest)"]["humaneval"]["pass_rate"],
        PRIOR_FP8_QUALITY["FP8+MTP=3 (study #2, latest)"]["mbpp"]["pass_rate"]),
    ("FP8+MTP=3\n@ mt=8192 (study #4)",
        safe(q_fp8, "humaneval", "pass_rate"),
        safe(q_fp8, "mbpp", "pass_rate")),
]

fig, ax = plt.subplots(figsize=(12, 6))
x = np.arange(len(groups))
w = 0.35
he_ys = [g[1] for g in groups]
mb_ys = [g[2] for g in groups]
ax.bar(x - w/2, he_ys, w, label="HumanEval pass@1", color="#1f77b4", alpha=0.85)
ax.bar(x + w/2, mb_ys, w, label="MBPP pass@1", color="#ff7f0e", alpha=0.85)
for xi, y in zip(x - w/2, he_ys):
    if y > 0: ax.text(xi, y + 0.6, f"{y:.1f}%", ha="center", fontsize=9)
for xi, y in zip(x + w/2, mb_ys):
    if y > 0: ax.text(xi, y + 0.6, f"{y:.1f}%", ha="center", fontsize=9)
ax.set_xticks(x, [g[0] for g in groups], fontsize=9)
ax.set_ylabel("pass@1 (%)", fontsize=11)
ax.set_title("Quality vs max_tokens — does mt=8192 recover empty_response failures?",
             fontsize=12)
ax.legend(loc="upper right", fontsize=10)
ax.grid(True, alpha=0.3, axis="y")
ax.set_ylim(0, 100)
plt.tight_layout()
plt.savefig(OUT / "quality_max_tokens.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"✓ {OUT/'quality_max_tokens.png'}")


# ──────────────────────────────────────────────────────────────────────────────
# PLOT 3: empty_response recovery stacked bars
# ──────────────────────────────────────────────────────────────────────────────
fig, (axh, axm) = plt.subplots(1, 2, figsize=(13, 5))

def stack(ax, title, n_total, datasets):
    # datasets: list of (label, n_correct, n_empty)
    xs = np.arange(len(datasets))
    correct = [d[1] for d in datasets]
    empty = [d[2] for d in datasets]
    wrong = [n_total - c - e for c, e in zip(correct, empty)]
    ax.bar(xs, correct, color="#2ca02c", label="correct", alpha=0.85)
    ax.bar(xs, wrong, bottom=correct, color="#d62728", label="wrong", alpha=0.85)
    ax.bar(xs, empty, bottom=[c+w for c, w in zip(correct, wrong)],
           color="#7f7f7f", label="empty_response", alpha=0.85)
    for x, (label, c, e) in zip(xs, datasets):
        ax.text(x, n_total + 1, f"{c}/{n_total}", ha="center", fontsize=10, fontweight="bold")
        if e > 0:
            ax.text(x, c + (n_total - c - e)/2 + e/2, f"empty={e}",
                    ha="center", fontsize=9, color="white")
    ax.set_xticks(xs, [d[0] for d in datasets], fontsize=9, rotation=15, ha="right")
    ax.set_ylabel("problems", fontsize=11)
    ax.set_title(title, fontsize=12)
    ax.legend(loc="lower right", fontsize=9)
    ax.set_ylim(0, n_total * 1.1)

he_datasets = [
    ("BF16+DFlash\nmt=4096",
     PRIOR_QUALITY["BF16+DFlash n=8"]["humaneval"]["n_correct"],
     PRIOR_QUALITY["BF16+DFlash n=8"]["humaneval"]["n_empty"]),
    ("BF16+DFlash\nmt=8192",
     safe(q_bf16, "humaneval", "n_correct"),
     safe(q_bf16, "humaneval", "n_empty")),
    ("FP8+MTP=3\nmt=8192",
     safe(q_fp8, "humaneval", "n_correct"),
     safe(q_fp8, "humaneval", "n_empty")),
]
stack(axh, "HumanEval (164 problems)", 164, he_datasets)

mb_datasets = [
    ("BF16+DFlash\nmt=4096",
     PRIOR_QUALITY["BF16+DFlash n=8"]["mbpp"]["n_correct"],
     PRIOR_QUALITY["BF16+DFlash n=8"]["mbpp"]["n_empty"]),
    ("BF16+DFlash\nmt=8192",
     safe(q_bf16, "mbpp", "n_correct"),
     safe(q_bf16, "mbpp", "n_empty")),
    ("FP8+MTP=3\nmt=8192",
     safe(q_fp8, "mbpp", "n_correct"),
     safe(q_fp8, "mbpp", "n_empty")),
]
stack(axm, "MBPP (257 problems)", 257, mb_datasets)

fig.suptitle("Empty-response recovery: mt=4096 → mt=8192",
             fontsize=13, y=1.02)
plt.tight_layout()
plt.savefig(OUT / "empty_response_recovery.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"✓ {OUT/'empty_response_recovery.png'}")

print(f"\nAll plots written to {OUT}")
