#!/usr/bin/env python3
"""Generate hero & section images for the 2026-05-11 v2-followup study
via Gemini 3.1 nano-banana (`gemini-3.1-flash-image-preview`).

Three images:
  - study_hero.png       — wide cinematic banner
  - speed_section.png    — accompanies the speed head-to-head section
  - quality_section.png  — accompanies the max_tokens recovery section

Prompts intentionally restrained (Tufte/Distill aesthetic) — no GPU/orb/neural-
net AI slop. Run idempotently; skips files that already exist non-trivially.
"""
import os
import pathlib
import sys

from google import genai

OUT = pathlib.Path(__file__).parent / "images"
OUT.mkdir(parents=True, exist_ok=True)

client = genai.Client(
    api_key=os.environ.get("GEMINI_API_KEY") or os.environ["GOOGLE_API_KEY"]
)
MODEL = "gemini-3.1-flash-image-preview"

PROMPTS = {
    "study_hero": (
        "A wide 16:9 hero banner in the visual style of a Distill.pub or Edward "
        "Tufte scientific publication. Composition: three slender vertical bars "
        "of different heights on the left side of the frame representing a "
        "head-to-head comparison; on the right side, two stacked horizontal "
        "bars showing a small grey segment shrinking dramatically between them "
        "(representing recovered failures). Connective fine hairline curves "
        "link the left and right halves. Off-white background (#fafaf7), "
        "muted ink-blue (#1f4068), restrained amber accent (#d99058), and a "
        "single thin magenta highlight. No GPUs, no neural-net clouds, no "
        "glowing orbs, no 3D renders, no particles. Flat, precise, "
        "publication-quality vector aesthetic. No readable text, no axis "
        "labels, no logos, no watermarks. Lots of negative space."
    ),
    "speed_section": (
        "A minimal scientific illustration of three vertical bars of different "
        "heights, left to right: medium (blue), tall (green), short (orange). "
        "Each bar has a thin error-bar whisker at the top. Off-white "
        "background, restrained colors, hairline borders. Tufte/Distill "
        "publication aesthetic. No readable text, no axis labels, no logos, "
        "no watermarks, no decoration. Lots of negative space."
    ),
    "quality_section": (
        "A minimal scientific illustration of two stacked horizontal bars, one "
        "above the other. The bottom bar has a small grey segment on its "
        "right side (~12% of width). The top bar has the grey segment reduced "
        "to almost nothing (~2%), with a thin downward arrow on the right "
        "side annotating the shrinkage. Off-white background, muted ink-blue "
        "for the dominant segment, restrained grey for the small segment. "
        "Tufte/Distill publication aesthetic. No readable text, no axis "
        "labels, no logos, no watermarks. Lots of negative space."
    ),
}

for name, prompt in PROMPTS.items():
    out_path = OUT / f"{name}.png"
    if out_path.exists() and out_path.stat().st_size > 1000:
        print(f"[skip] {out_path} exists ({out_path.stat().st_size} bytes)")
        continue
    print(f"[gen]  {name}: {prompt[:80]}...")
    try:
        resp = client.models.generate_content(model=MODEL, contents=prompt)
        wrote = False
        for part in resp.candidates[0].content.parts:
            if getattr(part, "inline_data", None) and part.inline_data.data:
                out_path.write_bytes(part.inline_data.data)
                print(f"[ok]   {out_path} ({out_path.stat().st_size} bytes)")
                wrote = True
                break
        if not wrote:
            print(f"[warn] no image data for {name}")
    except Exception as e:
        print(f"[err]  {name}: {e}", file=sys.stderr)
        sys.exit(1)
