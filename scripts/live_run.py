"""Run the real pipeline against a local video. Costs real money (~$1).

Usage:
    OPENAI_API_KEY=sk-... .venv/bin/python scripts/live_run.py \
        "Sample input ad/claude-vs-claude-with-fos-v2_....mp4"
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import PROJECT_ROOT          # noqa: E402
from src.pipeline import generate_batch       # noqa: E402


def main() -> int:
    if len(sys.argv) < 2:
        sample = next((PROJECT_ROOT / "Sample input ad").glob("*.mp4"), None)
        if sample is None:
            print("Pass a video path.")
            return 1
        video = sample
    else:
        video = Path(sys.argv[1])

    work_dir = PROJECT_ROOT / "out" / "live"
    work_dir.mkdir(parents=True, exist_ok=True)

    outcome = generate_batch(video, work_dir, progress=lambda m: print(f"  {m}"))

    print(f"\nAd: {outcome.plan.ad_summary}")
    for warning in outcome.warnings:
        print(f"  warning: {warning}")

    print("\nResults:")
    for result in outcome.results:
        variant = result.variant
        status = "ok"
        if result.path is None:
            status = f"FAILED — {result.note}"
        elif result.flagged:
            status = f"FLAGGED — {result.note}"
        print(f"  {variant.index}. [{variant.hook_type}/{variant.treatment}] "
              f"{variant.headline!r} → {status}")
        if result.path:
            print(f"     {result.path}")

    flagged = sum(1 for r in outcome.results if r.flagged or r.path is None)
    print(f"\n{5 - flagged}/5 clean. Files in {work_dir / 'out'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
