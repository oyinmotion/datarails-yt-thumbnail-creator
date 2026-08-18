"""Orchestration. Knows the order of operations and nothing about the UI.

The hard contract: the caller always receives exactly five result rows. A failed
render is a row with path=None and a readable note, never a missing tile.

Rerolls run as a distinct second phase (not a loop inside each worker) so that
every variant's first attempt completes before any variant's reroll starts.
That phase separation is what makes "exactly one reroll per variant" a
deterministic guarantee under real thread scheduling, not just true on average.
"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from . import plan as plan_module
from . import postprocess, probe, qa, render
from .models import BatchPlan, Variant

log = logging.getLogger(__name__)

MAX_WORKERS = 5


@dataclass
class ThumbResult:
    variant: Variant
    path: Path | None
    flagged: bool = False
    note: str = ""


@dataclass
class BatchOutcome:
    plan: BatchPlan
    results: list[ThumbResult]
    warnings: list[str] = field(default_factory=list)


def _slug(variant: Variant) -> str:
    return f"{variant.index:02d}_{variant.hook_type}_{variant.treatment}"


def _other_frame(frames: dict[str, Path], used: str) -> Path | None:
    for name, path in frames.items():
        if name != used:
            return path
    return None


def _attempt(
    variant: Variant,
    frames: dict[str, Path],
    out_dir: Path,
    client,
    extra_instruction: str = "",
    frame_override: Path | None = None,
) -> dict:
    """One render + finalize + QA pass. Returns a small status dict, never raises."""
    try:
        raw = render.render_variant(
            variant, frames, client=client,
            extra_instruction=extra_instruction,
            frame_override=frame_override,
        )
    except render.RenderBlocked as exc:
        return {"status": "blocked", "note": f"blocked by content filter: {exc}"}
    except render.RenderError as exc:
        return {"status": "error", "note": str(exc)}

    path = postprocess.finalize(raw, out_dir / f"{_slug(variant)}.png")
    result = qa.check(path, variant.headline, client=client)
    if result.ok:
        return {"status": "ok", "path": path}
    return {"status": "qa_fail", "path": path, "note": "; ".join(result.problems)}


def generate_batch(
    video: Path,
    work_dir: Path,
    headline_override: str | None = None,
    context: str | None = None,
    client=None,
    progress: Callable[[str], None] | None = None,
) -> BatchOutcome:
    def say(message: str) -> None:
        log.info(message)
        if progress:
            progress(message)

    work_dir = Path(work_dir)
    frames_dir = work_dir / "frames"
    out_dir = work_dir / "out"
    out_dir.mkdir(parents=True, exist_ok=True)
    warnings: list[str] = []

    say("Pulling frames out of the ad…")
    frame_paths = probe.extract_frames(video, frames_dir)
    frames = {p.name: p for p in frame_paths}

    audio: Path | None = None
    try:
        audio = probe.extract_audio(video, work_dir)
    except probe.ProbeError as exc:
        warnings.append(f"Couldn't read the audio, so hooks come from the "
                        f"picture alone. ({exc})")

    say("Reading the ad and planning five hooks…")
    batch_plan = plan_module.build_plan(
        frame_paths, audio, headline_override, context, client=client,
    )
    if not batch_plan.transcript_used and not warnings:
        warnings.append("No transcript was available; hooks come from the "
                        "frames alone.")

    say("Rendering 5 thumbnails…")
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        first_pass = list(pool.map(
            lambda v: _attempt(v, frames, out_dir, client),
            batch_plan.variants,
        ))

    results: list[ThumbResult | None] = [None] * len(batch_plan.variants)
    # (index into results, variant, extra_instruction, frame_override)
    reroll_specs: list[tuple[int, Variant, str, Path | None]] = []

    for i, (variant, outcome) in enumerate(zip(batch_plan.variants, first_pass)):
        if outcome["status"] == "ok":
            results[i] = ThumbResult(variant=variant, path=outcome["path"])
        elif outcome["status"] == "blocked":
            # Expected occasionally on real faces — retry with a different frame.
            reroll_specs.append((
                i, variant, "", _other_frame(frames, variant.frame_id),
            ))
        elif outcome["status"] == "error":
            reroll_specs.append((i, variant, "", None))
        else:  # qa_fail
            note = outcome["note"]
            extra = (
                f"The previous attempt failed verification: {note}. "
                "Render the headline larger, fully inside the frame, with more "
                "space around it, and make every word unmistakably legible."
            )
            reroll_specs.append((i, variant, extra, None))

    if reroll_specs:
        say("Rerolling variants that need a second pass…")
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
            second_pass = list(pool.map(
                lambda spec: _attempt(
                    spec[1], frames, out_dir, client,
                    extra_instruction=spec[2], frame_override=spec[3],
                ),
                reroll_specs,
            ))

        for (i, variant, _extra, _override), outcome in zip(reroll_specs, second_pass):
            if outcome["status"] == "ok":
                results[i] = ThumbResult(variant=variant, path=outcome["path"])
            elif outcome["status"] == "qa_fail":
                # Second failure: still hand it over, flagged. The user decides.
                results[i] = ThumbResult(
                    variant=variant, path=outcome["path"], flagged=True,
                    note=f"text may be unreadable — {outcome['note']}",
                )
            else:  # blocked or error again — no image to show
                results[i] = ThumbResult(
                    variant=variant, path=None, flagged=True,
                    note=outcome["note"],
                )

    say("Done.")
    return BatchOutcome(plan=batch_plan, results=results, warnings=warnings)
