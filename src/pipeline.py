"""Orchestration. Knows the order of operations and nothing about the UI.

The hard contract: the caller always receives exactly five result rows. A failed
render is a row with path=None and a readable note, never a missing tile.
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


def _one_variant(
    variant: Variant,
    frames: dict[str, Path],
    out_dir: Path,
    client,
) -> ThumbResult:
    """Render, finalize, verify. One reroll on failure, then flag and move on."""
    extra_instruction = ""
    frame_override: Path | None = None
    last_note = ""

    for attempt in (1, 2):
        try:
            raw = render.render_variant(
                variant, frames, client=client,
                extra_instruction=extra_instruction,
                frame_override=frame_override,
            )
        except render.RenderBlocked as exc:
            last_note = f"blocked by content filter: {exc}"
            frame_override = _other_frame(frames, variant.frame_id)
            continue
        except render.RenderError as exc:
            last_note = str(exc)
            continue

        path = postprocess.finalize(raw, out_dir / f"{_slug(variant)}.png")
        result = qa.check(path, variant.headline, client=client)
        if result.ok:
            return ThumbResult(variant=variant, path=path)

        last_note = "; ".join(result.problems)
        if attempt == 1:
            extra_instruction = (
                f"The previous attempt failed verification: {last_note}. "
                "Render the headline larger, fully inside the frame, with more "
                "space around it, and make every word unmistakably legible."
            )
            continue

        # Second failure: still hand it over, flagged. The user decides.
        return ThumbResult(
            variant=variant, path=path, flagged=True,
            note=f"text may be unreadable — {last_note}",
        )

    return ThumbResult(variant=variant, path=None, flagged=True, note=last_note)


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
        results = list(pool.map(
            lambda v: _one_variant(v, frames, out_dir, client),
            batch_plan.variants,
        ))

    say("Done.")
    return BatchOutcome(plan=batch_plan, results=results, warnings=warnings)
