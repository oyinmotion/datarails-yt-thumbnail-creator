"""Orchestration. Knows the order of operations and nothing about the UI.

The hard contract: the caller always receives exactly five result rows. A failed
render is a row with path=None and a readable note, never a missing tile.
"""

from __future__ import annotations

import logging
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from . import backoff
from . import plan as plan_module
from . import postprocess, probe, qa, render
from .models import BatchPlan, Variant

log = logging.getLogger(__name__)

MAX_WORKERS = 5

# Module-level so tests can swap in a no-op and stay instant.
DEFAULT_SLEEPER = time.sleep


@dataclass
class ThumbResult:
    variant: Variant
    path: Path | None
    flagged: bool = False
    note: str = ""
    # True when the render was handed over without the legibility model ever
    # reading it back — see QAResult.unverified.
    unverified: bool = False


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
    sleeper=None,
) -> ThumbResult:
    """Render, finalize, verify. One reroll on failure, then flag and move on.

    Every failure mode is contained here. The five variants run in a thread
    pool, and `pool.map` re-raises the first exception it sees — which would
    lose all five rows over one bad response — so nothing may escape.
    """
    sleeper = sleeper or DEFAULT_SLEEPER
    # Deterministic per-variant offset: without it all five variants wake from
    # backoff at the same instant and re-collide with the same rate limit.
    offset = variant.index * backoff.STAGGER
    extra_instruction = ""
    frame_override: Path | None = None
    last_note = ""

    try:
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
                if attempt == 1:
                    backoff.wait(attempt, sleeper=sleeper, offset=offset)
                continue
            except render.RenderError as exc:
                last_note = str(exc)
                if attempt == 1:
                    backoff.wait(attempt, sleeper=sleeper, offset=offset)
                continue

            path = postprocess.finalize(raw, out_dir / f"{_slug(variant)}.png")
            result = qa.check(path, variant.headline, client=client)
            if result.ok:
                return ThumbResult(
                    variant=variant, path=path,
                    unverified=result.unverified,
                )

            last_note = "; ".join(result.problems)
            if attempt == 1:
                extra_instruction = (
                    f"The previous attempt failed verification: {last_note}. "
                    "Render the headline larger, fully inside the frame, with "
                    "more space around it, and make every word unmistakably "
                    "legible."
                )
                # No backoff here: the API answered fine, the type was just
                # illegible. Backoff exists for rate limits and transport
                # failures, and making the user wait for a taste reroll is
                # pure cost.
                continue

            # Second failure: still hand it over, flagged. The user decides.
            return ThumbResult(
                variant=variant, path=path, flagged=True,
                note=f"text may be unreadable — {last_note}",
            )
    except Exception as exc:
        # Anything unforeseen — a malformed payload, a PIL failure, a bug —
        # costs exactly one tile instead of the whole batch.
        log.warning("variant %s failed unexpectedly", variant.index,
                    exc_info=True)
        return ThumbResult(
            variant=variant, path=None, flagged=True,
            note=f"something went wrong rendering this one ({exc})",
        )

    return ThumbResult(variant=variant, path=None, flagged=True, note=last_note)


def generate_batch(
    video: Path,
    work_dir: Path,
    headline_override: str | None = None,
    context: str | None = None,
    client=None,
    progress: Callable[[str], None] | None = None,
    sleeper=None,
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
        sleeper=sleeper,
    )
    if not batch_plan.transcript_used and not warnings:
        warnings.append("No transcript was available; hooks come from the "
                        "frames alone.")

    say("Rendering 5 thumbnails…")
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        results = list(pool.map(
            lambda v: _one_variant(v, frames, out_dir, client, sleeper=sleeper),
            batch_plan.variants,
        ))

    unverified = sum(1 for r in results if r.unverified)
    if unverified:
        warnings.append(
            f"{unverified} of {len(results)} thumbnails could not be "
            "text-checked — verify the headlines yourself."
        )

    say("Done.")
    return BatchOutcome(plan=batch_plan, results=results, warnings=warnings)
