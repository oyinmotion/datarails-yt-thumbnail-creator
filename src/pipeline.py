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
from . import branding, postprocess, probe, qa, render
from .config import PRIMARY_RATIO, RATIOS
from .models import BatchPlan, Variant

log = logging.getLogger(__name__)

# Five variants times three ratios is fifteen renders per batch, so the pool
# is wider than the old one-render-per-variant shape.
MAX_WORKERS = 8

# Module-level so tests can swap in a no-op and stay instant.
DEFAULT_SLEEPER = time.sleep


@dataclass
class RenderOutcome:
    """One variant at one aspect ratio."""
    variant: Variant
    ratio: str
    path: Path | None
    flagged: bool = False
    note: str = ""
    unverified: bool = False


@dataclass
class ThumbResult:
    """One concept, rendered at every ratio."""
    variant: Variant
    paths: dict[str, Path] = field(default_factory=dict)
    flagged: bool = False
    note: str = ""
    # True when a render was handed over without the legibility model ever
    # reading it back — see QAResult.unverified.
    unverified: bool = False

    @property
    def path(self) -> Path | None:
        """The 16:9 render — the YouTube thumbnail, and the tile preview."""
        return self.paths.get(PRIMARY_RATIO)


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


def _one_render(
    variant: Variant,
    ratio: str,
    frames: dict[str, Path],
    out_dir: Path,
    client,
    sleeper=None,
    people_in_ad: bool = True,
) -> RenderOutcome:
    """Render, finalize, verify. One reroll on failure, then flag and move on.

    Every failure mode is contained here. The five variants run in a thread
    pool, and `pool.map` re-raises the first exception it sees — which would
    lose all five rows over one bad response — so nothing may escape.
    """
    sleeper = sleeper or DEFAULT_SLEEPER
    (gen_w, gen_h), final_size = RATIOS[ratio]
    gen_size = f"{gen_w}x{gen_h}"
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
                    people_in_ad=people_in_ad,
                    gen_size=gen_size,
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

            path = postprocess.finalize(
                raw, out_dir / f"{_slug(variant)}_{ratio}.png",
                final_size=final_size,
            )
            # The likeness gate compares against the frame this render was
            # actually built from, which is the override after a blocked reroll.
            source_frame = frame_override or frames.get(variant.frame_id)
            result = qa.check(
                path, variant.headline,
                reference_frame=source_frame, client=client,
                people_in_ad=people_in_ad, expected_size=final_size,
            )
            if result.ok:
                # The logo is composited here, after verification: stamping
                # before the legibility read could hide warped type behind it.
                path = branding.stamp_logo(path)
                return RenderOutcome(
                    variant=variant, ratio=ratio, path=path,
                    unverified=result.unverified,
                )

            last_note = "; ".join(result.problems)
            if attempt == 1:
                if result.likeness in ("DIFFERENT", "NOBODY"):
                    extra_instruction = (
                        f"The previous attempt failed verification: {last_note}. "
                        "You generated a person who is not in the reference "
                        "frames. Do not invent, replace or beautify anyone. Cut "
                        "the exact person out of the reference frame and place "
                        "them in front of the background, keeping their face, "
                        "age, hair, facial hair and clothing identical, with "
                        "their face large and clearly visible."
                    )
                else:
                    extra_instruction = (
                        f"The previous attempt failed verification: {last_note}. "
                        "Render the headline larger, fully inside the frame, "
                        "with more space around it, and make every word "
                        "unmistakably legible."
                    )
                # No backoff here: the API answered fine, the type was just
                # illegible. Backoff exists for rate limits and transport
                # failures, and making the user wait for a taste reroll is
                # pure cost.
                continue

            # Second failure: still hand it over, flagged. The user decides.
            path = branding.stamp_logo(path)
            return RenderOutcome(
                variant=variant, ratio=ratio, path=path, flagged=True,
                note=f"text may be unreadable — {last_note}",
            )
    except Exception as exc:
        # Anything unforeseen — a malformed payload, a PIL failure, a bug —
        # costs exactly one tile instead of the whole batch.
        log.warning("variant %s failed unexpectedly", variant.index,
                    exc_info=True)
        return RenderOutcome(
            variant=variant, ratio=ratio, path=None, flagged=True,
            note=f"something went wrong rendering this one ({exc})",
        )

    return RenderOutcome(
        variant=variant, ratio=ratio, path=None, flagged=True, note=last_note
    )


def _group_by_variant(
    outcomes: list[RenderOutcome], variants: list[Variant]
) -> list[ThumbResult]:
    """Collapse per-ratio outcomes into one row per concept.

    Preserves the hard contract: exactly one row per variant, in matrix order,
    even if every ratio of that variant failed.
    """
    by_index: dict[int, ThumbResult] = {
        v.index: ThumbResult(variant=v) for v in variants
    }
    for outcome in outcomes:
        row = by_index[outcome.variant.index]
        if outcome.path is not None:
            row.paths[outcome.ratio] = outcome.path
        if outcome.flagged:
            row.flagged = True
        if outcome.unverified:
            row.unverified = True
        if outcome.note:
            label = "" if outcome.ratio == PRIMARY_RATIO else f"{outcome.ratio}: "
            row.note = f"{row.note}; {label}{outcome.note}".lstrip("; ")
    return [by_index[v.index] for v in variants]


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

    if not batch_plan.people_in_ad:
        warnings.append(
            "This ad has no people in it, so the thumbnails are built without "
            "any person and the house-style references are withheld."
        )

    ratios = list(RATIOS)
    jobs = [(v, r) for v in batch_plan.variants for r in ratios]
    say(f"Rendering {len(batch_plan.variants)} concepts × {len(ratios)} "
        f"ratios = {len(jobs)} images…")
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        outcomes = list(pool.map(
            lambda job: _one_render(
                job[0], job[1], frames, out_dir, client, sleeper=sleeper,
                people_in_ad=batch_plan.people_in_ad,
            ),
            jobs,
        ))

    results = _group_by_variant(outcomes, batch_plan.variants)

    unverified = sum(1 for r in results if r.unverified)
    if unverified:
        warnings.append(
            f"{unverified} of {len(results)} thumbnails could not be "
            "text-checked — verify the headlines yourself."
        )

    say("Done.")
    return BatchOutcome(plan=batch_plan, results=results, warnings=warnings)
