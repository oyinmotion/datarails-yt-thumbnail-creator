"""Orchestration. Knows the order of operations and nothing about the UI.

Two artifacts per tile per ratio: the ART (text-free, from the model, cached at
generation size under work_dir/art) and the FINAL (headline set by typeset.py,
logo stamped, downscaled once, under work_dir/out).

The hard contract: the caller always receives exactly one result row per
planned concept. A failed render is a row with path=None and a readable note,
never a missing tile.
"""

from __future__ import annotations

import logging
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from PIL import Image

from . import backoff
from . import plan as plan_module
from . import branding, postprocess, probe, qa, render, typeset
from .config import PRIMARY_RATIO, RATIOS
from .models import DEFAULT_VARIANTS, BatchPlan, Variant

log = logging.getLogger(__name__)

# Five variants times three ratios is fifteen renders per batch. The pool is at
# least that wide so a full batch is ONE wave: with eight workers the second
# wave could not start until the slowest render of the first had finished,
# which was the single biggest chunk of wall clock a user sat through. Rate
# limits are handled by the staggered per-concept backoff, not by queueing.
MAX_WORKERS = 15

# Module-level so tests can swap in a no-op and stay instant.
DEFAULT_SLEEPER = time.sleep

ZONE_REROLL_INSTRUCTION = (
    "The previous attempt painted detail into the reserved area. Keep that area "
    "completely calm — flat colour or a soft gradient only — and move every "
    "person, object, spark and light effect out of it."
)


@dataclass
class RenderOutcome:
    """One variant at one aspect ratio."""
    variant: Variant
    ratio: str
    path: Path | None
    art_path: Path | None = None
    style: str = ""
    headline: str = ""
    flagged: bool = False
    note: str = ""
    unverified: bool = False
    # How many image calls this tile made, and how many of those returned an
    # image (and so were billed). A reroll is a second call; a 5xx is a call
    # that was not billed.
    attempts: int = 0
    billed: int = 0


@dataclass
class ThumbResult:
    """One concept, rendered at every ratio."""
    variant: Variant
    paths: dict[str, Path] = field(default_factory=dict)
    art_paths: dict[str, Path] = field(default_factory=dict)
    style: str = ""              # current look; differs from variant.style after swap
    headline: str = ""           # current headline; differs from the plan after an edit
    flagged: bool = False
    note: str = ""
    # True when a render was handed over without the likeness model ever
    # reading it back — see QAResult.unverified.
    unverified: bool = False
    render_calls: int = 0
    images_billed: int = 0

    @property
    def path(self) -> Path | None:
        """The 16:9 render — the YouTube thumbnail, and the tile preview."""
        return self.paths.get(PRIMARY_RATIO)


@dataclass
class BatchOutcome:
    plan: BatchPlan
    results: list[ThumbResult]
    warnings: list[str] = field(default_factory=list)
    # What the batch actually cost in image calls. The pre-run caption assumes
    # one call per image; rerolls push the real number above that.
    planned_renders: int = 0
    render_calls: int = 0
    images_billed: int = 0
    # What per-tile actions need later in the session.
    frames: dict[str, Path] = field(default_factory=dict)
    people_in_ad: bool = True
    work_dir: Path | None = None

    @property
    def rerolls(self) -> int:
        return max(0, self.render_calls - self.planned_renders)


def _slug(variant: Variant) -> str:
    return f"{variant.index:02d}_{variant.hook_type}_{variant.treatment}"


def _other_frame(frames: dict[str, Path], used: str) -> Path | None:
    for name, path in frames.items():
        if name != used:
            return path
    return None


def compose(
    art_path: Path, headline: str, style: str, treatment: str, ratio: str,
    out_path: Path, scrim: bool = False,
) -> tuple[Path, list[str]]:
    """Art on disk → delivered thumbnail on disk. Pure Pillow, ~100ms.

    typeset (native size) → logo → ONE downscale → size/2MB guard.
    """
    with Image.open(art_path) as opened:
        art = opened.convert("RGB")
    set_ = typeset.typeset(art, headline, style, treatment, ratio, scrim=scrim)
    branded = branding.stamp_logo_image(set_.image)
    written = postprocess.finalize_image(branded, out_path, final_size=RATIOS[ratio][1])
    return written, list(set_.notes)


def _one_render(
    variant: Variant,
    ratio: str,
    frames: dict[str, Path],
    art_dir: Path,
    out_dir: Path,
    client,
    sleeper=None,
    people_in_ad: bool = True,
    style: str | None = None,
    headline: str | None = None,
) -> RenderOutcome:
    """Render art, gate it, compose. One reroll at most, then flag and move on.

    Every failure mode is contained here. The tiles run in a thread pool, and
    `pool.map` re-raises the first exception it sees — which would lose every
    row over one bad response — so nothing may escape.
    """
    sleeper = sleeper or DEFAULT_SLEEPER
    style = style or variant.style
    headline = headline or variant.headline
    # Deterministic per-variant offset: without it all variants wake from
    # backoff at the same instant and re-collide with the same rate limit.
    offset = variant.index * backoff.STAGGER
    extra_instruction = ""
    frame_override: Path | None = None
    last_note = ""
    attempts = 0
    billed = 0
    art_path = art_dir / f"{_slug(variant)}_{ratio}.png"
    out_path = out_dir / f"{_slug(variant)}_{ratio}.png"

    def _done(**kw) -> RenderOutcome:
        return RenderOutcome(variant=variant, ratio=ratio, style=style, headline=headline,
                             attempts=attempts, billed=billed, **kw)

    try:
        zone_busy = False
        gate = qa.QAResult(ok=True)
        for attempt in (1, 2):
            try:
                attempts += 1
                raw = render.render_art(
                    variant, frames, client=client,
                    extra_instruction=extra_instruction,
                    frame_override=frame_override,
                    people_in_ad=people_in_ad,
                    style=style, ratio=ratio,
                )
                billed += 1
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

            art_dir.mkdir(parents=True, exist_ok=True)
            art_path.write_bytes(raw)

            # The likeness gate compares against the frame this render was
            # actually built from, which is the override after a blocked reroll.
            source_frame = frame_override or frames.get(variant.frame_id)
            gate = qa.likeness_gate(art_path, source_frame, client=client,
                                    people_in_ad=people_in_ad)
            with Image.open(art_path) as art:
                zone_busy = typeset.zone_is_busy(
                    art.convert("RGB"), typeset.zone_box(variant.treatment, ratio, art.size))

            if gate.ok and not zone_busy:
                path, notes = compose(art_path, headline, style, variant.treatment,
                                      ratio, out_path)
                return _done(path=path, art_path=art_path, unverified=gate.unverified,
                             flagged=bool(notes), note="; ".join(notes))

            problems = list(gate.problems)
            if zone_busy:
                problems.append("art painted into the text zone")
            last_note = "; ".join(problems)

            if attempt == 1:
                if not gate.ok and gate.likeness in ("DIFFERENT", "NOBODY"):
                    extra_instruction = (
                        f"The previous attempt failed verification: {last_note}. "
                        "You generated a person who is not in the reference "
                        "frames. Do not invent, replace or beautify anyone. Cut "
                        "the exact person out of the reference frame and place "
                        "them in front of the background, keeping their face, "
                        "age, hair, facial hair and clothing identical, with "
                        "their face large and clearly visible."
                    )
                    if zone_busy:
                        extra_instruction += " " + ZONE_REROLL_INSTRUCTION
                else:
                    extra_instruction = ZONE_REROLL_INSTRUCTION
                # No backoff: the API answered fine. Backoff exists for rate
                # limits and transport failures, not for a taste reroll.
                continue

            # Second failure: still hand it over, flagged. A zone that is still
            # busy gets a scrim under the type so the headline stays legible.
            path, notes = compose(art_path, headline, style, variant.treatment,
                                  ratio, out_path, scrim=zone_busy)
            return _done(path=path, art_path=art_path, flagged=True,
                         note="; ".join([last_note] + notes),
                         unverified=gate.unverified)
    except Exception as exc:
        # Anything unforeseen — a malformed payload, a PIL failure, a bug —
        # costs exactly one tile instead of the whole batch.
        log.warning("variant %s/%s failed unexpectedly", variant.index, ratio,
                    exc_info=True)
        return _done(path=None, art_path=art_path if art_path.exists() else None,
                     flagged=True,
                     note=f"something went wrong rendering this one ({exc})")

    return _done(path=None, flagged=True, note=last_note)


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
        row.style = outcome.style
        row.headline = outcome.headline
        row.render_calls += outcome.attempts
        row.images_billed += outcome.billed
        if outcome.path is not None:
            row.paths[outcome.ratio] = outcome.path
        if outcome.art_path is not None:
            row.art_paths[outcome.ratio] = outcome.art_path
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
    variant_count: int = DEFAULT_VARIANTS,
    sleeper=None,
) -> BatchOutcome:
    def say(message: str) -> None:
        log.info(message)
        if progress:
            progress(message)

    work_dir = Path(work_dir)
    frames_dir = work_dir / "frames"
    art_dir = work_dir / "art"
    out_dir = work_dir / "out"
    art_dir.mkdir(parents=True, exist_ok=True)
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
        sleeper=sleeper, variant_count=variant_count,
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
                job[0], job[1], frames, art_dir, out_dir, client, sleeper=sleeper,
                people_in_ad=batch_plan.people_in_ad,
            ),
            jobs,
        ))

    results = _group_by_variant(outcomes, batch_plan.variants)

    unverified = sum(1 for r in results if r.unverified)
    if unverified:
        warnings.append(
            f"{unverified} of {len(results)} thumbnails could not be "
            "checked for the actor's likeness — look at the faces yourself."
        )

    say("Done.")
    return BatchOutcome(
        plan=batch_plan, results=results, warnings=warnings,
        planned_renders=len(jobs),
        render_calls=sum(o.attempts for o in outcomes),
        images_billed=sum(o.billed for o in outcomes),
        frames=frames, people_in_ad=batch_plan.people_in_ad, work_dir=work_dir,
    )
