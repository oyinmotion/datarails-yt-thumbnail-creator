"""The house style pack.

refs/style/ is locked: the four approved thumbnails, always contributing at
least one reference to every render call. refs/winners/ grows as the team
approves outputs, which tightens the tool's taste over time. The guarantee that
a locked ref is always present is what bounds style drift.
"""

from __future__ import annotations

import shutil
import uuid
from pathlib import Path

from .config import REFS_STYLE_DIR, REFS_WINNERS_DIR
from .models import HOUSE_STYLE

_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp"}


def _images(directory: Path) -> list[Path]:
    if not directory.exists():
        return []
    return sorted(
        p for p in directory.iterdir()
        if p.suffix.lower() in _IMAGE_SUFFIXES
    )


def style_refs() -> list[Path]:
    return _images(REFS_STYLE_DIR)


def winner_refs(style: str | None = None, treatment: str | None = None) -> list[Path]:
    """Approved outputs, optionally narrowed by style and then treatment.

    Winners are named "{style}__{treatment}__{id}", so a winner only ever
    informs renders of its OWN style. A dark-cinematic winner must never be
    handed to a clean-corporate render.
    """
    winners = _images(REFS_WINNERS_DIR)
    if style is not None:
        winners = [p for p in winners if p.name.startswith(f"{style}__")]
    if treatment is not None:
        winners = [p for p in winners if f"__{treatment}__" in p.name]
    return winners


def pick_refs(
    style: str, treatment: str, limit: int = 3, people_in_ad: bool = True
) -> list[Path]:
    """References for one render call.

    For the house style, a locked reference is ALWAYS included — that is the
    drift guard that keeps the proven look proven.

    For every other style, no house reference is sent at all. The locked pack is
    four images of one high-contrast orange-and-blue treatment, so including one
    would drag a dark-cinematic or flat-graphic render straight back into the
    house look — which is the exact problem this style axis exists to solve.
    Those styles start prompt-only and accumulate their own references as the
    team approves outputs.
    """
    if not people_in_ad:
        # Every image in refs/style/ is a finished thumbnail, and every one
        # contains the two actors. Handing one to a motion-graphic ad is how a
        # stranger from a different shoot ends up in the output. Winners are
        # skipped for the same reason: they were approved for people-led ads.
        return []

    same_style_winners = winner_refs(style, treatment) + winner_refs(style)

    if style != HOUSE_STYLE:
        picked: list[Path] = []
        for candidate in same_style_winners:
            if len(picked) >= limit:
                break
            if candidate not in picked:
                picked.append(candidate)
        return picked

    locked = style_refs()
    if not locked:
        raise FileNotFoundError(
            f"No style references found in {REFS_STYLE_DIR}. Copy the approved "
            "thumbnails there before rendering."
        )

    picked = [locked[0]]                               # drift guard
    for candidate in same_style_winners + locked[1:]:
        if len(picked) >= limit:
            break
        if candidate not in picked:
            picked.append(candidate)
    return picked[:limit]


def save_winner(image_path: Path, style: str, treatment: str) -> Path:
    REFS_WINNERS_DIR.mkdir(parents=True, exist_ok=True)
    dest = (
        REFS_WINNERS_DIR
        / f"{style}__{treatment}__{uuid.uuid4().hex[:8]}{Path(image_path).suffix}"
    )
    shutil.copy2(image_path, dest)
    return dest
