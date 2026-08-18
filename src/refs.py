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


def winner_refs(treatment: str | None = None) -> list[Path]:
    winners = _images(REFS_WINNERS_DIR)
    if treatment is None:
        return winners
    return [p for p in winners if p.name.startswith(f"{treatment}__")]


def pick_refs(treatment: str, limit: int = 3) -> list[Path]:
    """References for one render call: matching winners first, style always."""
    style = style_refs()
    if not style:
        raise FileNotFoundError(
            f"No style references found in {REFS_STYLE_DIR}. Copy the approved "
            "thumbnails there before rendering."
        )

    picked: list[Path] = [style[0]]                    # drift guard
    for candidate in winner_refs(treatment) + style[1:] + winner_refs():
        if len(picked) >= limit:
            break
        if candidate not in picked:
            picked.append(candidate)
    return picked[:limit]


def save_winner(image_path: Path, treatment: str) -> Path:
    REFS_WINNERS_DIR.mkdir(parents=True, exist_ok=True)
    dest = REFS_WINNERS_DIR / f"{treatment}__{uuid.uuid4().hex[:8]}{Path(image_path).suffix}"
    shutil.copy2(image_path, dest)
    return dest
