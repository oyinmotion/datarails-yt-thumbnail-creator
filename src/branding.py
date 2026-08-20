"""Stamp the Datarails logo onto a finished thumbnail.

Deliberately NOT done by the image model. gpt-image-2 renders invented,
mangled versions of real wordmarks, and prompts/render.md forbids it from
drawing one at all. Compositing the real file is the only way to get the actual
mark, at a predictable size, in a predictable place, every time.
"""

from __future__ import annotations

import logging
from pathlib import Path

from PIL import Image, ImageStat

from .config import (
    LOGO_CORNER,
    LOGO_DARK,
    LOGO_LIGHT,
    LOGO_MARGIN_FRACTION,
    LOGO_WIDTH_FRACTION,
)

log = logging.getLogger(__name__)

# Above this mean luminance the corner counts as light, so the dark logo goes on.
LIGHT_CORNER_THRESHOLD = 140


def _corner_box(
    width: int, height: int, logo_w: int, logo_h: int, margin: int
) -> tuple[int, int]:
    """Top-left pixel where the logo goes, for the configured corner."""
    if LOGO_CORNER == "bottom-left":
        return margin, height - logo_h - margin
    if LOGO_CORNER == "bottom-right":
        return width - logo_w - margin, height - logo_h - margin
    if LOGO_CORNER == "top-left":
        return margin, margin
    if LOGO_CORNER == "top-right":
        return width - logo_w - margin, margin
    raise ValueError(f"unknown LOGO_CORNER: {LOGO_CORNER!r}")


def corner_is_light(image: Image.Image, box: tuple[int, int, int, int]) -> bool:
    """Is the region where the logo will sit light or dark?

    Decides which logo variant to use. A white wordmark on the clean-corporate
    style's near-white background would be invisible; this is what stops that.
    """
    region = image.convert("L").crop(box)
    if not region.width or not region.height:
        return False
    # ImageStat rather than getdata(), which Pillow 14 removes.
    return ImageStat.Stat(region).mean[0] > LIGHT_CORNER_THRESHOLD


def stamp_logo(image_path: Path, out_path: Path | None = None) -> Path:
    """Composite the logo onto the image. Returns the path written.

    Missing logo files are not fatal: a thumbnail without a logo is still a
    usable thumbnail, and losing a paid render over a missing asset would be
    worse. It logs and returns the image untouched.
    """
    image_path = Path(image_path)
    out_path = Path(out_path) if out_path else image_path

    if not LOGO_LIGHT.exists() or not LOGO_DARK.exists():
        log.warning("logo assets missing (%s, %s); leaving the image unbranded",
                    LOGO_LIGHT, LOGO_DARK)
        return image_path

    with Image.open(image_path) as opened:
        base = opened.convert("RGBA")

    width, height = base.size
    margin = max(1, round(width * LOGO_MARGIN_FRACTION))
    target_w = max(1, round(width * LOGO_WIDTH_FRACTION))

    # Measure the corner BEFORE pasting, using a probe the size the logo will be.
    with Image.open(LOGO_LIGHT) as probe:
        probe_h = max(1, round(probe.height * target_w / probe.width))
    x, y = _corner_box(width, height, target_w, probe_h, margin)
    light = corner_is_light(base, (x, y, x + target_w, y + probe_h))

    logo_path = LOGO_DARK if light else LOGO_LIGHT
    with Image.open(logo_path) as opened_logo:
        logo = opened_logo.convert("RGBA")
        logo_h = max(1, round(logo.height * target_w / logo.width))
        logo = logo.resize((target_w, logo_h), Image.LANCZOS)

    x, y = _corner_box(width, height, target_w, logo_h, margin)
    base.alpha_composite(logo, (x, y))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    base.convert("RGB").save(out_path, "PNG", optimize=True)
    return out_path
