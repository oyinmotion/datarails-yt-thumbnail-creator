"""Stamp the Datarails logo onto a finished thumbnail.

Deliberately NOT done by the image model. gpt-image-2 renders invented,
mangled versions of real wordmarks, and prompts/render.md forbids it from
drawing one at all. Compositing the real file is the only way to get the actual
mark, at a predictable size, every time.

The logo must sit on flat colour — never over texture, type or graphics. A fixed
corner cannot promise that, because what is in a corner changes with every
render. So each candidate position is scored for flatness and the calmest one
wins; if nothing on the image is calm enough, a solid brand-coloured plate is
drawn to make a flat area that did not exist.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter, ImageStat

from .config import (
    LOGO_BUSY_THRESHOLD,
    LOGO_CORNERS,
    LOGO_DARK,
    LOGO_LIGHT,
    LOGO_MARGIN_FRACTION,
    LOGO_PLATE_ALPHA,
    LOGO_PLATE_DARK,
    LOGO_PLATE_LIGHT,
    LOGO_PLATE_PAD_FRACTION,
    LOGO_WIDTH_FRACTION,
)

log = logging.getLogger(__name__)

# Above this mean luminance the area counts as light, so the dark logo goes on.
LIGHT_CORNER_THRESHOLD = 140

# Edge energy dominates the busy score: a photo of a wall and a wall of small
# text can share a luminance spread, but only one is full of edges.
EDGE_WEIGHT = 2.0


@dataclass
class LogoPlacement:
    corner: str
    box: tuple[int, int, int, int]
    score: float
    needs_plate: bool
    on_light: bool


def _positions(
    width: int, height: int, logo_w: int, logo_h: int, margin: int
) -> dict[str, tuple[int, int]]:
    centre_x = (width - logo_w) // 2
    return {
        "bottom-left": (margin, height - logo_h - margin),
        "top-left": (margin, margin),
        "top-right": (width - logo_w - margin, margin),
        "bottom-centre": (centre_x, height - logo_h - margin),
        "top-centre": (centre_x, margin),
        "bottom-right": (width - logo_w - margin, height - logo_h - margin),
    }


def busy_score(image: Image.Image, box: tuple[int, int, int, int]) -> float:
    """How much is going on in this region. Lower is flatter.

    Combines luminance spread with edge energy. Flat colour scores near zero;
    texture, type and graphics all score high, which is what we need to avoid.
    """
    region = image.convert("L").crop(box)
    if not region.width or not region.height:
        return float("inf")
    spread = ImageStat.Stat(region).stddev[0]
    edges = ImageStat.Stat(region.filter(ImageFilter.FIND_EDGES)).mean[0]
    return spread + EDGE_WEIGHT * edges


def region_is_light(image: Image.Image, box: tuple[int, int, int, int]) -> bool:
    """Is the region light or dark? Decides which logo variant to use."""
    region = image.convert("L").crop(box)
    if not region.width or not region.height:
        return False
    return ImageStat.Stat(region).mean[0] > LIGHT_CORNER_THRESHOLD


# Kept under the old name so existing callers and tests keep working.
corner_is_light = region_is_light


def plan_placement(
    image: Image.Image, logo_w: int, logo_h: int, margin: int
) -> LogoPlacement:
    """Pick the flattest candidate, and say whether it still needs a plate."""
    width, height = image.size
    positions = _positions(width, height, logo_w, logo_h, margin)

    scored: list[tuple[float, int, str, tuple[int, int, int, int]]] = []
    for preference, corner in enumerate(LOGO_CORNERS):
        if corner not in positions:
            continue
        x, y = positions[corner]
        box = (x, y, x + logo_w, y + logo_h)
        # Preference index breaks ties, so equally flat corners resolve to the
        # documented order rather than to dict iteration order.
        scored.append((busy_score(image, box), preference, corner, box))

    scored.sort(key=lambda item: (round(item[0], 3), item[1]))
    score, _preference, corner, box = scored[0]

    if score > LOGO_BUSY_THRESHOLD:
        # Nothing is naturally flat, so a plate is going down regardless — and a
        # plate flattens whatever is underneath it. Flatness has stopped being a
        # reason to move, so honour the layout preference instead and keep the
        # mark out of the corner YouTube stamps its duration badge on.
        score, _preference, corner, box = min(scored, key=lambda item: item[1])

    return LogoPlacement(
        corner=corner,
        box=box,
        score=score,
        needs_plate=score > LOGO_BUSY_THRESHOLD,
        on_light=region_is_light(image, box),
    )


def _draw_plate(
    base: Image.Image, box: tuple[int, int, int, int], on_light: bool
) -> None:
    """Lay a solid brand-coloured plate behind the logo, in place."""
    x0, y0, x1, y1 = box
    pad = max(2, round((y1 - y0) * LOGO_PLATE_PAD_FRACTION))
    plate_box = (
        max(0, x0 - pad), max(0, y0 - pad),
        min(base.width, x1 + pad), min(base.height, y1 + pad),
    )
    colour = LOGO_PLATE_LIGHT if on_light else LOGO_PLATE_DARK
    radius = max(4, round((y1 - y0) * 0.35))

    overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
    ImageDraw.Draw(overlay).rounded_rectangle(
        plate_box, radius=radius, fill=(*colour, LOGO_PLATE_ALPHA)
    )
    base.alpha_composite(overlay)


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

    # Probe with the light logo's proportions; both variants are the same shape.
    with Image.open(LOGO_LIGHT) as probe:
        target_h = max(1, round(probe.height * target_w / probe.width))

    placement = plan_placement(base, target_w, target_h, margin)

    if placement.needs_plate:
        # Nowhere on this render is flat enough, so make somewhere flat. The
        # plate goes down first, then the region is re-measured: the logo
        # variant must suit the PLATE it now sits on, not the art underneath.
        _draw_plate(base, placement.box, placement.on_light)
        on_light = region_is_light(base, placement.box)
        log.info("logo plated at %s (busy score %.1f)",
                 placement.corner, placement.score)
    else:
        on_light = placement.on_light
        log.info("logo placed bare at %s (busy score %.1f)",
                 placement.corner, placement.score)

    logo_path = LOGO_DARK if on_light else LOGO_LIGHT
    with Image.open(logo_path) as opened_logo:
        logo = opened_logo.convert("RGBA")
        logo_h = max(1, round(logo.height * target_w / logo.width))
        logo = logo.resize((target_w, logo_h), Image.LANCZOS)

    base.alpha_composite(logo, (placement.box[0], placement.box[1]))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    base.convert("RGB").save(out_path, "PNG", optimize=True)
    return out_path
