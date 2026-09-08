"""The headline as a layer. The only module that draws text.

The image model renders art with a reserved, calm zone; this module sets the
headline into that zone in Poppins Black, with the type treatment the style
calls for, and composites it down onto the art. Deterministic: same inputs,
same pixels. Nothing here calls an API.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter, ImageFont, ImageStat

from . import branding
from .config import (
    CAPTION_FONT,
    CAPTION_FONT_FRACTION,
    CREAM,
    HEADLINE_FALLBACK_FONT,
    HEADLINE_FONT,
    NAVY,
    ORANGE,
    PINK,
    TEXT_BOTTOM_RESERVE,
    TEXT_FLOOR_FRACTION,
    TEXT_SUPERSAMPLE,
    WHITE,
    ZONE_EDGE_THRESHOLD,
)

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class Zone:
    """Where the headline goes, as fractions of the canvas."""
    x0: float
    y0: float
    x1: float
    y1: float
    align: str          # "centre" | "left"


# 16:9 and 1:1 share the "wide" zone; 9:16 gets a "tall" zone across the top.
# Every zone stays above TEXT_BOTTOM_RESERVE so the logo has somewhere to land.
_WIDE: dict[str, Zone] = {
    "split_screen":    Zone(0.30, 0.12, 0.70, 0.60, "centre"),
    "face_closeup":    Zone(0.06, 0.10, 0.52, 0.70, "left"),
    "full_bleed":      Zone(0.15, 0.08, 0.85, 0.44, "centre"),
    "text_dominant":   Zone(0.06, 0.10, 0.62, 0.72, "left"),
    "product_forward": Zone(0.10, 0.08, 0.90, 0.30, "centre"),
}
_TALL: dict[str, Zone] = {
    "split_screen":    Zone(0.06, 0.08, 0.94, 0.36, "centre"),
    "face_closeup":    Zone(0.06, 0.08, 0.94, 0.40, "left"),
    "full_bleed":      Zone(0.06, 0.08, 0.94, 0.38, "centre"),
    "text_dominant":   Zone(0.06, 0.08, 0.94, 0.50, "left"),
    "product_forward": Zone(0.06, 0.08, 0.94, 0.30, "centre"),
}
_FAMILY: dict[str, dict[str, Zone]] = {"16x9": _WIDE, "1x1": _WIDE, "9x16": _TALL}

# Where the actor goes so the zone stays clear. Spoken to the model.
_ACTOR: dict[str, str] = {
    "split_screen": "one actor on each side, the seam of light between them",
    "face_closeup": "the actor's face fills the RIGHT half of the frame",
    "full_bleed": "the actors occupy the lower two-thirds of the frame",
    "text_dominant": "the actor stands offset to the RIGHT, still large",
    "product_forward": "the product surface and the actor sit BELOW the reserved band",
}


def zone_for(treatment: str, ratio: str) -> Zone:
    return _FAMILY[ratio][treatment]          # KeyError is a programming error


def zone_box(treatment: str, ratio: str, size: tuple[int, int]) -> tuple[int, int, int, int]:
    """The zone in pixels for a canvas of `size`, clamped above the logo band."""
    zone = zone_for(treatment, ratio)
    width, height = size
    y1 = min(zone.y1, 1.0 - TEXT_BOTTOM_RESERVE)
    return (
        round(width * zone.x0), round(height * zone.y0),
        round(width * zone.x1), round(height * y1),
    )


def _side(zone: Zone) -> str:
    if zone.x1 <= 0.55:
        return "left"
    if zone.x0 >= 0.45:
        return "right"
    return "centre" if (zone.x1 - zone.x0) < 0.8 else "full width"


def zone_instruction(treatment: str, ratio: str) -> str:
    """Prose for the render prompt: where to leave room, and that we add the text."""
    zone = zone_for(treatment, ratio)
    return (
        f"Reserve a calm area at the {_side(zone)} of the frame, from "
        f"{round(zone.x0 * 100)}% to {round(zone.x1 * 100)}% across and from "
        f"{round(zone.y0 * 100)}% to {round(zone.y1 * 100)}% down. Inside it: flat "
        "colour or a soft gradient only — no face, no hand, no object, no sparks, "
        "no logo, no pattern. The headline is added afterwards by us. You render "
        "NO text of any kind anywhere in the image: no words, letters, numbers, "
        f"captions, watermarks or UI. Composition: {_ACTOR[treatment]}."
    )


# --- fitting -----------------------------------------------------------------
class TypesetError(RuntimeError):
    """The headline could not be set at all (no usable font)."""


@dataclass
class Fit:
    font: ImageFont.FreeTypeFont
    lines: list[str]
    size: int
    fits: bool          # False when even the floor size overflows the box


def load_font(size: int) -> ImageFont.FreeTypeFont:
    """Poppins Black, else SemiBold, else a TypesetError naming both."""
    for path in (HEADLINE_FONT, HEADLINE_FALLBACK_FONT):
        try:
            return ImageFont.truetype(str(path), size)
        except OSError:
            log.warning("font %s not loadable", path)
    raise TypesetError(
        f"No headline font could be loaded (tried {HEADLINE_FONT.name} and "
        f"{HEADLINE_FALLBACK_FONT.name}); the fonts ship in assets/fonts/poppins."
    )


def floor_px(canvas_height: int) -> int:
    return round(canvas_height * TEXT_FLOOR_FRACTION)


def _cap_height(font: ImageFont.FreeTypeFont) -> int:
    bbox = font.getbbox("H")
    return bbox[3] - bbox[1]


def _wrap(words: list[str], font: ImageFont.FreeTypeFont, max_width: int) -> list[str]:
    """Greedy fill: as many words per line as fit the width."""
    lines: list[str] = []
    current: list[str] = []
    for word in words:
        candidate = " ".join(current + [word])
        if current and font.getlength(candidate) > max_width:
            lines.append(" ".join(current))
            current = [word]
        else:
            current.append(word)
    if current:
        lines.append(" ".join(current))
    return lines


def _block_height(font: ImageFont.FreeTypeFont, n_lines: int) -> int:
    cap = _cap_height(font)
    leading = round(cap * 1.18)          # tight display leading
    return cap + leading * (n_lines - 1)


def _font_at(font_path: Path, size: int) -> ImageFont.FreeTypeFont:
    # The configured headline face goes through the fallback chain; any other
    # face a caller names explicitly is loaded as-is.
    if Path(font_path) == HEADLINE_FONT:
        return load_font(size)
    return ImageFont.truetype(str(font_path), size)


def fit_headline(
    text: str, font_path: Path, box: tuple[int, int, int, int],
    floor: int, max_lines: int = 3,
) -> Fit:
    """The largest size at which the headline fits `box` in <= max_lines.

    Sizes step down by 4px from a generous start. If nothing fits at the floor,
    return the floor-size fit with fits=False so the caller can flag the tile;
    the text is still drawn — a flagged headline beats a missing one.
    """
    words = " ".join((text or "").upper().split()).split()
    x0, y0, x1, y1 = box
    max_w, max_h = x1 - x0, y1 - y0

    def _fits(font: ImageFont.FreeTypeFont, lines: list[str]) -> bool:
        return (len(lines) <= max_lines
                and all(font.getlength(line) <= max_w for line in lines)
                and _block_height(font, len(lines)) <= max_h)

    # Start where a single line would fill the height; nothing can be bigger.
    size = max(floor, max_h)
    while size > floor:
        font = _font_at(font_path, size)
        lines = _wrap(words, font, max_w)
        if _fits(font, lines):
            return Fit(font=font, lines=lines, size=size, fits=True)
        size -= 4

    font = _font_at(font_path, floor)
    lines = _wrap(words, font, max_w)
    return Fit(font=font, lines=lines, size=floor, fits=_fits(font, lines))


# --- type treatments, scrim, typeset() ------------------------------------------
@dataclass(frozen=True)
class TypeTreatment:
    """How a style sets its headline. Mirrors the retired TYPE clauses of STYLE_BRIEF."""
    fill: tuple[int, int, int]
    stroke: tuple[int, int, int] | None
    stroke_frac: float                          # of font size
    shadow: tuple[int, int, int, int] | None    # RGBA
    shadow_frac: float                          # offset, of font size
    adaptive: bool = False                      # flat_graphic: fill from the zone's luminance


TYPE_TREATMENTS: dict[str, TypeTreatment] = {
    "house_energy":    TypeTreatment(WHITE, NAVY, 0.07, (0, 0, 0, 166), 0.06),
    "dark_cinematic":  TypeTreatment(CREAM, None, 0.0, None, 0.0),
    "flat_graphic":    TypeTreatment(NAVY, None, 0.0, None, 0.0, adaptive=True),
    "clean_corporate": TypeTreatment(NAVY, None, 0.0, None, 0.0),
}


@dataclass(frozen=True)
class PillTreatment:
    """The caption pill per style: fill, text colour, optional edge."""
    fill: tuple[int, int, int]
    text: tuple[int, int, int]
    edge: tuple[int, int, int] | None = None


PILL_TREATMENTS: dict[str, PillTreatment] = {
    "house_energy":    PillTreatment(CREAM, NAVY, PINK),       # the approved refs' look
    "dark_cinematic":  PillTreatment(NAVY, CREAM),
    "flat_graphic":    PillTreatment(ORANGE, NAVY),
    "clean_corporate": PillTreatment(NAVY, CREAM),
}


@dataclass
class TypesetResult:
    image: Image.Image
    notes: list[str]
    scrimmed: bool = False
    overflowed: bool = False
    # Where things landed, in art pixels. For tests and for the UI's overlays.
    headline_box: tuple[int, int, int, int] | None = None
    caption_box: tuple[int, int, int, int] | None = None


def edge_energy(image: Image.Image, box: tuple[int, int, int, int]) -> float:
    """Mean edge response over the region. Near zero for flat colour AND for
    smooth gradients; high wherever there are faces, sparks, type or texture."""
    region = image.convert("L").crop(box)
    if not region.width or not region.height:
        return float("inf")
    return ImageStat.Stat(region.filter(ImageFilter.FIND_EDGES)).mean[0]


def zone_is_busy(image: Image.Image, box: tuple[int, int, int, int]) -> bool:
    """Did the model paint detail into the reserved zone?

    Edge energy only. branding.busy_score adds luminance spread, which is right
    for a logo corner but wrong here: a calm gradient has a large spread.
    """
    return edge_energy(image, box) > ZONE_EDGE_THRESHOLD


def _draw_scrim(canvas: Image.Image, box: tuple[int, int, int, int], dark: bool) -> None:
    """A soft plate under the type, in place. Dark under light type, light under dark."""
    x0, y0, x1, y1 = box
    pad = round((y1 - y0) * 0.08)
    colour = (*NAVY, 150) if dark else (*CREAM, 170)
    overlay = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
    ImageDraw.Draw(overlay).rounded_rectangle(
        (max(0, x0 - pad), max(0, y0 - pad),
         min(canvas.width, x1 + pad), min(canvas.height, y1 + pad)),
        radius=max(8, pad), fill=colour,
    )
    canvas.alpha_composite(overlay)


def _pill_metrics(canvas_height: int) -> tuple[int, int, int]:
    """(font size, pill height, gap above the pill), all in 1x art pixels."""
    size = max(12, round(canvas_height * CAPTION_FONT_FRACTION))
    pill_h = round(size * 1.9)
    gap = round(size * 0.9)
    return size, pill_h, gap


def _load_caption_font(size: int) -> ImageFont.FreeTypeFont:
    try:
        return ImageFont.truetype(str(CAPTION_FONT), size)
    except OSError:
        log.warning("caption font %s not loadable; using the headline face", CAPTION_FONT)
        return load_font(size)


def typeset(
    art: Image.Image, headline: str, style: str, treatment: str, ratio: str,
    *, caption: str | None = None, scrim: bool = False,
) -> TypesetResult:
    """Set `headline` into the zone on a copy of `art`, at the art's own size.

    Text is drawn on a transparent layer at TEXT_SUPERSAMPLE x the art's size and
    resampled down before compositing, so edges are anti-aliased. The art itself
    is never resampled here — the single downscale to delivery size happens
    later, in postprocess.finalize_image().
    """
    canvas = art.convert("RGBA")
    box = zone_box(treatment, ratio, canvas.size)
    zone = zone_for(treatment, ratio)
    look = TYPE_TREATMENTS[style]
    notes: list[str] = []

    fill = look.fill
    if look.adaptive:
        fill = NAVY if branding.region_is_light(canvas, box) else CREAM

    if scrim:
        dark_type = sum(fill) < 384
        _draw_scrim(canvas, box, dark=not dark_type)
        notes.append("set over a scrim — art ignored the text zone")

    # A caption takes a fixed slice off the bottom of the zone before the
    # headline is fitted, so the pair always fits together.
    caption = " ".join((caption or "").split()) or None
    head_box = box
    if caption:
        _c_size, pill_h, gap = _pill_metrics(canvas.height)
        head_box = (box[0], box[1], box[2], max(box[1] + 1, box[3] - pill_h - gap))

    fit = fit_headline(headline, HEADLINE_FONT, head_box, floor=floor_px(canvas.height))
    if not fit.fits:
        notes.append("headline too long for this layout")

    s = TEXT_SUPERSAMPLE
    layer = Image.new("RGBA", (canvas.width * s, canvas.height * s), (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer)
    font = load_font(fit.size * s)
    cap = _cap_height(font)
    leading = round(cap * 1.18)
    stroke_w = round(fit.size * s * look.stroke_frac) if look.stroke else 0
    shadow_d = round(fit.size * s * look.shadow_frac) if look.shadow else 0
    top_offset = font.getbbox("H")[1]       # keeps the cap where we measured it

    bx0, by0, bx1, by1 = (v * s for v in head_box)
    block_h = cap + leading * (len(fit.lines) - 1)
    y = by0 + max(0, ((by1 - by0) - block_h) // 2)
    block_top = y
    for line in fit.lines:
        w = font.getlength(line)
        x = bx0 if zone.align == "left" else bx0 + ((bx1 - bx0) - w) / 2
        if look.shadow:
            draw.text((x + shadow_d, y - top_offset + shadow_d), line, font=font,
                      fill=look.shadow, stroke_width=stroke_w, stroke_fill=look.shadow)
        draw.text((x, y - top_offset), line, font=font, fill=(*fill, 255),
                  stroke_width=stroke_w,
                  stroke_fill=(*look.stroke, 255) if look.stroke else None)
        y += leading
    block_bottom = block_top + block_h
    headline_box = (round(bx0 / s), round(block_top / s), round(bx1 / s), round(block_bottom / s))

    caption_box = None
    if caption:
        c_size, pill_h, gap = _pill_metrics(canvas.height)
        pill = PILL_TREATMENTS[style]
        cfont = _load_caption_font(c_size * s)
        tb = cfont.getbbox(caption)
        text_w, text_h = tb[2] - tb[0], tb[3] - tb[1]
        pad_x = round(c_size * s * 0.7)
        ph = pill_h * s
        pw = text_w + 2 * pad_x
        # Below the headline block, aligned like it, clamped to the zone.
        zx0, zy0, zx1, zy1 = (v * s for v in box)
        py0 = min(block_bottom + gap * s, zy1 - ph)
        px0 = zx0 if zone.align == "left" else zx0 + ((zx1 - zx0) - pw) / 2
        px0 = max(zx0, min(px0, zx1 - pw))
        edge_w = round(ph * 0.09) if pill.edge else 0
        draw.rounded_rectangle((px0, py0, px0 + pw, py0 + ph), radius=ph / 2,
                               fill=(*pill.fill, 255),
                               outline=(*pill.edge, 255) if pill.edge else None,
                               width=edge_w)
        draw.text((px0 + pad_x - tb[0], py0 + (ph - text_h) / 2 - tb[1]), caption,
                  font=cfont, fill=(*pill.text, 255))
        caption_box = (round(px0 / s), round(py0 / s), round((px0 + pw) / s), round((py0 + ph) / s))

    layer = layer.resize(canvas.size, Image.LANCZOS)
    canvas.alpha_composite(layer)
    return TypesetResult(image=canvas.convert("RGB"), notes=notes,
                         scrimmed=scrim, overflowed=not fit.fits,
                         headline_box=headline_box, caption_box=caption_box)
