"""The headline as a layer. The only module that draws text.

The image model renders art with a reserved, calm zone; this module sets the
headline into that zone in Poppins Black, with the type treatment the style
calls for, and composites it down onto the art. Deterministic: same inputs,
same pixels. Nothing here calls an API.
"""

from __future__ import annotations

from dataclasses import dataclass

from .config import TEXT_BOTTOM_RESERVE


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
