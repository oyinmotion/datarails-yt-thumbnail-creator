"""Plan data model. Pure data — no I/O, no API calls.

The hook/treatment/style pairing is fixed in code, not chosen by the model, so
every batch spans the space instead of clustering on one idea.

Style is deliberately NOT a field on Variant. It is derived from the slot index
via style_for(), so the planner never sees it, cannot get it wrong, and no new
key enters the structured-output schema. See the comment on Variant.index.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, field_validator

HookType = Literal["stat", "question", "conflict", "pain", "outcome"]
Treatment = Literal[
    "split_screen", "face_closeup", "full_bleed", "text_dominant",
    "product_forward",
]
Style = Literal[
    "house_energy", "dark_cinematic", "flat_graphic", "clean_corporate",
]

# The proven look. Its locked reference pack is always sent with it; the other
# styles deliberately get no house references at all, or they collapse back
# into this one. See refs.pick_refs().
HOUSE_STYLE: Style = "house_energy"

# (index, hook_type, treatment, style) — locked pairing.
# house_energy appears twice, on the two slots it suits best, so every batch
# carries two safe-to-ship options plus three genuine alternatives.
MATRIX: list[tuple[int, HookType, Treatment, Style]] = [
    (1, "stat", "split_screen", "house_energy"),
    (2, "question", "face_closeup", "dark_cinematic"),
    (3, "conflict", "full_bleed", "house_energy"),
    (4, "pain", "text_dominant", "flat_graphic"),
    (5, "outcome", "product_forward", "clean_corporate"),
]

TREATMENT_BRIEF: dict[str, str] = {
    "split_screen": (
        "Both actors face off, one on each side, a hard vertical seam of light "
        "between them. Headline centered on the seam."
    ),
    "face_closeup": (
        "One actor's face fills roughly half the frame with a clear reaction. "
        "Headline stacked in the remaining space."
    ),
    "full_bleed": (
        "A single dramatic energy burst fills the frame behind both actors. "
        "Headline centered and dominant."
    ),
    "text_dominant": (
        "Typography carries the frame and the headline is the loudest element, "
        "set very large. The actor is offset to one side rather than centred — "
        "but still close, still big enough that their face and expression read "
        "clearly at thumbnail size. Type dominance means the words are bigger, "
        "not that the person is small or turned away."
    ),
    "product_forward": (
        "The FinanceOS product surface or its mark is visible and legible, with "
        "one actor presenting it. Headline supports rather than competes."
    ),
}

# Each brief must fully specify palette, lighting, subject treatment AND type
# treatment, because prompts/render.md no longer states any of them globally —
# that global block was what made all five renders look identical.
STYLE_BRIEF: dict[str, str] = {
    "house_energy": (
        "The proven Datarails look. Extremely high contrast, built to stop a "
        "scroll. BACKGROUND: splits deep navy blue against vivid orange with hot "
        "white light where they meet, carrying embers, sparks or light rays, lit "
        "cinematically. SUBJECT: the cut-out people from the footage stand in "
        "front of it with a subtle light rim separating them. TYPE: heavy "
        "condensed sans, all caps, pure white with a thick dark outline and a "
        "hard drop shadow."
    ),
    "dark_cinematic": (
        "Restrained and expensive, like a prestige drama poster. BACKGROUND: "
        "near-black, with one hard warm key light raking in from the side and a "
        "deep falloff into shadow. No colour split, no embers, no sparks, no "
        "glow effects. A single orange accent at most. SUBJECT: the cut-out "
        "person from the footage, placed against that darkness with a faint warm "
        "rim light along one edge so they separate from it — their face stays "
        "bright enough to read clearly and is NOT lost in shadow. TYPE: heavy "
        "condensed sans, all caps, off-white, tight tracking, no outline — "
        "separation comes from the darkness behind it."
    ),
    "flat_graphic": (
        "A bold flat-colour treatment. BACKGROUND: two or three solid flat "
        "colour fields — deep navy, vivid orange, off-white — with hard "
        "geometric edges. No gradients, no glow, no sparks, no photographic "
        "scenery. SUBJECT: the cut-out person from the footage sits on those "
        "colour fields with a crisp offset shadow, kept LARGE in the frame with "
        "their expression fully readable — this is a thumbnail, not a minimal "
        "print poster, so never shrink them or turn them away. TYPE: very large, "
        "all caps, heavy grotesque, navy on the orange field or knocked out to "
        "off-white, aligned hard to the layout."
    ),
    "clean_corporate": (
        "Calm software credibility, bright and modern. BACKGROUND: a clean, "
        "light, near-white or very pale grey field with a soft even wash of "
        "light. No drama, no sparks, no dark vignette. SUBJECT: the cut-out "
        "person from the footage, placed on that light background with a soft "
        "contact shadow so they separate from it. They keep the clothing they "
        "are wearing in the footage — do not put them in different clothes, a "
        "different setting, an office, or at a desk, and do not replace them "
        "with anyone else. TYPE: heavy sans, all caps, deep navy on the light "
        "background, no outline and no shadow — contrast alone carries it. One "
        "vivid orange accent at most."
    ),
}

MAX_HEADLINE_WORDS = 5


def style_for(index: int) -> Style:
    """The style locked to a slot. Derived, never model-chosen."""
    for row_index, _hook, _treatment, style in MATRIX:
        if row_index == index:
            return style
    raise ValueError(f"no style for slot {index}; valid slots are 1-{len(MATRIX)}")


class Variant(BaseModel):
    # No Field(ge=..., le=...) here, and no Field(min_length=...) on
    # BatchPlan.variants: both emit JSON Schema keywords (minimum/maximum,
    # minItems/maxItems) that the strict structured-output schema may reject,
    # which would 400 every planner call. The same rules are enforced in
    # Python by BatchPlan.validate_matrix() instead. field_validators are safe
    # — they run after parsing and emit no schema keywords.
    index: int
    hook_type: HookType
    treatment: Treatment
    headline: str
    frame_id: str
    second_frame_id: str | None = None
    scene_direction: str
    rationale: str

    @property
    def style(self) -> Style:
        """The visual style locked to this slot. Not a model-supplied field."""
        return style_for(self.index)

    @field_validator("headline")
    @classmethod
    def clean_headline(cls, v: str) -> str:
        v = " ".join(v.split()).rstrip(".").strip().upper()
        if not v:
            raise ValueError("headline cannot be empty")
        if len(v.split()) > MAX_HEADLINE_WORDS:
            raise ValueError(
                f"headline must be {MAX_HEADLINE_WORDS} words or fewer, got "
                f"{len(v.split())}: {v!r}"
            )
        return v


class BatchPlan(BaseModel):
    ad_summary: str
    transcript_used: bool
    variants: list[Variant]

    def validate_matrix(self) -> None:
        """Fail loudly if the model drifted off the locked pairing.

        This is the *only* guard on variant count and index range — see the
        comment on Variant.index for why they cannot be schema constraints.
        """
        if len(self.variants) != len(MATRIX):
            raise ValueError(
                f"plan does not follow the locked matrix: expected exactly "
                f"{len(MATRIX)} variants, got {len(self.variants)}"
            )

        indexes = sorted(v.index for v in self.variants)
        expected_indexes = sorted(row[0] for row in MATRIX)
        if indexes != expected_indexes:
            raise ValueError(
                f"plan does not follow the locked matrix: indexes must be "
                f"{expected_indexes}, got {indexes}"
            )

        # Compare only the planner-facing columns. Style is derived from the
        # slot, so the model is never asked for it and cannot violate it.
        actual = [(v.index, v.hook_type, v.treatment) for v in self.variants]
        expected = [(row[0], row[1], row[2]) for row in MATRIX]
        if sorted(actual) != sorted(expected):
            raise ValueError(
                f"plan does not follow the locked matrix.\n"
                f"expected: {sorted(expected)}\ngot:      {sorted(actual)}"
            )
