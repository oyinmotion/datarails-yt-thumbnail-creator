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
        "Typography carries the frame; the actor is smaller and offset to one "
        "side. The headline is the subject."
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
        "scroll. Background splits deep navy blue against vivid orange with hot "
        "white light where they meet, carrying embers, sparks or light rays. "
        "Cinematic dramatic lighting. The people are cut out cleanly with a "
        "subtle light rim, standing in front of the scene rather than inside "
        "it. Headline in a heavy condensed sans, all caps, pure white with a "
        "thick dark outline and a hard drop shadow."
    ),
    "dark_cinematic": (
        "Restrained and expensive, like a prestige drama poster. Near-black "
        "background with one hard key light raking across from the side and a "
        "deep falloff into shadow. No colour split, no embers, no sparks, no "
        "glow effects. A single warm orange accent only — the key light itself "
        "or one thin rule. The person is lit naturally within the darkness "
        "with a faint rim separating them from it. Headline in a heavy "
        "condensed sans, all caps, off-white, tight tracking, NO outline: "
        "separation comes from the surrounding darkness. Quiet, not loud."
    ),
    "flat_graphic": (
        "A flat editorial poster, not a photograph. The background is two or "
        "three solid flat colour fields — deep navy, vivid orange, off-white — "
        "with hard geometric edges. No gradients, no glow, no sparks, no "
        "photographic scenery of any kind. The person is cut out and placed on "
        "the colour fields like a sticker, with a crisp offset shadow. "
        "Typography is the design: enormous, all caps, in a heavy grotesque, "
        "either navy on the orange field or knocked out to off-white, aligned "
        "hard to the layout's grid. Swiss poster discipline."
    ),
    "clean_corporate": (
        "Calm software credibility, like a SaaS landing page hero. Light, airy "
        "background in near-white or very pale grey with soft even studio "
        "light. No drama, no sparks, no dark vignette, no rim light. The "
        "person is naturally lit and looks like a real professional in a real "
        "room. Generous margins and real breathing space. Headline in a "
        "medium-heavy sans, all caps, deep navy on the light background, with "
        "no outline and no shadow — contrast alone carries it. One vivid "
        "orange accent at most."
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
