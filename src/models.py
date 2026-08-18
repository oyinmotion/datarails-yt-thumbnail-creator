"""Plan data model. Pure data — no I/O, no API calls.

The hook/treatment pairing is fixed in code, not chosen by the model, so every
batch spans the space instead of clustering on one idea.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, field_validator

HookType = Literal["stat", "question", "conflict", "pain", "outcome"]
Treatment = Literal[
    "split_screen", "face_closeup", "full_bleed", "text_dominant",
    "product_forward",
]

# (index, hook_type, treatment) — locked pairing.
MATRIX: list[tuple[int, HookType, Treatment]] = [
    (1, "stat", "split_screen"),
    (2, "question", "face_closeup"),
    (3, "conflict", "full_bleed"),
    (4, "pain", "text_dominant"),
    (5, "outcome", "product_forward"),
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

MAX_HEADLINE_WORDS = 5


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

        actual = [(v.index, v.hook_type, v.treatment) for v in self.variants]
        if sorted(actual) != sorted(MATRIX):
            raise ValueError(
                f"plan does not follow the locked matrix.\n"
                f"expected: {sorted(MATRIX)}\ngot:      {sorted(actual)}"
            )
