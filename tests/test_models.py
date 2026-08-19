import json

import pytest
from openai.lib._pydantic import to_strict_json_schema
from pydantic import ValidationError

from src.models import (
    HOUSE_STYLE,
    MATRIX,
    STYLE_BRIEF,
    BatchPlan,
    Variant,
    style_for,
)


def _variant(index=1, hook="stat", treatment="split_screen", headline="47K OVER"):
    return Variant(
        index=index,
        hook_type=hook,
        treatment=treatment,
        headline=headline,
        frame_id="scene_004.jpg",
        second_frame_id=None,
        scene_direction="orange versus blue split, sparks at the seam",
        rationale="the ad names a budget overrun",
    )


def _plan(variants=None):
    return BatchPlan(
        ad_summary="two colleagues get different answers from the same AI",
        transcript_used=True,
        variants=variants or [_variant(i, h, t) for i, h, t, _s in MATRIX],
    )


def test_matrix_has_five_unique_rows():
    assert len(MATRIX) == 5
    assert len({row[1] for row in MATRIX}) == 5   # 5 distinct hooks
    assert len({row[2] for row in MATRIX}) == 5   # 5 distinct treatments
    assert [row[0] for row in MATRIX] == [1, 2, 3, 4, 5]


def test_valid_plan_passes_matrix_validation():
    _plan().validate_matrix()


def test_plan_with_four_variants_is_rejected():
    # Enforced in validate_matrix(), not by Field(min_length=...): a length
    # constraint would put minItems/maxItems in the strict schema.
    with pytest.raises(ValueError, match="matrix"):
        _plan(variants=[_variant(i, h, t) for i, h, t, _s in MATRIX[:4]]).validate_matrix()


def test_plan_with_six_variants_is_rejected():
    six = [_variant(i, h, t) for i, h, t, _s in MATRIX] + [_variant(1, "stat", "split_screen")]
    with pytest.raises(ValueError, match="matrix"):
        _plan(variants=six).validate_matrix()


def test_plan_with_an_out_of_range_index_is_rejected():
    bad = [_variant(i, h, t) for i, h, t, _s in MATRIX]
    bad[0].index = 9
    with pytest.raises(ValueError, match="matrix"):
        _plan(variants=bad).validate_matrix()


def test_plan_with_a_duplicated_index_is_rejected():
    bad = [_variant(i, h, t) for i, h, t, _s in MATRIX]
    bad[1].index = 1
    with pytest.raises(ValueError, match="matrix"):
        _plan(variants=bad).validate_matrix()


def test_strict_schema_emits_no_length_or_range_keywords():
    """Regression guard: these four keywords 400 the structured-output call.

    Field(min_length=5, max_length=5) on variants and Field(ge=1, le=5) on
    index used to put all four in the emitted schema, which would have failed
    every planner call. The rules live in validate_matrix() now.
    """
    schema = json.dumps(to_strict_json_schema(BatchPlan))
    for keyword in ("minItems", "maxItems", "minimum", "maximum"):
        assert keyword not in schema, f"{keyword} is back in the strict schema"


def test_headline_validator_adds_no_schema_keyword():
    """A field_validator runs after parsing and must not shape the schema."""
    schema = to_strict_json_schema(BatchPlan)
    headline = schema["$defs"]["Variant"]["properties"]["headline"]
    assert set(headline) <= {"type", "title", "description"}


def test_plan_with_wrong_pairing_is_rejected():
    bad = [_variant(i, h, t) for i, h, t, _s in MATRIX]
    bad[0].treatment = "full_bleed"          # stat must pair with split_screen
    with pytest.raises(ValueError, match="matrix"):
        _plan(variants=bad).validate_matrix()


def test_unknown_treatment_is_rejected():
    with pytest.raises(ValidationError):
        _variant(treatment="interpretive_dance")


def test_headline_longer_than_five_words_is_rejected():
    with pytest.raises(ValidationError):
        _variant(headline="this headline has far too many words in it")


def test_headline_trailing_period_is_stripped():
    assert _variant(headline="SAME AI DIFFERENT ANSWER.").headline.endswith("ANSWER")


def test_headline_is_uppercased_and_trimmed():
    assert _variant(headline="  same ai  ").headline == "SAME AI"


# --- style axis -----------------------------------------------------------


def test_matrix_carries_a_style_on_every_row():
    assert all(len(row) == 4 for row in MATRIX)


def test_the_batch_spans_four_distinct_styles():
    styles = [row[3] for row in MATRIX]
    assert len(set(styles)) == 4, (
        "a batch must offer genuine variety, not five of one look"
    )


def test_the_house_style_appears_exactly_twice():
    """Two safe-to-ship options plus three genuine alternatives."""
    styles = [row[3] for row in MATRIX]
    assert styles.count(HOUSE_STYLE) == 2


def test_every_style_in_the_matrix_has_a_brief():
    """A missing brief would leave {style_brief} unfilled in the render prompt."""
    for row in MATRIX:
        assert row[3] in STYLE_BRIEF


def test_no_style_brief_is_orphaned():
    assert set(STYLE_BRIEF) == {row[3] for row in MATRIX}


def test_style_for_returns_the_locked_style_of_each_slot():
    for index, _hook, _treatment, style in MATRIX:
        assert style_for(index) == style


def test_style_for_rejects_an_unknown_slot():
    with pytest.raises(ValueError, match="slot"):
        style_for(99)


def test_a_variant_derives_its_style_from_its_slot():
    for index, hook, treatment, style in MATRIX:
        assert _variant(index, hook, treatment).style == style


def test_style_is_not_a_model_field():
    """It must stay derived: a field would enter the strict schema and let the
    planner contradict the locked matrix."""
    assert "style" not in Variant.model_fields


def test_each_style_brief_specifies_its_own_type_treatment():
    """render.md no longer states type treatment globally, so every brief must.

    Checks for type direction by any name — a brief may say "Headline in a heavy
    condensed sans" or "Typography is the design"; both direct the type.
    """
    for style, brief in STYLE_BRIEF.items():
        lowered = brief.lower()
        assert any(
            word in lowered for word in ("headline", "typography", "type")
        ), f"{style} brief must direct the type treatment"
        assert any(
            word in lowered for word in ("sans", "grotesque", "caps")
        ), f"{style} brief must name a typeface character or case"


def test_the_divergent_briefs_forbid_the_house_treatment():
    """Otherwise the model reverts to sparks and an orange/blue split."""
    for style in ("dark_cinematic", "flat_graphic", "clean_corporate"):
        brief = STYLE_BRIEF[style].lower()
        assert "no sparks" in brief or "no glow" in brief or "no drama" in brief
