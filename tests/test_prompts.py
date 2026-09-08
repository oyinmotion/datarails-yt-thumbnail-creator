import pytest

from src import prompts
from src.models import MATRIX, Variant


def _variant(
    treatment="split_screen",
    headline="SAME AI DIFFERENT ANSWER",
    index=1,
    hook="stat",
):
    return Variant(
        index=index, hook_type=hook, treatment=treatment, headline=headline,
        frame_id="scene_001.jpg", second_frame_id=None,
        scene_direction="hard vertical seam of light between them",
        rationale="the ad contrasts two answers",
    )


def test_the_prompt_files_load():
    for name in ("planner", "render", "qa_likeness"):
        assert len(prompts.load(name)) > 200


def test_the_legibility_prompt_is_gone():
    with pytest.raises(FileNotFoundError):
        prompts.load("qa_legibility")


def test_missing_prompt_file_raises():
    with pytest.raises(FileNotFoundError):
        prompts.load("does_not_exist")


def test_render_prompt_carries_the_treatment_brief():
    text = prompts.render_prompt(_variant(treatment="product_forward"))
    assert "FinanceOS" in text


def test_render_prompt_has_no_unfilled_placeholders():
    text = prompts.render_prompt(_variant())
    assert "{" not in text and "}" not in text


def test_planner_prompt_states_the_headline_word_limit():
    text = prompts.planner_prompt(transcript="hello", context=None,
                                  headline_override=None)
    assert "five words" in text.lower()


def test_planner_prompt_includes_every_matrix_row():
    text = prompts.planner_prompt(transcript=None, context=None,
                                  headline_override=None)
    for _, hook, treatment, _style in MATRIX:
        assert hook in text
        assert treatment in text


def test_planner_prompt_notes_missing_transcript():
    text = prompts.planner_prompt(transcript=None, context=None,
                                  headline_override=None)
    assert "no transcript" in text.lower()


def test_planner_prompt_passes_through_override_and_context():
    text = prompts.planner_prompt(
        transcript="t", context="targeting CFOs", headline_override="ONE TRUTH",
    )
    assert "ONE TRUTH" in text
    assert "targeting CFOs" in text


# --- style axis -----------------------------------------------------------


def test_render_prompt_injects_the_slot_style_brief():
    """Slot 2 is dark_cinematic, so its brief must appear, not the house one."""
    text = prompts.render_prompt(
        _variant(index=2, hook="question", treatment="face_closeup")
    )
    assert "near-black" in text.lower()


def test_render_prompt_uses_the_house_brief_for_a_house_slot():
    text = prompts.render_prompt(
        _variant(index=1, hook="stat", treatment="split_screen")
    )
    assert "proven Datarails look" in text


def test_two_different_slots_get_visibly_different_style_direction():
    """The whole point: the render prompts must not be near-identical."""
    house = prompts.render_prompt(
        _variant(index=1, hook="stat", treatment="split_screen")
    )
    flat = prompts.render_prompt(
        _variant(index=4, hook="pain", treatment="text_dominant")
    )
    assert house != flat
    assert "flat-colour treatment" in flat
    assert "flat-colour treatment" not in house


def test_render_prompt_no_longer_hardcodes_one_palette_globally():
    """A clean-corporate render must not be told to use orange-and-blue."""
    text = prompts.render_prompt(
        _variant(index=5, hook="outcome", treatment="product_forward")
    )
    assert "deep navy blue and vivid orange, with hot white" not in text


def test_render_prompt_tells_the_model_not_to_revert_to_the_house_look():
    text = prompts.render_prompt(
        _variant(index=2, hook="question", treatment="face_closeup")
    )
    assert "do not fall back" in text.lower()


def test_every_slot_renders_without_leaving_a_placeholder():
    for index, hook, treatment, _style in MATRIX:
        text = prompts.render_prompt(_variant(index=index, hook=hook,
                                              treatment=treatment))
        assert "{" not in text and "}" not in text


def test_the_planner_prompt_never_mentions_style():
    """Style is derived from the slot; asking the planner for it invites drift."""
    text = prompts.planner_prompt(transcript="t", context=None,
                                  headline_override=None).lower()
    for style in ("dark_cinematic", "flat_graphic", "clean_corporate"):
        assert style not in text


def test_the_planner_is_told_exactly_how_many_concepts_to_return():
    for count, word in ((1, "one"), (2, "two"), (5, "five")):
        text = prompts.planner_prompt("t", None, None, variant_count=count)
        assert f"exactly {word} variant" in text


def test_a_short_batch_lists_only_its_own_rows():
    text = prompts.planner_prompt("t", None, None, variant_count=2)
    assert "| 1 | stat |" in text and "| 2 | question |" in text
    assert "| 3 | conflict |" not in text
    assert "| 5 | outcome |" not in text


def test_a_short_batch_is_told_not_to_add_the_others():
    text = prompts.planner_prompt("t", None, None, variant_count=3)
    assert "Do not add the others" in text


def test_a_full_batch_gets_no_such_warning():
    text = prompts.planner_prompt("t", None, None, variant_count=5)
    assert "Do not add the others" not in text
    for row in MATRIX:
        assert f"| {row[0]} | {row[1]} |" in text


def test_one_concept_reads_as_singular():
    assert "exactly one variant," in prompts.planner_prompt(
        "t", None, None, variant_count=1
    )


# --- art only ---------------------------------------------------------------


def test_render_prompt_never_mentions_the_headline_text():
    text = prompts.render_prompt(_variant(headline="SAME AI DIFFERENT ANSWER"))
    assert "SAME AI DIFFERENT ANSWER" not in text
    # The only permitted mentions say that WE add it; nothing may ask the model
    # to render, place or style a headline.
    stripped = (text.lower()
                .replace("the headline is added afterwards by us", "")
                .replace("we add the headline ourselves", ""))
    assert "headline" not in stripped


def test_render_prompt_forbids_all_text_and_reserves_the_zone():
    flat = " ".join(prompts.render_prompt(
        _variant(treatment="face_closeup", index=2, hook="question")).split()).lower()
    assert "no text of any kind" in flat
    assert "reserve a calm area" in flat
    assert "left" in flat, "face_closeup reserves the left"


def test_render_prompt_can_be_asked_for_another_style():
    text = prompts.render_prompt(_variant(index=1), style="flat_graphic")
    assert "flat-colour treatment" in text
    assert "proven Datarails look" not in text


def test_render_prompt_uses_the_tall_zone_for_9x16():
    v = _variant(treatment="face_closeup", index=2, hook="question")
    assert prompts.render_prompt(v, ratio="16x9") != prompts.render_prompt(v, ratio="9x16")


def test_style_briefs_no_longer_describe_type():
    from src.models import STYLE_BRIEF, TREATMENT_BRIEF
    for brief in STYLE_BRIEF.values():
        assert "TYPE:" not in brief
    for brief in TREATMENT_BRIEF.values():
        assert "headline" not in brief.lower()
