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


def test_all_three_prompt_files_load():
    for name in ("planner", "render", "qa_legibility"):
        assert len(prompts.load(name)) > 200


def test_missing_prompt_file_raises():
    with pytest.raises(FileNotFoundError):
        prompts.load("does_not_exist")


def test_render_prompt_contains_the_exact_headline_in_quotes():
    text = prompts.render_prompt(_variant())
    assert '"SAME AI DIFFERENT ANSWER"' in text


def test_render_prompt_carries_the_treatment_brief():
    text = prompts.render_prompt(_variant(treatment="product_forward"))
    assert "FinanceOS" in text


def test_render_prompt_forbids_extra_text():
    text = prompts.render_prompt(_variant()).lower()
    assert "other text" in text
    assert "do not add" in text


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
    assert "Near-black background" in text


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
    assert "flat editorial poster" in flat
    assert "flat editorial poster" not in house


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
