import pytest

from src import prompts
from src.models import MATRIX, Variant


def _variant(treatment="split_screen", headline="SAME AI DIFFERENT ANSWER"):
    return Variant(
        index=1, hook_type="stat", treatment=treatment, headline=headline,
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
    for _, hook, treatment in MATRIX:
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
