import io
import zipfile
from pathlib import Path

from PIL import Image

import app
from src.models import MATRIX, Variant
from src.pipeline import ThumbResult


def _result(tmp_path, index, hook, treatment, missing=False):
    variant = Variant(
        index=index, hook_type=hook, treatment=treatment,
        headline=f"HOOK {index}", frame_id="scene_001.jpg",
        second_frame_id=None, scene_direction="sparks", rationale="from the ad",
    )
    if missing:
        return ThumbResult(variant=variant, path=None, flagged=True,
                           note="render failed")
    path = tmp_path / f"{index:02d}_{hook}_{treatment}.png"
    Image.new("RGB", (1920, 1080), (10, 20, 40)).save(path, "PNG")
    return ThumbResult(variant=variant, path=path)


def test_zip_contains_every_successful_render(tmp_path):
    results = [_result(tmp_path, i, h, t) for i, h, t, _s in MATRIX]
    with zipfile.ZipFile(io.BytesIO(app.zip_bytes(results))) as archive:
        assert len(archive.namelist()) == 5


def test_zip_skips_failed_renders(tmp_path):
    results = [_result(tmp_path, i, h, t) for i, h, t, _s in MATRIX]
    results[2] = _result(tmp_path, 3, "conflict", "full_bleed", missing=True)
    with zipfile.ZipFile(io.BytesIO(app.zip_bytes(results))) as archive:
        assert len(archive.namelist()) == 4


def test_zip_entries_keep_their_descriptive_names(tmp_path):
    results = [_result(tmp_path, 1, "stat", "split_screen")]
    with zipfile.ZipFile(io.BytesIO(app.zip_bytes(results))) as archive:
        assert archive.namelist() == ["01_stat_split_screen.png"]


def test_batch_folder_name_strips_the_extension_and_dates_it():
    name = app.batch_folder_name("claude-vs-claude-v2.mp4", "2026-08-18")
    assert name == "thumbnails — claude-vs-claude-v2 — 2026-08-18"


def test_batch_folder_name_survives_a_name_with_no_extension():
    assert app.batch_folder_name("ad", "2026-08-18") == (
        "thumbnails — ad — 2026-08-18"
    )


def test_should_show_outcome_matching_link():
    link = "https://drive.google.com/file/d/abc123/view"
    assert app.should_show_outcome(link, link) is True


def test_should_show_outcome_different_link():
    stored = "https://drive.google.com/file/d/abc123/view"
    current = "https://drive.google.com/file/d/xyz789/view"
    assert app.should_show_outcome(stored, current) is False


def test_should_show_outcome_no_stored_link():
    assert app.should_show_outcome(None, "https://drive.google.com/file/d/abc123/view") is False


def test_should_show_outcome_ignores_surrounding_whitespace():
    link = "https://drive.google.com/file/d/abc123/view"
    assert app.should_show_outcome(f"  {link}  ", link) is True


def test_should_show_outcome_empty_current_link_does_not_match_stored():
    stored = "https://drive.google.com/file/d/abc123/view"
    assert app.should_show_outcome(stored, "") is False


# --- OAuth state (I10) ------------------------------------------------------
def test_oauth_state_matches_when_the_callback_echoes_it():
    assert app.oauth_state_matches("abc123", "abc123") is True


def test_oauth_state_rejects_a_different_state():
    """A forged callback carrying someone else's code must not be exchanged."""
    assert app.oauth_state_matches("abc123", "evil999") is False


def test_oauth_state_rejects_a_missing_state_when_one_is_on_record():
    assert app.oauth_state_matches("abc123", None) is False
    assert app.oauth_state_matches("abc123", "") is False


def test_oauth_state_allows_the_callback_when_no_state_was_stored():
    """Streamlit starts a fresh session on the redirect back from Google, so an
    absent stored state means unverifiable, not hostile — refusing it would
    lock every user out."""
    assert app.oauth_state_matches(None, "abc123") is True
