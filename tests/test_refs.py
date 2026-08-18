import shutil

import pytest

from src import refs


@pytest.fixture
def isolated_refs(tmp_path, monkeypatch):
    style = tmp_path / "style"
    winners = tmp_path / "winners"
    style.mkdir()
    winners.mkdir()
    for name in ("a.png", "b.png", "c.png", "d.png"):
        (style / name).write_bytes(b"\x89PNG fake")
    monkeypatch.setattr(refs, "REFS_STYLE_DIR", style)
    monkeypatch.setattr(refs, "REFS_WINNERS_DIR", winners)
    return style, winners


def test_style_refs_finds_the_locked_pack(isolated_refs):
    assert len(refs.style_refs()) == 4


def test_real_style_pack_is_populated():
    """The four approved samples must actually be in the repo."""
    assert len(refs.style_refs()) >= 4


def test_pick_refs_always_includes_at_least_one_locked_style_ref(isolated_refs):
    style, winners = isolated_refs
    for i in range(6):
        (winners / f"split_screen__{i}.png").write_bytes(b"\x89PNG fake")
    picked = refs.pick_refs("split_screen", limit=3)
    assert len(picked) == 3
    assert any(p.parent == style for p in picked), (
        "style drift guard: a locked reference must always be in the mix"
    )


def test_pick_refs_prefers_winners_of_the_same_treatment(isolated_refs):
    _, winners = isolated_refs
    (winners / "split_screen__1.png").write_bytes(b"\x89PNG fake")
    (winners / "face_closeup__1.png").write_bytes(b"\x89PNG fake")
    picked = refs.pick_refs("split_screen", limit=3)
    names = [p.name for p in picked]
    assert "split_screen__1.png" in names
    assert "face_closeup__1.png" not in names


def test_pick_refs_respects_the_limit(isolated_refs):
    assert len(refs.pick_refs("full_bleed", limit=2)) == 2


def test_save_winner_tags_the_file_with_its_treatment(isolated_refs, tmp_path):
    src_file = tmp_path / "chosen.png"
    src_file.write_bytes(b"\x89PNG fake")
    saved = refs.save_winner(src_file, "text_dominant")
    assert saved.name.startswith("text_dominant__")
    assert saved.exists()
    assert refs.winner_refs("text_dominant") == [saved]
