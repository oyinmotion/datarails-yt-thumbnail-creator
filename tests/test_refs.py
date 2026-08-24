import shutil

import pytest

from src import refs


@pytest.fixture
def refs_enabled(monkeypatch):
    """Turn style references back on.

    They are off by default because every file in the pack contains recognisable
    people (see config.SEND_STYLE_REFS). The selection logic below is still
    exercised so the feature stays working for a future people-free pack.
    """
    monkeypatch.setattr(refs, "SEND_STYLE_REFS", True)


@pytest.fixture
def isolated_refs(tmp_path, monkeypatch, refs_enabled):
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


def test_house_style_always_includes_at_least_one_locked_style_ref(isolated_refs):
    """The drift guard: the proven look must stay anchored to the locked pack."""
    style, winners = isolated_refs
    for i in range(6):
        (winners / f"house_energy__split_screen__{i}.png").write_bytes(b"\x89PNG fake")
    picked = refs.pick_refs("house_energy", "split_screen", limit=3)
    assert len(picked) == 3
    assert any(p.parent == style for p in picked), (
        "style drift guard: a locked reference must always be in the mix"
    )


def test_a_divergent_style_gets_no_house_references_at_all(isolated_refs):
    """The whole point of the style axis.

    The locked pack is four images of one high-contrast orange-and-blue look.
    Sending even one to a dark-cinematic render drags it back to the house
    style, which is the problem the style axis exists to solve.
    """
    style, _winners = isolated_refs
    for divergent in ("dark_cinematic", "flat_graphic", "clean_corporate"):
        picked = refs.pick_refs(divergent, "face_closeup", limit=3)
        assert not any(p.parent == style for p in picked), (
            f"{divergent} must not receive a locked house reference"
        )


def test_a_divergent_style_starts_with_no_references_at_all(isolated_refs):
    """Prompt-only until the team approves some of its own."""
    assert refs.pick_refs("dark_cinematic", "face_closeup", limit=3) == []


def test_a_divergent_style_uses_its_own_winners_once_they_exist(isolated_refs):
    _style, winners = isolated_refs
    mine = winners / "dark_cinematic__face_closeup__1.png"
    mine.write_bytes(b"\x89PNG fake")
    picked = refs.pick_refs("dark_cinematic", "face_closeup", limit=3)
    assert picked == [mine]


def test_one_style_never_borrows_another_styles_winners(isolated_refs):
    """A flat-graphic winner must never inform a clean-corporate render."""
    _style, winners = isolated_refs
    (winners / "flat_graphic__text_dominant__1.png").write_bytes(b"\x89PNG fake")
    assert refs.pick_refs("clean_corporate", "product_forward", limit=3) == []


def test_house_style_prefers_winners_of_the_same_treatment(isolated_refs):
    """Same-treatment winners rank ahead of other-treatment ones.

    Priority, not exclusion: an other-treatment winner of the SAME style still
    shares the visual identity, so it is useful filler once the same-treatment
    ones are in. What matters is the ordering.
    """
    _style, winners = isolated_refs
    (winners / "house_energy__split_screen__1.png").write_bytes(b"\x89PNG fake")
    (winners / "house_energy__full_bleed__1.png").write_bytes(b"\x89PNG fake")
    names = [p.name for p in refs.pick_refs("house_energy", "split_screen", limit=3)]
    assert "house_energy__split_screen__1.png" in names
    assert names.index("house_energy__split_screen__1.png") < names.index(
        "house_energy__full_bleed__1.png"
    )


def test_a_same_treatment_winner_wins_the_last_slot_over_other_treatments(
    isolated_refs,
):
    """With only one slot left after the drift guard, same-treatment takes it."""
    _style, winners = isolated_refs
    (winners / "house_energy__full_bleed__1.png").write_bytes(b"\x89PNG fake")
    (winners / "house_energy__split_screen__1.png").write_bytes(b"\x89PNG fake")
    names = [p.name for p in refs.pick_refs("house_energy", "split_screen", limit=2)]
    assert names[1] == "house_energy__split_screen__1.png"


def test_pick_refs_respects_the_limit(isolated_refs):
    assert len(refs.pick_refs("house_energy", "full_bleed", limit=2)) == 2


def test_pick_refs_never_returns_duplicates(isolated_refs):
    _style, winners = isolated_refs
    (winners / "house_energy__split_screen__1.png").write_bytes(b"\x89PNG fake")
    picked = refs.pick_refs("house_energy", "split_screen", limit=3)
    assert len(picked) == len(set(picked))


def test_save_winner_tags_the_file_with_its_style_and_treatment(
    isolated_refs, tmp_path
):
    src_file = tmp_path / "chosen.png"
    src_file.write_bytes(b"\x89PNG fake")
    saved = refs.save_winner(src_file, "dark_cinematic", "face_closeup")
    assert saved.name.startswith("dark_cinematic__face_closeup__")
    assert saved.exists()
    assert refs.winner_refs("dark_cinematic", "face_closeup") == [saved]


def test_save_winner_called_twice_keeps_both_files(isolated_refs, tmp_path):
    src_file = tmp_path / "chosen.png"
    src_file.write_bytes(b"\x89PNG fake")
    first = refs.save_winner(src_file, "flat_graphic", "text_dominant")
    second = refs.save_winner(src_file, "flat_graphic", "text_dominant")
    assert first != second
    assert first.exists() and second.exists()
    found = refs.winner_refs("flat_graphic", "text_dominant")
    assert sorted(found) == sorted([first, second])


def test_winner_refs_unfiltered_returns_every_style(isolated_refs, tmp_path):
    src_file = tmp_path / "chosen.png"
    src_file.write_bytes(b"\x89PNG fake")
    refs.save_winner(src_file, "house_energy", "split_screen")
    refs.save_winner(src_file, "dark_cinematic", "face_closeup")
    assert len(refs.winner_refs()) == 2


# --- style references are off by default ----------------------------------


def test_no_references_are_sent_by_default(isolated_refs, monkeypatch):
    """The bug this closes: thumbnails containing people from a different ad.

    Every file in refs/style/ is a finished thumbnail of the Claude-vs-Claude
    actors. Sent as a "style" reference into any other ad, the model composited
    those two men in. Prose carries the style now.
    """
    monkeypatch.setattr(refs, "SEND_STYLE_REFS", False)
    for style in ("house_energy", "dark_cinematic", "flat_graphic",
                  "clean_corporate"):
        assert refs.pick_refs(style, "split_screen", people_in_ad=True) == []
        assert refs.pick_refs(style, "split_screen", people_in_ad=False) == []


def test_the_real_style_pack_is_never_sent_while_it_contains_people(monkeypatch):
    """Guards the default itself: flipping SEND_STYLE_REFS on is a decision that
    requires a people-free pack, not an accident."""
    from src import config
    assert config.SEND_STYLE_REFS is False, (
        "refs/style/ still contains photographs of people; turning this on "
        "reintroduces strangers into other ads' thumbnails"
    )


def test_saving_a_winner_still_works_with_references_off(isolated_refs, tmp_path,
                                                        monkeypatch):
    """Starring must not crash just because references are not being sent."""
    monkeypatch.setattr(refs, "SEND_STYLE_REFS", False)
    src_file = tmp_path / "chosen.png"
    src_file.write_bytes(b"\x89PNG fake")
    saved = refs.save_winner(src_file, "house_energy", "split_screen")
    assert saved.exists()
