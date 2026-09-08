import io

from PIL import Image

from src import qa
from src.config import FINAL_H, FINAL_W


def _write(path, w=FINAL_W, h=FINAL_H):
    Image.new("RGB", (w, h), (10, 20, 40)).save(path, "PNG")
    return path


def test_hard_checks_pass_for_a_correct_file(tmp_path):
    assert qa.hard_checks(_write(tmp_path / "ok.png")) == []


def test_hard_checks_flag_wrong_dimensions(tmp_path):
    problems = qa.hard_checks(_write(tmp_path / "small.png", 1280, 720))
    assert any("1920" in p for p in problems)


def test_hard_checks_flag_oversize_files(tmp_path, monkeypatch):
    path = _write(tmp_path / "big.png")
    monkeypatch.setattr(qa, "MAX_BYTES", 10)
    assert any("2 MB" in p or "too large" in p for p in qa.hard_checks(path))


def test_hard_checks_flag_a_corrupt_file(tmp_path):
    path = tmp_path / "broken.png"
    path.write_bytes(b"not an image")
    assert qa.hard_checks(path)



# --- likeness gate --------------------------------------------------------


class FakeLikenessClient:
    def __init__(self, answer):
        self.answer = answer
        self.calls = []
        outer = self

        class _Responses:
            def create(self, **kwargs):
                outer.calls.append(kwargs)
                return type("R", (), {"output_text": outer.answer})()

        self.responses = _Responses()


def _frame(tmp_path, name="scene_001.jpg"):
    path = tmp_path / name
    Image.new("RGB", (1280, 720), (60, 60, 60)).save(path, "JPEG")
    return path


def test_a_render_of_the_right_actor_passes(tmp_path):
    result = qa.likeness_gate(_write(tmp_path / "a.png"), _frame(tmp_path),
                              client=FakeLikenessClient("SAME"))
    assert result.ok and result.likeness == "SAME" and result.checked
    assert not result.unverified


def test_an_invented_person_fails_the_tile(tmp_path):
    result = qa.likeness_gate(_write(tmp_path / "b.png"), _frame(tmp_path),
                              client=FakeLikenessClient("DIFFERENT"))
    assert not result.ok
    assert any("not the actor" in p for p in result.problems)


def test_a_render_with_no_person_fails(tmp_path):
    result = qa.likeness_gate(_write(tmp_path / "c.png"), _frame(tmp_path),
                              client=FakeLikenessClient("NOBODY"))
    assert not result.ok
    assert any("no person" in p for p in result.problems)


def test_an_unclear_verdict_does_not_block_the_render(tmp_path):
    result = qa.likeness_gate(_write(tmp_path / "d.png"), _frame(tmp_path),
                              client=FakeLikenessClient("UNCLEAR"))
    assert result.ok and result.likeness == "UNCLEAR"


def test_no_reference_frame_means_no_call_and_an_unverified_pass(tmp_path):
    client = FakeLikenessClient("SAME")
    result = qa.likeness_gate(_write(tmp_path / "f.png"), None, client=client)
    assert result.ok and result.likeness is None and client.calls == []
    assert result.unverified


def test_a_likeness_outage_passes_unverified_rather_than_losing_the_render(tmp_path):
    class Broken:
        class responses:
            @staticmethod
            def create(**kwargs):
                raise RuntimeError("vision down")

    result = qa.likeness_gate(_write(tmp_path / "g.png"), _frame(tmp_path), client=Broken())
    assert result.ok and result.likeness is None and result.unverified


def test_the_likeness_check_sends_the_render_and_the_frame_in_that_order(tmp_path):
    client = FakeLikenessClient("SAME")
    qa.likeness_gate(_write(tmp_path / "h.png"), _frame(tmp_path), client=client)
    content = client.calls[0]["input"][0]["content"]
    images = [c for c in content if c["type"] == "input_image"]
    assert len(images) == 2
    assert "same real person" in content[0]["text"]


def test_a_people_free_ad_fails_when_a_person_appears(tmp_path):
    for verdict in ("SAME", "DIFFERENT"):
        result = qa.likeness_gate(_write(tmp_path / f"p{verdict}.png"), _frame(tmp_path),
                                  client=FakeLikenessClient(verdict), people_in_ad=False)
        assert not result.ok
        assert any("invented" in p for p in result.problems)


def test_a_people_free_ad_passes_when_there_is_nobody(tmp_path):
    result = qa.likeness_gate(_write(tmp_path / "none.png"), _frame(tmp_path),
                              client=FakeLikenessClient("NOBODY"), people_in_ad=False)
    assert result.ok and result.likeness == "NOBODY"


def test_the_legibility_check_is_gone():
    assert not hasattr(qa, "check")
    assert not hasattr(qa, "headline_is_legible")
