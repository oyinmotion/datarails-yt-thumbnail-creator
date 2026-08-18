import io

from PIL import Image

from src import qa
from src.config import FINAL_H, FINAL_W


def _write(path, w=FINAL_W, h=FINAL_H):
    Image.new("RGB", (w, h), (10, 20, 40)).save(path, "PNG")
    return path


class FakeResponses:
    def __init__(self, text):
        self.text = text
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return type("R", (), {"output_text": self.text})()


class FakeClient:
    def __init__(self, text):
        self.responses = FakeResponses(text)


def test_normalize_strips_case_and_punctuation():
    assert qa.normalize("Same AI, Different Answer!") == [
        "SAME", "AI", "DIFFERENT", "ANSWER",
    ]


def test_normalize_collapses_whitespace():
    assert qa.normalize("  47K   OVER  ") == ["47K", "OVER"]


def test_legible_when_every_word_appears_in_order():
    assert qa.headline_is_legible("SAME AI DIFFERENT ANSWER",
                                  "same ai different answer")


def test_legible_when_the_model_reads_extra_words():
    assert qa.headline_is_legible("47K OVER", "financeos 47K over budget")


def test_not_legible_when_a_word_is_missing():
    assert not qa.headline_is_legible("SAME AI DIFFERENT ANSWER",
                                      "same ai answer")


def test_not_legible_when_words_are_out_of_order():
    assert not qa.headline_is_legible("PROBABLY VS PROVEN",
                                      "proven vs probably")


def test_not_legible_when_nothing_is_readable():
    assert not qa.headline_is_legible("47K OVER", "NONE")


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


def test_check_passes_when_the_headline_reads_back(tmp_path):
    result = qa.check(_write(tmp_path / "a.png"), "47K OVER",
                      client=FakeClient("47K OVER"))
    assert result.ok
    assert result.problems == []


def test_check_fails_and_names_the_problem_when_text_is_unreadable(tmp_path):
    result = qa.check(_write(tmp_path / "b.png"), "SAME AI DIFFERENT ANSWER",
                      client=FakeClient("same different"))
    assert not result.ok
    assert any("headline" in p.lower() for p in result.problems)
    assert result.transcribed == "same different"


def test_check_sends_a_downscaled_320px_image(tmp_path):
    client = FakeClient("47K OVER")
    qa.check(_write(tmp_path / "c.png"), "47K OVER", client=client)
    sent = client.responses.calls[0]["input"][0]["content"]
    image_entry = next(c for c in sent if c["type"] == "input_image")
    assert image_entry["image_url"].startswith("data:image/png;base64,")


def test_check_survives_a_vision_failure_without_blocking_the_batch(tmp_path):
    class Broken:
        class responses:
            @staticmethod
            def create(**kwargs):
                raise RuntimeError("vision down")

    result = qa.check(_write(tmp_path / "d.png"), "47K OVER", client=Broken())
    assert result.ok            # unverified, but not a failure
    assert result.transcribed is None


def test_an_unavailable_vision_model_marks_the_result_unverified(tmp_path):
    class Broken:
        class responses:
            @staticmethod
            def create(**kwargs):
                raise RuntimeError("vision down")

    result = qa.check(_write(tmp_path / "e.png"), "47K OVER", client=Broken())
    assert result.ok
    assert result.unverified, "a silent fail-open is what I7 is about"


def test_a_verified_pass_is_not_marked_unverified(tmp_path):
    result = qa.check(_write(tmp_path / "f.png"), "47K OVER",
                      client=FakeClient("47K OVER"))
    assert not result.unverified


def test_a_failed_check_is_not_marked_unverified(tmp_path):
    result = qa.check(_write(tmp_path / "g.png"), "47K OVER",
                      client=FakeClient("nothing here"))
    assert not result.ok
    assert not result.unverified
