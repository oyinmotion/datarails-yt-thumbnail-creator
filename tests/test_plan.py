from pathlib import Path

import pytest

from src import plan
from src.models import MATRIX, BatchPlan, Variant


def _valid_plan():
    return BatchPlan(
        ad_summary="two colleagues, same AI, different answers",
        transcript_used=True,
        variants=[
            Variant(
                index=i, hook_type=h, treatment=t,
                headline=f"HOOK {i}", frame_id="scene_001.jpg",
                second_frame_id=None, scene_direction="sparks at the seam",
                rationale="from the dialogue",
            )
            for i, h, t in MATRIX
        ],
    )


class FakeTranscriptions:
    def __init__(self, text="they disagree about the forecast", fail=False):
        self.text_value = text
        self.fail = fail

    def create(self, **kwargs):
        if self.fail:
            raise RuntimeError("transcription service unavailable")
        return type("T", (), {"text": self.text_value})()


class FakeResponses:
    def __init__(self, parsed, fail_times=0):
        self.parsed = parsed
        self.fail_times = fail_times
        self.calls = []

    def parse(self, **kwargs):
        self.calls.append(kwargs)
        if self.fail_times > 0:
            self.fail_times -= 1
            raise RuntimeError("bad json")
        return type("R", (), {"output_parsed": self.parsed})()


class FakeClient:
    def __init__(self, parsed=None, fail_times=0, transcribe_fail=False):
        self.responses = FakeResponses(parsed or _valid_plan(), fail_times)
        self.audio = type(
            "A", (), {"transcriptions": FakeTranscriptions(fail=transcribe_fail)}
        )()


@pytest.fixture
def frames(tmp_path):
    paths = []
    for i in range(3):
        p = tmp_path / f"scene_{i:03d}.jpg"
        p.write_bytes(b"\xff\xd8\xff fake jpeg")
        paths.append(p)
    return paths


def test_transcribe_returns_text(tmp_path):
    audio = tmp_path / "audio.m4a"
    audio.write_bytes(b"fake")
    assert plan.transcribe(audio, client=FakeClient()) == (
        "they disagree about the forecast"
    )


def test_transcribe_failure_returns_none_instead_of_raising(tmp_path):
    audio = tmp_path / "audio.m4a"
    audio.write_bytes(b"fake")
    assert plan.transcribe(audio, client=FakeClient(transcribe_fail=True)) is None


def test_build_plan_returns_five_matrix_variants(frames, tmp_path):
    audio = tmp_path / "audio.m4a"
    audio.write_bytes(b"fake")
    result = plan.build_plan(frames, audio, client=FakeClient())
    assert len(result.variants) == 5
    result.validate_matrix()


def test_build_plan_sends_every_frame_as_an_image(frames, tmp_path):
    client = FakeClient()
    plan.build_plan(frames, None, client=client)
    sent = client.responses.calls[0]
    content = sent["input"][0]["content"]
    images = [c for c in content if c["type"] == "input_image"]
    assert len(images) == len(frames)


def test_build_plan_lists_frame_filenames_in_the_prompt(frames):
    client = FakeClient()
    plan.build_plan(frames, None, client=client)
    text = client.responses.calls[0]["input"][0]["content"][0]["text"]
    for f in frames:
        assert f.name in text


def test_build_plan_marks_transcript_unused_when_audio_is_missing(frames):
    result = plan.build_plan(frames, None, client=FakeClient())
    assert result.transcript_used is False


def test_build_plan_retries_once_on_bad_response(frames):
    client = FakeClient(fail_times=1)
    result = plan.build_plan(frames, None, client=client)
    assert len(client.responses.calls) == 2
    assert len(result.variants) == 5


def test_build_plan_raises_plan_error_after_second_failure(frames):
    with pytest.raises(plan.PlanError):
        plan.build_plan(frames, None, client=FakeClient(fail_times=2))


def test_build_plan_rejects_a_plan_that_breaks_the_matrix(frames):
    broken = _valid_plan()
    broken.variants[0].treatment = "full_bleed"
    with pytest.raises(plan.PlanError, match="matrix"):
        plan.build_plan(frames, None, client=FakeClient(parsed=broken))


def test_build_plan_with_no_frames_raises(tmp_path):
    with pytest.raises(plan.PlanError):
        plan.build_plan([], None, client=FakeClient())
