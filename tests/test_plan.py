from pathlib import Path

import pytest
from pydantic import ValidationError

from src import backoff, plan
from src.models import MATRIX, BatchPlan, Variant


@pytest.fixture(autouse=True)
def no_real_sleeping(monkeypatch):
    """Backoff is real in production and instant in tests."""
    monkeypatch.setattr(plan, "DEFAULT_SLEEPER", lambda seconds: None)


def _validation_error() -> ValidationError:
    """A real pydantic ValidationError, the way the SDK's parse would raise it."""
    try:
        Variant(
            index=1, hook_type="stat", treatment="split_screen",
            headline="this headline has far too many words in it",
            frame_id="scene_001.jpg", second_frame_id=None,
            scene_direction="x", rationale="y",
        )
    except ValidationError as exc:
        return exc
    raise AssertionError("expected a ValidationError")


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
    def __init__(self, parsed, fail_times=0, error=None):
        self.parsed = parsed
        self.fail_times = fail_times
        self.error = error or RuntimeError("bad json")
        self.calls = []

    def parse(self, **kwargs):
        self.calls.append(kwargs)
        if self.fail_times > 0:
            self.fail_times -= 1
            raise self.error
        return type("R", (), {"output_parsed": self.parsed})()


class FakeClient:
    def __init__(self, parsed=None, fail_times=0, transcribe_fail=False,
                 error=None):
        self.responses = FakeResponses(parsed or _valid_plan(), fail_times,
                                       error)
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


def test_a_validation_error_is_retried_and_can_succeed(frames):
    """A six-word headline is the likeliest model slip and must cost a retry,
    not the whole batch — ValidationError subclasses ValueError, so it used to
    land in the matrix branch and kill the plan with an attempt unused."""
    client = FakeClient(fail_times=1, error=_validation_error())
    result = plan.build_plan(frames, None, client=client)
    assert len(client.responses.calls) == 2
    assert len(result.variants) == 5


def test_the_retry_tells_the_model_what_it_got_wrong(frames):
    client = FakeClient(fail_times=1, error=_validation_error())
    plan.build_plan(frames, None, client=client)
    second_prompt = client.responses.calls[1]["input"][0]["content"][0]["text"]
    assert "rejected" in second_prompt.lower()
    assert "5 words or fewer" in second_prompt
    assert "headline" in second_prompt.lower()


def test_a_persistent_validation_error_gives_its_own_message(frames):
    client = FakeClient(fail_times=2, error=_validation_error())
    with pytest.raises(plan.PlanError, match="hooks we can't use"):
        plan.build_plan(frames, None, client=client)
    assert len(client.responses.calls) == 2


def test_a_matrix_violation_does_not_consume_a_retry(frames):
    """The matrix is spelled out in the prompt; breaking it is not a slip a
    reroll fixes, so it must fail on the first attempt."""
    broken = _valid_plan()
    broken.variants[0].treatment = "full_bleed"
    client = FakeClient(parsed=broken)
    with pytest.raises(plan.PlanError, match="matrix"):
        plan.build_plan(frames, None, client=client)
    assert len(client.responses.calls) == 1


def test_a_retry_backs_off_before_trying_again(frames):
    delays = []
    client = FakeClient(fail_times=1)
    plan.build_plan(frames, None, client=client, sleeper=delays.append)
    assert delays == [backoff.delay_for(1)]
    assert delays[0] > 0


def test_a_validation_retry_also_backs_off(frames):
    delays = []
    client = FakeClient(fail_times=1, error=_validation_error())
    plan.build_plan(frames, None, client=client, sleeper=delays.append)
    assert delays == [backoff.delay_for(1)]


def test_the_final_attempt_does_not_sleep_pointlessly(frames):
    delays = []
    with pytest.raises(plan.PlanError):
        plan.build_plan(frames, None, client=FakeClient(fail_times=2),
                        sleeper=delays.append)
    assert len(delays) == 1        # after attempt 1 only
