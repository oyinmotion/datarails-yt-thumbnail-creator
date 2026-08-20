import io
from pathlib import Path

import pytest
from PIL import Image

from src import backoff, branding, pipeline, postprocess, probe, qa, render
from src import plan as plan_module
from src.config import FINAL_H, FINAL_W, GEN_SIZE
from src.models import MATRIX, BatchPlan, Variant
from src.qa import QAResult


def _plan():
    return BatchPlan(
        ad_summary="same AI, different answers",
        transcript_used=True,
        variants=[
            Variant(
                index=i, hook_type=h, treatment=t, headline=f"HOOK {i}",
                frame_id="scene_001.jpg", second_frame_id=None,
                scene_direction="sparks", rationale="from the ad",
            )
            for i, h, t, _s in MATRIX
        ],
    )


@pytest.fixture
def wired(monkeypatch, tmp_path):
    """Replace every external dependency with a deterministic fake."""
    frames = []
    for name in ("scene_001.jpg", "scene_002.jpg"):
        frame = tmp_path / name
        frame.write_bytes(b"\xff\xd8\xff fake")
        frames.append(frame)
    audio = tmp_path / "audio.m4a"
    audio.write_bytes(b"fake")

    monkeypatch.setattr(probe, "extract_frames",
                        lambda video, out_dir, max_frames=16: frames)
    monkeypatch.setattr(probe, "extract_audio", lambda video, out_dir: audio)
    monkeypatch.setattr(plan_module, "build_plan",
                        lambda *a, **k: _plan())
    monkeypatch.setattr(render, "render_variant",
                        lambda *a, **k: b"\x89PNG bytes")

    written = []

    def fake_finalize(image_bytes, out_path, final_size=None):
        Path(out_path).write_bytes(image_bytes)
        written.append(Path(out_path))
        return Path(out_path)

    monkeypatch.setattr(postprocess, "finalize", fake_finalize)
    # The stub files above are not real images, so the real logo stamp cannot
    # open them. Branding has its own tests in tests/test_branding.py.
    monkeypatch.setattr(branding, "stamp_logo", lambda path, out=None: path)
    monkeypatch.setattr(
        qa, "check",
        lambda path, headline, **k: QAResult(ok=True, transcribed=headline),
    )
    # Backoff is real in production and instant in tests.
    monkeypatch.setattr(pipeline, "DEFAULT_SLEEPER", lambda seconds: None)
    monkeypatch.setattr(plan_module, "DEFAULT_SLEEPER", lambda seconds: None)
    return written


def test_batch_returns_exactly_five_results(wired, tmp_path):
    outcome = pipeline.generate_batch(tmp_path / "ad.mp4", tmp_path / "work")
    assert len(outcome.results) == 5
    assert all(r.path is not None for r in outcome.results)
    assert not any(r.flagged for r in outcome.results)


def test_every_result_keeps_its_variant_metadata(wired, tmp_path):
    outcome = pipeline.generate_batch(tmp_path / "ad.mp4", tmp_path / "work")
    pairs = {(r.variant.hook_type, r.variant.treatment) for r in outcome.results}
    assert pairs == {(h, t) for _, h, t, _s in MATRIX}


def test_output_filenames_are_ordered_and_descriptive(wired, tmp_path):
    outcome = pipeline.generate_batch(tmp_path / "ad.mp4", tmp_path / "work")
    names = sorted(r.path.name for r in outcome.results)
    assert names[0].startswith("01_stat_split_screen")
    assert names[4].startswith("05_outcome_product_forward")


def test_failed_qa_triggers_exactly_one_reroll(wired, tmp_path, monkeypatch):
    calls = {"render": 0}

    def counting_render(*args, **kwargs):
        calls["render"] += 1
        return b"\x89PNG bytes"

    monkeypatch.setattr(render, "render_variant", counting_render)
    monkeypatch.setattr(
        qa, "check",
        lambda path, headline, **k: QAResult(
            ok=False, problems=["the headline isn't readable"],
        ),
    )
    outcome = pipeline.generate_batch(tmp_path / "ad.mp4", tmp_path / "work")
    # Five concepts x three ratios = 15 renders, each rerolled once.
    assert calls["render"] == 30
    assert len(outcome.results) == 5
    assert all(r.flagged for r in outcome.results)
    assert all(r.path is not None for r in outcome.results)


def test_reroll_passes_the_failure_reason_back_into_the_prompt(
    wired, tmp_path, monkeypatch
):
    seen = []

    def capturing_render(variant, frames, client=None, extra_instruction="",
                         frame_override=None, **kwargs):
        seen.append(extra_instruction)
        return b"\x89PNG bytes"

    monkeypatch.setattr(render, "render_variant", capturing_render)
    monkeypatch.setattr(
        qa, "check",
        lambda path, headline, **k: QAResult(ok=False, problems=["cut off"]),
    )
    pipeline.generate_batch(tmp_path / "ad.mp4", tmp_path / "work")
    assert any("cut off" in s for s in seen if s)


def test_a_render_that_always_fails_still_yields_a_result_row(
    wired, tmp_path, monkeypatch
):
    def always_fails(*args, **kwargs):
        raise render.RenderError("503 from the API")

    monkeypatch.setattr(render, "render_variant", always_fails)
    outcome = pipeline.generate_batch(tmp_path / "ad.mp4", tmp_path / "work")
    assert len(outcome.results) == 5
    assert all(r.path is None for r in outcome.results)
    assert all("503" in r.note for r in outcome.results)


def test_blocked_render_retries_with_a_different_frame(
    wired, tmp_path, monkeypatch
):
    attempts = []

    def blocked_then_ok(variant, frames, client=None, extra_instruction="",
                        frame_override=None, **kwargs):
        # Keyed on the argument, not a call counter: five variants render
        # concurrently, so a counter would be racy.
        attempts.append(((variant.index, kwargs.get("gen_size")), frame_override))
        if frame_override is None:
            raise render.RenderBlocked("filter refused")
        return b"\x89PNG bytes"

    monkeypatch.setattr(render, "render_variant", blocked_then_ok)
    outcome = pipeline.generate_batch(tmp_path / "ad.mp4", tmp_path / "work")
    assert len(outcome.results) == 5
    assert all(r.path is not None for r in outcome.results)

    # Order-independent: group by variant instead of relying on global append
    # order. One variant is handled start-to-finish by a single worker, so
    # each variant's own two attempts are in order even though the five
    # variants interleave with each other under concurrency.
    by_variant: dict[int, list[Path | None]] = {}
    for index, override in attempts:
        by_variant.setdefault(index, []).append(override)

    assert len(by_variant) == 15   # five concepts x three ratios
    for index, overrides in by_variant.items():
        assert len(overrides) == 2, (
            f"{index} should render exactly twice (attempt + reroll)"
        )
        first, second = overrides
        assert first is None, (
            f"variant {index}'s first attempt should use its planned frame"
        )
        assert second is not None, (
            f"variant {index}'s reroll must try a different frame than the "
            "blocked one"
        )


def test_missing_audio_is_a_warning_not_a_failure(wired, tmp_path, monkeypatch):
    def no_audio(video, out_dir):
        raise probe.ProbeError("That video has no usable audio track.")

    monkeypatch.setattr(probe, "extract_audio", no_audio)
    outcome = pipeline.generate_batch(tmp_path / "ad.mp4", tmp_path / "work")
    assert len(outcome.results) == 5
    assert any("audio" in w.lower() for w in outcome.warnings)


def test_progress_callback_reports_each_phase(wired, tmp_path):
    messages = []
    pipeline.generate_batch(
        tmp_path / "ad.mp4", tmp_path / "work", progress=messages.append,
    )
    joined = " ".join(messages).lower()
    assert "frame" in joined
    assert "plan" in joined or "reading" in joined
    assert "render" in joined


def test_one_variant_blowing_up_unexpectedly_costs_only_that_tile(
    wired, tmp_path, monkeypatch
):
    """pool.map re-raises, so an exception that is not a RenderError used to
    take all five variants down with it."""
    def explode_for_variant_three(variant, frames, client=None,
                                  extra_instruction="", frame_override=None,
                                  **kwargs):
        if variant.index == 3:
            raise TypeError("'NoneType' object is not subscriptable")
        return b"\x89PNG bytes"

    monkeypatch.setattr(render, "render_variant", explode_for_variant_three)
    outcome = pipeline.generate_batch(tmp_path / "ad.mp4", tmp_path / "work")

    assert len(outcome.results) == 5
    by_index = {r.variant.index: r for r in outcome.results}
    assert by_index[3].path is None
    assert by_index[3].flagged
    assert "something went wrong" in by_index[3].note
    assert all(by_index[i].path is not None for i in (1, 2, 4, 5))


def test_an_unexpected_failure_in_finalize_also_costs_only_one_tile(
    wired, tmp_path, monkeypatch
):
    calls = {"n": 0}
    real_fake = postprocess.finalize

    def sometimes_broken(image_bytes, out_path, final_size=None, **kwargs):
        if "03_" in Path(out_path).name:
            raise OSError("cannot identify image file")
        calls["n"] += 1
        return real_fake(image_bytes, out_path, final_size)

    monkeypatch.setattr(postprocess, "finalize", sometimes_broken)
    outcome = pipeline.generate_batch(tmp_path / "ad.mp4", tmp_path / "work")
    assert len(outcome.results) == 5
    assert sum(1 for r in outcome.results if r.path is None) == 1


def test_a_failed_render_backs_off_before_the_reroll(wired, tmp_path, monkeypatch):
    delays = []
    monkeypatch.setattr(pipeline, "DEFAULT_SLEEPER", delays.append)

    def always_fails(*args, **kwargs):
        raise render.RenderError("429 rate limit")

    monkeypatch.setattr(render, "render_variant", always_fails)
    pipeline.generate_batch(tmp_path / "ad.mp4", tmp_path / "work")

    # One backoff per render, so fifteen. The stagger is per CONCEPT, so the
    # three ratios of one concept share a delay while the five concepts differ —
    # which is what stops all five re-colliding with the same rate limit.
    assert len(delays) == 15
    assert all(d > 0 for d in delays)
    assert sorted(set(delays)) == sorted(
        backoff.delay_for(1, offset=i * backoff.STAGGER) for i in range(1, 6)
    )
    from collections import Counter
    assert set(Counter(delays).values()) == {3}, (
        "each concept's three ratios should share that concept's delay"
    )


def test_backoff_grows_between_attempts():
    """The delay after a second failure is longer than after the first."""
    assert backoff.delay_for(2) > backoff.delay_for(1)


def test_a_qa_reroll_does_not_make_the_user_wait(wired, tmp_path, monkeypatch):
    """Backoff is for rate limits, not for an illegible headline."""
    delays = []
    monkeypatch.setattr(pipeline, "DEFAULT_SLEEPER", delays.append)
    monkeypatch.setattr(
        qa, "check",
        lambda path, headline, **k: QAResult(
            ok=False, problems=["the headline isn't readable"],
        ),
    )
    pipeline.generate_batch(tmp_path / "ad.mp4", tmp_path / "work")
    assert delays == []


def test_an_unverified_batch_warns_the_user(wired, tmp_path, monkeypatch):
    """A QA outage fails open, which must never be silent."""
    monkeypatch.setattr(
        qa, "check",
        lambda path, headline, **k: QAResult(ok=True, transcribed=None),
    )
    outcome = pipeline.generate_batch(tmp_path / "ad.mp4", tmp_path / "work")
    assert all(r.unverified for r in outcome.results)
    assert any("could not be text-checked" in w for w in outcome.warnings)
    assert any("5 of 5" in w for w in outcome.warnings)


def test_a_verified_batch_does_not_warn(wired, tmp_path):
    outcome = pipeline.generate_batch(tmp_path / "ad.mp4", tmp_path / "work")
    assert not any("text-checked" in w for w in outcome.warnings)


# --- T11: the real finalize feeding the real hard checks --------------------
def _real_render_bytes() -> bytes:
    """A genuine PNG at the size gpt-image-2 is actually asked for."""
    width, height = (int(v) for v in GEN_SIZE.split("x"))
    image = Image.new("RGB", (width, height), (12, 24, 48))
    buffer = io.BytesIO()
    image.save(buffer, "PNG")
    return buffer.getvalue()


class FakeVision:
    """Only the vision call is stubbed. It reads back every headline in the
    batch, so each variant's own words appear in order."""

    class responses:
        @staticmethod
        def create(**kwargs):
            return type("R", (), {"output_text": "HOOK 1 2 3 4 5"})()


def test_a_real_2048x1152_render_survives_finalize_and_hard_checks(
    tmp_path, monkeypatch
):
    """The one integration the suite used to fake on both sides. If finalize
    ever stopped downscaling, hard_checks would flag all five tiles and nobody
    would know until a live run."""
    frames = []
    for name in ("scene_001.jpg", "scene_002.jpg"):
        frame = tmp_path / name
        frame.write_bytes(b"\xff\xd8\xff fake")
        frames.append(frame)
    audio = tmp_path / "audio.m4a"
    audio.write_bytes(b"fake")

    monkeypatch.setattr(probe, "extract_frames",
                        lambda video, out_dir, max_frames=16: frames)
    monkeypatch.setattr(probe, "extract_audio", lambda video, out_dir: audio)
    monkeypatch.setattr(plan_module, "build_plan", lambda *a, **k: _plan())
    monkeypatch.setattr(render, "render_variant",
                        lambda *a, **k: _real_render_bytes())
    monkeypatch.setattr(pipeline, "DEFAULT_SLEEPER", lambda seconds: None)
    # postprocess.finalize and qa.check are the REAL ones here.

    outcome = pipeline.generate_batch(
        tmp_path / "ad.mp4", tmp_path / "work", client=FakeVision(),
    )

    assert len(outcome.results) == 5
    assert all(r.path is not None for r in outcome.results), [
        r.note for r in outcome.results
    ]
    assert not any(r.flagged for r in outcome.results), [
        r.note for r in outcome.results
    ]
    assert not any(r.unverified for r in outcome.results)

    for result in outcome.results:
        assert qa.hard_checks(result.path) == []
        with Image.open(result.path) as im:
            assert im.size == (FINAL_W, FINAL_H)


def test_an_invented_person_triggers_a_reroll_with_likeness_advice(
    wired, tmp_path, monkeypatch
):
    """A likeness failure must not be met with 'render the headline larger'."""
    instructions = []

    def capturing_render(variant, frames, client=None, extra_instruction="",
                         frame_override=None, **kwargs):
        instructions.append(extra_instruction)
        return b"\x89PNG bytes"

    monkeypatch.setattr(render, "render_variant", capturing_render)
    monkeypatch.setattr(
        qa, "check",
        lambda path, headline, reference_frame=None, **k: QAResult(
            ok=False,
            problems=["the person in this thumbnail is not the actor from the ad"],
            likeness="DIFFERENT",
        ),
    )
    outcome = pipeline.generate_batch(tmp_path / "ad.mp4", tmp_path / "work")
    assert len(outcome.results) == 5
    reroll_advice = [i for i in instructions if i]
    assert reroll_advice, "a failed likeness check must produce a reroll"
    assert any("invent" in i for i in reroll_advice)
    assert not any("headline larger" in i for i in reroll_advice), (
        "type advice is useless when the defect is an invented person"
    )


def test_the_likeness_gate_receives_the_frame_the_render_used(
    wired, tmp_path, monkeypatch
):
    seen = []

    def capturing_check(path, headline, reference_frame=None, **kwargs):
        seen.append(reference_frame)
        return QAResult(ok=True, transcribed=headline, likeness="SAME")

    monkeypatch.setattr(qa, "check", capturing_check)
    pipeline.generate_batch(tmp_path / "ad.mp4", tmp_path / "work")
    # Three ratios per concept now.
    assert len(seen) == 15
    assert all(f is not None for f in seen), (
        "without a reference frame the likeness gate silently does nothing"
    )
    assert all(f.name.startswith("scene_") for f in seen)
