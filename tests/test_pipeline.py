from pathlib import Path

import pytest

from src import pipeline, postprocess, probe, qa, render
from src import plan as plan_module
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
            for i, h, t in MATRIX
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

    def fake_finalize(image_bytes, out_path):
        Path(out_path).write_bytes(image_bytes)
        written.append(Path(out_path))
        return Path(out_path)

    monkeypatch.setattr(postprocess, "finalize", fake_finalize)
    monkeypatch.setattr(qa, "check", lambda *a, **k: QAResult(ok=True))
    return written


def test_batch_returns_exactly_five_results(wired, tmp_path):
    outcome = pipeline.generate_batch(tmp_path / "ad.mp4", tmp_path / "work")
    assert len(outcome.results) == 5
    assert all(r.path is not None for r in outcome.results)
    assert not any(r.flagged for r in outcome.results)


def test_every_result_keeps_its_variant_metadata(wired, tmp_path):
    outcome = pipeline.generate_batch(tmp_path / "ad.mp4", tmp_path / "work")
    pairs = {(r.variant.hook_type, r.variant.treatment) for r in outcome.results}
    assert pairs == {(h, t) for _, h, t in MATRIX}


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
    assert calls["render"] == 10          # five variants, one reroll each
    assert len(outcome.results) == 5
    assert all(r.flagged for r in outcome.results)
    assert all(r.path is not None for r in outcome.results)


def test_reroll_passes_the_failure_reason_back_into_the_prompt(
    wired, tmp_path, monkeypatch
):
    seen = []

    def capturing_render(variant, frames, client=None, extra_instruction="",
                         frame_override=None):
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
                        frame_override=None):
        # Keyed on the argument, not a call counter: five variants render
        # concurrently, so a counter would be racy.
        attempts.append((variant.index, frame_override))
        if frame_override is None:
            raise render.RenderBlocked("filter refused")
        return b"\x89PNG bytes"

    monkeypatch.setattr(render, "render_variant", blocked_then_ok)
    outcome = pipeline.generate_batch(tmp_path / "ad.mp4", tmp_path / "work")
    assert len(outcome.results) == 5
    assert all(r.path is not None for r in outcome.results)
    assert all(override is not None for _, override in attempts[5:]), (
        "the reroll must try a different frame than the blocked one"
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
