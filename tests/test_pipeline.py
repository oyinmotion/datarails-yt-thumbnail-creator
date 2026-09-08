import io
from pathlib import Path

import pytest
from PIL import Image

from src import backoff, branding, pipeline, postprocess, probe, qa, render, typeset
from src import plan as plan_module
from src.config import FINAL_H, FINAL_W, GEN_SIZE, RATIOS

_REAL_TYPESET = typeset.typeset
_REAL_STAMP = branding.stamp_logo_image
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


def _flat_png(ratio="16x9", colour=(30, 60, 120)) -> bytes:
    """Real PNG bytes at the ratio's generation size, so finalize runs for real."""
    buf = io.BytesIO()
    Image.new("RGB", RATIOS[ratio][0], colour).save(buf, "PNG")
    return buf.getvalue()


@pytest.fixture
def wired(monkeypatch, tmp_path):
    """Replace every external dependency with a deterministic fake.

    typeset and the logo stamp are swapped for fast identities here — fifteen
    real 2048px typesets per test would make the suite crawl. The integration
    test below and the zone test restore the real ones.
    """
    frames = []
    for name in ("scene_001.jpg", "scene_002.jpg"):
        frame = tmp_path / name
        Image.new("RGB", (1280, 720), (60, 60, 60)).save(frame, "JPEG")
        frames.append(frame)
    audio = tmp_path / "audio.m4a"
    audio.write_bytes(b"fake")

    monkeypatch.setattr(probe, "extract_frames",
                        lambda video, out_dir, max_frames=16: frames)
    monkeypatch.setattr(probe, "extract_audio", lambda video, out_dir: audio)
    monkeypatch.setattr(plan_module, "build_plan", lambda *a, **k: _plan())
    monkeypatch.setattr(render, "render_art",
                        lambda *a, **k: _flat_png(k.get("ratio", "16x9")))
    monkeypatch.setattr(
        qa, "likeness_gate",
        lambda path, frame, **k: QAResult(ok=True, likeness="SAME", checked=True),
    )
    monkeypatch.setattr(
        typeset, "typeset",
        lambda art, headline, style, treatment, ratio, scrim=False:
            typeset.TypesetResult(image=art, notes=[], scrimmed=scrim),
    )
    monkeypatch.setattr(branding, "stamp_logo_image", lambda img: img)
    # Backoff is real in production and instant in tests.
    monkeypatch.setattr(pipeline, "DEFAULT_SLEEPER", lambda seconds: None)
    monkeypatch.setattr(plan_module, "DEFAULT_SLEEPER", lambda seconds: None)
    return frames


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


def test_a_failed_likeness_triggers_exactly_one_reroll(wired, tmp_path, monkeypatch):
    calls = {"render": 0}

    def counting_render(*args, **kwargs):
        calls["render"] += 1
        return _flat_png(kwargs.get("ratio", "16x9"))

    monkeypatch.setattr(render, "render_art", counting_render)
    monkeypatch.setattr(
        qa, "likeness_gate",
        lambda path, frame, **k: QAResult(
            ok=False, problems=["the person in this thumbnail is not the actor from the ad"],
            likeness="DIFFERENT", checked=True,
        ),
    )
    outcome = pipeline.generate_batch(tmp_path / "ad.mp4", tmp_path / "work")
    # Five concepts x three ratios = 15 renders, each rerolled once.
    assert calls["render"] == 30
    assert len(outcome.results) == 5
    assert all(r.flagged for r in outcome.results)
    assert all(r.path is not None for r in outcome.results)


def test_a_render_that_always_fails_still_yields_a_result_row(
    wired, tmp_path, monkeypatch
):
    def always_fails(*args, **kwargs):
        raise render.RenderError("503 from the API")

    monkeypatch.setattr(render, "render_art", always_fails)
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
        attempts.append(((variant.index, kwargs.get("ratio")), frame_override))
        if frame_override is None:
            raise render.RenderBlocked("filter refused")
        return _flat_png(kwargs.get("ratio", "16x9"))

    monkeypatch.setattr(render, "render_art", blocked_then_ok)
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
        return _flat_png(kwargs.get("ratio", "16x9"))

    monkeypatch.setattr(render, "render_art", explode_for_variant_three)
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
    real = postprocess.finalize_image

    def sometimes_broken(image, out_path, final_size=None, **kwargs):
        if "03_" in Path(out_path).name:
            raise OSError("cannot identify image file")
        return real(image, out_path, final_size or (FINAL_W, FINAL_H))

    monkeypatch.setattr(postprocess, "finalize_image", sometimes_broken)
    outcome = pipeline.generate_batch(tmp_path / "ad.mp4", tmp_path / "work")
    assert len(outcome.results) == 5
    assert sum(1 for r in outcome.results if r.path is None) == 1


def test_a_failed_render_backs_off_before_the_reroll(wired, tmp_path, monkeypatch):
    delays = []
    monkeypatch.setattr(pipeline, "DEFAULT_SLEEPER", delays.append)

    def always_fails(*args, **kwargs):
        raise render.RenderError("429 rate limit")

    monkeypatch.setattr(render, "render_art", always_fails)
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


def test_an_unverified_batch_warns_the_user(wired, tmp_path, monkeypatch):
    """A QA outage fails open, which must never be silent."""
    monkeypatch.setattr(
        qa, "likeness_gate", lambda path, frame, **k: QAResult(ok=True, checked=False),
    )
    outcome = pipeline.generate_batch(tmp_path / "ad.mp4", tmp_path / "work")
    assert all(r.unverified for r in outcome.results)
    assert any("could not be checked" in w and "likeness" in w for w in outcome.warnings)
    assert any("5 of 5" in w for w in outcome.warnings)


def test_a_verified_batch_does_not_warn(wired, tmp_path):
    outcome = pipeline.generate_batch(tmp_path / "ad.mp4", tmp_path / "work")
    assert not any("could not be checked" in w for w in outcome.warnings)


# --- the real typeset, logo and single downscale, end to end -----------------
def test_a_real_2048x1152_art_becomes_a_1920x1080_thumbnail_with_the_headline(tmp_path, monkeypatch):
    """The integration the suite must never fake on both sides: real typeset,
    real logo stamp, real single downscale, real hard checks."""
    frames = []
    for name in ("scene_001.jpg", "scene_002.jpg"):
        frame = tmp_path / name
        Image.new("RGB", (1280, 720), (60, 60, 60)).save(frame, "JPEG")
        frames.append(frame)
    audio = tmp_path / "audio.m4a"
    audio.write_bytes(b"fake")
    monkeypatch.setattr(probe, "extract_frames", lambda video, out_dir, max_frames=16: frames)
    monkeypatch.setattr(probe, "extract_audio", lambda video, out_dir: audio)
    monkeypatch.setattr(plan_module, "build_plan", lambda *a, **k: _plan())
    # Mid-grey art: light type (house, cinematic, flat) reads above it and dark
    # type (clean corporate's navy) reads below it, so one assertion covers all.
    monkeypatch.setattr(render, "render_art",
                        lambda *a, **k: _flat_png(k.get("ratio", "16x9"), (110, 110, 110)))
    monkeypatch.setattr(qa, "likeness_gate",
                        lambda path, frame, **k: QAResult(ok=True, likeness="SAME", checked=True))
    monkeypatch.setattr(pipeline, "DEFAULT_SLEEPER", lambda seconds: None)
    # typeset, stamp_logo_image and finalize_image are the REAL ones here.

    outcome = pipeline.generate_batch(tmp_path / "ad.mp4", tmp_path / "work")

    for result in outcome.results:
        assert result.path is not None and not result.flagged, result.note
        assert qa.hard_checks(result.path) == []
        with Image.open(result.path) as im:
            assert im.size == (FINAL_W, FINAL_H)
            box = typeset.zone_box(result.variant.treatment, "16x9", im.size)
            levels = im.crop(box).convert("L").get_flattened_data()
            assert max(levels) > 200 or min(levels) < 40, (
                f"no type in the zone for {result.variant.treatment}/{result.style}")
        with Image.open(result.art_paths["16x9"]) as art:
            assert art.size == (2048, 1152), "art is cached at generation size"


def test_the_legibility_model_is_never_called(wired, tmp_path):
    assert not hasattr(qa, "check")
    outcome = pipeline.generate_batch(tmp_path / "ad.mp4", tmp_path / "work")
    assert all(not r.unverified for r in outcome.results)


def test_every_result_records_its_headline_and_style(wired, tmp_path):
    outcome = pipeline.generate_batch(tmp_path / "ad.mp4", tmp_path / "work")
    for r in outcome.results:
        assert r.headline == r.variant.headline
        assert r.style == r.variant.style
        assert set(r.art_paths) == set(r.paths) == {"16x9", "1x1", "9x16"}


def test_the_outcome_carries_what_reroll_needs(wired, tmp_path):
    outcome = pipeline.generate_batch(tmp_path / "ad.mp4", tmp_path / "work")
    assert set(outcome.frames) == {"scene_001.jpg", "scene_002.jpg"}
    assert outcome.people_in_ad is True
    assert outcome.work_dir == tmp_path / "work"


def test_art_painted_into_the_zone_is_rerolled_once_then_scrimmed(wired, tmp_path, monkeypatch):
    monkeypatch.setattr(typeset, "typeset", _REAL_TYPESET)      # the scrim note comes from here
    monkeypatch.setattr(plan_module, "build_plan", lambda *a, **k: _plan_of(1))

    def noisy(*args, **kwargs):
        w, h = RATIOS[kwargs.get("ratio", "16x9")][0]
        buf = io.BytesIO()
        Image.effect_noise((w, h), 80).convert("RGB").save(buf, "PNG")
        return buf.getvalue()

    seen = []

    def capturing(*args, **kwargs):
        seen.append(kwargs.get("extra_instruction", ""))
        return noisy(**kwargs)

    monkeypatch.setattr(render, "render_art", capturing)
    outcome = pipeline.generate_batch(tmp_path / "ad.mp4", tmp_path / "work", variant_count=1)
    assert outcome.render_calls == 6, "one reroll per ratio"
    assert any("reserved" in s for s in seen if s)
    r = outcome.results[0]
    assert r.flagged and "scrim" in r.note
    assert all(p is not None for p in r.paths.values())


def test_an_invented_person_triggers_a_reroll_with_likeness_advice(
    wired, tmp_path, monkeypatch
):
    """A likeness failure must not be met with 'render the headline larger'."""
    instructions = []

    def capturing_render(variant, frames, client=None, extra_instruction="",
                         frame_override=None, **kwargs):
        instructions.append(extra_instruction)
        return _flat_png(kwargs.get("ratio", "16x9"))

    monkeypatch.setattr(render, "render_art", capturing_render)
    monkeypatch.setattr(
        qa, "likeness_gate",
        lambda path, frame, **k: QAResult(
            ok=False,
            problems=["the person in this thumbnail is not the actor from the ad"],
            likeness="DIFFERENT", checked=True,
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

    def capturing_gate(path, reference_frame, **kwargs):
        seen.append(reference_frame)
        return QAResult(ok=True, likeness="SAME", checked=True)

    monkeypatch.setattr(qa, "likeness_gate", capturing_gate)
    pipeline.generate_batch(tmp_path / "ad.mp4", tmp_path / "work")
    # Three ratios per concept now.
    assert len(seen) == 15
    assert all(f is not None for f in seen), (
        "without a reference frame the likeness gate silently does nothing"
    )
    assert all(f.name.startswith("scene_") for f in seen)


def _plan_of(count):
    from src.models import matrix_for
    return BatchPlan(
        ad_summary="same AI, different answers",
        transcript_used=True,
        variants=[
            Variant(
                index=i, hook_type=h, treatment=t, headline=f"HOOK {i}",
                frame_id="scene_001.jpg", second_frame_id=None,
                scene_direction="sparks", rationale="from the ad",
            )
            for i, h, t, _s in matrix_for(count)
        ],
    )


def test_asking_for_two_concepts_renders_two_rows_and_six_images(
    wired, tmp_path, monkeypatch
):
    monkeypatch.setattr(plan_module, "build_plan", lambda *a, **k: _plan_of(2))
    calls = {"n": 0}

    def counting(*args, **kwargs):
        calls["n"] += 1
        return _flat_png(kwargs.get("ratio", "16x9"))

    monkeypatch.setattr(render, "render_art", counting)
    outcome = pipeline.generate_batch(
        tmp_path / "ad.mp4", tmp_path / "work", variant_count=2,
    )
    assert len(outcome.results) == 2
    assert calls["n"] == 6, "two concepts at three ratios"
    assert all(len(r.paths) == 3 for r in outcome.results)


def test_the_requested_count_reaches_the_planner(wired, tmp_path, monkeypatch):
    seen = {}

    def capturing_plan(*args, **kwargs):
        seen["count"] = kwargs.get("variant_count")
        return _plan_of(kwargs.get("variant_count", 5))

    monkeypatch.setattr(plan_module, "build_plan", capturing_plan)
    pipeline.generate_batch(
        tmp_path / "ad.mp4", tmp_path / "work", variant_count=3,
    )
    assert seen["count"] == 3


def test_one_concept_still_produces_a_row(wired, tmp_path, monkeypatch):
    monkeypatch.setattr(plan_module, "build_plan", lambda *a, **k: _plan_of(1))
    outcome = pipeline.generate_batch(
        tmp_path / "ad.mp4", tmp_path / "work", variant_count=1,
    )
    assert len(outcome.results) == 1
    assert outcome.results[0].path is not None


# --- one wave, and what it cost -------------------------------------------
def test_a_full_batch_renders_in_a_single_wave():
    """Five concepts x three ratios is fifteen renders. With fewer workers than
    that the batch is two waves, and the second wave only starts when the
    slowest render of the first is done — the single biggest chunk of wall
    clock a user waits through."""
    from src.config import RATIOS
    assert pipeline.MAX_WORKERS >= len(MATRIX) * len(RATIOS)


def test_a_clean_batch_reports_fifteen_renders_and_no_rerolls(wired, tmp_path):
    outcome = pipeline.generate_batch(tmp_path / "ad.mp4", tmp_path / "work")
    assert outcome.render_calls == 15
    assert outcome.rerolls == 0


def test_rerolls_are_counted_so_the_real_cost_can_be_shown(
    wired, tmp_path, monkeypatch
):
    """The pre-run caption estimates one render per image. Rerolls make the
    real bill higher, and until now invisibly so."""
    monkeypatch.setattr(
        qa, "likeness_gate",
        lambda path, frame, **k: QAResult(ok=False, problems=["not the actor"],
                                          likeness="DIFFERENT", checked=True),
    )
    outcome = pipeline.generate_batch(tmp_path / "ad.mp4", tmp_path / "work")
    assert outcome.render_calls == 30
    assert outcome.rerolls == 15


def test_a_render_that_never_answered_still_counts_its_attempts(
    wired, tmp_path, monkeypatch
):
    def always_fails(*args, **kwargs):
        raise render.RenderError("503 from the API")

    monkeypatch.setattr(render, "render_art", always_fails)
    outcome = pipeline.generate_batch(tmp_path / "ad.mp4", tmp_path / "work")
    # Two attempts per job, none produced an image. Failed calls are not
    # billed, so the money figure excludes them — see cost_line.
    assert outcome.render_calls == 30
    assert outcome.images_billed == 0


# --- per-tile actions: retitle and reroll ------------------------------------
def test_retitle_recomposes_every_ratio_for_free(wired, tmp_path, monkeypatch):
    monkeypatch.setattr(plan_module, "build_plan", lambda *a, **k: _plan_of(1))
    outcome = pipeline.generate_batch(tmp_path / "ad.mp4", tmp_path / "work")
    monkeypatch.setattr(typeset, "typeset", _REAL_TYPESET)   # so the re-set is visible
    calls = {"n": 0}

    def no_render(*a, **k):
        calls["n"] += 1
        raise AssertionError("retitle must not render")

    monkeypatch.setattr(render, "render_art", no_render)
    before = {r: outcome.results[0].paths[r].read_bytes() for r in outcome.results[0].paths}
    new = pipeline.retitle(outcome.results[0], "new line here")
    assert calls["n"] == 0, "no image call"
    assert new.headline == "NEW LINE HERE"
    assert set(new.paths) == {"16x9", "1x1", "9x16"}
    for ratio, path in new.paths.items():
        assert path.read_bytes() != before[ratio], f"{ratio} was re-set"
    assert new.art_paths == outcome.results[0].art_paths, "the art is untouched"
    assert not new.flagged


def test_retitle_flags_a_headline_that_does_not_fit(wired, tmp_path, monkeypatch):
    monkeypatch.setattr(plan_module, "build_plan", lambda *a, **k: _plan_of(1))
    outcome = pipeline.generate_batch(tmp_path / "ad.mp4", tmp_path / "work")
    monkeypatch.setattr(typeset, "typeset", _REAL_TYPESET)
    new = pipeline.retitle(outcome.results[0], "EXTRAORDINARILY LONG WORDS EVERYWHERE HERE")
    assert new.flagged and "too long" in new.note


def test_reroll_rerenders_all_three_ratios_and_bills_them(wired, tmp_path, monkeypatch):
    outcome = pipeline.generate_batch(tmp_path / "ad.mp4", tmp_path / "work")
    calls = {"n": 0}

    def counting(*a, **k):
        calls["n"] += 1
        return _flat_png(k.get("ratio", "16x9"), (90, 20, 20))

    monkeypatch.setattr(render, "render_art", counting)
    before_calls, before_billed = outcome.render_calls, outcome.images_billed
    new = pipeline.reroll(outcome.results[0], outcome)
    assert calls["n"] == 3
    assert outcome.render_calls == before_calls + 3
    assert outcome.images_billed == before_billed + 3
    assert new.headline == outcome.results[0].headline
    assert new.style == outcome.results[0].style
    assert new.variant.index == outcome.results[0].variant.index
    assert all(p.exists() for p in new.paths.values())


def test_reroll_with_a_style_swaps_the_look(wired, tmp_path, monkeypatch):
    outcome = pipeline.generate_batch(tmp_path / "ad.mp4", tmp_path / "work")
    seen = []

    def capturing(*a, **k):
        seen.append(k.get("style"))
        return _flat_png(k.get("ratio", "16x9"))

    monkeypatch.setattr(render, "render_art", capturing)
    new = pipeline.reroll(outcome.results[0], outcome, style="dark_cinematic")
    assert seen == ["dark_cinematic"] * 3
    assert new.style == "dark_cinematic"
    assert new.variant.style == "house_energy", "the matrix slot is unchanged; the row is off-matrix"


def test_reroll_keeps_the_hard_contract_when_the_api_fails(wired, tmp_path, monkeypatch):
    monkeypatch.setattr(plan_module, "build_plan", lambda *a, **k: _plan_of(1))
    outcome = pipeline.generate_batch(tmp_path / "ad.mp4", tmp_path / "work")
    assert outcome.images_billed == 3

    def fails(*a, **k):
        raise render.RenderError("503")

    monkeypatch.setattr(render, "render_art", fails)
    new = pipeline.reroll(outcome.results[0], outcome)
    assert new.variant.index == 1
    assert all(new.paths.get(r) is None for r in ("16x9", "1x1", "9x16"))
    assert "503" in new.note
    assert outcome.images_billed == 3, "failed calls are not billed"
    assert outcome.render_calls == 3 + 6, "two attempts per ratio were made"
