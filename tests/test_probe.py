from pathlib import Path

import pytest
from PIL import Image

from src import probe
from src.config import PROJECT_ROOT

SAMPLE = next((PROJECT_ROOT / "Sample input ad").glob("*.mp4"), None)
needs_sample = pytest.mark.skipif(SAMPLE is None, reason="sample ad not present")


@needs_sample
def test_video_duration_matches_the_sample_ad():
    assert probe.video_duration(SAMPLE) == pytest.approx(46.6, abs=0.5)


@needs_sample
def test_extract_frames_returns_real_jpegs_capped_at_max(tmp_path):
    frames = probe.extract_frames(SAMPLE, tmp_path, max_frames=8)
    assert 1 <= len(frames) <= 8
    for f in frames:
        assert f.exists() and f.stat().st_size > 0
        with Image.open(f) as im:
            assert im.format == "JPEG"
            assert im.width == 1280


@needs_sample
def test_extract_frames_are_distinct_moments(tmp_path):
    """Scene selection must not return the same frame repeatedly."""
    frames = probe.extract_frames(SAMPLE, tmp_path, max_frames=6)
    digests = {f.read_bytes()[:2048] for f in frames}
    assert len(digests) == len(frames)


@needs_sample
def test_extract_audio_produces_a_small_mono_file(tmp_path):
    audio = probe.extract_audio(SAMPLE, tmp_path)
    assert audio.exists()
    assert audio.stat().st_size > 1024
    # Mono 16kHz 32kbps over ~47s must be far smaller than the 86MB source.
    assert audio.stat().st_size < 1_000_000


def test_missing_video_raises_probe_error(tmp_path):
    with pytest.raises(probe.ProbeError):
        probe.extract_frames(tmp_path / "nope.mp4", tmp_path)


def test_interval_timestamps_are_evenly_spread_and_inside_the_video():
    ts = probe._interval_timestamps(duration=40.0, count=4)
    assert len(ts) == 4
    assert all(0 < t < 40.0 for t in ts)
    assert ts == sorted(ts)
