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
    # The sample ad is a multi-cut skit and yields the full 8. Asserting a
    # meaningful floor catches an extraction that silently collapses to one
    # frame, while still tolerating a small shift in scene detection.
    assert 4 <= len(frames) <= 8
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


@needs_sample
def test_extract_frames_interval_fallback_with_real_ffmpeg(tmp_path, monkeypatch):
    """Force interval-sampling fallback by raising MIN_SCENE_FRAMES impossibly high."""
    monkeypatch.setattr(probe, "MIN_SCENE_FRAMES", 999)
    frames = probe.extract_frames(SAMPLE, tmp_path, max_frames=4)

    # Assert count matches request
    assert len(frames) == 4

    # Assert all filenames use interval pattern (not scene pattern)
    for f in frames:
        assert f.name.startswith("interval_")
        assert f.name.endswith(".jpg")

    # Assert all are real JPEG images of correct width
    for f in frames:
        assert f.exists() and f.stat().st_size > 0
        with Image.open(f) as im:
            assert im.format == "JPEG"
            assert im.width == 1280

    # Assert frames are distinct (no byte-level duplicates)
    digests = {f.read_bytes()[:2048] for f in frames}
    assert len(digests) == len(frames)


# --- binary lookup -----------------------------------------------------------
# Streamlit Cloud can neither apt-install ffmpeg (expired Debian 11 source in its
# image) nor reliably download one at runtime (static-ffmpeg wrote into
# site-packages and needed github.com; it failed in production). imageio-ffmpeg
# ships the binary INSIDE its wheel, so the fallback is a file that is already
# there. These tests never touch the network: the module is faked in sys.modules
# and shutil.which is stubbed.

def _fake_imageio_ffmpeg(monkeypatch, exe: str | None, on_path: dict[str, str] | None = None):
    import sys, types
    calls = []

    def get_ffmpeg_exe():
        calls.append("get_ffmpeg_exe")
        if exe is None:
            raise RuntimeError("No ffmpeg exe could be found")
        return exe

    fake = types.ModuleType("imageio_ffmpeg")
    fake.get_ffmpeg_exe = get_ffmpeg_exe
    monkeypatch.setitem(sys.modules, "imageio_ffmpeg", fake)
    found = dict(on_path or {})
    monkeypatch.setattr(probe.shutil, "which", lambda name: found.get(name))
    return calls


def test_binary_falls_back_to_the_wheel_bundled_ffmpeg_when_not_on_path(monkeypatch):
    calls = _fake_imageio_ffmpeg(monkeypatch, "/site-packages/imageio_ffmpeg/binaries/ffmpeg-linux")
    assert probe._binary("ffmpeg") == "/site-packages/imageio_ffmpeg/binaries/ffmpeg-linux"
    assert calls == ["get_ffmpeg_exe"]


def test_binary_prefers_system_ffmpeg_and_skips_fallback(monkeypatch):
    calls = _fake_imageio_ffmpeg(monkeypatch, "/bundled", on_path={"ffmpeg": "/opt/homebrew/bin/ffmpeg"})
    assert probe._binary("ffmpeg") == "/opt/homebrew/bin/ffmpeg"
    assert calls == []


def test_binary_raises_probe_error_when_fallback_also_fails(monkeypatch):
    _fake_imageio_ffmpeg(monkeypatch, None)
    with pytest.raises(probe.ProbeError, match="ffmpeg is not installed"):
        probe._binary("ffmpeg")


def test_nothing_asks_for_ffprobe_any_more():
    """imageio-ffmpeg ships ffmpeg only, so the tool must never need ffprobe."""
    import inspect
    assert '_binary("ffprobe")' not in inspect.getsource(probe)


# --- duration without ffprobe ---------------------------------------------------
FFMPEG_I_STDERR = """Input #0, mov,mp4,m4a,3gp,3g2,mj2, from 'ad.mp4':
  Metadata:
    major_brand     : isom
  Duration: 00:00:46.60, start: 0.000000, bitrate: 14825 kb/s
  Stream #0:0[0x1](und): Video: h264 (High) (avc1 / 0x31637634), yuv420p
At least one output file must be specified
"""


def test_parse_duration_reads_ffmpeg_i_output():
    assert probe._parse_duration(FFMPEG_I_STDERR) == pytest.approx(46.6)


def test_parse_duration_handles_hours():
    assert probe._parse_duration("  Duration: 01:02:03.50, start: 0") == pytest.approx(3723.5)


def test_parse_duration_rejects_a_non_video():
    with pytest.raises(probe.ProbeError, match="doesn't look like a video"):
        probe._parse_duration("ad.mp4: Invalid data found when processing input\n")
