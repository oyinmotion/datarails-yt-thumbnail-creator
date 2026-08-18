"""ffmpeg wrappers. No network, no AI, no Streamlit."""

from __future__ import annotations

import logging
import shutil
import subprocess
from pathlib import Path

from .config import FRAME_WIDTH, MAX_FRAMES, MIN_SCENE_FRAMES, SCENE_THRESHOLD

log = logging.getLogger(__name__)


class ProbeError(RuntimeError):
    """Raised when ffmpeg/ffprobe cannot read the video."""


def _binary(name: str) -> str:
    path = shutil.which(name)
    if not path:
        raise ProbeError(
            f"{name} is not installed. On Streamlit Cloud this comes from "
            "packages.txt; locally, `brew install ffmpeg`."
        )
    return path


def _run(cmd: list[str], doing: str) -> str:
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        log.warning(
            "%s failed while %s: %s", cmd[0], doing, result.stderr.strip()
        )
        raise ProbeError(f"Couldn't finish {doing}.")
    return result.stdout


def video_duration(video: Path) -> float:
    if not Path(video).exists():
        raise ProbeError("That video file doesn't exist.")
    out = _run(
        [
            _binary("ffprobe"), "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            str(video),
        ],
        doing="reading the video's duration",
    )
    try:
        return float(out.strip())
    except ValueError as exc:
        raise ProbeError("That file doesn't look like a video.") from exc


def _interval_timestamps(duration: float, count: int) -> list[float]:
    """Evenly spread timestamps, avoiding the very first and last frame."""
    step = duration / (count + 1)
    return [round(step * (i + 1), 3) for i in range(count)]


def extract_frames(
    video: Path, out_dir: Path, max_frames: int = MAX_FRAMES
) -> list[Path]:
    """Scene-change frames, falling back to even sampling on flat footage."""
    video = Path(video)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if not video.exists():
        raise ProbeError("That video file doesn't exist.")

    _run(
        [
            _binary("ffmpeg"), "-v", "error", "-i", str(video),
            "-vf", f"select='gt(scene,{SCENE_THRESHOLD})',scale={FRAME_WIDTH}:-2",
            "-vsync", "vfr", "-q:v", "3",
            str(out_dir / "scene_%03d.jpg"),
        ],
        doing="pulling scene-change frames",
    )
    frames = sorted(out_dir.glob("scene_*.jpg"))

    if len(frames) < MIN_SCENE_FRAMES:
        # Flat footage (one continuous shot): sample by time instead.
        for f in frames:
            f.unlink()
        duration = video_duration(video)
        for i, ts in enumerate(_interval_timestamps(duration, max_frames)):
            _run(
                [
                    _binary("ffmpeg"), "-v", "error", "-ss", str(ts),
                    "-i", str(video), "-frames:v", "1",
                    "-vf", f"scale={FRAME_WIDTH}:-2", "-q:v", "3",
                    str(out_dir / f"interval_{i:03d}.jpg"),
                ],
                doing="pulling a frame",
            )
        frames = sorted(out_dir.glob("interval_*.jpg"))

    if not frames:
        raise ProbeError("Couldn't pull any frames out of that video.")

    # Keep an even spread rather than only the first N.
    if len(frames) > max_frames:
        stride = len(frames) / max_frames
        keep = {frames[int(i * stride)] for i in range(max_frames)}
        for f in frames:
            if f not in keep:
                f.unlink()
        frames = sorted(keep)

    return frames


def extract_audio(video: Path, out_dir: Path) -> Path:
    """Mono 16kHz m4a — small enough to upload for transcription."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    audio = out_dir / "audio.m4a"
    _run(
        [
            _binary("ffmpeg"), "-v", "error", "-y", "-i", str(video),
            "-vn", "-ac", "1", "-ar", "16000", "-c:a", "aac", "-b:a", "32k",
            str(audio),
        ],
        doing="extracting the audio track",
    )
    if not audio.exists() or audio.stat().st_size == 0:
        raise ProbeError("That video has no usable audio track.")
    return audio
