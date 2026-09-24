"""Bounded source-video operations shared by Text Query and Text Manipulation."""

from dataclasses import dataclass
from hashlib import sha256
import math
from pathlib import Path

import cv2
import numpy as np


@dataclass(frozen=True)
class VideoInfo:
    """Metadata needed to validate and address a stationary source video."""

    fps: float
    frame_count: int
    duration_seconds: float
    width: int
    height: int


def probe_video(path: str) -> VideoInfo:
    """Inspect metadata without decoding the full source stream."""
    capture = cv2.VideoCapture(path)
    if not capture.isOpened():
        raise ValueError("Could not open the selected video.")
    try:
        fps = float(capture.get(cv2.CAP_PROP_FPS) or 0)
        if not math.isfinite(fps) or fps <= 0:
            fps = 12.0
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        if frame_count < 1 or width < 1 or height < 1:
            raise ValueError("The selected video contains no readable frames.")
        return VideoInfo(fps, frame_count, frame_count / fps, width, height)
    finally:
        capture.release()


def source_digest(path: str) -> str:
    """Return a content hash used to keep cached scenes tied to their source."""
    digest = sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_uniform_samples(path: str, sample_fps: float, maximum_seconds: float) -> tuple[list[np.ndarray], list[float], VideoInfo]:
    """Decode evenly spaced RGB frames for semantic retrieval only."""
    if sample_fps <= 0 or maximum_seconds <= 0:
        raise ValueError("Video sampling rates and limits must be positive.")
    info = probe_video(path)
    if info.duration_seconds > maximum_seconds + 1e-3:
        raise ValueError(
            f"Text Query supports videos up to {maximum_seconds:g} seconds; trim the source video before searching it."
        )
    count = max(1, int(math.ceil(info.duration_seconds * sample_fps)))
    capture = cv2.VideoCapture(path)
    frames: list[np.ndarray] = []
    timestamps: list[float] = []
    try:
        for index in range(count):
            seconds = min(index / sample_fps, max(0.0, info.duration_seconds - 1 / info.fps))
            capture.set(cv2.CAP_PROP_POS_MSEC, seconds * 1000)
            ok, frame = capture.read()
            if not ok:
                continue
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            timestamps.append(seconds)
    finally:
        capture.release()
    if not frames:
        raise ValueError("The selected video contains no readable retrieval samples.")
    return frames, timestamps, info


def interval_start(anchor_seconds: float, clip_seconds: float, source_seconds: float) -> float:
    """Center a requested clip on an anchor while preserving its real duration."""
    if clip_seconds <= 0 or source_seconds <= 0:
        raise ValueError("Interval timing must be positive.")
    return max(0.0, min(anchor_seconds - clip_seconds / 2, max(0.0, source_seconds - clip_seconds)))


def read_interval(path: str, start_seconds: float, duration_seconds: float, output_fps: float) -> tuple[list[np.ndarray], float]:
    """Decode a temporal interval without spatial crop, resize, or aspect change."""
    if start_seconds < 0 or duration_seconds <= 0 or output_fps <= 0:
        raise ValueError("Interval timing must be positive.")
    info = probe_video(path)
    count = max(1, int(round(duration_seconds * output_fps)))
    capture = cv2.VideoCapture(path)
    frames: list[np.ndarray] = []
    try:
        for index in range(count):
            seconds = start_seconds + index / output_fps
            if seconds >= info.duration_seconds:
                break
            capture.set(cv2.CAP_PROP_POS_MSEC, seconds * 1000)
            ok, frame = capture.read()
            if not ok:
                break
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    finally:
        capture.release()
    if not frames:
        raise ValueError("The requested source interval contains no readable frames.")
    return frames, float(output_fps)


def read_editing_episode(path: str, fps: float, maximum_frames: int) -> tuple[list[np.ndarray], float]:
    """Load one short edit episode and reject accidental long-video processing."""
    if fps <= 0 or maximum_frames < 1:
        raise ValueError("Editing episode settings must be positive.")
    info = probe_video(path)
    maximum_seconds = maximum_frames / fps
    if info.duration_seconds > maximum_seconds + 1e-3:
        raise ValueError(
            f"Text Manipulation accepts one episode up to {maximum_seconds:.2f} seconds "
            f"({maximum_frames} frames at {fps:g} fps). Use a Text Query result or trim the source video."
        )
    return read_interval(path, 0, min(info.duration_seconds, maximum_seconds), fps)


def timestamp_label(seconds: float) -> str:
    """Format a stable caption timestamp without leaking machine-local dates."""
    minutes, remainder = divmod(max(0.0, seconds), 60)
    return f"{int(minutes):02d}:{remainder:04.1f}"
