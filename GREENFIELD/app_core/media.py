"""Bounded, deterministic media I/O used by interactive feature adapters."""
from pathlib import Path
import math
import cv2
import imageio.v2 as imageio
import numpy as np

def read_video(path: str, maximum: int = 90) -> tuple[list[np.ndarray], float]:
    """Decode a bounded RGB episode so a browser request cannot exhaust memory."""
    capture = cv2.VideoCapture(path)
    if not capture.isOpened(): raise ValueError("Could not open the selected video.")
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 12.0); frames = []
    while len(frames) < maximum:
        ok, frame = capture.read()
        if not ok: break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    capture.release()
    if not frames: raise ValueError("The selected video contains no readable frames.")
    return frames, fps

def read_recent_window(path: str, seconds: float, fps: int, max_side: int) -> tuple[list[np.ndarray], float]:
    """Sample up to the latest requested duration at fixed cadence.

    Short samples remain usable: all available history is selected rather than
    padding it with duplicated frames.
    """
    if seconds <= 0 or fps <= 0 or max_side < 2: raise ValueError("Future View sampling settings must be positive.")
    capture = cv2.VideoCapture(path)
    if not capture.isOpened(): raise ValueError("Could not open the selected video.")
    try:
        source_fps = float(capture.get(cv2.CAP_PROP_FPS))
        if not math.isfinite(source_fps) or source_fps <= 0: source_fps = float(fps)
        count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        if count < 1: raise ValueError("The selected video contains no readable frames.")
        available = count / source_fps
        output_count = max(1, min(int(round(seconds * fps)), int(math.floor(available * fps))))
        start_seconds = max(0.0, available - seconds)
        frames = []
        for offset in range(output_count):
            index = min(count - 1, round((start_seconds + offset / fps) * source_fps))
            capture.set(cv2.CAP_PROP_POS_FRAMES, index); ok, frame = capture.read()
            if not ok: raise ValueError("The selected video ended while Future View was sampling it.")
            height, width = frame.shape[:2]; scale = min(1.0, max_side / max(height, width))
            size = (max(2, round(width * scale / 2) * 2), max(2, round(height * scale / 2) * 2))
            frames.append(cv2.cvtColor(cv2.resize(frame, size), cv2.COLOR_BGR2RGB))
        return frames, float(fps)
    finally:
        capture.release()

def write_video(frames: list[np.ndarray], path: Path, fps: float) -> Path:
    """Write a browser-playable H.264 result inside a caller-owned artifact path."""
    with imageio.get_writer(path, fps=max(1, fps), codec="libx264", quality=8, macro_block_size=1) as writer:
        for frame in frames: writer.append_data(np.asarray(frame, dtype=np.uint8))
    return path
