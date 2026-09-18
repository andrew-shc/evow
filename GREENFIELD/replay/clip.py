"""Extract a short, timestamped RGB clip and write browser-playable video."""

import math
from pathlib import Path

import cv2
import imageio.v2 as imageio
import numpy as np

from .settings import Settings


def extract_clip(
    video_path: str,
    start_seconds: float,
    frame_count: int,
    settings: Settings,
    run_dir: Path,
) -> np.ndarray:
    if not video_path or not Path(video_path).is_file():
        raise ValueError("Upload or record a video before starting a run.")
    if start_seconds < 0:
        raise ValueError("Clip start must be non-negative.")

    capture = cv2.VideoCapture(video_path)
    if not capture.isOpened():
        raise ValueError("The video could not be opened.")
    try:
        source_fps = float(capture.get(cv2.CAP_PROP_FPS))
        if not math.isfinite(source_fps) or source_fps <= 0:
            source_fps = float(settings.fps)
        start_frame = round(start_seconds * source_fps)
        capture.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        frames = []
        source_index = start_frame
        last_frame = None
        for i in range(frame_count):
            target_index = start_frame + round(i * source_fps / settings.fps)
            while source_index <= target_index:
                ok, last_frame = capture.read()
                if not ok:
                    raise ValueError(
                        f"The video ends before {frame_count} frames can be sampled "
                        f"from {start_seconds:.2f}s."
                    )
                source_index += 1
            height, width = last_frame.shape[:2]
            scale = min(1.0, settings.max_side / max(height, width))
            output_width = max(2, round(width * scale / 2) * 2)
            output_height = max(2, round(height * scale / 2) * 2)
            resized = cv2.resize(last_frame, (output_width, output_height))
            frames.append(cv2.cvtColor(resized, cv2.COLOR_BGR2RGB))
    finally:
        capture.release()

    stacked = np.stack(frames, axis=0)
    np.save(run_dir / "frames.npy", stacked)
    write_video(stacked, run_dir / "source.mp4", settings.fps)
    imageio.imwrite(run_dir / "source.png", stacked[0])
    return stacked


def write_video(frames: np.ndarray, path: Path, fps: int) -> None:
    # H.264/yuv420p is playable in the browser. Input dimensions are even.
    with imageio.get_writer(
        path, fps=fps, codec="libx264", quality=8, macro_block_size=1,
    ) as writer:
        for frame in frames:
            writer.append_data(np.asarray(frame, dtype=np.uint8))
