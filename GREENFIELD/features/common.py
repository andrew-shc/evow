"""Small shared helpers for the three local camera feature pages."""

from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import cv2
import imageio.v2 as imageio
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
OUTPUTS = ROOT / "ASSETS" / "features"


def run_dir(feature: str) -> Path:
    """Create one ignored artifact directory for a feature invocation."""
    target = OUTPUTS / feature / f"{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}_{uuid4().hex[:8]}"
    target.mkdir(parents=True, exist_ok=True)
    return target


def read_video(path: str, maximum: int = 90) -> tuple[list[np.ndarray], float]:
    """Decode a bounded RGB episode using OpenCV, keeping interactive runs fast."""
    capture = cv2.VideoCapture(path)
    if not capture.isOpened():
        raise ValueError("Could not open the selected video.")
    fps = capture.get(cv2.CAP_PROP_FPS) or 12.0
    frames: list[np.ndarray] = []
    while len(frames) < maximum:
        ok, frame = capture.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    capture.release()
    if not frames:
        raise ValueError("The selected video contains no readable frames.")
    return frames, float(fps)


def write_video(frames: list[np.ndarray], path: Path, fps: float) -> str:
    """Write a browser-playable H.264 result beneath ASSETS."""
    with imageio.get_writer(path, fps=max(1, fps), codec="libx264", quality=8, macro_block_size=1) as writer:
        for frame in frames:
            writer.append_data(np.asarray(frame, dtype=np.uint8))
    return str(path)


def flow_html(title: str, mode: str, stages: list[tuple[str, str]]) -> str:
    """Render a shallow collapsible flow used by each new feature page."""
    rows = "".join(
        f'<details><summary>{number}. {name}</summary><pre>{detail}</pre></details>'
        for number, (name, detail) in enumerate(stages, 1)
    )
    return f'''<section style="background:transparent;color:#17212b;border:0;border-radius:0;padding:4px 0;font:14px system-ui">

{rows}</section>'''


def guide_html(*steps: str) -> str:
    """Render a quiet numbered route without decorative controls."""
    labels = " → ".join(f"{index}. {step}" for index, step in enumerate(steps, 1))
    return f'<div style="margin:0 0 12px;color:#425466;font:13px system-ui">{labels}</div>'
