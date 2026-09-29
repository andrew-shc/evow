"""Read locally attached V4L2 cameras into bounded dashboard source clips."""

from datetime import datetime, timezone
from pathlib import Path
import glob
import threading
import time

import cv2
import numpy as np

from GREENFIELD.app_core.media import write_video


class LiveCamera:
    """Serialize direct camera access for dashboard previews and clip capture."""

    def __init__(self, assets: Path) -> None:
        self._assets = assets / "live_camera" / "captures"
        self._lock = threading.Lock()

    def choices(self) -> list[tuple[str, str]]:
        """Return local Linux video devices without opening them."""
        choices = []
        for device in sorted(glob.glob("/dev/video*")):
            name_file = Path("/sys/class/video4linux") / Path(device).name / "name"
            try:
                name = name_file.read_text().strip()
            except OSError:
                name = "Camera"
            choices.append((f"{name} ({device})", device))
        return choices

    def preview(self, device: str | None) -> tuple[np.ndarray | None, str]:
        """Read one RGB frame for the polling preview without retaining a handle."""
        if not device:
            return None, "Select a camera to preview it."
        with self._lock:
            capture = cv2.VideoCapture(device, cv2.CAP_V4L2)
            try:
                ok, frame = capture.read() if capture.isOpened() else (False, None)
            finally:
                capture.release()
        if not ok or frame is None:
            return None, f"Could not read {device}. Check that it is connected and accessible."
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), f"Live from {device}."
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), ""
    def capture(self, device: str | None, seconds: float, fps: float = 12) -> tuple[str | None, str]:
        """Record a finite source clip so existing pipelines retain file semantics."""
        if not device:
            return None, "Select a camera before capturing."
        seconds = max(1.0, min(float(seconds), 30.0))
        interval = 1 / fps
        frames: list[np.ndarray] = []
        deadline = time.monotonic() + seconds
        next_frame = time.monotonic()
        with self._lock:
            camera = cv2.VideoCapture(device, cv2.CAP_V4L2)
            if not camera.isOpened():
                camera.release()
                return None, f"Could not open {device}. Check camera permissions."
            try:
                while time.monotonic() < deadline:
                    ok, frame = camera.read()
                    if ok:
                        now = time.monotonic()
                        if now >= next_frame:
                            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                            next_frame = now + interval
                    else:
                        break
            finally:
                camera.release()
        if len(frames) < 2:
            return None, "The camera did not provide enough frames to create a clip."
        self._assets.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        path = self._assets / f"live-camera-{stamp}.mp4"
        write_video(frames, path, fps)
        return str(path), f"Captured {len(frames) / fps:.1f} seconds from {device}."
