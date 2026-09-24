"""Fit a native dynamic Gaussian scene and render a nearby camera path."""

import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

from .pose import ViewRequest
from .settings import Settings


def _worker(command: list[str], log_path: Path, stage: str, detail: str, progress_path: Path | None = None):
    """Keep CUDA model memory isolated while streaming progress into the run trace."""
    with log_path.open("w") as log_file:
        process = subprocess.Popen(command, stdout=log_file, stderr=subprocess.STDOUT)
        started = time.monotonic()
        last_step = None
        yield {"stage": stage, "status": "running", "detail": detail}
        while process.poll() is None:
            time.sleep(0.05)
            progress = None
            if progress_path and progress_path.is_file():
                try:
                    progress = json.loads(progress_path.read_text())
                except json.JSONDecodeError:
                    pass
            # The Gaussian worker uses this file for more than optimizer
            # updates. Its later export/render records report completed
            # keyframes, not a training ``step`` or image loss. Only consume
            # the training-shaped records here; otherwise keep the ordinary
            # heartbeat alive until the subprocess exits.
            if progress and "step" in progress and progress.get("step") != last_step:
                last_step = progress["step"]
                yield {
                    "stage": stage, "status": "running",
                    "detail": f"Fitting 3D Gaussians · step {progress['step']}/{progress['total']} · image loss {progress['loss']}",
                    **({"preview": Path(progress["preview"])} if progress.get("preview") else {}),
                    "metrics": {"step": progress["step"], "image_loss": progress["loss"],
                                "gaussians": progress["gaussians"]},
                }
            else:
                yield {
                    "stage": stage, "status": "running",
                    "detail": f"{detail} · {time.monotonic() - started:.2f}s elapsed",
                }
    if process.returncode:
        tail = "\n".join(log_path.read_text(errors="replace").splitlines()[-15:])
        raise RuntimeError(f"{stage} failed. Log: {log_path}\n{tail}")


def run_explicit(frames: np.ndarray, request: ViewRequest, source_fov_degrees: float, run_dir: Path, settings: Settings):
    if not settings.depth_checkpoint.is_file():
        raise FileNotFoundError(f"Missing depth initializer weights: {settings.depth_checkpoint}")
    if not (settings.depth_repo / "video_depth_anything" / "video_depth.py").is_file():
        raise FileNotFoundError("Video-Depth-Anything checkout is missing.")
    output_dir = run_dir / "explicit"
    output_dir.mkdir()
    depth_path = output_dir / "depths.npz"
    depth_command = [
        sys.executable, "-m", "GREENFIELD.replay.depth_worker",
        "--frames", str(run_dir / "frames.npy"), "--out", str(depth_path),
    ]
    yield from _worker(
        depth_command, run_dir / "depth.log", "3D initialization",
        "Estimating a depth prior to initialize a native Gaussian scene.",
    )
    if not depth_path.is_file():
        raise RuntimeError("Depth initialization produced no file.")
    yield {
        "stage": "3D initialization", "status": "complete",
        "detail": "The depth prior and optical flow are ready for Gaussian fitting.",
        "preview": run_dir / "source.png", "metrics": {"frames": len(frames)},
    }

    command = [
        sys.executable, "-m", "GREENFIELD.replay.gaussian_worker",
        "--frames", str(run_dir / "frames.npy"), "--depths", str(depth_path),
        "--out", str(output_dir), "--render-yaw", str(request.yaw_degrees),
        "--render-shift", str(request.lateral_shift), "--render-fov", str(request.fov_degrees),
        "--source-fov", str(source_fov_degrees),
    ]
    yield from _worker(
        command, run_dir / "gaussian.log", "4D Gaussian fitting",
        "Optimizing persistent 3D Gaussians across the selected frames.",
        output_dir / "progress.json",
    )
    models = sorted(output_dir.glob("splat_*.splat"))
    video = output_dir / "rendered.mp4"
    if len(models) != len(frames) or not video.is_file() or not (output_dir / "scene.pt").is_file():
        raise RuntimeError("Gaussian fitting did not save a complete 4D scene.")
    yield {
        "stage": "Native 4D scene", "status": "complete",
        "detail": "The persistent Gaussian scene, time-varying splats, and viewpoint video are ready.",
        "preview": output_dir / "render_000.png",
        "metrics": {"frames": len(models), "representation": "3D Gaussians with temporal deformation"},
    }
    return {"video": video, "models": models}
