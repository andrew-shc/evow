"""Run explicit 4D Gaussian reconstruction in isolated, traceable workers."""

from pathlib import Path
import subprocess
import sys
import time

import numpy as np

from .overrides import settings_json
from .pose import ViewRequest
from .settings import Settings
from .worker_events import read_events


def _log_tail(path: Path, lines: int = 16) -> str:
    """Keep trace events useful without copying an unbounded worker log."""
    if not path.is_file():
        return ""
    return "\n".join(path.read_text(errors="replace").splitlines()[-lines:])


def _worker(
    command: list[str],
    log_path: Path,
    events_path: Path,
    startup_stage: str,
    startup_detail: str,
):
    """Forward every atomically recorded worker event, including final events."""
    with log_path.open("w") as log_file:
        process = subprocess.Popen(command, stdout=log_file, stderr=subprocess.STDOUT)
        last_sequence = 0
        yield {
            "stage_id": startup_stage, "status": "running", "detail": startup_detail,
            "log_path": log_path,
        }
        while True:
            for event in read_events(events_path, last_sequence):
                last_sequence = max(last_sequence, int(event["sequence"]))
                metrics = dict(event.get("metrics") or {})
                tail = _log_tail(log_path)
                if tail:
                    metrics["log_tail"] = tail
                yield {
                    "stage_id": event["stage_id"], "status": event["status"],
                    "detail": event["detail"], "metrics": metrics,
                    "preview": Path(event["preview"]) if event.get("preview") else None,
                    "execution": event.get("execution", "serial"), "log_path": log_path,
                }
            if process.poll() is not None:
                break
            time.sleep(0.05)
    # The atomic sidecar can receive events between the final polling pass and
    # child exit, so drain once after the process is known to be complete.
    for event in read_events(events_path, last_sequence):
        last_sequence = max(last_sequence, int(event["sequence"]))
        metrics = dict(event.get("metrics") or {})
        tail = _log_tail(log_path)
        if tail:
            metrics["log_tail"] = tail
        yield {
            "stage_id": event["stage_id"], "status": event["status"],
            "detail": event["detail"], "metrics": metrics,
            "preview": Path(event["preview"]) if event.get("preview") else None,
            "execution": event.get("execution", "serial"), "log_path": log_path,
        }
    if process.returncode:
        raise RuntimeError(f"Explicit worker failed. Log: {log_path}\n{_log_tail(log_path)}")


def depth_command(run_dir: Path, depth_path: Path, settings: Settings, events_path: Path | None = None) -> list[str]:
    """Build the isolated depth-worker command with its event sidecar."""
    command = [
        sys.executable, "-m", "GREENFIELD.replay.depth_worker",
        "--frames", str(run_dir / "frames.npy"), "--out", str(depth_path),
        "--settings-json", settings_json(settings),
    ]
    if events_path is not None:
        command.extend(("--events", str(events_path)))
    return command


def gaussian_command(
    run_dir: Path,
    output_dir: Path,
    depth_path: Path,
    request: ViewRequest,
    source_fov_degrees: float,
    settings: Settings,
    events_path: Path | None = None,
) -> list[str]:
    """Build the isolated Gaussian-worker command with its event sidecar."""
    command = [
        sys.executable, "-m", "GREENFIELD.replay.gaussian_worker",
        "--frames", str(run_dir / "frames.npy"), "--depths", str(depth_path),
        "--out", str(output_dir), "--render-yaw", str(request.yaw_degrees),
        "--render-shift", str(request.lateral_shift), "--render-fov", str(request.fov_degrees),
        "--source-fov", str(source_fov_degrees), "--settings-json", settings_json(settings),
    ]
    if events_path is not None:
        command.extend(("--events", str(events_path)))
    return command


def run_explicit(
    frames: np.ndarray,
    request: ViewRequest,
    source_fov_degrees: float,
    run_dir: Path,
    settings: Settings,
):
    """Yield one serial update for each real explicit-reconstruction operation."""
    if not settings.depth_checkpoint.is_file():
        raise FileNotFoundError(f"Missing depth initializer weights: {settings.depth_checkpoint}")
    if not (settings.depth_repo / "video_depth_anything" / "video_depth.py").is_file():
        raise FileNotFoundError("Video-Depth-Anything checkout is missing.")

    output_dir = run_dir / "explicit"
    output_dir.mkdir()
    depth_path = output_dir / "depths.npz"
    yield from _worker(
        depth_command(run_dir, depth_path, settings, run_dir / "depth_events.json"),
        run_dir / "depth.log", run_dir / "depth_events.json", "load_depth_model",
        "Starting Video Depth Anything worker.",
    )
    if not depth_path.is_file():
        raise RuntimeError("Depth initialization produced no file.")

    gaussian_events = output_dir / "worker_events.json"
    yield from _worker(
        gaussian_command(run_dir, output_dir, depth_path, request, source_fov_degrees, settings, gaussian_events),
        run_dir / "gaussian.log", gaussian_events, "load_gaussian_inputs",
        "Starting 4D Gaussian worker.",
    )
    models = sorted(output_dir.glob("splat_*.splat"))
    video = output_dir / "rendered.mp4"
    if len(models) != len(frames) or not video.is_file() or not (output_dir / "scene.pt").is_file():
        raise RuntimeError("Gaussian fitting did not save a complete 4D scene.")
    return {"video": video, "models": models}
