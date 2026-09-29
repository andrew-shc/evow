"""Prepare an AnyView episode and forward its isolated atomic stages."""

import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

import numpy as np

from .overrides import settings_json
from .pose import ViewRequest, intrinsics, target_cam_to_world
from .settings import Settings
from .worker_events import read_events


def anyview_command(episode: Path, output_dir: Path, settings: Settings, events_path: Path | None = None) -> list[str]:
    """Build the isolated AnyView worker call and give it an event sidecar."""
    command = [
        sys.executable, "-m", "GREENFIELD.replay.anyview_worker",
        "--episode", str(episode), "--out", str(output_dir), "--settings-json", settings_json(settings),
    ]
    if events_path is not None:
        command.extend(("--events", str(events_path)))
    return command


def prepare_episode(
    frames: np.ndarray,
    request: ViewRequest,
    source_fov_degrees: float,
    run_dir: Path,
    settings: Settings,
) -> Path:
    """Write the local AnyView-compatible RGB and camera episode."""
    episode = run_dir / "anyview_episode"
    (episode / "rgb").mkdir(parents=True)
    (episode / "lowdim").mkdir()
    shutil.copyfile(run_dir / "source.mp4", episode / "rgb" / "cam1.mp4")

    count, height, width = frames.shape[:3]
    source_intrinsics = np.repeat(intrinsics(height, width, source_fov_degrees)[None], count, axis=0)
    target_intrinsics = np.repeat(intrinsics(height, width, request.fov_degrees)[None], count, axis=0)
    source_poses = np.repeat(np.eye(4, dtype=np.float32)[None], count, axis=0)
    target_poses = np.repeat(target_cam_to_world(request)[None], count, axis=0)
    for camera, poses, intrinsics_for_camera in (("cam1", source_poses, source_intrinsics), ("cam0", target_poses, target_intrinsics)):
        np.savez_compressed(
            episode / "lowdim" / f"{camera}.npz",
            camera=np.repeat(camera, count), timestep=np.arange(count, dtype=np.int64),
            intrinsics=intrinsics_for_camera, extrinsics=poses,
        )
    metadata = {
        "info": {"name": "evow", "storage": "videos"}, "cameras": ["cam0", "cam1"],
        "resolution": [height, width], "num_frames": count, "framerate": settings.fps,
        "rgb": {"extension": "mp4"}, "extrinsics": {"transform": "cam2world"},
        "specific": {"roles": {"input": "cam1", "target": "cam0"}, "scale_factor": 0.125},
    }
    (episode / "metadata.json").write_text(json.dumps(metadata, indent=2))
    return episode


def _log_tail(path: Path, lines: int = 16) -> str:
    if not path.is_file():
        return ""
    return "\n".join(path.read_text(errors="replace").splitlines()[-lines:])


def _worker(command: list[str], log_path: Path, events_path: Path):
    """Forward every event in an atomically published AnyView sidecar."""
    with log_path.open("w") as log_file:
        process = subprocess.Popen(command, cwd=Path.cwd(), stdout=log_file, stderr=subprocess.STDOUT)
        last_sequence = 0
        yield {
            "stage_id": "load_anyview_vae", "status": "running",
            "detail": "Starting AnyView worker.", "log_path": log_path,
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
        raise RuntimeError(f"AnyView failed. Log: {log_path}\n{_log_tail(log_path)}")


def run_implicit(
    frames: np.ndarray,
    request: ViewRequest,
    source_fov_degrees: float,
    run_dir: Path,
    settings: Settings,
):
    """Yield the real serial AnyView operations in their execution order."""
    required = (settings.anyview_checkpoint, settings.anyview_tokenizer, settings.anyview_text_embedding)
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing AnyView weights: " + ", ".join(missing))
    if not (settings.anyview_repo / "scripts" / "infer.py").is_file():
        raise FileNotFoundError("AnyView-DVS checkout is missing.")

    yield {
        "stage_id": "package_anyview_episode", "status": "running",
        "detail": "Packaging source RGB and virtual cameras.",
    }
    episode = prepare_episode(frames, request, source_fov_degrees, run_dir, settings)
    yield {
        "stage_id": "package_anyview_episode", "status": "complete",
        "detail": "Saved AnyView episode.", "preview": run_dir / "source.png",
        "metrics": {"frames": len(frames)},
    }

    output_dir = run_dir / "implicit"
    output_dir.mkdir()
    events_path = output_dir / "worker_events.json"
    yield from _worker(
        anyview_command(episode, output_dir, settings, events_path),
        run_dir / "anyview.log", events_path,
    )
    video = output_dir / "pred.mp4"
    first_frame = output_dir / "frames" / "000000.png"
    if not video.is_file() or not first_frame.is_file():
        raise RuntimeError(f"AnyView produced no playable result. See {run_dir / 'anyview.log'}")
    return {"video": video, "models": []}
