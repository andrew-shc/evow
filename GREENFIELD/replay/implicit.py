"""Prepare an AnyView episode and run its video diffusion model in isolation."""

import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

import numpy as np

from .pose import ViewRequest, intrinsics, target_cam_to_world
from .settings import Settings


def prepare_episode(
    frames: np.ndarray,
    request: ViewRequest,
    source_fov_degrees: float,
    run_dir: Path,
    settings: Settings,
) -> Path:
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
            camera=np.repeat(camera, count),
            timestep=np.arange(count, dtype=np.int64),
            intrinsics=intrinsics_for_camera,
            extrinsics=poses,
        )

    metadata = {
        "info": {"name": "evow", "storage": "videos"},
        "cameras": ["cam0", "cam1"],
        "resolution": [height, width],
        "num_frames": count,
        "framerate": settings.fps,
        "rgb": {"extension": "mp4"},
        "extrinsics": {"transform": "cam2world"},
        "specific": {
            "roles": {"input": "cam1", "target": "cam0"},
            # AnyView trains with translation channels scaled by scene family.
            "scale_factor": 0.125,
        },
    }
    (episode / "metadata.json").write_text(json.dumps(metadata, indent=2))
    return episode


def run_implicit(
    frames: np.ndarray,
    request: ViewRequest,
    source_fov_degrees: float,
    run_dir: Path,
    settings: Settings,
):
    required = (
        settings.anyview_checkpoint,
        settings.anyview_tokenizer,
        settings.anyview_text_embedding,
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing AnyView weights: " + ", ".join(missing))
    if not (settings.anyview_repo / "scripts" / "infer.py").is_file():
        raise FileNotFoundError("AnyView-DVS checkout is missing.")

    episode = prepare_episode(frames, request, source_fov_degrees, run_dir, settings)
    yield {
        "stage": "Camera setup", "status": "complete",
        "detail": "Saved the fixed source camera and requested virtual camera.",
        "preview": run_dir / "source.png",
    }
    output_dir = run_dir / "implicit"
    output_dir.mkdir()
    log_path = run_dir / "anyview.log"
    command = [
        sys.executable, "-m", "GREENFIELD.replay.anyview_worker",
        "--episode", str(episode), "--out", str(output_dir),
    ]
    with log_path.open("w") as log_file:
        process = subprocess.Popen(
            command, cwd=settings.root, stdout=log_file, stderr=subprocess.STDOUT,
        )
        started = time.monotonic()
        yield {
            "stage": "Video diffusion", "status": "running",
            "detail": "AnyView is generating the synchronized target-view clip.",
        }
        while process.poll() is None:
            time.sleep(0.05)
            yield {
                "stage": "Video diffusion", "status": "running",
                "detail": f"Generating target view · {time.monotonic() - started:.2f}s elapsed",
            }
    if process.returncode:
        tail = "\n".join(log_path.read_text(errors="replace").splitlines()[-12:])
        raise RuntimeError(f"AnyView failed. Log: {log_path}\n{tail}")

    video = output_dir / "pred.mp4"
    first_frame = output_dir / "frames" / "000000.png"
    if not video.is_file() or not first_frame.is_file():
        raise RuntimeError(f"AnyView produced no playable result. See {log_path}")
    yield {
        "stage": "Generated view", "status": "complete",
        "detail": "The target-view video is ready.",
        "preview": first_frame,
        "metrics": {"frames": len(frames), "elapsed_seconds": round(time.monotonic() - started, 1)},
    }
    return {"video": video, "models": [], "previews": [run_dir / "source.png", first_frame]}
