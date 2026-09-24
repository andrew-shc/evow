"""Reusable cached dynamic-Gaussian scene builds for text feature episodes."""

from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path
import subprocess
import sys
import time
from uuid import uuid4

import numpy as np

from GREENFIELD.app_core.contracts import StageEvent
from GREENFIELD.app_core.models import require_checkpoint, require_package
from GREENFIELD.replay.settings import load_settings as load_replay_settings
from .text_settings import TextSettings


@dataclass(frozen=True)
class CachedScene:
    """A complete source-view dynamic Gaussian scene held outside any one run."""

    key: str
    root: Path
    scene_dir: Path
    rendered: Path
    checkpoint: Path
    splats: tuple[Path, ...]
    width: int
    height: int
    fps: float
    source_fov_degrees: float


def scene_cache_key(source_hash: str, start_seconds: float, frame_count: int, fps: float, settings: TextSettings) -> str:
    """Version cached scenes by content, exact temporal interval, and geometry settings."""
    payload = {
        "source_sha256": source_hash,
        "start_seconds": round(start_seconds, 4),
        "frame_count": frame_count,
        "fps": round(fps, 4),
        "scene_config": settings.scene_config_token,
    }
    return sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:24]


def _complete_scene(root: Path) -> CachedScene | None:
    """Return a validated cache entry only after its atomically written manifest exists."""
    manifest_path = root / "manifest.json"
    if not manifest_path.is_file():
        return None
    try:
        manifest = json.loads(manifest_path.read_text())
        scene_dir = root / "scene"
        checkpoint = scene_dir / "scene.pt"
        rendered = scene_dir / "rendered.mp4"
        splats = tuple(sorted(scene_dir.glob("splat_*.splat")))
        if not checkpoint.is_file() or not rendered.is_file() or len(splats) != int(manifest["frame_count"]):
            return None
        return CachedScene(
            key=str(manifest["key"]),
            root=root,
            scene_dir=scene_dir,
            rendered=rendered,
            checkpoint=checkpoint,
            splats=splats,
            width=int(manifest["width"]),
            height=int(manifest["height"]),
            fps=float(manifest["fps"]),
            source_fov_degrees=float(manifest["source_fov_degrees"]),
        )
    except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError):
        return None


def _log_tail(path: Path, lines: int = 4) -> str:
    """Provide useful worker status without streaming an unbounded log into Gradio."""
    if not path.is_file():
        return "Worker log has not produced output yet."
    return "\n".join(path.read_text(errors="replace").splitlines()[-lines:]) or "Worker log is still empty."


def _worker(command: list[str], log_path: Path, stage: str, detail: str, progress_path: Path | None = None):
    """Run GPU work out-of-process so model memory is released after a scene build."""
    with log_path.open("w") as log_file:
        process = subprocess.Popen(command, stdout=log_file, stderr=subprocess.STDOUT)
        started = time.monotonic()
        last_step = None
        while process.poll() is None:
            time.sleep(0.1)
            progress = None
            if progress_path and progress_path.is_file():
                try:
                    progress = json.loads(progress_path.read_text())
                except json.JSONDecodeError:
                    pass
            if progress and "step" in progress and progress["step"] != last_step:
                last_step = progress["step"]
                yield StageEvent(
                    stage, "running",
                    "Fitting dynamic 3D Gaussians · step {}/{} · image loss {}".format(
                        progress["step"], progress["total"], progress["loss"]
                    ),
                    metrics={"step": progress["step"], "image_loss": progress["loss"],
                             "gaussians": progress["gaussians"], "log_file": log_path.name,
                             "log_tail": _log_tail(log_path)},
                )
            else:
                yield StageEvent(
                    stage, "running", f"{detail} · {time.monotonic() - started:.1f}s elapsed",
                    metrics={"log_file": log_path.name, "log_tail": _log_tail(log_path)},
                )
    if process.returncode:
        raise RuntimeError(f"{stage} failed. Log: {log_path}\n{_log_tail(log_path, 15)}")


def ensure_scene_steps(frames: list[np.ndarray], fps: float, source_hash: str, start_seconds: float, settings: TextSettings, stage: str):
    """Yield cache/build progress, then return the dynamic scene for an episode."""
    if not frames:
        raise ValueError("Explicit 3D processing needs at least one source frame.")
    # Cache hits do not need a CUDA device or depth checkpoint; this lets a saved
    # scene remain inspectable and reusable after a worker restart.
    key = scene_cache_key(source_hash, start_seconds, len(frames), fps, settings)
    cache_root = settings.root / "ASSETS" / "scenes"
    target = cache_root / key
    existing = _complete_scene(target)
    if existing:
        yield StageEvent(stage, "complete", "Reused cached dynamic 3D Gaussian scene.", metrics={"cache": "hit", "scene_key": key})
        return existing
    if target.exists():
        raise RuntimeError(f"Scene cache entry {key} is incomplete. Remove only that cache directory, then retry.")
    replay = load_replay_settings()
    require_checkpoint(replay.depth_checkpoint, "Video Depth Anything")
    require_package("gsplat", "replay")
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("Explicit 3D text features require CUDA and gsplat.")
    cache_root.mkdir(parents=True, exist_ok=True)
    temporary = cache_root / f".{key}_{uuid4().hex}.building"
    temporary.mkdir(parents=False)
    height, width = frames[0].shape[:2]
    try:
        np.save(temporary / "frames.npy", np.stack(frames))
        depths = temporary / "depths.npz"
        yield StageEvent(stage, "running", "Estimating depth and motion to initialize the cached 3D scene.", metrics={"cache": "miss", "scene_key": key})
        yield from _worker(
            [sys.executable, "-m", "GREENFIELD.replay.depth_worker", "--frames", str(temporary / "frames.npy"), "--out", str(depths)],
            temporary / "depth.log", stage, "Estimating depth prior and optical flow.",
        )
        if not depths.is_file():
            raise RuntimeError("Depth estimation did not save a cacheable depth prior.")
        scene_dir = temporary / "scene"
        yield from _worker(
            [sys.executable, "-m", "GREENFIELD.replay.gaussian_worker", "--frames", str(temporary / "frames.npy"),
             "--depths", str(depths), "--out", str(scene_dir), "--render-yaw", "0", "--render-shift", "0",
             "--render-fov", str(settings.source_fov_degrees), "--source-fov", str(settings.source_fov_degrees)],
            temporary / "gaussian.log", stage, "Optimizing the reusable dynamic Gaussian scene.", scene_dir / "progress.json",
        )
        manifest = {
            "key": key, "frame_count": len(frames), "fps": fps, "width": width, "height": height,
            "source_fov_degrees": settings.source_fov_degrees, "source_hash": source_hash,
            "start_seconds": start_seconds, "scene_config": settings.scene_config_token,
        }
        (temporary / "manifest.json").write_text(json.dumps(manifest, indent=2))
        temporary.replace(target)
    except Exception:
        # The incomplete build remains distinguishable and is never accepted as a cache hit.
        raise
    scene = _complete_scene(target)
    if scene is None:
        raise RuntimeError("Dynamic Gaussian worker did not save a complete cache entry.")
    yield StageEvent(stage, "complete", "Saved reusable dynamic 3D Gaussian scene.", metrics={"cache": "miss", "scene_key": key, "splats": len(scene.splats)})
    return scene
