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
from GREENFIELD.app_core.timing import format_stopwatch
from GREENFIELD.app_core.models import require_checkpoint, require_package
from GREENFIELD.replay.settings import Settings, load_settings as load_replay_settings
from GREENFIELD.replay.overrides import reconstruction_fingerprint, settings_json
from .text_settings import TextSettings


@dataclass(frozen=True)
class CachedScene:
    """A complete source-view dynamic Gaussian scene held outside any one run.

    ``cache_hit`` reports whether this instance was returned from an existing
    complete cache entry (True) or freshly built during the current call (False).
    It is a runtime fact, never persisted in the manifest, so a caller can emit a
    truthful "cache hit" row without re-deriving the cache key itself.
    """

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
    cache_hit: bool = False


def scene_cache_key(
    source_hash: str,
    start_seconds: float,
    frame_count: int,
    fps: float,
    settings: TextSettings,
    replay_settings: Settings | None = None,
) -> str:
    """Version cached scenes by content, exact temporal interval, and geometry settings.

    The fingerprint covers every reconstruction knob the Gaussian worker reads, so
    tuning any of them (stride, steps, losses, rates, background) yields a fresh
    cache entry instead of silently reusing an incompatible scene. ``replay_settings``
    is the exact effective value whose fingerprint keys the cache; the same value is
    serialized into the worker commands, so a key can never describe settings other
    than the ones the build actually used. Calling without it keeps tracked defaults.
    """
    replay = replay_settings if replay_settings is not None else load_replay_settings()
    payload = {
        "source_sha256": source_hash,
        "start_seconds": round(start_seconds, 4),
        "frame_count": frame_count,
        "fps": round(fps, 4),
        "scene_config": settings.scene_config_token,
        "reconstruction": reconstruction_fingerprint(replay),
    }
    return sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:24]


def _complete_scene(root: Path, *, cache_hit: bool = True) -> CachedScene | None:
    """Return a validated cache entry only after its atomically written manifest exists.

    ``cache_hit`` tags the returned instance: the lookup path reads an existing
    entry (True) while the build path validates what it just wrote (False).
    """
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
            cache_hit=cache_hit,
        )
    except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError):
        return None


def _log_tail(path: Path, lines: int = 4) -> str:
    """Provide useful worker status without streaming an unbounded log into Gradio."""
    if not path.is_file():
        return "Worker log has not produced output yet."
    return "\n".join(path.read_text(errors="replace").splitlines()[-lines:]) or "Worker log is still empty."


def _worker(command: list[str], log_path: Path, stage: str, detail: str, progress_path: Path | None = None, *, stage_id: str = ""):
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
                    stage_id=stage_id,
                )
            else:
                yield StageEvent(
                    stage, "running", f"{detail} · {format_stopwatch(time.monotonic() - started)} elapsed",
                    metrics={"log_file": log_path.name, "log_tail": _log_tail(log_path)},
                    stage_id=stage_id,
                )
    if process.returncode:
        raise RuntimeError(f"{stage} failed. Log: {log_path}\n{_log_tail(log_path, 15)}")


def ensure_scene_steps(frames: list[np.ndarray], fps: float, source_hash: str, start_seconds: float, settings: TextSettings, stage: str, *, stage_id: str = ""):
    """Yield cache/build progress, then return the dynamic scene for an episode.

    ``stage_id`` is the owning feature's stable flow identity for the display
    ``stage``; it lets persisted events resolve by id instead of by label.
    """
    if not frames:
        raise ValueError("Explicit 3D processing needs at least one source frame.")
    # Cache hits do not need a CUDA device or depth checkpoint; this lets a saved
    # scene remain inspectable and reusable after a worker restart. Resolve the
    # effective replay settings once so this exact value both keys the cache and
    # feeds the worker commands below.
    replay = load_replay_settings()
    key = scene_cache_key(source_hash, start_seconds, len(frames), fps, settings, replay)
    cache_root = settings.root / "ASSETS" / "scenes"
    target = cache_root / key
    existing = _complete_scene(target)
    if existing:
        # A cache hit is tick-free: the calling feature owns the single terminal
        # and reads ``existing.cache_hit`` to report it truthfully.
        return existing
    if target.exists():
        raise RuntimeError(f"Scene cache entry {key} is incomplete. Remove only that cache directory, then retry.")
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
        # Tick-only relay: the caller owns the row's single terminal.
        yield StageEvent(stage, "running", "Estimating depth and motion to initialize the cached 3D scene.", metrics={"cache": "miss", "scene_key": key}, stage_id=stage_id)
        yield from _worker(
            [sys.executable, "-m", "GREENFIELD.replay.depth_worker", "--frames", str(temporary / "frames.npy"), "--out", str(depths),
             "--settings-json", settings_json(replay)],
            temporary / "depth.log", stage, "Estimating depth prior and optical flow.",
            stage_id=stage_id,
        )
        if not depths.is_file():
            raise RuntimeError("Depth estimation did not save a cacheable depth prior.")
        scene_dir = temporary / "scene"
        yield from _worker(
            [sys.executable, "-m", "GREENFIELD.replay.gaussian_worker", "--frames", str(temporary / "frames.npy"),
             "--depths", str(depths), "--out", str(scene_dir), "--render-yaw", "0", "--render-shift", "0",
             "--render-fov", str(settings.source_fov_degrees), "--source-fov", str(settings.source_fov_degrees),
             "--settings-json", settings_json(replay)],
            temporary / "gaussian.log", stage, "Optimizing the reusable dynamic Gaussian scene.", scene_dir / "progress.json",
            stage_id=stage_id,
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
    scene = _complete_scene(target, cache_hit=False)
    if scene is None:
        raise RuntimeError("Dynamic Gaussian worker did not save a complete cache entry.")
    return scene
