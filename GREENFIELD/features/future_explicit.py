"""Explicit Future View worker wrapper and constant-velocity scene extrapolation."""
from dataclasses import dataclass
from pathlib import Path
import json
import subprocess, sys
import time
import numpy as np
from GREENFIELD.app_core.contracts import StageEvent
from GREENFIELD.app_core.models import require_checkpoint
from GREENFIELD.replay.settings import load_settings
from GREENFIELD.replay.splat import timeline_keyframe_durations, timeline_keyframe_indices


def _log_tail(path: Path, lines: int = 4) -> str:
    """Return a compact worker-log tail for the live execution card."""
    if not path.is_file():
        return "Worker log has not produced output yet."
    return "\n".join(path.read_text(errors="replace").splitlines()[-lines:]) or "Worker log is still empty."


@dataclass(frozen=True)
class ExplicitFutureArtifacts:
    """Forecast video plus the ordered observed-to-generated scene timeline."""

    forecast: Path
    splats: tuple[Path, ...]
    observed_indices: tuple[int, ...]
    forecast_indices: tuple[int, ...]
    observed_durations: tuple[float, ...]
    forecast_durations: tuple[float, ...]


def ordered_future_splats(output: Path, observed_indices: tuple[int, ...], forecast_indices: tuple[int, ...]) -> tuple[Path, ...]:
    """Require every sparse native scene frame needed by the browser timeline."""
    paths = tuple(
        [output / f"splat_{index:03d}.splat" for index in observed_indices]
        + [output / f"forecast_splat_{index:03d}.splat" for index in forecast_indices]
    )
    if not paths or any(not path.is_file() for path in paths):
        raise RuntimeError("Explicit 3D worker did not save a complete observed-to-future splat timeline.")
    return paths

def extrapolate_motion(motion: np.ndarray, history_fps: int, output_fps: int, output_frames: int, fit_frames: int) -> np.ndarray:
    """Least-squares velocity from recent learned deformation, sampled into future."""
    recent = motion[-min(fit_frames, len(motion)):]
    times = np.arange(len(recent), dtype=np.float32)
    slope = ((times - times.mean())[:, None, None] * (recent - recent.mean(axis=0))).sum(axis=0) / max(((times - times.mean()) ** 2).sum(), 1e-6)
    steps = np.arange(1, output_frames + 1, dtype=np.float32)[:, None, None] * (history_fps / output_fps)
    return motion[-1][None] + steps * slope[None]

def _worker(command: list[str], log_path: Path, stage: str, detail: str, progress_path: Path | None = None):
    """Stream a subprocess stage without holding its CUDA context in Gradio."""
    with log_path.open("w") as log_file:
        process = subprocess.Popen(command, stdout=log_file, stderr=subprocess.STDOUT)
        started = time.monotonic()
        last_step = None
        yield StageEvent(stage, "running", detail, metrics={"log_file": log_path.name, "log_tail": _log_tail(log_path)})
        while process.poll() is None:
            time.sleep(0.05)
            progress = None
            if progress_path and progress_path.is_file():
                try:
                    progress = json.loads(progress_path.read_text())
                except json.JSONDecodeError:
                    pass
            if progress and "step" in progress and progress.get("step") != last_step:
                last_step = progress["step"]
                yield StageEvent(stage, "running",
                                 "Fitting dynamic 3D Gaussians · step {}/{} · image loss {}".format(progress["step"], progress["total"], progress["loss"]),
                                 metrics={"step": progress["step"], "image_loss": progress["loss"], "gaussians": progress["gaussians"], "log_file": log_path.name, "log_tail": _log_tail(log_path)})
            else:
                # Match 4D Replay: publish every worker poll so an expensive
                # subprocess never looks frozen between optimizer updates.
                elapsed = time.monotonic() - started
                yield StageEvent(stage, "running", f"{detail} · {elapsed:.1f}s elapsed",
                                 metrics={"elapsed_worker_seconds": round(elapsed, 2),
                                          "log_file": log_path.name, "log_tail": _log_tail(log_path)})
    if process.returncode:
        tail = "\n".join(log_path.read_text(errors="replace").splitlines()[-15:])
        raise RuntimeError(f"{stage} failed. Log: {log_path}\n{tail}")


def run_explicit(frames: list[np.ndarray], artifacts: Path, settings):
    """Yield actual reconstruction progress, then return the finished scene artifacts."""
    replay = load_settings()
    require_checkpoint(replay.depth_checkpoint, "Video Depth Anything")
    if not (replay.depth_repo / "video_depth_anything" / "video_depth.py").is_file():
        raise RuntimeError("Video-Depth-Anything checkout is missing for Explicit 3D Future View.")
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("Explicit 3D Future View requires CUDA and gsplat.")
    np.save(artifacts / "frames.npy", np.stack(frames))
    depth = artifacts / "depths.npz"
    yield from _worker([sys.executable, "-m", "GREENFIELD.replay.depth_worker", "--frames",
                        str(artifacts / "frames.npy"), "--out", str(depth)],
                       artifacts / "depth.log", "Prepare selected method",
                       "Estimating a depth prior and optical-flow motion from the observed history.")
    if not depth.is_file():
        raise RuntimeError("Depth estimation did not save a depth prior.")
    yield StageEvent("Prepare selected method", "complete",
                     "Depth prior and observed motion are ready to initialize the Gaussian scene.",
                     metrics={"observed_frames": len(frames)})
    output = artifacts / "explicit"
    output.mkdir(exist_ok=True)
    yield from _worker([sys.executable, "-m", "GREENFIELD.replay.gaussian_worker", "--frames",
                        str(artifacts / "frames.npy"), "--depths", str(depth), "--out", str(output),
                        "--render-yaw", "0", "--render-shift", "0", "--render-fov", "70",
                        "--source-fov", "70", "--forecast-frames", str(settings.output_frames),
                        "--forecast-fps", str(settings.output_fps), "--motion-fit-frames",
                        str(settings.motion_fit_frames), "--history-fps", str(settings.history_fps),
                        "--observed-splat-keyframe-fps", str(settings.observed_splat_timeline_fps),
                        "--forecast-splat-keyframe-fps", str(settings.forecast_splat_timeline_fps)],
                       artifacts / "gaussian.log", "Generate future scenario",
                       "Fitting dynamic 3D Gaussians, then extrapolating and exporting the future scene.",
                       output / "progress.json")
    result = output / "forecast.mp4"
    if not result.is_file():
        raise RuntimeError("Explicit 3D worker did not save forecast.mp4.")
    observed_indices = timeline_keyframe_indices(len(frames), settings.history_fps, settings.observed_splat_timeline_fps)
    forecast_indices = timeline_keyframe_indices(settings.output_frames, settings.output_fps, settings.forecast_splat_timeline_fps)
    return ExplicitFutureArtifacts(
        result, ordered_future_splats(output, observed_indices, forecast_indices),
        observed_indices, forecast_indices,
        timeline_keyframe_durations(observed_indices, settings.history_fps),
        timeline_keyframe_durations(forecast_indices, settings.output_fps),
    )
