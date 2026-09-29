"""Explicit Future View worker wrapper and constant-velocity scene extrapolation."""
from dataclasses import dataclass
from pathlib import Path
import subprocess, sys
import time
import numpy as np
from GREENFIELD.app_core.contracts import StageEvent
from GREENFIELD.app_core.models import require_checkpoint
from GREENFIELD.replay.settings import load_settings
from GREENFIELD.replay.worker_events import read_events
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


@dataclass(frozen=True)
class WorkerEventRow:
    """Map one Replay worker stage onto the Future View row that owns it.

    ``terminal_detail`` is set only on the worker stage whose completion closes
    the owning row; the relay then emits the single parent terminal itself so a
    worker never ends up owning two terminals for one Future stage.
    """

    stage_id: str
    label: str
    terminal_detail: str | None = None


# The depth worker publishes three atomic substeps, all owned by ``depth_infer``.
# No substep closes the row: the parent waits until it has verified the depth
# prior so a missing file cannot follow a premature "complete".
_DEPTH_EVENT_ROWS = {
    "load_depth_model": WorkerEventRow("depth_infer", "Infer temporal depth"),
    "infer_temporal_depth": WorkerEventRow("depth_infer", "Infer temporal depth"),
    "write_depth_prior": WorkerEventRow("depth_infer", "Infer temporal depth"),
}

# The Gaussian worker runs fit, motion extrapolation, render/export, and encode in
# one process. Each worker stage is relaid as nested ticks under its owning row;
# completing the last worker stage of a row closes that Future row exactly once.
_GAUSSIAN_EVENT_ROWS = {
    "load_gaussian_inputs": WorkerEventRow("gaussian_fit", "Fit dynamic Gaussians"),
    "initialize_gaussians": WorkerEventRow("gaussian_fit", "Fit dynamic Gaussians"),
    "fit_4d_scene": WorkerEventRow("gaussian_fit", "Fit dynamic Gaussians",
                                   "Fitted the persistent 4D Gaussian scene."),
    "motion_extrapolate": WorkerEventRow("motion_extrapolate", "Extrapolate scene motion",
                                         "Extrapolated the learned deformation into the future window."),
    "render_export_splats": WorkerEventRow("splat_render_export", "Render and export splats",
                                           "Rendered the requested view and exported every observed-to-future splat keyframe."),
    "encode_rendered_video": WorkerEventRow("encode_video", "Encode forecast video",
                                            "Encoded the observed and forecast videos."),
    "persist_scene_artifacts": WorkerEventRow("validate_artifacts", "Validate forecast artifacts"),
}

# Used when the worker publishes no event sidecar at all (for example a crash
# before its first emit): the parent still reports one honest boundary per row
# with unknown metrics instead of fabricating step counts or leaving rows running.
_GAUSSIAN_FALLBACK_ROWS = (
    WorkerEventRow("gaussian_fit", "Fit dynamic Gaussians",
                   "Ran the isolated 4D Gaussian worker; it reported no finer progress."),
    WorkerEventRow("motion_extrapolate", "Extrapolate scene motion",
                   "Ran the isolated 4D Gaussian worker; it reported no finer progress."),
    WorkerEventRow("splat_render_export", "Render and export splats",
                   "Ran the isolated 4D Gaussian worker; it reported no finer progress."),
    WorkerEventRow("encode_video", "Encode forecast video",
                   "Ran the isolated 4D Gaussian worker; it reported no finer progress."),
)


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


def _relay_event(event: dict, rows: dict[str, WorkerEventRow], log_path: Path, terminated: set[str]):
    """Turn one worker event into parent-row ticks, downgrading its status.

    A worker lifecycle is never a parent lifecycle: the worker's ``complete``
    becomes a nested ``running`` tick, and the parent emits at most one terminal
    per row (driven by ``WorkerEventRow.terminal_detail``).
    """
    row = rows.get(str(event.get("stage_id", "")))
    if row is None:
        return
    status = str(event.get("status", "running"))
    if status == "error":
        raise RuntimeError(f"{row.label} failed: {event.get('detail', 'the worker reported an error.')}")
    metrics = dict(event.get("metrics") or {})
    metrics["log_file"] = log_path.name
    tail = _log_tail(log_path)
    if tail:
        metrics["log_tail"] = tail
    yield StageEvent(row.label, "running", str(event.get("detail", "")), metrics=metrics, stage_id=row.stage_id)
    if row.terminal_detail and status != "running" and row.stage_id not in terminated:
        terminated.add(row.stage_id)
        yield StageEvent(row.label, "complete", row.terminal_detail, metrics=metrics, stage_id=row.stage_id)


def _relay_worker(command: list[str], log_path: Path, events_path: Path, rows: dict[str, WorkerEventRow],
                  *, startup_stage: str, startup_detail: str):
    """Stream one isolated worker's atomic events as Future View parent ticks.

    Returns the parent stage ids this call terminated. The atomic sidecar is read
    without assuming it exists yet or holds valid JSON, and it is drained once
    more after exit because a worker can publish its final event just before it
    is reaped.
    """
    startup = rows[startup_stage]
    with log_path.open("w") as log_file:
        process = subprocess.Popen(command, stdout=log_file, stderr=subprocess.STDOUT)
        last_sequence = 0
        terminated: set[str] = set()
        yield StageEvent(startup.label, "running", startup_detail,
                         metrics={"log_file": log_path.name, "log_tail": _log_tail(log_path)},
                         stage_id=startup.stage_id)
        while True:
            for event in read_events(events_path, last_sequence):
                last_sequence = max(last_sequence, int(event.get("sequence", 0)))
                yield from _relay_event(event, rows, log_path, terminated)
            if process.poll() is not None:
                break
            time.sleep(0.05)
        for event in read_events(events_path, last_sequence):
            last_sequence = max(last_sequence, int(event.get("sequence", 0)))
            yield from _relay_event(event, rows, log_path, terminated)
    if process.returncode:
        raise RuntimeError(f"Explicit 3D worker failed. Log: {log_path}\n{_log_tail(log_path, 15)}")
    return terminated


def _fallback_worker_rows(terminated: set[str]):
    """Close any Gaussian row the worker never reported, without fake metrics."""
    for row in _GAUSSIAN_FALLBACK_ROWS:
        if row.stage_id in terminated:
            continue
        yield StageEvent(row.label, "running", "The isolated 4D Gaussian worker reported no finer progress.", stage_id=row.stage_id)
        yield StageEvent(row.label, "complete", row.terminal_detail, stage_id=row.stage_id)


def gaussian_worker_command(artifacts: Path, output: Path, depth: Path, settings, events_path: Path | None = None) -> list[str]:
    """Build the explicit reconstruction command from the per-run render pose.

    Kept separate from ``run_explicit`` so the editable camera pose and timeline
    rates can be asserted without launching a CUDA worker. Passing an
    ``events_path`` opts the worker into the atomic sidecar the relay reads.
    """
    command = [sys.executable, "-m", "GREENFIELD.replay.gaussian_worker", "--frames",
               str(artifacts / "frames.npy"), "--depths", str(depth), "--out", str(output),
               "--render-yaw", str(settings.render_yaw), "--render-shift", str(settings.render_shift),
               "--render-fov", str(settings.render_fov), "--source-fov", str(settings.source_fov_degrees),
               "--forecast-frames", str(settings.output_frames),
               "--forecast-fps", str(settings.output_fps), "--motion-fit-frames",
               str(settings.motion_fit_frames), "--history-fps", str(settings.history_fps),
               "--observed-splat-keyframe-fps", str(settings.observed_splat_timeline_fps),
               "--forecast-splat-keyframe-fps", str(settings.forecast_splat_timeline_fps)]
    if events_path is not None:
        command.extend(("--events", str(events_path)))
    return command


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
    depth_events = artifacts / "depth_events.json"
    yield from _relay_worker(
        [sys.executable, "-m", "GREENFIELD.replay.depth_worker", "--frames",
         str(artifacts / "frames.npy"), "--out", str(depth), "--events", str(depth_events)],
        artifacts / "depth.log", depth_events, _DEPTH_EVENT_ROWS,
        startup_stage="load_depth_model",
        startup_detail="Starting the Video Depth Anything worker.",
    )
    if not depth.is_file():
        raise RuntimeError("Depth estimation did not save a depth prior.")
    yield StageEvent("Infer temporal depth", "complete",
                     "Depth prior and observed motion are ready to initialize the Gaussian scene.",
                     metrics={"observed_frames": len(frames)},
                     stage_id="depth_infer")
    output = artifacts / "explicit"
    output.mkdir(exist_ok=True)
    gaussian_events = output / "worker_events.json"
    # The Gaussian worker terminates ``gaussian_fit`` mid-process, then reports
    # motion, render/export, and encode, so the parent rows stay in order.
    terminated = yield from _relay_worker(
        gaussian_worker_command(artifacts, output, depth, settings, gaussian_events),
        artifacts / "gaussian.log", gaussian_events, _GAUSSIAN_EVENT_ROWS,
        startup_stage="load_gaussian_inputs",
        startup_detail="Starting the 4D Gaussian worker.",
    )
    yield from _fallback_worker_rows(terminated)
    result = output / "forecast.mp4"
    if not result.is_file():
        raise RuntimeError("Explicit 3D worker did not save forecast.mp4.")
    observed_indices = timeline_keyframe_indices(len(frames), settings.history_fps, settings.observed_splat_timeline_fps)
    forecast_indices = timeline_keyframe_indices(settings.output_frames, settings.output_fps, settings.forecast_splat_timeline_fps)
    yield StageEvent("Validate forecast artifacts", "running", "Checking the forecast video and every timeline splat frame.", stage_id="validate_artifacts")
    paths = ordered_future_splats(output, observed_indices, forecast_indices)
    yield StageEvent("Validate forecast artifacts", "complete", "Forecast video and every timeline splat frame are present.",
                     metrics={"splats": len(paths), "forecast": result.name},
                     stage_id="validate_artifacts")
    return ExplicitFutureArtifacts(
        result, paths,
        observed_indices, forecast_indices,
        timeline_keyframe_durations(observed_indices, settings.history_fps),
        timeline_keyframe_durations(forecast_indices, settings.output_fps),
    )
