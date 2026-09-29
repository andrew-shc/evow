"""Coordinate a Replay run and stream its producer-owned trace to Gradio."""

from datetime import datetime, timezone
import json
from pathlib import Path
from uuid import uuid4

from .clip import persist_episode, sample_episode
from .explicit import run_explicit
from .implicit import run_implicit
from .overrides import ReplayOverrides
from .pose import ViewRequest
from .settings import load_settings
from .trace import RunTrace


def _run_directory() -> Path:
    settings = load_settings()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = settings.run_root / f"{stamp}_{uuid4().hex[:8]}"
    run_dir.mkdir(parents=True)
    return run_dir


def _snapshot(
    run_dir: Path,
    trace: RunTrace,
    message: str,
    video: Path | None = None,
    models: list[Path] | None = None,
) -> dict:
    previews = []
    for event in trace.record["events"]:
        if "preview" in event:
            path = run_dir / event["preview"]
            if path.is_file():
                # V1 traces recorded a human stage label but had no stable id.
                # A preview must never make an otherwise valid saved result fail
                # to open; the workflow renderer will show that record as legacy.
                stage_id = str(event.get("stage_id") or event.get("stage") or "preview")
                previews.append((str(path), stage_id))
    return {
        "run_id": run_dir.name,
        "message": message,
        "trace": trace.record,
        "previews": previews,
        "source_video": str(run_dir / "source.mp4") if (run_dir / "source.mp4").is_file() else None,
        "video": str(video) if video else None,
        "models": [str(path) for path in (models or [])],
    }


def _title(stage_id: str) -> str:
    """Keep raw trace events readable while workflow specs own visible labels."""
    return stage_id.replace("_", " ").title()


def _add_update(trace: RunTrace, update: dict) -> None:
    """Forward one worker/coordinator event without collapsing its stable id."""
    stage_id = str(update["stage_id"])
    preview = Path(update["preview"]) if update.get("preview") else None
    log_path = Path(update["log_path"]) if update.get("log_path") else None
    trace.add(
        str(update.get("stage", _title(stage_id))),
        str(update["status"]),
        str(update["detail"]),
        preview,
        update.get("metrics"),
        stage_id=stage_id,
        execution=str(update.get("execution", "serial")),
        log_path=log_path,
    )


def execute_run(
    video_path: str,
    mode: str,
    start_seconds: float,
    frame_count: int,
    source_fov_degrees: float,
    yaw_degrees: float,
    lateral_shift: float,
    fov_degrees: float,
    overrides: ReplayOverrides | None = None,
):
    """Yield display snapshots after each real serial pipeline transition."""
    if overrides is not None:
        overrides.validate()
    settings = overrides.apply(load_settings()) if overrides is not None else load_settings()
    if mode not in ("implicit", "explicit"):
        raise ValueError("Choose Implicit video or Explicit 3D.")
    if frame_count not in (13, 29, 41):
        raise ValueError("Choose 13, 29, or 41 frames.")
    if not 35.0 <= source_fov_degrees <= 110.0:
        raise ValueError("Choose a source camera field of view between 35 and 110 degrees.")
    request = ViewRequest(yaw_degrees, lateral_shift, fov_degrees)
    request.validate()

    run_dir = _run_directory()
    traced_request = {
        "source_name": Path(video_path).name if video_path else None,
        "start_seconds": start_seconds,
        "frame_count": frame_count,
        "fps": settings.fps,
        "source_fov_degrees": source_fov_degrees,
        "yaw_degrees": yaw_degrees,
        "lateral_shift_scene_units": lateral_shift,
        "fov_degrees": fov_degrees,
        "effective_settings": ReplayOverrides.from_settings(settings).as_dict(),
    }
    trace = RunTrace(run_dir, mode, traced_request)
    active_stage = "validate_request"
    try:
        _add_update(trace, {
            "stage_id": "validate_request", "status": "running",
            "detail": "Checking request and reconstruction settings.",
        })
        _add_update(trace, {
            "stage_id": "validate_request", "status": "complete",
            "detail": "Request is valid.",
        })

        active_stage = "sample_episode"
        _add_update(trace, {
            "stage": "Input clip", "stage_id": active_stage, "status": "running",
            "detail": "Decoding and sampling the fixed-camera episode.",
        })
        yield _snapshot(run_dir, trace, "Sampling input episode.")
        frames = sample_episode(video_path, start_seconds, frame_count, settings)
        _add_update(trace, {
            "stage": "Input clip", "stage_id": active_stage, "status": "complete",
            "detail": "Decoded the requested RGB episode.",
            "metrics": {"frames": frame_count, "fps": settings.fps},
        })

        active_stage = "persist_source"
        _add_update(trace, {
            "stage_id": active_stage, "status": "running",
            "detail": "Writing source frames, preview, and browser clip.",
        })
        yield _snapshot(run_dir, trace, "Persisting input episode.")
        persist_episode(frames, settings, run_dir)
        trace.update_request({"stationary_view_assumed": True})
        _add_update(trace, {
            "stage_id": active_stage, "status": "complete",
            "detail": "Saved source artifacts.", "preview": run_dir / "source.png",
            "metrics": {"frames": frame_count, "fps": settings.fps, "stationary_view_assumed": True},
        })
        yield _snapshot(run_dir, trace, "Input episode ready.")

        backend = (
            run_implicit(frames, request, source_fov_degrees, run_dir, settings)
            if mode == "implicit" else
            run_explicit(frames, request, source_fov_degrees, run_dir, settings)
        )
        while True:
            try:
                update = next(backend)
            except StopIteration as finished:
                result = finished.value
                break
            active_stage = str(update["stage_id"])
            _add_update(trace, update)
            yield _snapshot(run_dir, trace, str(update["detail"]))

        active_stage = "save_run_manifest"
        _add_update(trace, {
            "stage_id": active_stage, "status": "running",
            "detail": "Writing the result manifest and final trace.",
        })
        yield _snapshot(run_dir, trace, "Saving run manifest.")
        video = Path(result["video"])
        models = list(result["models"])
        (run_dir / "result.json").write_text(json.dumps({
            "mode": mode,
            "video": str(video.relative_to(run_dir)),
            "models": [str(path.relative_to(run_dir)) for path in models],
        }, indent=2))
        _add_update(trace, {
            "stage_id": active_stage, "status": "complete",
            "detail": "Saved result and execution trace.",
        })
        yield _snapshot(run_dir, trace, "Run complete.", video, models)
    except Exception as error:
        # The active producer stage, not a generic catch-all row, owns failure.
        events = trace.record["events"]
        latest = next((event for event in reversed(events) if event["stage_id"] == active_stage), None)
        if latest and latest["status"] == "running":
            _add_update(trace, {
                "stage": latest["stage"], "stage_id": active_stage, "status": "error",
                "detail": str(error), "log_path": run_dir / latest["log_path"] if latest.get("log_path") else None,
            })
        yield _snapshot(run_dir, trace, f"Run failed: {error}")


def list_saved_runs() -> list[tuple[str, str]]:
    settings = load_settings()
    choices = []
    for result_path in sorted(settings.run_root.glob("*/result.json"), reverse=True):
        run_dir = result_path.parent
        result = json.loads(result_path.read_text())
        trace = json.loads((run_dir / "trace.json").read_text())
        if not (trace["request"].get("stationary_view_assumed") or trace["request"].get("static_camera_verified")):
            continue
        source_name = trace["request"].get("source_name") or "recording"
        method = "Explicit 4D Gaussian" if result["mode"] == "explicit" else "Implicit video diffusion"
        choices.append((f"{method} · {source_name} · {run_dir.name}", run_dir.name))
    return choices


def load_saved_run(run_id: str) -> dict:
    if not run_id or "/" in run_id or "\\" in run_id:
        raise ValueError("Choose a saved run.")
    run_dir = load_settings().run_root / run_id
    result = json.loads((run_dir / "result.json").read_text())
    trace = RunTrace.__new__(RunTrace)
    trace.run_dir = run_dir
    trace.record = json.loads((run_dir / "trace.json").read_text())
    video = run_dir / result["video"]
    models = [run_dir / path for path in result["models"]]
    return _snapshot(run_dir, trace, "Saved run loaded.", video, models)


def saved_stage(run_id: str, index: int) -> tuple[str | None, str]:
    if not run_id or "/" in run_id or "\\" in run_id:
        return None, "Choose a saved run."
    run_dir = load_settings().run_root / run_id
    trace = json.loads((run_dir / "trace.json").read_text())
    events = trace["events"]
    if not events:
        return None, "This run has no stages yet."
    event = events[max(0, min(int(index), len(events) - 1))]
    preview = run_dir / event["preview"] if "preview" in event else None
    image = str(preview) if preview and preview.is_file() else None
    return image, f"**{event['stage']}** · {event['status']} · {event['detail']}"
