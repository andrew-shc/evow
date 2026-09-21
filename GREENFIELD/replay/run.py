"""Coordinate one replay run and expose saved traces to the local interface."""

from datetime import datetime, timezone
import json
from pathlib import Path
from uuid import uuid4

from .clip import extract_clip
from .explicit import run_explicit
from .implicit import run_implicit
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
    workflow_changed: bool = True,
) -> dict:
    previews = []
    for event in trace.record["events"]:
        if "preview" in event:
            path = run_dir / event["preview"]
            if path.is_file():
                previews.append((str(path), event["stage"]))
    return {
        "run_id": run_dir.name,
        "message": message,
        "trace": trace.record,
        "workflow_changed": workflow_changed,
        "previews": previews,
        "source_video": str(run_dir / "source.mp4") if (run_dir / "source.mp4").is_file() else None,
        "video": str(video) if video else None,
        "models": [str(path) for path in (models or [])],
    }


def execute_run(
    video_path: str,
    mode: str,
    start_seconds: float,
    frame_count: int,
    source_fov_degrees: float,
    yaw_degrees: float,
    lateral_shift: float,
    fov_degrees: float,
):
    settings = load_settings()
    if mode not in ("implicit", "explicit"):
        raise ValueError("Choose Implicit video or Explicit 3D.")
    if frame_count not in (13, 29, 41):
        raise ValueError("Choose 13, 29, or 41 frames.")
    if not 35.0 <= source_fov_degrees <= 110.0:
        raise ValueError("Choose a source camera field of view between 35 and 110 degrees.")
    request = ViewRequest(yaw_degrees, lateral_shift, fov_degrees)
    request.validate()
    run_dir = _run_directory()
    trace = RunTrace(
        run_dir, mode,
        {
            "source_name": Path(video_path).name if video_path else None,
            "start_seconds": start_seconds,
            "frame_count": frame_count,
            "fps": settings.fps,
            "source_fov_degrees": source_fov_degrees,
            "yaw_degrees": yaw_degrees,
            "lateral_shift_scene_units": lateral_shift,
            "fov_degrees": fov_degrees,
        },
    )

    try:
        frames = extract_clip(video_path, start_seconds, frame_count, settings, run_dir)
        trace.update_request({"stationary_view_assumed": True})
        trace.add(
            "Input clip", "complete", "Sampled the same short stationary-view episode for both methods.",
            run_dir / "source.png",
            {"frames": frame_count, "fps": settings.fps, "stationary_view_assumed": True},
        )
        yield _snapshot(run_dir, trace, "Input clip ready.")
        # A whole HTML replacement collapses every native <details> card. Only
        # refresh it when the visible stage or its status changes.
        last_workflow_state = ("Input clip", "complete")

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
            workflow_changed = False
            if update.get("record", True):
                trace.add(
                    update["stage"], update["status"], update["detail"],
                    update.get("preview"), update.get("metrics"),
                )
                workflow_state = (update["stage"], update["status"])
                workflow_changed = workflow_state != last_workflow_state
                last_workflow_state = workflow_state
            yield _snapshot(
                run_dir, trace, update["detail"], workflow_changed=workflow_changed,
            )

        video = Path(result["video"])
        models = list(result["models"])
        (run_dir / "result.json").write_text(json.dumps({
            "mode": mode,
            "video": str(video.relative_to(run_dir)),
            "models": [str(path.relative_to(run_dir)) for path in models],
        }, indent=2))
        trace.add("Run complete", "complete", "The result and stage trace are saved.")
        yield _snapshot(run_dir, trace, "Run complete.", video, models)
    except Exception as error:
        trace.add("Run failed", "error", str(error))
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
        label = f"{method} · {source_name} · {run_dir.name}"
        choices.append((label, run_dir.name))
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
