"""Render Replay's versioned, atomic execution trace without browser scripts."""

from dataclasses import dataclass
from html import escape
import json
from typing import Any
from urllib.parse import quote

from GREENFIELD.app_core.model_catalog import ANYVIEW, GSPLAT, VIDEO_DEPTH_ANYTHING
from GREENFIELD.app_core.timing import format_stopwatch
from GREENFIELD.app_core.ui import detail_rows, help_icon

from .settings import load_settings


@dataclass(frozen=True)
class Step:
    """One producer-owned stage and its compact program inspector metadata."""

    stage_id: str
    name: str
    source_file: str
    source_location: str
    function_chain: str
    inputs: str
    outputs: str
    controls: str
    help_text: str
    models: tuple[Any, ...] = ()


SHARED = (
    Step("validate_request", "Validate request", "GREENFIELD/replay/run.py", "def execute_run(", "ReplayOverrides.validate() → ViewRequest.validate()", "request · settings", "validated request", "All controls", "Checks every typed setting before creating files."),
    Step("sample_episode", "Sample episode", "GREENFIELD/replay/clip.py", "def sample_episode(", "cv2.VideoCapture → resize → RGB", "source · time · frames", "RGB frame tensor", "Start · Frames · FPS · Max side", "Decodes only the requested fixed-camera episode."),
    Step("persist_source", "Persist source", "GREENFIELD/replay/clip.py", "def persist_episode(", "numpy.save → write_video → imageio.imwrite", "RGB frame tensor", "frames.npy · source.mp4 · source.png", "FPS", "Writes replayable source artifacts."),
)
EXPLICIT = (
    Step("load_depth_model", "Load depth model", "GREENFIELD/replay/depth_worker.py", "def main()", "VideoDepthAnything → load_state_dict", "checkpoint", "ready depth model", "Depth input", "Loads the temporal depth initializer.", (VIDEO_DEPTH_ANYTHING,)),
    Step("infer_temporal_depth", "Infer temporal depth", "GREENFIELD/replay/depth_worker.py", "def main()", "VideoDepthAnything.infer_video_depth", "source frames", "relative depths", "Depth input", "Estimates depth aligned across time.", (VIDEO_DEPTH_ANYTHING,)),
    Step("write_depth_prior", "Write depth prior", "GREENFIELD/replay/depth_worker.py", "def main()", "numpy.savez_compressed", "relative depths", "depths.npz", "None", "Persists the depth input used by reconstruction."),
    Step("load_gaussian_inputs", "Load scene inputs", "GREENFIELD/replay/gaussian_worker.py", "def main()", "numpy.load → scene_depths → camera tensors", "frames · depths", "scene tensors", "Source FOV", "Loads reconstruction inputs on CUDA.", (GSPLAT,)),
    Step("initialize_gaussians", "Initialize Gaussians", "GREENFIELD/replay/gaussian_worker.py", "def _initial_gaussians(", "depth lift → Farneback flow", "frames · depths", "primitives · motion", "Gaussian stride", "Seeds persistent primitives and motion offsets.", (GSPLAT,)),
    Step("fit_4d_scene", "Fit 4D scene", "GREENFIELD/replay/gaussian_worker.py", "for step in range(settings.gaussian_steps):", "gsplat.rasterization → Adam", "primitives · observations", "optimized scene", "Gaussian steps · advanced", "Optimizes one time-varying Gaussian scene.", (GSPLAT,)),
    Step("render_export_splats", "Render + export splats", "GREENFIELD/replay/gaussian_worker.py", "for index in range(count):", "rasterization → save_splat", "optimized scene", "view frames · splat sequence", "Vertical-axis rotation · Lateral translation · Output FOV", "This fused loop renders each view frame and exports its matching splat.", (GSPLAT,)),
    Step("encode_rendered_video", "Encode view video", "GREENFIELD/replay/gaussian_worker.py", "write_video(np.stack(all_renders)", "imageio H.264 encode", "rendered frames", "rendered.mp4", "FPS", "Encodes browser-playable output."),
    Step("persist_scene_artifacts", "Persist scene", "GREENFIELD/replay/gaussian_worker.py", "torch.save({", "imageio.imwrite → torch.save → JSON", "optimized scene", "scene.pt · scene.json · preview", "None", "Writes the checkpoint and scene metadata."),
)
IMPLICIT = (
    Step("package_anyview_episode", "Package episode", "GREENFIELD/replay/implicit.py", "def prepare_episode(", "copy source → write camera NPZ + metadata", "source frames · camera request", "AnyView episode", "Source FOV · virtual view", "Creates the local AnyView input format."),
    Step("load_anyview_vae", "Load video tokenizer", "GREENFIELD/replay/anyview_worker.py", "def main()", "load_vae", "tokenizer checkpoint", "ready VAE", "None", "Loads the AnyView video tokenizer.", (ANYVIEW,)),
    Step("load_anyview_diffusion", "Load diffusion model", "GREENFIELD/replay/anyview_worker.py", "def main()", "load_pipeline", "AnyView checkpoint", "ready transformer", "None", "Loads the target-view diffusion transformer.", (ANYVIEW,)),
    Step("load_anyview_episode", "Load episode", "GREENFIELD/replay/anyview_worker.py", "def main()", "load_episode → reanchor_world2cam", "AnyView episode", "frames · cameras", "None", "Loads RGB frames and both camera trajectories.", (ANYVIEW,)),
    Step("encode_source_rgb", "Encode source RGB", "GREENFIELD/replay/anyview_worker.py", "source_latent = vae.encode_rgb", "resize_video → VAE encode_rgb", "source frames", "RGB latent", "None", "Encodes the observed camera video.", (ANYVIEW,)),
    Step("encode_target_camera", "Encode target rays", "GREENFIELD/replay/anyview_worker.py", "target_cameras = module.prepare_cams_latent", "Plücker rays → VAE encode_cams", "target camera", "target camera latent", "Vertical-axis rotation · Lateral translation · Output FOV", "Encodes the requested virtual camera.", (ANYVIEW,)),
    Step("encode_source_camera", "Encode source rays", "GREENFIELD/replay/anyview_worker.py", "source_cameras = module.prepare_cams_latent", "Plücker rays → VAE encode_cams", "source camera", "source camera latent", "Source FOV", "Encodes the stationary source camera.", (ANYVIEW,)),
    Step("build_anyview_conditioning", "Build conditioning", "GREENFIELD/replay/anyview_worker.py", "entries = build_dvs_entries", "build_dvs_entries", "RGB + camera latents", "DVS entries", "None", "Combines source and camera conditioning.", (ANYVIEW,)),
    Step("sample_target_video", "Sample target video", "GREENFIELD/replay/anyview_worker.py", "samples = pipeline.generate", "AnyView pipeline.generate", "DVS entries", "target latent streams", "AnyView steps", "Denoises the requested target-view video.", (ANYVIEW,)),
    Step("decode_target_video", "Decode target RGB", "GREENFIELD/replay/anyview_worker.py", "predicted_video = vae.decode_rgb", "unpack entries → VAE decode_rgb", "target latent streams", "RGB target frames", "None", "Decodes generated video latents."),
    Step("write_target_video", "Write target MP4", "GREENFIELD/replay/anyview_worker.py", "module.save_video", "imageio video writer", "RGB target frames", "pred.mp4", "FPS", "Writes browser-playable output."),
    Step("write_frames_metadata", "Write frames + metadata", "GREENFIELD/replay/anyview_worker.py", "module.save_frames", "imageio.imwrite → json.dumps", "RGB target frames", "PNG frames · info.json", "None", "Writes inspectable result frames and metadata."),
)
FINAL = Step("save_run_manifest", "Save run manifest", "GREENFIELD/replay/run.py", "(run_dir / \"result.json\").write_text", "json.dumps → RunTrace._save", "result · trace", "result.json · trace.json", "None", "Persists the final replay record.")
LEGACY = Step("legacy_trace", "Legacy trace", "GREENFIELD/replay/trace.py", "class RunTrace", "recorded V1 events", "legacy trace", "raw recorded events", "None", "This saved run predates atomic lifecycle records.")

# The implicit route has the longest profile: three shared, twelve model
# operations, and the final manifest. Components are preallocated up to this
# limit so streaming header updates never collapse an accordion the user opened.
MAX_FLOW_STAGES = 16


def _steps(trace: dict[str, Any]) -> tuple[Step, ...]:
    if int(trace.get("workflow_version", 1)) < 2:
        return (LEGACY,)
    mode = str(trace.get("mode", "explicit"))
    return (*SHARED, *(EXPLICIT if mode == "explicit" else IMPLICIT), FINAL)


def workflow_visible(trace: dict[str, Any], index: int) -> bool:
    """Tell the preallocated Gradio row whether this profile owns it."""
    return 0 <= index < len(_steps(trace))


def _events_for(step: Step, trace: dict[str, Any]) -> list[dict[str, Any]]:
    events = list(trace.get("events", []))
    if step is LEGACY:
        return events
    return [event for event in events if event.get("stage_id") == step.stage_id]


def _latest(step: Step, trace: dict[str, Any]) -> dict[str, Any] | None:
    events = _events_for(step, trace)
    return events[-1] if events else None


def _state(event: dict[str, Any] | None) -> str:
    if event is None:
        return "waiting"
    return "failed" if event.get("status") == "error" else str(event.get("status", "waiting"))


def _duration(events: list[dict[str, Any]], event: dict[str, Any] | None) -> str:
    if event is None or event.get("status") == "skipped":
        return "—"
    if "stage_elapsed_seconds" in event:
        return format_stopwatch(float(event["stage_elapsed_seconds"]))
    start = float(events[0].get("elapsed_seconds", 0)) if events else 0
    return format_stopwatch(max(0.0, float(event.get("elapsed_seconds", 0)) - start))


def workflow_group_summary(trace: dict[str, Any], indexes: range) -> tuple[str, str, bool]:
    """Aggregate only the atomic records physically nested under one group.

    The parent status is a compact roll-up, never a guessed dependency state:
    failure and active work take precedence, while a group becomes complete only
    once every stage it owns has reached a terminal lifecycle event.
    """
    if int(trace.get("workflow_version", 1)) < 2:
        return "waiting", "—", False
    steps = _steps(trace)
    latest_events = [
        _latest(steps[index], trace)
        for index in indexes
        if index < len(steps)
    ]
    if not latest_events or not any(latest_events):
        return "waiting", "—", False
    states = [_state(event) for event in latest_events]
    elapsed = sum(
        float(event.get("stage_elapsed_seconds", 0))
        for event in latest_events
        if event and event.get("status") != "skipped"
    )
    concurrent = any(
        event and event.get("status") == "running" and event.get("execution") == "concurrent"
        for event in latest_events
    )
    if "failed" in states:
        state = "failed"
    elif "running" in states:
        state = "running"
    elif all(state in {"complete", "skipped"} for state in states):
        state = "skipped" if all(state == "skipped" for state in states) else "complete"
    else:
        state = "waiting"
    return state, format_stopwatch(elapsed), concurrent


def workflow_header(trace: dict[str, Any], index: int) -> tuple[str, str]:
    """Return one minimal, truthful persistent-accordion header."""
    steps = _steps(trace)
    if index >= len(steps):
        return "Unused stage", "evow-stage-waiting"
    step = steps[index]
    event = _latest(step, trace)
    # A V1 record can end with a stale running event. It is evidence, not a
    # trustworthy atomic lifecycle state, so never restyle it as live work.
    state = "waiting" if step is LEGACY else _state(event)
    label = {
        "complete": "Completed", "running": "Running", "waiting": "Waiting",
        "failed": "Failed", "skipped": "Skipped",
    }.get(state, "Waiting")
    if state == "running" and event and event.get("execution") == "concurrent":
        label = "Running (concurrent/async)"
    return f"{index + 1}. {step.name} · {label} · {_duration(_events_for(step, trace), event)}", f"evow-stage-{state}"


def _source_line(step: Step) -> int:
    path = load_settings().root / step.source_file
    for number, line in enumerate(path.read_text().splitlines(), 1):
        if step.source_location in line:
            return number
    return 1


def _source_excerpt(step: Step, lines: int = 18) -> str:
    path = load_settings().root / step.source_file
    source = path.read_text().splitlines()
    start = max(0, _source_line(step) - 1)
    return "\n".join(source[start:start + lines])


def _source_url(step: Step) -> str:
    path = load_settings().root / step.source_file
    return "/gradio_api/file=" + quote(str(path), safe="/") + "#:~:text=" + quote(step.source_location, safe="")


def _preview(trace: dict[str, Any], event: dict[str, Any] | None) -> str:
    if not event or not event.get("preview") or not trace.get("run_id"):
        return ""
    path = load_settings().run_root / str(trace["run_id"]) / str(event["preview"])
    if not path.is_file() or path.suffix.lower() not in {".png", ".jpg", ".jpeg", ".webp"}:
        return ""
    url = "/gradio_api/file=" + quote(str(path), safe="/")
    return f'<img class="evow-preview" src="{escape(url, quote=True)}" alt="Stage preview">'


def _log_link(trace: dict[str, Any], event: dict[str, Any] | None) -> str:
    if not event or not event.get("log_path") or not trace.get("run_id"):
        return ""
    path = load_settings().run_root / str(trace["run_id"]) / str(event["log_path"])
    if not path.is_file():
        return ""
    url = "/gradio_api/file=" + quote(str(path), safe="/")
    return f'<p><a class="evow-code-link" href="{escape(url, quote=True)}" target="_blank" rel="noopener">full worker log</a></p>'


def _tick_html(events: list[dict[str, Any]]) -> str:
    if not events:
        return ""
    ticks = []
    for item in events:
        metrics = item.get("metrics") or {}
        ticks.append(
            '<details class="evow-tick"><summary>{} · {}</summary><p>{}</p>{}</details>'.format(
                escape(str(item.get("status", "waiting"))),
                escape(format_stopwatch(float(item.get("stage_elapsed_seconds", 0)))),
                escape(str(item.get("detail", ""))), detail_rows(metrics),
            )
        )
    return '<details class="evow-ticks"><summary>events ({})</summary>{}</details>'.format(len(ticks), "".join(ticks))


def _models_html(step: Step) -> str:
    if not step.models:
        return ""
    rows = "".join(
        '<li><a class="evow-code-link" href="{}" target="_blank" rel="noopener">{}</a> <code>{}</code></li>'.format(
            escape(model.url, quote=True), escape(model.name), escape(model.identifier)
        ) for model in step.models
    )
    return "<ul>" + rows + "</ul>"


def workflow_card_html(trace: dict[str, Any], index: int) -> str:
    """Render code, artifacts, logs, and raw records for one persistent row."""
    steps = _steps(trace)
    if index >= len(steps):
        return ""
    step = steps[index]
    events = _events_for(step, trace)
    event = events[-1] if events else None
    if step is LEGACY:
        raw = escape(json.dumps(list(trace.get("events", [])), indent=2))
        return f'<article class="evow-stage-card"><p>Legacy trace: atomic terminal states were not recorded.</p><pre>{raw}</pre></article>'

    metrics = (event or {}).get("metrics") or {}
    duration = "Not started" if event is None else _duration(events, event)
    raw = escape(json.dumps(event or {}, indent=2))
    source = f"{step.source_file}:{_source_line(step)}"
    contract = detail_rows((("Inputs", step.inputs), ("Outputs", step.outputs), ("Control", step.controls)))
    live = detail_rows({"Stage time": duration, **{key: value for key, value in metrics.items() if key != "log_tail"}})
    log_tail = metrics.get("log_tail")
    return f"""<article class="evow-stage-card"><style>
.evow-stage-card{{color:#253646;line-height:1.4}}.evow-stage-card *{{box-sizing:border-box}}.evow-stage-detail{{display:grid;grid-template-columns:1fr 1fr;gap:14px}}.evow-stage-detail section{{min-width:0}}.evow-stage-detail h4{{margin:0 0 7px;font-size:12px;color:#425466;text-transform:uppercase;letter-spacing:.05em}}.evow-stage-detail p{{margin:0 0 8px;font-size:12px}}.evow-stage-card pre{{max-height:240px;margin:0 0 8px;overflow:auto;padding:8px;border:1px solid #d5dfe7;border-radius:5px;background:#f6f9fb;font:11px ui-monospace,SFMono-Regular,monospace;white-space:pre-wrap}}.evow-code-link{{color:#0b5e96;text-decoration:underline;font:12px ui-monospace,SFMono-Regular,monospace}}.evow-kv{{display:grid;gap:5px;margin:0}}.evow-kv div{{display:grid;grid-template-columns:minmax(90px,35%) 1fr;gap:8px}}.evow-kv dt{{color:#526575;font-size:12px}}.evow-kv dd{{margin:0;overflow-wrap:anywhere}}.evow-preview{{display:block;max-width:100%;max-height:230px;margin:0 0 8px;border:1px solid #c7d4de;border-radius:5px}}.evow-ticks,.evow-tick{{margin:7px 0;font-size:12px}}.evow-tick{{border-left:2px solid #93c5fd;padding-left:7px}}.evow-ticks summary,.evow-tick summary{{cursor:pointer}}@media(max-width:760px){{.evow-stage-detail{{grid-template-columns:1fr}}}}
</style><div class="evow-stage-detail"><section><h4>Program {help_icon(step.help_text, step.name)}</h4><p><a class="evow-code-link" href="{escape(_source_url(step), quote=True)}" target="_blank" rel="noopener">{escape(source)}</a></p><pre>{escape(_source_excerpt(step))}</pre><p><code>{escape(step.function_chain)}</code></p>{_models_html(step)}{contract}</section><section><h4>Live record</h4><p>{escape(str((event or {}).get("detail", "Waiting for this operation.")))}</p>{_preview(trace, event)}{live}{_tick_html(events)}{_log_link(trace, event)}{'<h4>Worker tail</h4><pre>' + escape(str(log_tail)) + '</pre>' if log_tail else ''}<details class="evow-raw"><summary>raw event JSON</summary><pre>{raw}</pre></details></section></div></article>"""


def workflow_labels(mode: str) -> tuple[str, ...]:
    """Return labels for callers that need the selected V2 flow profile."""
    return tuple(step.name for step in _steps({"workflow_version": 2, "mode": mode}))
