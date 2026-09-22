"""Render a two-level, program-aware execution flow for each replay run."""

from dataclasses import dataclass
from html import escape
import json
from pathlib import Path
from typing import Any
from urllib.parse import quote

from .settings import load_settings


@dataclass(frozen=True)
class Step:
    """A compact flow node plus the implementation details revealed when opened."""

    name: str
    purpose: str
    source_file: str
    function_chain: str
    inputs: str
    outputs: str
    controls: str
    trace_stages: tuple[str, ...]


SHARED = (
    Step("Choose source", "Accept the recorded file or browser camera capture.", "GREENFIELD/replay/app.py", "_run(video, mode, start, count, yaw, shift, fov)", "MP4 or browser capture", "source path passed to the run coordinator", "Stationary-view source · included sample · saved run", ("Input clip",)),
    Step("Extract episode", "Decode one short RGB episode from the source at 12 fps.", "GREENFIELD/replay/clip.py", "extract_clip() → cv2.VideoCapture → write_video()", "source path · start seconds · 13/29/41 frames", "frames.npy · source.mp4 · source.png", "Clip start · episode length", ("Input clip",)),
)
EXPLICIT = (
    Step("Infer depth through time", "Estimate temporally consistent relative depth for the observed view.", "GREENFIELD/replay/depth_worker.py", "depth_worker.main() → VideoDepthAnything.infer_video_depth()", "frames.npy", "explicit/depths.npz", "Representation method: Explicit", ("Video depth prior", "3D initialization")),
    Step("Initialize 3D Gaussians", "Lift observed pixels into native 3D Gaussian primitives with initial motion.", "GREENFIELD/replay/gaussian_worker.py", "_initial_gaussians() → Farneback optical flow", "RGB frames · depths.npz", "centers · scales · colors · opacity · per-frame motion", "Representation method: Explicit", ("Gaussian initialization", "3D initialization")),
    Step("Fit the 4D scene", "Optimize one persistent Gaussian scene across every selected time step.", "GREENFIELD/replay/gaussian_worker.py", "gsplat.rasterization() → torch.optim.Adam → progress.json", "Gaussian state · observed frames", "optimized shared primitives and temporal deformation", "Episode length sets the observed motion window", ("4D Gaussian fitting",)),
    Step("Render requested view", "Render the learned 4D scene from the requested virtual camera.", "GREENFIELD/replay/gaussian_worker.py", "gsplat.rasterization() → splat.save_splat() → imageio video writer", "scene state · yaw · lateral shift · field of view", "rendered.mp4 · scene.pt · splat_####.splat", "Turn · sideways offset · field of view", ("Native 4D scene",)),
)
IMPLICIT = (
    Step("Prepare camera episode", "Package frames and target camera for the video generator.", "GREENFIELD/replay/implicit.py", "prepare_episode() → camera matrices", "RGB episode · virtual camera", "model episode and camera conditioning", "Representation method: Implicit", ("Camera setup",)),
    Step("Encode video and camera", "Transform images and camera rays into model conditioning.", "GREENFIELD/replay/anyview_worker.py", "VAE encode_rgb() → prepare_cams_latent()", "RGB frames · camera matrices", "video latents · Plücker camera rays", "Representation method: Implicit", ("Video diffusion",)),
    Step("Generate nearby video", "Denoise a target-view video conditioned on the source episode.", "GREENFIELD/replay/anyview_worker.py", "AnyView pipeline.generate()", "conditioning latents · virtual camera", "generated video latents", "Turn · sideways offset · field of view", ("Video diffusion",)),
    Step("Decode result", "Decode and save the generated target-view video.", "GREENFIELD/replay/anyview_worker.py", "VAE decode_rgb() → imageio.mimsave()", "generated latents", "rendered.mp4 · preview PNGs", "No additional control", ("Generated view",)),
)
FINAL = Step("Save run", "Persist all run settings, events, metrics, previews, and output locations.", "GREENFIELD/replay/trace.py", "RunTrace.add() → trace.json · result.json", "trace events · generated artifacts", "replayable run directory under ASSETS/replay/runs", "Load a saved run to inspect it again", ("Run complete",))


def _steps(mode: str) -> tuple[Step, ...]:
    """Return the implementation path for the selected representation."""
    return (*SHARED, *(EXPLICIT if mode == "explicit" else IMPLICIT), FINAL)


def _latest(events: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Keep the newest meaningful live update for each backend stage."""
    return {str(event["stage"]): event for event in events if event.get("record", True)}


def _event(step: Step, latest: dict[str, dict[str, Any]]) -> dict[str, Any] | None:
    """Find the trace event corresponding to a displayed implementation step."""
    matches = [latest[name] for name in step.trace_stages if name in latest]
    return matches[-1] if matches else None


def _state(index: int, steps: tuple[Step, ...], latest: dict[str, dict[str, Any]]) -> str:
    """Show waiting, active, complete, or failed without inventing progress."""
    if latest.get("Run complete", {}).get("status") == "complete":
        return "complete"
    event = _event(steps[index], latest)
    if event:
        # Trace producers historically called terminal failures "error". Keep
        # that wire format, but present one stable user-facing state everywhere.
        status = str(event.get("status", "waiting"))
        return "failed" if status == "error" else status
    return "complete" if any(_event(later, latest) for later in steps[index + 1:]) else "waiting"


def _stage_duration(step: Step, events: list[dict[str, Any]], event: dict[str, Any] | None) -> str:
    """Show a stage's own elapsed time instead of the total run age."""
    if event is None:
        return "—"
    matching = [entry for entry in events if entry.get("stage") in step.trace_stages]
    started = float(matching[0].get("elapsed_seconds", 0)) if matching else 0.0
    return f"{max(0.0, float(event.get('elapsed_seconds', 0)) - started):.2f}s"


STATUS_LABELS = {
    "complete": "Completed",
    "running": "Running",
    "waiting": "Waiting",
    "failed": "Failed",
}


def workflow_header(trace: dict[str, Any], index: int) -> tuple[str, str]:
    """Return the visible label and semantic CSS class for one accordion.

    Accordion content is never replaced to communicate status. Gradio updates
    only these header properties, preserving whether the user has it open.
    """
    mode = str(trace.get("mode", "explicit"))
    steps = _steps(mode)
    events = list(trace.get("events", []))
    latest = _latest(events)
    step = steps[index]
    state = _state(index, steps, latest)
    duration = _stage_duration(step, events, _event(step, latest))
    label = STATUS_LABELS.get(state, "Waiting")
    return (
        f"{index + 1}. {step.name} · {label} · {duration}",
        f"evow-stage-{state}",
    )


def _preview(trace: dict[str, Any], event: dict[str, Any] | None) -> str:
    """Embed a recorded stage preview from the local run directory when present."""
    if not event or not event.get("preview") or not trace.get("run_id"):
        return ""
    path = load_settings().run_root / str(trace["run_id"]) / str(event["preview"])
    if not path.is_file() or path.suffix.lower() not in {".png", ".jpg", ".jpeg", ".webp"}:
        return ""
    source = "/gradio_api/file=" + quote(str(path), safe="/")
    return f'<img class="evow-preview" src="{escape(source, quote=True)}" alt="Recorded stage preview">'


# Text fragments open the served source file at the function responsible for each stage.
SOURCE_LOCATIONS = {
    ("GREENFIELD/replay/app.py", "Choose source"): "def _run(",
    ("GREENFIELD/replay/clip.py", "Extract episode"): "def extract_clip(",
    ("GREENFIELD/replay/depth_worker.py", "Infer depth through time"): "def main()",
    ("GREENFIELD/replay/gaussian_worker.py", "Initialize 3D Gaussians"): "def _initial_gaussians(",
    ("GREENFIELD/replay/gaussian_worker.py", "Fit the 4D scene"): "def main()",
    ("GREENFIELD/replay/gaussian_worker.py", "Render requested view"): "def main()",
    ("GREENFIELD/replay/implicit.py", "Prepare camera episode"): "def prepare_episode(",
    ("GREENFIELD/replay/anyview_worker.py", "Encode video and camera"): "def main()",
    ("GREENFIELD/replay/anyview_worker.py", "Generate nearby video"): "def main()",
    ("GREENFIELD/replay/anyview_worker.py", "Decode result"): "def main()",
    ("GREENFIELD/replay/trace.py", "Save run"): "def add(",
}


def _source_location(step: Step) -> str:
    """Return the source entry point associated with a displayed stage."""
    return SOURCE_LOCATIONS[(step.source_file, step.name)]


def _code_link(step: Step) -> str:
    """Link a flow stage to its served local file and entry-point text fragment."""
    return "/gradio_api/file=" + quote(str(load_settings().root / step.source_file), safe="/") + "#:~:text=" + quote(_source_location(step), safe="")


def _source_line(step: Step) -> int:
    """Find the current one-based line number for a workflow entry point."""
    source_path = load_settings().root / step.source_file
    for line_number, line in enumerate(source_path.read_text().splitlines(), 1):
        if _source_location(step) in line:
            return line_number
    raise ValueError(f"Could not locate {_source_location(step)!r} in {step.source_file}.")


def _program_contract(step: Step) -> str:
    """Serialize static implementation facts in a readable, copyable form."""
    return escape(json.dumps({
        "source_file": f"{step.source_file}:{_source_line(step)}",
        "source_location": _source_location(step),
        "function_chain": step.function_chain,
        "inputs": step.inputs,
        "outputs": step.outputs,
        "editable_control": step.controls,
    }, indent=2))


def workflow_html(trace: dict[str, Any]) -> str:
    """Return a script-free two-level flow safe to update while a run streams."""
    mode = str(trace.get("mode", "explicit"))
    steps = _steps(mode)
    latest = _latest(list(trace.get("events", [])))
    cards: list[str] = []
    for index, step in enumerate(steps):
        event = _event(step, latest)
        state = _state(index, steps, latest)
        detail = str(event.get("detail", "Waiting for this operation.")) if event else "Waiting for this operation."
        runtime = {"status": state, "elapsed_seconds": event.get("elapsed_seconds", 0) if event else 0,
                   "metrics": event.get("metrics") if event else None, "event": event or None}
        if index == 0:
            runtime["run_request"] = trace.get("request", {})
        open_attribute = " open" if event and state in {"running", "error"} else ""
        source_url = _code_link(step)
        source_display = f"{step.source_file}:{_source_line(step)}"
        duration = _stage_duration(step, list(trace.get("events", [])), event)
        cards.append(
            f'<details class="evow-step evow-{escape(state)}"{open_attribute}>'
            f'<summary><span class="evow-number">{index + 1}</span><span class="evow-name">{escape(step.name)}</span>'
            f'<span class="evow-purpose">{escape(step.purpose)}</span><span class="evow-duration">{escape(duration)}</span></summary>'
            '<div class="evow-detail">'
            '<section><h4>Program</h4>'
            f'<p><a class="evow-code-link" href="{escape(source_url, quote=True)}" target="_blank" rel="noopener">{escape(source_display)}</a></p><p><code>{escape(step.function_chain)}</code></p>'
            f'<pre>{_program_contract(step)}</pre></section>'
            '<section><h4>Live run record</h4>'
            f'<p>{escape(detail)}</p>{_preview(trace, event)}'
            f'<pre>{escape(json.dumps(runtime, indent=2))}</pre></section>'
            '</div></details>'
        )
    return f'''<section class="evow-flow"><style>
.evow-flow{{background:transparent;color:#17212b;border:0;border-radius:0;padding:4px 0;font:14px system-ui,sans-serif}}.evow-flow *{{box-sizing:border-box}}.evow-flow h3{{margin:0 0 7px;font-size:19px}}.evow-flow>p{{margin:0 0 14px;color:#344454;line-height:1.5}}.evow-legend{{display:flex;flex-wrap:wrap;gap:8px;margin:0 0 14px}}.evow-tag{{padding:6px 9px;border:1px solid #b9c8d5;border-radius:99px;background:#edf3f7;color:#243545;font-size:12px}}
.evow-steps{{display:grid;gap:9px}}.evow-step{{border:1px solid #b7c5d0;border-radius:8px;background:#fff}}.evow-step summary{{display:grid;grid-template-columns:24px minmax(150px,1fr) minmax(180px,2fr) auto;align-items:center;gap:9px;cursor:pointer;padding:12px;list-style:none;font-weight:400}}.evow-step summary::-webkit-details-marker{{display:none}}.evow-number{{display:grid;place-items:center;width:23px;height:23px;border-radius:50%;background:#d9e7f0;font-size:12px}}.evow-purpose{{color:#526575;font-weight:400;font-size:12px}}.evow-duration{{color:#425466;font-size:12px;font-variant-numeric:tabular-nums}}.evow-name{{font-weight:400!important}}
.evow-detail{{display:grid;grid-template-columns:1fr 1fr;gap:14px;padding:13px;border-top:1px solid #d5dfe7;color:#253646;line-height:1.45}}.evow-detail section{{min-width:0}}.evow-detail h4{{margin:0 0 7px;font-size:12px;color:#425466;text-transform:uppercase;letter-spacing:.05em}}.evow-detail p{{margin:0 0 8px}}.evow-detail code{{color:#173f59;white-space:normal;overflow-wrap:anywhere;font-weight:600}}.evow-detail pre{{max-height:260px;margin:0;overflow:auto;padding:9px;border:1px solid #d5dfe7;border-radius:5px;background:#f6f9fb;color:#17212b;white-space:pre-wrap;font:12px ui-monospace,SFMono-Regular,monospace}}.evow-code-link{{color:#0b5e96;text-decoration:underline;font:12px ui-monospace,SFMono-Regular,monospace}}.evow-preview{{display:block;max-width:100%;max-height:260px;margin:0 0 9px;border:1px solid #c7d4de;border-radius:5px}}
.evow-complete{{border-left:5px solid #2c7a5e}}.evow-running{{border-left:5px solid #1677b8;background:#f0f8fd}}.evow-error{{border-left:5px solid #bd3d3d;background:#fff5f5}}.evow-waiting{{border-left:5px solid #a6b4c0;background:#f8fafb}}@media(max-width:760px){{.evow-step summary{{grid-template-columns:24px 1fr auto}}.evow-purpose{{grid-column:2/4}}.evow-detail{{grid-template-columns:1fr}}}}
</style><div class="evow-steps">{"".join(cards)}</div></section>'''


def workflow_labels(mode: str) -> tuple[str, ...]:
    """Return stable accordion labels for the selected execution path."""
    return tuple(f"{index + 1}. {step.name}" for index, step in enumerate(_steps(mode)))


def workflow_card_html(trace: dict[str, Any], index: int) -> str:
    """Render one live record within a persistent Gradio accordion."""
    mode = str(trace.get("mode", "explicit"))
    steps = _steps(mode)
    step = steps[index]
    events = list(trace.get("events", []))
    latest = _latest(events)
    event = _event(step, latest)
    state = _state(index, steps, latest)
    detail = str(event.get("detail", "Waiting for this operation.")) if event else "Waiting for this operation."
    runtime = {"status": state, "elapsed_seconds": event.get("elapsed_seconds", 0) if event else 0, "metrics": event.get("metrics") if event else None, "event": event or None}
    if index == 0:
        runtime["run_request"] = trace.get("request", {})
    source_url = _code_link(step)
    source_display = f"{step.source_file}:{_source_line(step)}"
    return f"""<article class=\"evow-stage-card evow-{escape(state)}\"><style>.evow-stage-card{{color:#253646;line-height:1.45}}.evow-stage-card *{{box-sizing:border-box}}.evow-stage-purpose{{margin:0 0 13px;color:#526575;font-size:12px}}.evow-stage-detail{{display:grid;grid-template-columns:1fr 1fr;gap:14px}}.evow-stage-detail section{{min-width:0}}.evow-stage-detail h4{{margin:0 0 7px;font-size:12px;color:#425466;text-transform:uppercase;letter-spacing:.05em}}.evow-stage-detail p{{margin:0 0 8px}}.evow-stage-detail code{{color:#173f59;white-space:normal;overflow-wrap:anywhere;font-weight:600}}.evow-stage-detail pre{{max-height:260px;margin:0;overflow:auto;padding:9px;border:1px solid #d5dfe7;border-radius:5px;background:#f6f9fb;color:#17212b;white-space:pre-wrap;font:12px ui-monospace,SFMono-Regular,monospace}}.evow-code-link{{color:#0b5e96;text-decoration:underline;font:12px ui-monospace,SFMono-Regular,monospace}}.evow-preview{{display:block;max-width:100%;max-height:260px;margin:0 0 9px;border:1px solid #c7d4de;border-radius:5px}}</style><p class=\"evow-stage-purpose\">{escape(step.purpose)}</p><div class=\"evow-stage-detail\"><section><h4>Program</h4><p><a class=\"evow-code-link\" href=\"{escape(source_url, quote=True)}\" target=\"_blank\" rel=\"noopener\">{escape(source_display)}</a></p><p><code>{escape(step.function_chain)}</code></p><pre>{_program_contract(step)}</pre></section><section><h4>Live run record</h4><p>{escape(detail)}</p>{_preview(trace, event)}<pre>{escape(json.dumps(runtime, indent=2))}</pre></section></div></article>"""
