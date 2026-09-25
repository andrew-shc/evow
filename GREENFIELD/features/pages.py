"""Replay-shell controllers with consistent sample and saved-run source controls."""

from dataclasses import replace
from html import escape
from pathlib import Path
import shutil
from typing import Callable, Iterator

import gradio as gr

from GREENFIELD.app_core.contracts import FeatureResult, StageEvent, StageSpec
from GREENFIELD.app_core.media import read_video, read_recent_window
from GREENFIELD.app_core.trace import FeatureTrace, list_runs
from GREENFIELD.app_core.ui import APP_CSS, help_icon, run_total_update, source_and_saved_controls, stage_html, stage_label
from GREENFIELD.app_core.splat_viewer import empty_splat_html, splat_html
from GREENFIELD.app_core.model_catalog import models_for
from . import editing, future, selection

ASSETS = Path(__file__).resolve().parents[2] / "ASSETS"
SAMPLE_VIDEO = ASSETS / "replay" / "samples" / "trees_swaying_pexels_12644693.mp4"


def _execute_stream(feature: str, mode: str, source: str, request: object, adapter: Callable[..., Iterator[StageEvent | FeatureResult]]) -> Iterator[tuple[FeatureResult | None, dict]]:
    """Persist and yield every typed adapter event using Replay's live contract."""
    if not source:
        raise gr.Error("Choose a stationary-view video first.")
    if feature == "future":
        from .future_settings import load_future_settings
        settings = load_future_settings()
        frames, fps = read_recent_window(source, settings.history_seconds, settings.history_fps, settings.max_side)
    else:
        frames, fps = read_video(source)
    trace = FeatureTrace(ASSETS, feature, mode, {**request.__dict__, "source_name": Path(source).name, "fps": fps})
    active_stage = "Source"
    try:
        source_copy = trace.artifacts.file("source.mp4")
        shutil.copy2(source, source_copy)
        trace.record["source_video"] = source_copy.name
        trace._save()
        result = None
        for item in adapter(frames, fps, request, trace.artifacts):
            if isinstance(item, StageEvent):
                active_stage = item.stage
                trace.add(item)
                yield None, trace.record
            else:
                result = item
        if result is None:
            raise RuntimeError("The feature adapter returned no result.")
        trace.finish(result)
        yield result, trace.record
    except Exception as error:
        trace.add(StageEvent(active_stage, "error", str(error)))
        trace.add(StageEvent("Run failed", "error", str(error)))
        raise gr.Error(str(error)) from error


def _execute_source_adapter_stream(feature: str, mode: str, source: str, request: object, adapter: Callable[..., Iterator[StageEvent | FeatureResult]]) -> Iterator[tuple[FeatureResult | None, dict]]:
    """Persist source-path adapters that must control their own bounded decoding."""
    if not source:
        raise gr.Error("Choose a stationary-view video first.")
    trace = FeatureTrace(ASSETS, feature, mode, {**request.__dict__, "source_name": Path(source).name})
    active_stage = "Inspect source video"
    try:
        source_copy = trace.artifacts.file("source.mp4")
        shutil.copy2(source, source_copy)
        trace.record["source_video"] = source_copy.name
        trace._save()
        result = None
        for item in adapter(str(source_copy), request, trace.artifacts):
            if isinstance(item, StageEvent):
                active_stage = item.stage
                trace.add(item)
                yield None, trace.record
            else:
                result = item
        if result is None:
            raise RuntimeError("The feature adapter returned no result.")
        trace.finish(result)
        yield result, trace.record
    except Exception as error:
        trace.add(StageEvent(active_stage, "error", str(error)))
        trace.add(StageEvent("Run failed", "error", str(error)))
        raise gr.Error(str(error)) from error


def _safe_result_file(run_dir: Path, relative_path: object) -> Path:
    """Resolve only artifact paths owned by the selected saved run."""
    if not isinstance(relative_path, str):
        raise ValueError("Saved result has an invalid artifact path.")
    root = run_dir.resolve()
    candidate = (root / relative_path).resolve()
    if root not in candidate.parents or not candidate.is_file():
        raise ValueError("Saved result references an unavailable artifact.")
    return candidate


def _selection_gallery_items(clips: list[object], run_dir: Path, projected: bool = False) -> list[tuple[str, str]]:
    """Load one named visualization variant from run-owned query artifacts."""
    items: list[tuple[str, str]] = []
    key = "projected_highlighted" if projected else "highlighted"
    label = "Explicit 3D projection" if projected else "2D semantic mask"
    for clip in clips:
        if not isinstance(clip, dict):
            continue
        try:
            path = _safe_result_file(run_dir, clip.get(key)); start = float(clip["start_seconds"]); end = float(clip["end_seconds"]); score = float(clip["score"])
        except (KeyError, TypeError, ValueError):
            continue
        confidence = float(clip.get("grounding_confidence") or 0); coverage = float(clip.get("projected_mask_coverage" if projected else "semantic_mask_coverage") or 0)
        warning = " · LOW SPATIAL SPECIFICITY" if projected and clip.get("projected_low_specificity") else ""
        items.append((str(path), f"{label} · {start:05.1f}s–{end:05.1f}s · similarity {score:.3f} · grounding {confidence:.3f} · coverage {coverage:.1%}{warning}"))
    return items


def _live_selection_gallery(result: FeatureResult, projected: bool = False) -> list[tuple[str, str]]:
    """Build a live 2D or Explicit 3D gallery without saved-run lookup."""
    label = "Explicit 3D projection" if projected else "2D semantic mask"; items = []
    for clip in result.clips:
        path = clip.projected_highlighted if projected else clip.highlighted
        if path is None: continue
        coverage = clip.projected_mask_coverage if projected else clip.semantic_mask_coverage
        warning = " · LOW SPATIAL SPECIFICITY" if projected and clip.projected_low_specificity else ""
        items.append((str(path), f"{label} · {clip.start_seconds:05.1f}s–{clip.end_seconds:05.1f}s · similarity {clip.score:.3f} · grounding {(clip.grounding_confidence or 0):.3f} · coverage {(coverage or 0):.1%}{warning}"))
    return items


def _selection_provenance(trace: dict) -> str:
    """Render request identity beside results, making a stale display evident."""
    request = trace.get("request") or {}; metadata = ((trace.get("result") or {}).get("metadata") or {}); details = metadata.get("provenance") or {}
    return f"**Active Text Query run:** `{escape(str(trace.get('run_id', 'starting')))}` · query: `{escape(str(request.get('query', '')).strip())}` · mode: `{escape(str(trace.get('mode', '')))}` · source SHA-256: `{escape(str(details.get('source_sha256', 'calculating…')))}` · pipeline: `{escape(str(details.get('pipeline_revision', 'text-query-v2')))}`"


def _selection_viewer_update(metadata: dict, run_dir: Path) -> dict:
    """Restore the selected structure as an orbitable solid-color 4D splat timeline."""
    viewer = metadata.get("viewer") if isinstance(metadata, dict) else None
    if not isinstance(viewer, dict):
        return gr.update(value=empty_splat_html(), visible=False)
    try:
        paths = [_safe_result_file(run_dir, value) for value in viewer.get("splat_paths", [])]
    except ValueError:
        return gr.update(value=empty_splat_html(), visible=False)
    if not paths:
        return gr.update(value=empty_splat_html(), visible=False)
    durations = viewer.get("durations")
    if not isinstance(durations, list) or len(durations) != len(paths): durations = [1 / 12] * len(paths)
    return gr.update(value=splat_html([str(path) for path in paths], durations, None), visible=True)


def _selection_clip_choices() -> list[tuple[str, str]]:
    """List raw saved query clips that can become a short editing source."""
    import json

    choices: list[tuple[str, str]] = []
    root = ASSETS / "selection" / "runs"
    for trace_path in sorted(root.glob("*/trace.json"), reverse=True):
        try:
            record = json.loads(trace_path.read_text())
            clips = (record.get("result") or {}).get("clips") or []
            for index, clip in enumerate(clips):
                if not isinstance(clip, dict):
                    continue
                _safe_result_file(trace_path.parent, clip.get("source"))
                label = "{}–{} · {}".format(
                    float(clip["start_seconds"]), float(clip["end_seconds"]), trace_path.parent.name
                )
                choices.append((label, f"{trace_path.parent.name}:{index}"))
        except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError):
            continue
    return choices


def _load_selection_clip(choice: str) -> str:
    """Load the raw, never-highlighted source artifact from a saved query run."""
    import json

    if not choice or choice.count(":") != 1:
        raise gr.Error("Choose a saved Text Query clip.")
    run_id, raw_index = choice.split(":", 1)
    if "/" in run_id or "\\" in run_id:
        raise gr.Error("Choose a valid saved Text Query clip.")
    try:
        index = int(raw_index)
        run_dir = ASSETS / "selection" / "runs" / run_id
        record = json.loads((run_dir / "trace.json").read_text())
        clip = ((record.get("result") or {}).get("clips") or [])[index]
        return str(_safe_result_file(run_dir, clip.get("source")))
    except (OSError, ValueError, KeyError, IndexError, TypeError, json.JSONDecodeError) as error:
        raise gr.Error(f"Could not load Text Query clip: {error}") from error


EDITING_EXPLICIT_MODES = frozenset({"explicit"})


def _editing_viewer_update(metadata: dict, run_dir: Path) -> dict:
    """Build a saved or live explicit-edit splat timeline from run-owned assets."""
    viewer = metadata.get("viewer") if isinstance(metadata, dict) else None
    if not isinstance(viewer, dict):
        return gr.update(value=empty_splat_html(), visible=False)
    relative_paths = viewer.get("splat_paths")
    if not isinstance(relative_paths, list):
        return gr.update(value=empty_splat_html(), visible=False)
    try:
        paths = [_safe_result_file(run_dir, path) for path in relative_paths]
    except ValueError:
        return gr.update(value=empty_splat_html(), visible=False)
    if any(path.suffix != ".splat" for path in paths):
        return gr.update(value=empty_splat_html(), visible=False)
    durations = viewer.get("durations")
    if not isinstance(durations, list) or len(durations) != len(paths) or any(not isinstance(value, (int, float)) or value <= 0 for value in durations):
        durations = [1 / 12] * len(paths)
    return gr.update(value=splat_html([str(path) for path in paths], durations, None), visible=True)


def _editing_method_change(mode: str) -> dict:
    """Hide retained explicit scene output when the selected family is implicit."""
    return gr.update(visible=mode in EDITING_EXPLICIT_MODES)

def _method_specs(feature: str, specs: tuple[StageSpec, ...], mode: str) -> tuple[StageSpec, ...]:
    """Keep a shared stage layout, but reveal only the active method's models."""
    selected_ids = {model.identifier for model in models_for(feature, mode)}
    return tuple(replace(spec, model_refs=tuple(model for model in spec.model_refs if model.identifier in selected_ids)) for spec in specs)


def _flow(feature: str, specs: tuple[StageSpec, ...]) -> tuple[list[gr.Accordion], list[gr.HTML], gr.Markdown]:
    specs = _method_specs(feature, specs, "explicit")
    headers, cards = [], []
    with gr.Accordion("Show internals", open=False, elem_classes="internal-flow"):
        for number, spec in enumerate(specs):
            label, css_class = stage_label(number, spec)
            with gr.Accordion(label, open=False, elem_classes=["evow-stage", css_class]) as header:
                headers.append(header)
                cards.append(gr.HTML(stage_html(spec)))
        total = gr.Markdown("**Run total:** —", elem_classes="evow-run-total")
    return headers, cards, total


def _updates(feature: str, specs: tuple[StageSpec, ...], trace: dict) -> tuple[dict, ...]:
    specs = _method_specs(feature, specs, str(trace.get("mode", "explicit")))
    latest = {event["stage"]: event for event in trace["events"]}
    header_changes, card_changes = [], []
    for number, spec in enumerate(specs):
        stage_events = [event for event in trace["events"] if event["stage"] == spec.name]
        event = latest.get(spec.name)
        stage_started_at = float(stage_events[0]["elapsed_seconds"]) if stage_events else None
        label, css_class = stage_label(number, spec, event, stage_started_at)
        header_changes.append(gr.update(label=label, elem_classes=["evow-stage", css_class]))
        card_changes.append(gr.update(value=stage_html(spec, event, stage_started_at, stage_events)))
    # Gradio registers all accordion headers before all HTML cards. Keeping the
    # update order identical prevents a card update being applied to the next
    # stage header, which otherwise appears as duplicate or missing stages.
    return tuple(header_changes + card_changes + [run_total_update(trace)])

def _method_flow_change(feature: str, specs: tuple[StageSpec, ...], mode: str) -> tuple[dict, ...]:
    """Refresh dormant flow cards immediately when a methodology is selected."""
    return _updates(feature, specs, {"mode": mode, "events": []})



def _sample() -> str:
    if not SAMPLE_VIDEO.is_file():
        raise gr.Error("The bundled stationary-camera sample is unavailable.")
    return str(SAMPLE_VIDEO)


def _load_saved(feature: str, run_id: str) -> tuple[str, str | None, list[list] | None, dict]:
    """Restore a new-format saved run while rejecting invalid IDs and old partial traces."""
    if not run_id or "/" in run_id or "\\" in run_id:
        raise gr.Error("Choose a saved run.")
    run_dir = ASSETS / feature / "runs" / run_id
    try:
        import json
        record = json.loads((run_dir / "trace.json").read_text())
        if record.get("feature") != feature:
            raise ValueError("That saved run belongs to a different feature.")
        source = _safe_result_file(run_dir, record.get("source_video", "source.mp4"))
        result = record.get("result") or {}
        primary = _safe_result_file(run_dir, result["primary"]) if result.get("primary") else None
        return str(source), str(primary) if primary and primary.is_file() else None, result.get("rows"), record
    except (OSError, TypeError, ValueError, KeyError) as error:
        raise gr.Error(f"Could not load saved run: {error}") from error


def _page(feature: str, title: str, source_label: str, controls: Callable[[], tuple], specs: tuple[StageSpec, ...], handler: Callable[..., tuple], result_factory: Callable[[], gr.components.Component], is_table: bool = False) -> gr.Blocks:
    """Compose the shared source controls, setup column, output, and internals."""
    with gr.Blocks(css=APP_CSS, analytics_enabled=False) as page:
        gr.Markdown(f"# {title}")
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("## 1. Source " + help_icon("Searches a stationary-view upload for up to five non-overlapping four-second clips. Explicit 3D is an estimate from one fixed camera.", "Source limits"))
                source, sample, saved, load = source_and_saved_controls(
                    source_label, list_runs(ASSETS, feature), f"ASSETS/{feature}/runs/",
                    SAMPLE_VIDEO.is_file(), f"evow-{feature}-saved-runs",
                )
                inputs = controls()
                gr.Markdown("## 4. Generate")
                button = gr.Button("Generate" if title == "Future View" else "Search" if title == "Text Query" else "Apply", variant="primary")
            with gr.Column(scale=2):
                gr.Markdown("## 5. Output", elem_classes="evow-section-heading")
                output = result_factory()
        gr.Markdown("## Internal Execution Flow", elem_classes="evow-section-heading evow-flow-heading")
        headers, cards, run_total = _flow(feature, specs)
        button.click(handler, [source, *inputs], [output, *headers, *cards, run_total], concurrency_limit=1, show_progress="hidden")
        sample.click(_sample, None, source)
        def load_saved(run_id):
            loaded_source, primary, rows, trace = _load_saved(feature, run_id)
            return loaded_source, rows if is_table else primary, *_updates(feature, specs, trace)
        load.click(load_saved, saved, [source, output, *headers, *cards, run_total])
    return page


FUTURE_EXPLICIT_MODES = frozenset({"explicit"})


def _future_viewer_update(metadata: dict, run_dir: Path) -> dict:
    """Build a saved or live observed-history-to-generated-future splat panel."""
    viewer = metadata.get("viewer")
    if not isinstance(viewer, dict):
        return gr.update(value=empty_splat_html(), visible=True)
    relative_paths = viewer.get("splat_paths", [])
    if not isinstance(relative_paths, list) or not all(isinstance(path, str) for path in relative_paths):
        return gr.update(value=empty_splat_html(), visible=True)
    root = run_dir.resolve()
    paths = [(root / relative_path).resolve() for relative_path in relative_paths]
    if not paths or any(root not in path.parents or not path.is_file() or path.suffix != ".splat" for path in paths):
        return gr.update(value=empty_splat_html(), visible=True)
    observed_frames = int(viewer.get("observed_frames", 0))
    observed_fps = float(viewer.get("observed_fps", 0))
    forecast_frames = int(viewer.get("forecast_frames", 0))
    forecast_fps = float(viewer.get("forecast_fps", 0))
    if observed_frames < 1 or forecast_frames < 1 or observed_fps <= 0 or forecast_fps <= 0 or len(paths) != observed_frames + forecast_frames:
        return gr.update(value=empty_splat_html(), visible=True)
    durations = viewer.get("splat_durations")
    if not isinstance(durations, list) or len(durations) != len(paths) or any(not isinstance(duration, (int, float)) or duration <= 0 for duration in durations):
        durations = [1 / observed_fps] * observed_frames + [1 / forecast_fps] * forecast_frames
    return gr.update(value=splat_html([str(path) for path in paths], durations, observed_frames), visible=True)


def _future_method_change(mode: str) -> dict:
    """Hide the retained scene only outside the Explicit 3D method family."""
    visible = mode in FUTURE_EXPLICIT_MODES
    return gr.update(visible=visible)


def _future_timing_update(trace: dict) -> dict:
    """Show the total once, outside stage cards whose clocks are local."""
    events = trace.get("events", [])
    elapsed = float(events[-1].get("elapsed_seconds", 0)) if events else 0
    return gr.update(value=f"**Run total:** {elapsed:.2f}s")


def _execute_future_stream(source: str, mode: str, request: object, settings):
    """Yield each persisted Future View stage so Gradio renders real progress."""
    if not source: raise gr.Error("Choose a stationary-view video first.")
    # Create the trace before reading so the source stage has a genuine start
    # boundary rather than appearing as an instantaneous completed event.
    trace = FeatureTrace(ASSETS, "future", mode, {**request.__dict__, "source_name": Path(source).name})
    active_stage = "Sample input tail"
    try:
        trace.add(StageEvent(active_stage, "running", "Sampling the final stationary-camera history window."))
        yield None, trace.record
        frames, fps = read_recent_window(source, settings.history_seconds, settings.history_fps, settings.max_side)
        source_copy = trace.artifacts.file("source.mp4")
        shutil.copy2(source, source_copy)
        trace.record["source_video"] = source_copy.name
        trace.record["request"]["fps"] = fps
        trace._save()
        trace.add(StageEvent(active_stage, "complete",
                             f"Read the tail end of the input video: {len(frames)} frames / {len(frames) / fps:.2f}s at {fps:g} fps.",
                             metrics={"input_frames": len(frames), "input_seconds": round(len(frames) / fps, 2),
                                      "input_fps": fps, "input_segment": "tail end of source video"}))
        yield None, trace.record
        result = None
        for item in future.run(frames, fps, request, trace.artifacts, settings):
            if isinstance(item, StageEvent):
                active_stage = item.stage
                trace.add(item); yield None, trace.record
            else: result = item
        if result is None: raise RuntimeError("The Future View adapter returned no result.")
        trace.finish(result); yield result, trace.record
    except Exception as error:
        # Replay records both the stage that failed and a generic terminal event.
        # The former keeps the visible configured stage from being left running.
        trace.add(StageEvent(active_stage, "error", str(error)))
        trace.add(StageEvent("Run failed", "error", str(error)))
        raise gr.Error(str(error)) from error
def _future_effective_settings(base, history_seconds, history_fps, output_seconds, output_fps, chunk_frames, motion_fit_frames, observed_fps, forecast_fps, max_side):
    """Validate editable defaults and return an immutable per-run settings snapshot."""
    values = (history_seconds, history_fps, output_seconds, output_fps, chunk_frames, motion_fit_frames, observed_fps, forecast_fps, max_side)
    if any(float(value) <= 0 for value in values):
        raise gr.Error("Every Future View configuration value must be greater than zero.")
    return replace(base, history_seconds=float(history_seconds), history_fps=int(history_fps), output_seconds=int(output_seconds), output_fps=int(output_fps), chunk_frames=int(chunk_frames), motion_fit_frames=int(motion_fit_frames), observed_splat_timeline_fps=int(observed_fps), forecast_splat_timeline_fps=int(forecast_fps), max_side=int(max_side))

def future_configuration_rows(settings) -> list[list[str]]:
    """Return the effective, read-only Future View defaults for the UI."""
    return [
        ["Implicit model", settings.model_id],
        ["Observed history", f"{settings.history_seconds:g}s at {settings.history_fps} fps"],
        ["Generated future", f"{settings.output_seconds}s at {settings.output_fps} fps ({settings.output_frames} frames)"],
        ["SVD rollout chunk", f"{settings.chunk_frames} frames"],
        ["Explicit motion fit", f"{settings.motion_fit_frames} observed frames"],
        ["3D timeline", f"observed {settings.observed_splat_timeline_fps} fps · forecast {settings.forecast_splat_timeline_fps} fps"],
        ["Input cap", f"{settings.max_side}px longest side"],
    ]

def _future_configuration_markdown(settings) -> str:
    rows = future_configuration_rows(settings)
    return "| Default | Effective value |\n| --- | --- |\n" + "\n".join(f"| {name} | `{value}` |" for name, value in rows)


def build_future() -> gr.Blocks:
    """Build Future View with editable per-run configuration defaults."""
    from .future_settings import load_future_settings
    settings = load_future_settings()
    with gr.Blocks(css=APP_CSS, analytics_enabled=False) as page:
        gr.Markdown("# Future View")
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("## 1. Source " + help_icon("Uses only the final stationary-camera history window. Shorter videos use all available history.", "Observed-history limits"))
                source, sample, saved, load = source_and_saved_controls("Observed camera history", list_runs(ASSETS, "future"), "ASSETS/future/runs/", SAMPLE_VIDEO.is_file(), "evow-future-saved-runs")
                gr.Markdown("## 2. Methodology " + help_icon("Choose Explicit 3D for Video Depth Anything + gsplat reconstruction, or Implicit video for Stable Video Diffusion rollout. Exact active models appear in the internal-flow stages.", "Methodology"))
                mode = gr.Radio([("Explicit 3D", "explicit"), ("Implicit 3D", "implicit")], value="explicit", show_label=False, elem_classes="evow-method-choice")
                gr.Markdown("## 3. Forecast")
                with gr.Accordion("Configuration", open=False):
                    # Tiny inline info icon replaces the former standalone helper sentence, so the
                    # guidance now lives behind the same click popover as the rest of the dashboard
                    # (and still collapses with the accordion, being its first child).
                    with gr.Row():
                        history_seconds = gr.Number(settings.history_seconds, minimum=0.1, label="History seconds", container=False)
                        history_fps = gr.Number(settings.history_fps, minimum=1, precision=0, label="History fps", container=False)
                    with gr.Row():
                        output_seconds = gr.Number(settings.output_seconds, minimum=1, precision=0, label="Output seconds", container=False)
                        output_fps = gr.Number(settings.output_fps, minimum=1, precision=0, label="Output fps", container=False)
                    with gr.Row():
                        chunk_frames = gr.Number(settings.chunk_frames, minimum=1, precision=0, label="SVD chunk frames", container=False)
                        motion_fit_frames = gr.Number(settings.motion_fit_frames, minimum=1, precision=0, label="Motion fit frames", container=False)
                    with gr.Row():
                        observed_fps = gr.Number(settings.observed_splat_timeline_fps, minimum=1, precision=0, label="Observed 3D timeline fps", container=False)
                        forecast_fps = gr.Number(settings.forecast_splat_timeline_fps, minimum=1, precision=0, label="Forecast 3D timeline fps", container=False)
                    max_side = gr.Number(settings.max_side, minimum=64, precision=0, label="Maximum input side", container=False)
                    seed = gr.Number(value=0, precision=0, label="Seed", container=False)
                gr.Markdown("## 4. Generate")
                button = gr.Button("Generate", variant="primary")
            with gr.Column(scale=2):
                gr.Markdown("## 5. Output", elem_classes="evow-section-heading")
                output = gr.Video(label="Generated future estimate", interactive=False, elem_classes=["evow-media", "evow-output-slot"], show_download_button=True)
                splat_viewer = gr.HTML(value=empty_splat_html(), elem_id="future-splat-viewer-output")
        gr.Markdown("## Internal Execution Flow", elem_classes="evow-section-heading evow-flow-heading")
        headers, cards, run_total = _flow("future", future.STAGES)
        def handle(video, selected_mode, selected_seed, *values):
            effective = _future_effective_settings(settings, *values)
            request = future.FutureRequest(selected_mode, int(selected_seed))
            for result, trace in _execute_future_stream(video, selected_mode, request, effective):
                viewer = _future_viewer_update(result.metadata, result.primary.parent.parent) if result and selected_mode in FUTURE_EXPLICIT_MODES else gr.update(visible=False) if result else gr.update()
                yield str(result.primary) if result else None, viewer, *_updates("future", future.STAGES, trace)
        def load_saved(run_id):
            loaded_source, primary, _rows, trace = _load_saved("future", run_id)
            result = trace.get("result") or {}
            viewer = _future_viewer_update(result.get("metadata", {}), ASSETS / "future" / "runs" / run_id) if trace.get("mode") in FUTURE_EXPLICIT_MODES else gr.update(visible=False)
            return loaded_source, primary, viewer, *_updates("future", future.STAGES, trace)
        mode.change(lambda selected: _method_flow_change("future", future.STAGES, selected), mode, [*headers, *cards, run_total])
        config_inputs = [history_seconds, history_fps, output_seconds, output_fps, chunk_frames, motion_fit_frames, observed_fps, forecast_fps, max_side]
        button.click(handle, [source, mode, seed, *config_inputs], [output, splat_viewer, *headers, *cards, run_total], concurrency_limit=1, show_progress="hidden")
        mode.change(_future_method_change, mode, splat_viewer)
        sample.click(_sample, None, source)
        load.click(load_saved, saved, [source, output, splat_viewer, *headers, *cards, run_total])
    return page


def build_selection() -> gr.Blocks:
    """Build Text Query around a gallery of playable, non-cropped highlights."""
    with gr.Blocks(css=APP_CSS, analytics_enabled=False) as page:
        gr.Markdown("# Text Query")
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("## 1. Source " + help_icon("Searches a stationary-view upload for up to five non-overlapping four-second clips. Explicit 3D is an estimate from one fixed camera.", "Source limits"))
                source, sample, saved, load = source_and_saved_controls(
                    "Source video (up to 5 minutes)", list_runs(ASSETS, "selection"), "ASSETS/selection/runs/",
                    SAMPLE_VIDEO.is_file(), "evow-selection-saved-runs",
                )
                gr.Markdown("## 2. Query")
                query = gr.Textbox(label="Query", placeholder="dark clouds above moving tree branches", container=False)
                gr.Markdown("## 3. Methodology " + help_icon("Choose Explicit 3D to lift tracked masks through Video Depth Anything + gsplat, or Implicit 3D for direct video masks. Exact active models appear in the internal-flow stages.", "Methodology"))
                mode = gr.Radio(
                    [("Explicit 3D", "explicit"),
                     ("Implicit 3D", "implicit")],
                    value="explicit", show_label=False, elem_classes="evow-method-choice",
                )
                gr.Markdown("## 4. Search")
                button = gr.Button("Search", variant="primary")
            with gr.Column(scale=2):
                gr.Markdown("## 5. Highlighted matching clips " + help_icon("Explicit mode shows selected 4D Gaussian structure in solid magenta; it is an orbitable structure viewer, not a projected video mask.", "3D structure viewer"), elem_classes="evow-section-heading")
                semantic_gallery = gr.Gallery(label="2D semantic highlights (Grounding DINO + SAM2)", columns=2, rows=3, file_types=["video"], type="filepath", show_download_button=True, elem_classes="evow-media")
                splat_viewer = gr.HTML(value=empty_splat_html(), elem_id="selection-splat-viewer", visible=False)
        gr.Markdown("## Internal Execution Flow", elem_classes="evow-section-heading evow-flow-heading")
        headers, cards, run_total = _flow("selection", selection.STAGES)

        def handle(video, text, selected_mode):
            # Never leave a prior query visible during a fresh long-running search.
            yield [], gr.update(value=empty_splat_html(), visible=False), *_updates("selection", selection.STAGES, {"mode": selected_mode, "events": []})
            request = selection.SelectionRequest(selected_mode, text)
            for result, trace in _execute_source_adapter_stream("selection", selected_mode, video, request, selection.run):
                semantic = _live_selection_gallery(result) if result else gr.update()
                viewer = _selection_viewer_update(result.metadata, result.clips[0].source.parent.parent) if result and selected_mode == "explicit" else gr.update(value=empty_splat_html(), visible=False)
                yield semantic, viewer, *_updates("selection", selection.STAGES, trace)

        def load_saved(run_id):
            loaded_source, _primary, _rows, trace = _load_saved("selection", run_id); clips = ((trace.get("result") or {}).get("clips") or []); run_dir = ASSETS / "selection" / "runs" / run_id
            metadata = ((trace.get("result") or {}).get("metadata") or {})
            viewer = _selection_viewer_update(metadata, run_dir) if trace.get("mode") == "explicit" else gr.update(value=empty_splat_html(), visible=False)
            return loaded_source, _selection_gallery_items(clips, run_dir), viewer, *_updates("selection", selection.STAGES, trace)
        mode.change(lambda selected: _method_flow_change("selection", selection.STAGES, selected), mode, [*headers, *cards, run_total])

        button.click(handle, [source, query, mode], [semantic_gallery, splat_viewer, *headers, *cards, run_total], concurrency_limit=1, show_progress="hidden")
        sample.click(_sample, None, source)
        load.click(load_saved, saved, [source, semantic_gallery, splat_viewer, *headers, *cards, run_total])
    return page


def _editing_result_caption(run_id: str, request: editing.EditingRequest, primary: Path) -> str:
    """Make the result's unique run and text conditioning visible beside playback."""
    method = "Explicit 3D" if request.mode == "explicit" else "Implicit 3D"
    return (f"**Generated result** · run `{escape(run_id)}` · {method} · seed `{request.seed}` · "
            f"prompt: “{escape(request.prompt.strip())}” · artifact `{escape(primary.name)}`")


def build_editing() -> gr.Blocks:
    """Build Text Manipulation as a single-pass text-conditioned edit."""
    with gr.Blocks(css=APP_CSS, analytics_enabled=False) as page:
        gr.Markdown("# Text Manipulation")
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("## 1. Source " + help_icon("Upload one episode up to 6.75 seconds, or load a raw Text Query result. The yellow query overlay is never used as edit input; explicit geometry is estimated.", "Editing source limits"))
                source, sample, saved, load = source_and_saved_controls(
                    "Short source episode", list_runs(ASSETS, "editing"), "ASSETS/editing/runs/",
                    SAMPLE_VIDEO.is_file(), "evow-editing-saved-runs",
                )
                gr.Markdown("## 2. Instruction")
                prompt = gr.Textbox(label="Instruction", placeholder="flood the forest while preserving the fixed camera view", container=False)
                gr.Markdown("## 3. Methodology " + help_icon("Choose Explicit 3D to project scope through Video Depth Anything + gsplat, or Implicit 3D for direct VACE conditioning. Exact active models appear in the internal-flow stages.", "Methodology"))
                mode = gr.Radio(
                    [("Explicit 3D", "explicit"),
                     ("Implicit 3D", "implicit")],
                    value="explicit", show_label=False, elem_classes="evow-method-choice",
                )
                with gr.Accordion("Configuration", open=False):
                    seed = gr.Number(value=0, precision=0, label="Seed", container=False)
                gr.Markdown("## 4. Generate")
                button = gr.Button("Generate", variant="primary")
            with gr.Column(scale=2):
                gr.Markdown("## 5. Output", elem_classes="evow-section-heading")
                output = gr.Video(label="Generated edit / Explicit 3D render", interactive=False, elem_classes=["evow-media", "evow-output-slot"], show_download_button=True)
                result_caption = gr.Markdown("")
                splat_viewer = gr.HTML(value=empty_splat_html(), elem_id="editing-splat-viewer-output", visible=False)
        gr.Markdown("## Internal Execution Flow", elem_classes="evow-section-heading evow-flow-heading")
        headers, cards, run_total = _flow("editing", editing.STAGES)

        def handle(video, instruction, selected_mode, selected_seed):
            if not video:
                raise gr.Error("Choose a stationary-view video first.")
            request = editing.EditingRequest(selected_mode, instruction, int(selected_seed or 0))
            trace = FeatureTrace(ASSETS, "editing", selected_mode, {**request.__dict__, "source_name": Path(video).name})
            active_stage = "Inspect source episode"
            try:
                source_copy = trace.artifacts.file("source.mp4")
                shutil.copy2(video, source_copy)
                trace.record["source_video"] = source_copy.name
                trace._save()
                # Clear the prior result surface before any potentially long scene work.
                yield None, gr.update(value=empty_splat_html(), visible=False), "", *_updates("editing", editing.STAGES, trace.record)
                result = None
                for item in editing.run(str(source_copy), request, trace.artifacts):
                    if isinstance(item, StageEvent):
                        active_stage = item.stage
                        trace.add(item)
                        yield gr.update(), gr.update(), gr.update(), *_updates("editing", editing.STAGES, trace.record)
                    else:
                        result = item
                if result is None:
                    raise RuntimeError("Text Manipulation returned no generated result.")
                trace.finish(result)
                viewer = _editing_viewer_update(result.metadata, trace.run_dir) if request.mode in EDITING_EXPLICIT_MODES else gr.update(value=empty_splat_html(), visible=False)
                yield str(result.primary), viewer, _editing_result_caption(trace.run_dir.name, request, result.primary), *_updates("editing", editing.STAGES, trace.record)
            except Exception as error:
                trace.add(StageEvent(active_stage, "error", str(error)))
                trace.add(StageEvent("Run failed", "error", str(error)))
                raise gr.Error(str(error)) from error

        def load_saved(run_id):
            loaded_source, primary, _rows, record = _load_saved("editing", run_id)
            result = record.get("result") or {}
            request_data = record.get("request") or {}
            request = editing.EditingRequest(str(record.get("mode", "")), str(request_data.get("prompt", "")), int(request_data.get("seed", 0)))
            run_dir = ASSETS / "editing" / "runs" / run_id
            viewer = _editing_viewer_update(result.get("metadata", {}), run_dir) if record.get("mode") in EDITING_EXPLICIT_MODES else gr.update(value=empty_splat_html(), visible=False)
            caption = _editing_result_caption(run_id, request, Path(primary)) if primary else ""
            return loaded_source, primary, viewer, caption, *_updates("editing", editing.STAGES, record)
        mode.change(lambda selected: _method_flow_change("editing", editing.STAGES, selected), mode, [*headers, *cards, run_total])

        button.click(handle, [source, prompt, mode, seed], [output, splat_viewer, result_caption, *headers, *cards, run_total], concurrency_limit=1, show_progress="hidden")
        mode.change(_editing_method_change, mode, splat_viewer)
        sample.click(_sample, None, source)
        load.click(load_saved, saved, [source, output, splat_viewer, result_caption, *headers, *cards, run_total])
    return page
