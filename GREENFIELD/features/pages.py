"""Replay-shell controllers with consistent sample and saved-run source controls."""

from html import escape
from pathlib import Path
import shutil
from typing import Callable, Iterator

import gradio as gr

from GREENFIELD.app_core.contracts import FeatureResult, StageEvent, StageSpec
from GREENFIELD.app_core.media import read_video, read_recent_window
from GREENFIELD.app_core.trace import FeatureTrace, list_runs
from GREENFIELD.app_core.ui import APP_CSS, source_and_saved_controls, stage_html, stage_label
from GREENFIELD.app_core.splat_viewer import empty_splat_html, splat_html
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

def _flow(specs: tuple[StageSpec, ...]) -> tuple[list[gr.Accordion], list[gr.HTML]]:
    headers, cards = [], []
    with gr.Accordion("Show internals", open=False, elem_classes="internal-flow"):
        for number, spec in enumerate(specs):
            label, css_class = stage_label(number, spec)
            with gr.Accordion(label, open=False, elem_classes=["evow-stage", css_class]) as header:
                headers.append(header)
                cards.append(gr.HTML(stage_html(spec)))
    return headers, cards


def _updates(specs: tuple[StageSpec, ...], trace: dict) -> tuple[dict, ...]:
    latest = {event["stage"]: event for event in trace["events"]}
    header_changes, card_changes = [], []
    for number, spec in enumerate(specs):
        stage_events = [event for event in trace["events"] if event["stage"] == spec.name]
        event = latest.get(spec.name)
        stage_started_at = float(stage_events[0]["elapsed_seconds"]) if stage_events else None
        label, css_class = stage_label(number, spec, event, stage_started_at)
        header_changes.append(gr.update(label=label, elem_classes=["evow-stage", css_class]))
        card_changes.append(gr.update(value=stage_html(spec, event, stage_started_at)))
    # Gradio registers all accordion headers before all HTML cards. Keeping the
    # update order identical prevents a card update being applied to the next
    # stage header, which otherwise appears as duplicate or missing stages.
    return tuple(header_changes + card_changes)


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
                gr.Markdown("## 1. Source")
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
                headers, cards = _flow(specs)
        button.click(handler, [source, *inputs], [output, *headers, *cards], concurrency_limit=1, show_progress="hidden")
        sample.click(_sample, None, source)
        def load_saved(run_id):
            loaded_source, primary, rows, trace = _load_saved(feature, run_id)
            return loaded_source, rows if is_table else primary, *_updates(specs, trace)
        load.click(load_saved, saved, [source, output, *headers, *cards])
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


def _clear_gpu_memory() -> str:
    """Offer an owner-controlled recovery action after CUDA allocation failures."""
    from .model_adapters import clear_gpu_memory
    result = clear_gpu_memory()
    before = result["before_bytes"] / 1024**3
    after = result["after_bytes"] / 1024**3
    released = ", ".join(result["released"]) or "cached CUDA allocations"
    return f"**GPU memory cleared:** {released}. App allocation: {before:.2f} → {after:.2f} GiB."


def _execute_future_stream(source: str, mode: str, request: object):
    """Yield each persisted Future View stage so Gradio renders real progress."""
    if not source: raise gr.Error("Choose a stationary-view video first.")
    from .future_settings import load_future_settings
    settings = load_future_settings()
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
        for item in future.run(frames, fps, request, trace.artifacts):
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

def build_future() -> gr.Blocks:
    """Build the only feature page whose explicit result includes an orbitable scene."""
    from .future_settings import load_future_settings
    settings = load_future_settings()
    with gr.Blocks(css=APP_CSS, analytics_enabled=False) as page:
        gr.Markdown("# Future View")
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("## 1. Source")
                source, sample, saved, load = source_and_saved_controls(
                    "Observed camera history", list_runs(ASSETS, "future"), "ASSETS/future/runs/",
                    SAMPLE_VIDEO.is_file(), "evow-future-saved-runs",
                )
                gr.Markdown(f"Input tail: up to the final {settings.history_seconds:g} seconds of the uploaded video, resampled at {settings.history_fps} fps (up to {settings.history_seconds * settings.history_fps:g} frames). Shorter clips use all available history.")
                gr.Markdown("## 2. Methodology")
                mode = gr.Radio([("Explicit 3D: Per-run Scene Projection", "explicit"),
                                 ("Implicit 3D: Video Generation", "implicit")],
                                value="explicit", show_label=False, elem_classes="evow-method-choice")
                gr.Markdown("## 3. Forecast")
                seed = gr.Number(value=0, precision=0, label="Seed")
                clear_gpu = gr.Button("Clear GPU memory", variant="secondary")
                gpu_status = gr.Markdown("Clear cached feature models after a CUDA OOM, then retry.")
                gr.Markdown(f"Output: one generated {settings.output_seconds}-second future at {settings.output_fps} fps ({settings.output_frames} frames). Explicit 3D keeps observed history at {settings.observed_splat_timeline_fps} fps and exports the generated future at {settings.forecast_splat_timeline_fps} fps for smooth 3D playback; neither output is observed footage or a reliable prediction.")
                gr.Markdown("## 4. Generate")
                button = gr.Button("Generate", variant="primary")
            with gr.Column(scale=2):
                gr.Markdown("## 5. Output", elem_classes="evow-section-heading")
                output = gr.Video(label="Generated future estimate", interactive=False, elem_classes="evow-media", show_download_button=True)
                run_timing = gr.Markdown("**Run total:** —", elem_classes="evow-run-timing")
                splat_viewer = gr.HTML(value=empty_splat_html(), elem_id="future-splat-viewer-output")
                gr.Markdown("## Internal Execution Flow", elem_classes="evow-section-heading evow-flow-heading")
                headers, cards = _flow(future.STAGES)

        def handle(video, selected_mode, selected_seed):
            request = future.FutureRequest(selected_mode, int(selected_seed))
            for result, trace in _execute_future_stream(video, selected_mode, request):
                if result is None:
                    yield None, gr.update(), _future_timing_update(trace), *_updates(future.STAGES, trace)
                elif selected_mode in FUTURE_EXPLICIT_MODES:
                    yield str(result.primary), _future_viewer_update(result.metadata, result.primary.parent.parent), _future_timing_update(trace), *_updates(future.STAGES, trace)
                else:
                    yield str(result.primary), gr.update(visible=False), _future_timing_update(trace), *_updates(future.STAGES, trace)

        def load_saved(run_id):
            loaded_source, primary, _rows, trace = _load_saved("future", run_id)
            result = trace.get("result") or {}
            if trace.get("mode") in FUTURE_EXPLICIT_MODES:
                viewer = _future_viewer_update(result.get("metadata", {}), ASSETS / "future" / "runs" / run_id)
                return loaded_source, primary, viewer, _future_timing_update(trace), *_updates(future.STAGES, trace)
            return loaded_source, primary, gr.update(visible=False), _future_timing_update(trace), *_updates(future.STAGES, trace)

        button.click(handle, [source, mode, seed], [output, splat_viewer, run_timing, *headers, *cards], concurrency_limit=1, show_progress="hidden")
        mode.change(_future_method_change, mode, splat_viewer)
        clear_gpu.click(_clear_gpu_memory, None, gpu_status, queue=False, show_progress="hidden")
        sample.click(_sample, None, source)
        load.click(load_saved, saved, [source, output, splat_viewer, run_timing, *headers, *cards])
    return page


def build_selection() -> gr.Blocks:
    """Build Text Query around a gallery of playable, non-cropped highlights."""
    with gr.Blocks(css=APP_CSS, analytics_enabled=False) as page:
        gr.Markdown("# Text Query")
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("## 1. Source")
                source, sample, saved, load = source_and_saved_controls(
                    "Source video (up to 5 minutes)", list_runs(ASSETS, "selection"), "ASSETS/selection/runs/",
                    SAMPLE_VIDEO.is_file(), "evow-selection-saved-runs",
                )
                gr.Markdown("Text Query searches one stationary-view upload for up to five non-overlapping 4-second source clips. The highlighted region is never spatially cropped.")
                gr.Markdown("Explicit 3D region support is estimated from this one fixed camera view.")
                gr.Markdown("## 2. Text")
                query = gr.Textbox(label="Query", placeholder="dark clouds above moving tree branches")
                gr.Markdown("## 3. Methodology")
                mode = gr.Radio(
                    [("Explicit 3D: Dynamic Gaussian semantic regions", "explicit"),
                     ("Implicit 3D: Video-language retrieval and tracked masks", "implicit")],
                    value="explicit", show_label=False, elem_classes="evow-method-choice",
                )
                gr.Markdown("## 4. Search")
                button = gr.Button("Search", variant="primary")
            with gr.Column(scale=2):
                gr.Markdown("## 5. Highlighted matching clips", elem_classes="evow-section-heading")
                provenance = gr.Markdown("**Active Text Query run:** —")
                semantic_gallery = gr.Gallery(label="2D semantic highlights (Grounding DINO + SAM2)", columns=2, rows=3, file_types=["video"], type="filepath", show_download_button=True, elem_classes="evow-media")
                splat_viewer = gr.HTML(value=empty_splat_html(), elem_id="selection-splat-viewer", visible=False)
                gr.Markdown("Explicit mode shows selected 4D Gaussian structure in solid magenta; it is an orbitable structure viewer, not a projected video mask.")
                gr.Markdown("## Internal Execution Flow", elem_classes="evow-section-heading evow-flow-heading")
                headers, cards = _flow(selection.STAGES)

        def handle(video, text, selected_mode):
            # Never leave a prior query visible during a fresh long-running search.
            yield [], gr.update(value=empty_splat_html(), visible=False), "**Active Text Query run:** starting…", *_updates(selection.STAGES, {"events": []})
            request = selection.SelectionRequest(selected_mode, text)
            for result, trace in _execute_source_adapter_stream("selection", selected_mode, video, request, selection.run):
                semantic = _live_selection_gallery(result) if result else gr.update()
                viewer = _selection_viewer_update(result.metadata, result.clips[0].source.parent.parent) if result and selected_mode == "explicit" else gr.update(value=empty_splat_html(), visible=False)
                yield semantic, viewer, _selection_provenance(trace), *_updates(selection.STAGES, trace)

        def load_saved(run_id):
            loaded_source, _primary, _rows, trace = _load_saved("selection", run_id); clips = ((trace.get("result") or {}).get("clips") or []); run_dir = ASSETS / "selection" / "runs" / run_id
            metadata = ((trace.get("result") or {}).get("metadata") or {})
            viewer = _selection_viewer_update(metadata, run_dir) if trace.get("mode") == "explicit" else gr.update(value=empty_splat_html(), visible=False)
            return loaded_source, _selection_gallery_items(clips, run_dir), viewer, _selection_provenance(trace), *_updates(selection.STAGES, trace)

        button.click(handle, [source, query, mode], [semantic_gallery, splat_viewer, provenance, *headers, *cards], concurrency_limit=1, show_progress="hidden")
        sample.click(_sample, None, source)
        load.click(load_saved, saved, [source, semantic_gallery, splat_viewer, provenance, *headers, *cards])
    return page


def _editing_scope_details(prepared: editing.PreparedEdit) -> str:
    """Describe the exact persisted VACE mask that awaits owner approval."""
    warning = "\n\n⚠️ **Warning:** this named target covers more than 75% of the frame." if prepared.broad_warning else ""
    return f"**Scope preview:** `{prepared.scope}` · editable pixels: **{prepared.coverage:.1%}**.{warning}"


def _editing_result_caption(run_id: str, request: editing.EditingRequest, primary: Path) -> str:
    """Make the result's unique run and text conditioning visible beside playback."""
    method = "Explicit 3D" if request.mode == "explicit" else "Implicit 3D"
    return (f"**Generated result** · run `{escape(run_id)}` · {method} · seed `{request.seed}` · "
            f"prompt: “{escape(request.prompt.strip())}” · artifact `{escape(primary.name)}`")


def _prepared_edit_from_trace(trace: FeatureTrace) -> editing.PreparedEdit:
    """Recover approval details from run-owned trace data, never browser state."""
    for event in reversed(trace.record.get("events", [])):
        if event.get("stage") == "Review edit scope":
            metrics = event.get("metrics") or {}
            preview = _safe_result_file(trace.run_dir, event.get("preview"))
            return editing.PreparedEdit(
                preview=preview,
                scope=str(metrics["scope"]),
                coverage=float(metrics["mask_coverage"]),
                broad_warning=bool(metrics.get("broad_target_warning", False)),
            )
    raise ValueError("That run has no saved edit-scope preview.")


def build_editing() -> gr.Blocks:
    """Build Text Manipulation as a reviewable scope preview followed by approval."""
    with gr.Blocks(css=APP_CSS, analytics_enabled=False) as page:
        gr.Markdown("# Text Manipulation")
        prepared_state = gr.State(value=None)
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("## 1. Source")
                source, sample, saved, load = source_and_saved_controls(
                    "Short source episode", list_runs(ASSETS, "editing"), "ASSETS/editing/runs/",
                    SAMPLE_VIDEO.is_file(), "evow-editing-saved-runs",
                )
                gr.Markdown("Upload one episode up to 6.75 seconds, or load a raw Text Query result below. The visible yellow query overlay is never used as edit input.")
                gr.Markdown("All edited output is generated. Any explicit 3D geometry is estimated from the single fixed camera view.")
                query_clip = gr.Dropdown(choices=_selection_clip_choices(), label="Saved Text Query source clip")
                with gr.Row():
                    load_query_clip = gr.Button("Load query clip")
                    refresh_query_clips = gr.Button("Refresh query clips", variant="secondary")
                gr.Markdown("## 2. Instruction")
                prompt = gr.Textbox(label="Instruction", placeholder="flood the forest while preserving the fixed camera view")
                gr.Markdown("## 3. Methodology")
                mode = gr.Radio(
                    [("Explicit 3D: Dynamic Gaussian scene edit", "explicit"),
                     ("Implicit 3D: Text-conditioned video edit", "implicit")],
                    value="explicit", show_label=False, elem_classes="evow-method-choice",
                )
                seed = gr.Number(value=0, precision=0, label="Seed")
                clear_gpu = gr.Button("Clear GPU memory", variant="secondary")
                gpu_status = gr.Markdown("Clear cached feature models after a CUDA OOM, then retry.")
                gr.Markdown("## 4. Review and apply")
                preview_button = gr.Button("Preview edit scope", variant="secondary")
                confirm_button = gr.Button("Generate approved edit", variant="primary", interactive=False)
            with gr.Column(scale=2):
                gr.Markdown("## 5. Output", elem_classes="evow-section-heading")
                output = gr.Video(label="Generated edit / Explicit 3D render", interactive=False, elem_classes="evow-media", show_download_button=True)
                result_caption = gr.Markdown("")
                splat_viewer = gr.HTML(value=empty_splat_html(), elem_id="editing-splat-viewer-output", visible=False)
                scope_preview = gr.Video(label="Editable-region preview (yellow)", interactive=False, visible=False, elem_classes="evow-media")
                scope_details = gr.Markdown("")
                gr.Markdown("## Internal Execution Flow", elem_classes="evow-section-heading evow-flow-heading")
                headers, cards = _flow(editing.STAGES)

        def preview_scope(video, instruction, selected_mode, selected_seed):
            request = editing.EditingRequest(selected_mode, instruction, int(selected_seed or 0))
            if not video:
                raise gr.Error("Choose a stationary-view video first.")
            trace = FeatureTrace(ASSETS, "editing", selected_mode, {**request.__dict__, "source_name": Path(video).name})
            active_stage = "Inspect source episode"
            try:
                source_copy = trace.artifacts.file("source.mp4")
                shutil.copy2(video, source_copy)
                trace.record["source_video"] = source_copy.name
                trace._save()
                # Clear both prior result surfaces before any potentially long scene work.
                yield None, gr.update(value=empty_splat_html(), visible=False), gr.update(value=None, visible=False), gr.update(value="Preparing the editable-region preview…"), gr.update(value=""), gr.update(interactive=False), None, *_updates(editing.STAGES, trace.record)
                prepared = None
                for item in editing.prepare(str(source_copy), request, trace.artifacts):
                    if isinstance(item, StageEvent):
                        active_stage = item.stage
                        trace.add(item)
                        yield gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(interactive=False), None, *_updates(editing.STAGES, trace.record)
                    else:
                        prepared = item
                if prepared is None:
                    raise RuntimeError("Text Manipulation did not produce an edit-scope preview.")
                yield None, gr.update(value=empty_splat_html(), visible=False), str(prepared.preview), gr.update(value=_editing_scope_details(prepared)), gr.update(value=f"**Run prepared:** `{trace.run_dir.name}`. Approve this exact mask to start generation."), gr.update(interactive=True), {"run_id": trace.run_dir.name}, *_updates(editing.STAGES, trace.record)
            except Exception as error:
                trace.add(StageEvent(active_stage, "error", str(error)))
                trace.add(StageEvent("Run failed", "error", str(error)))
                raise gr.Error(str(error)) from error

        def generate_approved(state):
            if not isinstance(state, dict) or not isinstance(state.get("run_id"), str):
                raise gr.Error("Preview an edit scope before generating.")
            trace = FeatureTrace.resume(ASSETS, "editing", state["run_id"])
            request_data = trace.record.get("request") or {}
            request = editing.EditingRequest(str(request_data.get("mode", "")), str(request_data.get("prompt", "")), int(request_data.get("seed", 0)))
            prepared = _prepared_edit_from_trace(trace)
            source_path = _safe_result_file(trace.run_dir, trace.record.get("source_video", "source.mp4"))
            active_stage = "Generate edit proposal"
            try:
                yield gr.update(), gr.update(), gr.update(), gr.update(), gr.update(value=f"**Generating run:** `{trace.run_dir.name}`."), gr.update(interactive=False), state, *_updates(editing.STAGES, trace.record)
                result = None
                for item in editing.generate_prepared(str(source_path), request, trace.artifacts, prepared.scope, prepared.coverage, prepared.broad_warning):
                    if isinstance(item, StageEvent):
                        active_stage = item.stage
                        trace.add(item)
                        yield gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(interactive=False), state, *_updates(editing.STAGES, trace.record)
                    else:
                        result = item
                if result is None:
                    raise RuntimeError("Text Manipulation returned no generated result.")
                trace.finish(result)
                viewer = _editing_viewer_update(result.metadata, trace.run_dir) if request.mode in EDITING_EXPLICIT_MODES else gr.update(value=empty_splat_html(), visible=False)
                yield str(result.primary), viewer, gr.update(), gr.update(), gr.update(value=_editing_result_caption(trace.run_dir.name, request, result.primary)), gr.update(interactive=False), None, *_updates(editing.STAGES, trace.record)
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
            preview_name = result.get("metadata", {}).get("scope_preview")
            try:
                preview = str(_safe_result_file(run_dir, preview_name)) if preview_name else None
            except ValueError:
                preview = None
            details = ""
            if isinstance(result.get("metadata"), dict) and "mask_coverage" in result["metadata"]:
                restored = editing.PreparedEdit(Path(preview) if preview else run_dir / "scope_preview.mp4", str(result["metadata"].get("scope", "unknown")), float(result["metadata"]["mask_coverage"]), bool(result["metadata"].get("broad_target_warning", False)))
                details = _editing_scope_details(restored)
            caption = _editing_result_caption(run_id, request, Path(primary)) if primary else ""
            return loaded_source, primary, viewer, gr.update(value=preview, visible=bool(preview)), details, caption, gr.update(interactive=False), None, *_updates(editing.STAGES, record)

        preview_button.click(preview_scope, [source, prompt, mode, seed], [output, splat_viewer, scope_preview, scope_details, result_caption, confirm_button, prepared_state, *headers, *cards], concurrency_limit=1, show_progress="hidden")
        confirm_button.click(generate_approved, prepared_state, [output, splat_viewer, scope_preview, scope_details, result_caption, confirm_button, prepared_state, *headers, *cards], concurrency_limit=1, show_progress="hidden")
        mode.change(_editing_method_change, mode, splat_viewer)
        clear_gpu.click(_clear_gpu_memory, None, gpu_status, queue=False, show_progress="hidden")
        sample.click(_sample, None, source)
        load.click(load_saved, saved, [source, output, splat_viewer, scope_preview, scope_details, result_caption, confirm_button, prepared_state, *headers, *cards])
        load_query_clip.click(_load_selection_clip, query_clip, source)
        refresh_query_clips.click(lambda: gr.update(choices=_selection_clip_choices()), None, query_clip, queue=False)
    return page
