"""Replay-shell controllers with consistent sample and saved-run source controls."""

from pathlib import Path
import shutil
from typing import Callable, Iterator

import gradio as gr

from GREENFIELD.app_core.contracts import FeatureResult, StageEvent, StageSpec
from GREENFIELD.app_core.media import read_video, read_recent_window
from GREENFIELD.app_core.trace import FeatureTrace, list_runs
from GREENFIELD.app_core.ui import APP_CSS, source_and_saved_controls, stage_html, stage_label
from GREENFIELD.replay.splat_viewer import empty_splat_html, splat_html
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
        source = run_dir / str(record.get("source_video", "source.mp4"))
        if not source.is_file():
            raise ValueError("This older saved run does not include a reloadable source clip.")
        result = record.get("result") or {}
        primary = run_dir / result["primary"] if result.get("primary") else None
        return str(source), str(primary) if primary and primary.is_file() else None, result.get("rows"), record
    except (OSError, ValueError, KeyError) as error:
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


def _clear_future_gpu_memory() -> str:
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
        clear_gpu.click(_clear_future_gpu_memory, None, gpu_status, queue=False, show_progress="hidden")
        sample.click(_sample, None, source)
        load.click(load_saved, saved, [source, output, splat_viewer, run_timing, *headers, *cards])
    return page


def build_selection() -> gr.Blocks:
    def controls():
        gr.Markdown("## 2. Text")
        query = gr.Textbox(label="Query", placeholder="dark clouds above moving branches")
        gr.Markdown("## 3. Methodology")
        mode = gr.Radio([("Explicit 3D: Per-run Scene Search", "explicit"), ("Implicit 3D: Video Embeddings", "implicit")], value="explicit", show_label=False, elem_classes="evow-method-choice")
        return query, mode
    def handle(video, query, mode):
        for result, trace in _execute_stream("selection", mode, video, selection.SelectionRequest(mode, query), selection.run):
            yield (result.rows if result else gr.update()), *_updates(selection.STAGES, trace)
    return _page("selection", "Text Query", "Archive window", controls, selection.STAGES, handle, lambda: gr.Dataframe(headers=["Start", "End", "Score", "Method"], label="Playable interval candidates", interactive=False), is_table=True)


def build_editing() -> gr.Blocks:
    def controls():
        gr.Markdown("## 2. Instruction")
        prompt = gr.Textbox(label="Instruction", placeholder="make the scene warmer and brighter")
        gr.Markdown("## 3. Methodology")
        mode = gr.Radio([("Explicit 3D: Reconstructed Scene Edit", "explicit"), ("Implicit 3D: Video Edit", "implicit")], value="explicit", show_label=False, elem_classes="evow-method-choice")
        seed = gr.Number(value=0, precision=0, label="Seed")
        return prompt, mode, seed
    def handle(video, prompt, mode, seed):
        for result, trace in _execute_stream("editing", mode, video, editing.EditingRequest(mode, prompt, int(seed)), editing.run):
            yield (str(result.primary) if result else gr.update()), *_updates(editing.STAGES, trace)
    return _page("editing", "Text Manipulation", "Observed source", controls, editing.STAGES, handle, lambda: gr.Video(label="Generated edited view", interactive=False, elem_classes="evow-media", show_download_button=True))
