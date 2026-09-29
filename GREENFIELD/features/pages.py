"""Replay-shell controllers with consistent sample and saved-run source controls."""

from dataclasses import replace
from functools import partial
from html import escape
from urllib.parse import quote
from pathlib import Path
import math
import shutil
from typing import Callable, Iterator

import gradio as gr

from GREENFIELD.app_core.contracts import FeatureResult, StageEvent, StageSpec
from GREENFIELD.app_core.media import read_video, read_recent_window
from GREENFIELD.app_core.trace import FeatureTrace, list_runs
from GREENFIELD.app_core.timing import format_stopwatch
from GREENFIELD.app_core.ui import APP_CSS, SAVED_RUN_PLACEHOLDER_JS, flat_config_number, flat_config_slider, flat_config_textbox, flat_trim_range, trim_range_markup, help_icon, run_total_update, source_and_saved_controls, stage_html, stage_label
from GREENFIELD.app_core.splat_viewer import empty_splat_html, splat_html
from GREENFIELD.app_core.flow import CURRENT_FLOW_VERSION, LEGACY_FLOW_VERSION, RETIRED_STAGE_ALIASES, RETIRED_STAGE_ALIASES_BY_FEATURE, FeatureFlow, FlowStage, profile_id, resolve_flow, stage_applies
from GREENFIELD.app_core.super_stages import available_super_stages, rollup_stage_states, super_stage_update
from GREENFIELD.app_core.model_catalog import models_for
from . import editing, future, selection
from .text_media import probe_video

ASSETS = Path(__file__).resolve().parents[2] / "ASSETS"
SAMPLE_VIDEO = ASSETS / "replay" / "samples" / "trees_swaying_pexels_12644693.mp4"

# Each dashboard project gets its matching SP0X pair. Keep these page-owned so
# selecting a short source from one tab can never select a similarly numbered
# sample on another tab.
PROJECT_SAMPLE_VIDEOS = {
    "future": (
        ("Sample Video 1", ASSETS / "samples" / "sp02_app01_ARECIBO_OBS_SHORT.mp4"),
        ("Sample Video 2", ASSETS / "samples" / "sp02_app02_tsunami.mp4"),
    ),
    "selection": (
        ("Sample Video 1", ASSETS / "samples" / "sp03_app01_river_flooding.mp4"),
                ("Sample Video 2", ASSETS / "samples" / "sp03_app02_flood_leakage.avi"),
    ),
    "editing": (
        ("Sample Video 1", ASSETS / "samples" / "sp04_app01_windsock.mp4"),
        ("Sample Video 2", ASSETS / "samples" / "sp04_app02_model_house_timelapse.mp4"),
    ),
}


def _sample_videos(feature: str) -> tuple[tuple[str, Path], ...]:
    """Return the basic sample plus the two curated videos for one dashboard page."""
    return (("Sample Video (Basic)", SAMPLE_VIDEO), *PROJECT_SAMPLE_VIDEOS[feature])


def _stage_has_terminal(trace: FeatureTrace, stage_id: str) -> bool:
    """Whether the active stage already owns a terminal event in this trace.

    A trailing step can fail after its stage was already reported complete — for
    example a missing adapter artifact after training, or a post-encode save. The
    exception handler must not then emit a second ``error`` terminal for the same
    stage, or the run shows a stage that both succeeded and failed. When a
    terminal exists the caller emits only the generic run-failure event.
    """
    return any(
        event.get("stage_id") == stage_id and event.get("status") in {"complete", "error", "skipped"}
        for event in trace.record.get("events", [])
    )


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
    trace = _new_feature_trace(feature, mode, {**request.__dict__, "source_name": Path(source).name, "fps": fps})
    active_stage = "Source"
    # Track the declared id alongside the label so a failure after a stage has
    # begun repaints that real stage; a label alone is no longer a valid identity
    # once the flow namespace is enforced.
    active_stage_id = active_stage
    try:
        source_copy = trace.artifacts.file("source.mp4")
        shutil.copy2(source, source_copy)
        trace.record["source_video"] = source_copy.name
        trace._save()
        result = None
        for item in adapter(frames, fps, request, trace.artifacts):
            if isinstance(item, StageEvent):
                active_stage = item.stage
                active_stage_id = item.stage_id or active_stage
                trace.add(item)
                yield None, trace.record
            else:
                result = item
        if result is None:
            raise RuntimeError("The feature adapter returned no result.")
        trace.finish(result)
        yield result, trace.record
    except Exception as error:
        # Only fail the active stage if it has not already reached a terminal, so
        # a trailing failure never paints a second terminal on the same row.
        if not _stage_has_terminal(trace, active_stage_id):
            trace.add(StageEvent(active_stage, "error", str(error), stage_id=active_stage_id))
        trace.add(StageEvent("Run failed", "error", str(error)))
        yield None, trace.record
        raise gr.Error(str(error)) from error


def _execute_source_adapter_stream(feature: str, mode: str, source: str, request: object, adapter: Callable[..., Iterator[StageEvent | FeatureResult]]) -> Iterator[tuple[FeatureResult | None, dict]]:
    """Persist source-path adapters that must control their own bounded decoding."""
    if not source:
        raise gr.Error("Choose a stationary-view video first.")
    trace = _new_feature_trace(feature, mode, {**request.__dict__, "source_name": Path(source).name})
    # The pre-adapter window belongs to the flow's first declared stage. Seeding
    # both the label and its stable id means an early failure repaints the real
    # row; the old display-name id would be ignored by the collision-safe
    # renderer and leave the stage looking live.
    flow = _current_flow(feature)
    name_to_id = {stage.spec.name: stage.stage_id for stage in flow.stages}
    first_stage = flow.stages[0] if flow.stages else None
    active_stage = first_stage.spec.name if first_stage else "Source"
    active_stage_id = first_stage.stage_id if first_stage else active_stage
    try:
        source_copy = trace.artifacts.file("source.mp4")
        shutil.copy2(source, source_copy)
        trace.record["source_video"] = source_copy.name
        trace._save()
        result = None
        for item in adapter(str(source_copy), request, trace.artifacts):
            if isinstance(item, StageEvent):
                active_stage = item.stage
                # A legacy adapter may leave ``stage_id`` empty. Resolve it through
                # the declared name map and persist the resolved identity, rather
                # than passing the display label off as an explicit id.
                active_stage_id = item.stage_id or name_to_id.get(item.stage, "")
                if not item.stage_id and active_stage_id:
                    item = replace(item, stage_id=active_stage_id)
                trace.add(item)
                yield None, trace.record
            else:
                result = item
        if result is None:
            raise RuntimeError("The feature adapter returned no result.")
        trace.finish(result)
        yield result, trace.record
    except Exception as error:
        if not _stage_has_terminal(trace, active_stage_id):
            trace.add(StageEvent(active_stage, "error", str(error), stage_id=active_stage_id))
        trace.add(StageEvent("Run failed", "error", str(error)))
        yield None, trace.record
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


def _selection_player_html(items: list[tuple[str, str]]) -> str:
    """Pair each native player with its own source-interval record below it."""
    if not items:
        return '<p class="evow-empty-clips">No highlighted clips yet.</p>'
    clips = []
    for number, (path, caption) in enumerate(items, start=1):
        source = "/gradio_api/file=" + quote(path, safe="/")
        details = caption.split(" · ")
        interval = details[1] if len(details) > 1 else "Source interval unavailable"
        metrics = " · ".join(details[2:]) if len(details) > 2 else caption
        clips.append(
            '<article class="evow-clip-result">'
            f'<video controls preload="metadata" playsinline src="{escape(source, quote=True)}"></video>'
            '<div class="evow-clip-interval">'
            f'<strong>Match {number} · source {escape(interval)}</strong>'
            f'<p>{escape(metrics)}</p></div></article>'
        )
    return '<div class="evow-video-list">' + "".join(clips) + '</div>'


SELECTION_EXPLICIT_MODES = frozenset({"explicit"})


def _selection_provenance(trace: dict) -> str:
    """Render request identity beside results, making a stale display evident."""
    request = trace.get("request") or {}; metadata = ((trace.get("result") or {}).get("metadata") or {}); details = metadata.get("provenance") or {}
    return f"**Active Text Query run:** `{escape(str(trace.get('run_id', 'starting')))}` · query: `{escape(str(request.get('query', '')).strip())}` · mode: `{escape(str(trace.get('mode', '')))}` · source SHA-256: `{escape(str(details.get('source_sha256', 'calculating…')))}` · pipeline: `{escape(str(details.get('pipeline_revision', 'text-query-v2')))}`"


def _selection_viewer_update(metadata: dict, run_dir: Path) -> dict:
    """Read legacy single-viewer metadata without weakening paired current results."""
    viewer = metadata.get("viewer") if isinstance(metadata, dict) else None
    if not isinstance(viewer, dict):
        return gr.update(value=empty_splat_html(), visible=True)
    try:
        paths = [_safe_result_file(run_dir, value) for value in viewer.get("splat_paths", [])]
    except ValueError:
        return gr.update(value=empty_splat_html(), visible=True)
    durations = viewer.get("durations")
    if not paths:
        return gr.update(value=empty_splat_html(), visible=True)
    if not isinstance(durations, list) or len(durations) != len(paths):
        durations = [1 / 12] * len(paths)
    return gr.update(value=splat_html([str(path) for path in paths], durations, None), visible=True)


def _selection_pair_records(clips: list[object], run_dir: Path | None = None, explicit: bool = True) -> list[dict[str, object]]:
    """Build one selectable result record per match for either query method."""
    records: list[dict[str, object]] = []
    video_key = "projected_highlighted" if explicit else "highlighted"
    for index, clip in enumerate(clips):
        if isinstance(clip, dict):
            if run_dir is None:
                continue
            try:
                video = _safe_result_file(run_dir, clip.get(video_key))
                splats = [_safe_result_file(run_dir, value) for value in clip.get("selected_splats", [])] if explicit else []
                start, end = float(clip["start_seconds"]), float(clip["end_seconds"])
            except (ValueError, TypeError, KeyError):
                continue
        else:
            video = getattr(clip, video_key, None)
            splats = list(getattr(clip, "selected_splats", ())) if explicit else []
            start, end = float(getattr(clip, "start_seconds", 0)), float(getattr(clip, "end_seconds", 0))
        if video is None or (explicit and not splats):
            continue
        records.append({"label": f"Match {index + 1} · {start:05.1f}s–{end:05.1f}s", "video": str(video), "splats": [str(path) for path in splats]})
    return records


def _selection_pair_update(choice: str | None, pairs: list[dict[str, object]]) -> tuple[dict, dict]:
    """Update the single selected video and its optional Explicit 3D viewer."""
    try:
        pair = pairs[int(choice or 0)]
        video = str(pair["video"])
        splats = [str(path) for path in pair.get("splats", [])]
    except (IndexError, TypeError, ValueError, KeyError):
        return gr.update(value=None, visible=False), gr.update(value=empty_splat_html(), visible=False)
    viewer = gr.update(value=splat_html(splats, [1 / 12] * len(splats), None), visible=True) if splats else gr.update(value=empty_splat_html(), visible=False)
    return gr.update(value=video, visible=True), viewer


def _trim_range_bounds(video: str | None, maximum_seconds: float, target_id: str) -> tuple[dict, dict]:
    """Reset one custom two-handle trim bar and its hidden Gradio value."""
    if not video:
        duration = maximum_seconds
    else:
        try:
            duration = probe_video(video).duration_seconds
        except (OSError, ValueError):
            duration = maximum_seconds
    end = min(duration, maximum_seconds)
    return gr.update(value=trim_range_markup(target_id, 0, end, duration)), gr.update(value=f"0,{end:.3f}")


def _parse_trim_range(raw: object) -> tuple[float, float]:
    """Validate the custom browser range before a run can consume it."""
    try:
        start_text, end_text = str(raw).split(",", 1)
        start, end = float(start_text), float(end_text)
    except (TypeError, ValueError) as error:
        raise ValueError("Trim range must contain a numeric start and end.") from error
    if not math.isfinite(start) or not math.isfinite(end) or start < 0 or end <= start:
        raise ValueError("Trim end must be after trim start.")
    return start, end




def _selection_method_change(mode: str) -> dict:
    """Clear stale structure; a viewer appears only after an Explicit match exists."""
    return gr.update(value=empty_splat_html(), visible=False)


def _selection_config_visibility(mode: str) -> tuple[dict, ...]:
    """Show the Explicit scene group only for Explicit 3D Text Query runs.

    Each update sets ``visible`` alone, never ``value``, so switching methods
    cannot reset a knob the owner already tuned. The tuple order matches the
    groups registered in ``build_selection``.
    """
    return (
        gr.update(visible=True),  # Source
        gr.update(visible=True),  # Results
        gr.update(visible=True),  # Matching
        gr.update(visible=mode in SELECTION_EXPLICIT_MODES),  # Explicit scene
    )


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
        return gr.update(value=empty_splat_html(), visible=True)
    relative_paths = viewer.get("splat_paths")
    if not isinstance(relative_paths, list):
        return gr.update(value=empty_splat_html(), visible=True)
    try:
        paths = [_safe_result_file(run_dir, path) for path in relative_paths]
    except ValueError:
        return gr.update(value=empty_splat_html(), visible=True)
    if any(path.suffix != ".splat" for path in paths):
        return gr.update(value=empty_splat_html(), visible=True)
    durations = viewer.get("durations")
    if not isinstance(durations, list) or len(durations) != len(paths) or any(not isinstance(value, (int, float)) or value <= 0 for value in durations):
        durations = [1 / 12] * len(paths)
    return gr.update(value=splat_html([str(path) for path in paths], durations, None), visible=True)


def _editing_method_change(mode: str) -> dict:
    """Clear stale scene output while toggling between explicit and implicit methods."""
    return gr.update(
        value=empty_splat_html(),
        visible=mode in EDITING_EXPLICIT_MODES,
    )


def _editing_config_visibility(mode: str) -> tuple[dict, ...]:
    """Show the Scene group only for Explicit 3D Text Manipulation runs.

    Each update sets ``visible`` alone, never ``value``, so switching methods
    cannot reset a knob the owner already tuned. The tuple order matches the
    groups registered in ``build_editing``.
    """
    return (
        gr.update(visible=True),  # Episode
        gr.update(visible=True),  # Edit
        gr.update(visible=True),  # Matching
        gr.update(visible=mode in EDITING_EXPLICIT_MODES),  # Scene
        gr.update(visible=True),  # Run
    )


def _method_specs(feature: str, specs: tuple[StageSpec, ...], mode: str) -> tuple[StageSpec, ...]:
    """Keep a shared stage layout, but reveal only the active method's models."""
    selected_ids = {model.identifier for model in models_for(feature, mode)}
    return tuple(replace(spec, model_refs=tuple(model for model in spec.model_refs if model.identifier in selected_ids)) for spec in specs)


# The one registry mapping each feature to its current stage profile. Stage
# declarations and their rendering both go through it, so a feature's identity
# is declared in exactly one place.
_FLOWS = {
    "future": future.flow,
    "selection": selection.flow,
    "editing": editing.flow,
}


def _current_flow(feature: str) -> FeatureFlow:
    """Return the current ordered stage profile registered for ``feature``."""
    return _FLOWS[feature]()


def _new_feature_trace(feature: str, mode: str, request_dict: dict) -> FeatureTrace:
    """Start a trace bound to the exact flow profile that will later render it."""
    flow = _current_flow(feature)
    return FeatureTrace(
        ASSETS, feature, mode, request_dict,
        flow_version=flow.flow_version,
        profile_id=profile_id(feature, mode, flow.flow_version),
    )


def _retired_alias(feature: str, retired_identity: str) -> str | None:
    """Resolve a retired id/name for ``feature``, scoped before global.

    A feature-scoped alias wins because a retired identity can be a *different*
    feature's current id: Text Manipulation retired ``prepare_method`` onto its
    own ``prepare_mask`` while Future View globally retires it onto ``svd_load``.
    The global map stays the fallback for identities no feature has scoped.
    """
    scoped = RETIRED_STAGE_ALIASES_BY_FEATURE.get(feature, {})
    return scoped.get(retired_identity) or RETIRED_STAGE_ALIASES.get(retired_identity)


def _event_stage_id(event: dict, name_to_id: dict[str, str], legacy: bool, feature: str) -> str:
    """Resolve one persisted event to a current stage id, keeping namespaces apart.

    The id and name namespaces are deliberately handled as two separate lookups:

    * A current-version event carries a non-empty ``stage_id``. It matches ONLY
      that exact id. An unknown id is returned as-is so it selects no row and is
      ignored; it is never retried against the display-name map, which would let a
      stale or foreign id silently land on a same-spelled stage. An id-less
      current event still resolves by display name.
    * A legacy (pre-current) event is mapped best-effort onto the current rows,
      in this order: (a) a non-empty ``stage_id`` that exactly matches a current
      id wins; (b) a known retired identity from the feature-scoped aliases (then
      the global ``RETIRED_STAGE_ALIASES``) maps onto its current id — this is
      tried for the event's id *and* for its display name, because a v1 trace
      carried no id at all while a v2 trace carried the old id, and both the old
      ids and the old names are aliased; (c) the unique display-name match for an
      id-less event whose label is still a current name.
    * A name that is not a known display name or a known retired name resolves to
      an empty identity (which ``FeatureFlow`` validation never assigns to a
      stage) and is ignored.

    A numeric-looking id is a real id here, never a position.
    """
    stage_id = event.get("stage_id") or ""
    if stage_id and not legacy:
        return stage_id
    if stage_id:
        if stage_id in name_to_id.values():
            return stage_id
        alias = _retired_alias(feature, stage_id)
        if alias is not None and alias in name_to_id.values():
            return alias
    name = event.get("stage", "")
    if legacy:
        alias = _retired_alias(feature, name)
        if alias is not None and alias in name_to_id.values():
            return alias
    return name_to_id.get(name, "")


def _active_stage_id(events: list[dict], resolved_ids: list[str], known_ids: set[str]) -> str | None:
    """Return the newest configured event's stage id when that event is running."""
    for event, event_id in zip(reversed(events), reversed(resolved_ids)):
        if event_id in known_ids:
            return event_id if event.get("status") == "running" else None
    return None


def _recorded_by_legacy(stage: FlowStage, legacy_ids: frozenset[str], legacy_names: tuple[str, ...]) -> bool:
    """Whether a stage existed when a legacy trace was written.

    ``legacy_ids`` is the current-id set the trace's *exact* version recorded, so
    a v2 trace that already persisted ids marks only the rows it genuinely lacked
    and never a union row it did record. When the version is unknown below
    current (empty ``legacy_ids``) no per-version set exists, so we fall back to
    the flow's frozen v1 *display names* — the last-resort identity that survives
    a stage-set change. Mixing the two would let a current snake_case id
    masquerade as a legacy name, so the fallback is always name-based.
    """
    if legacy_ids:
        return stage.stage_id in legacy_ids
    return stage.spec.name in legacy_names


def _placeholder_event(stage: FlowStage, spec: StageSpec, detail: str) -> dict:
    """Build a synthetic skipped event for a row this trace legitimately lacks."""
    return {
        "stage": spec.name,
        "stage_id": stage.stage_id,
        "status": "skipped",
        "detail": detail,
        "elapsed_seconds": 0.0,
        "stage_elapsed_seconds": 0.0,
    }


def _skipped_placeholder(stage: FlowStage, mode: str, flow_version: int, legacy_ids: frozenset[str], legacy_names: tuple[str, ...], spec: StageSpec) -> dict | None:
    """Return a skipped row for a stage the run could not have executed, if any.

    A pre-current trace is resolved against the applicability its *own* version
    recorded, before the current flow's mode rules are consulted: a row present
    in that version's profile but without an event is simply historical (it
    renders Waiting/Interrupted), never "Not used by this method", even when the
    current flow restricts it to the other mode. Only a row genuinely absent from
    that version's profile is "Not recorded". A current trace keeps the
    mode-derived skip so a genuinely inapplicable row is not left waiting.
    """
    if flow_version < CURRENT_FLOW_VERSION:
        if not _recorded_by_legacy(stage, legacy_ids, legacy_names):
            return _placeholder_event(stage, spec, "Not recorded by this legacy run.")
        return None
    if not stage_applies(stage, mode):
        return _placeholder_event(stage, spec, "Not used by this method.")
    return None


def _flow(feature: str) -> tuple[list[gr.Accordion], list[gr.Accordion], list[gr.HTML], gr.Markdown]:
    """Render grouped, always-resident stages so live updates preserve open cards."""
    flow = _current_flow(feature)
    specs = _method_specs(feature, tuple(stage.spec for stage in flow.stages), "explicit")
    positions = {stage.stage_id: number for number, stage in enumerate(flow.stages)}
    groups, headers, cards = [], [], []
    with gr.Accordion("Internal Execution Flow", open=False, elem_classes="internal-flow"):
        for group in available_super_stages(flow.groups, set(positions)):
            with gr.Accordion(f"{group.label} · Waiting · —", open=False, elem_classes=["evow-flow-group", "evow-flow-group-waiting"]) as group_component:
                groups.append(group_component)
                for stage_id in group.stage_ids:
                    number = positions[stage_id]
                    label, css_class = stage_label(number, specs[number])
                    with gr.Accordion(label, open=False, elem_classes=["evow-stage", css_class]) as header:
                        headers.append(header)
                        cards.append(gr.HTML(stage_html(specs[number])))
        total = gr.Markdown("**Run total:** —", elem_classes="evow-run-total")
    return groups, headers, cards, total


def _updates(feature: str, trace: dict, is_saved: bool = False) -> tuple[dict, ...]:
    """Refresh grouped stages and child cards, matching events by stable stage id.

    ``is_saved`` says the trace was loaded from a completed-on-disk run rather
    than yielded live by a button click. It must be passed explicitly: a saved
    trace can legitimately have no ``result`` (a failed or interrupted run), and
    a live trace has no ``result`` until its final yield, so ``result`` cannot
    tell the two apart. Only a saved trace derives a dangling final ``running``
    tick as "Interrupted"; a live mid-run tick must keep streaming as running.
    """
    mode = str(trace.get("mode", "explicit"))
    flow_version = int(trace.get("flow_version", LEGACY_FLOW_VERSION))
    resolution = resolve_flow(feature, mode, flow_version, trace.get("profile_id", ""), _FLOWS)
    flow = resolution.flow
    # A version newer than this build cannot be interpreted at all: its events
    # may use ids this renderer has never heard of, so every row stays waiting.
    ignores_events = flow_version > CURRENT_FLOW_VERSION
    events = [] if ignores_events else trace.get("events", [])
    is_legacy = flow_version < CURRENT_FLOW_VERSION
    specs = _method_specs(feature, tuple(stage.spec for stage in flow.stages), mode)
    positions = {stage.stage_id: number for number, stage in enumerate(flow.stages)}
    known_ids = set(positions)
    name_to_id = {stage.spec.name: stage.stage_id for stage in flow.stages}

    resolved_ids = [_event_stage_id(event, name_to_id, is_legacy, feature) for event in events]
    events_by_id: dict[str, list[dict]] = {}
    for event_id, event in zip(resolved_ids, events):
        events_by_id.setdefault(event_id, []).append(event)
    latest_by_id = {event_id: stage_events[-1] for event_id, stage_events in events_by_id.items()}
    active_id = _active_stage_id(events, resolved_ids, known_ids)

    header_changes, card_changes, stage_states = [], [], []
    for number, stage in enumerate(flow.stages):
        stage_id = stage.stage_id
        spec = specs[number]
        stage_events = events_by_id.get(stage_id, [])
        event = latest_by_id.get(stage_id)
        displayed_events = stage_events
        display_override = None
        if event is None and not ignores_events:
            event = _skipped_placeholder(stage, mode, flow_version, resolution.legacy_ids, flow.legacy_stage_ids, spec)
            if event is not None:
                displayed_events = [event]
        elif event is not None and is_legacy and event.get("status") == "running" and stage_id != active_id:
            # Legacy traces are serial: an earlier running tick is visually
            # completed once a later stage has begun. This runs before the
            # interrupted derivation so only the trace's final dangling tick is
            # treated as cut off; an earlier one merely predates a later stage.
            event = {**event, "status": "complete", "detail": "Completed before the next pipeline stage began."}
            displayed_events = [*stage_events[:-1], event]
        elif event is not None and is_saved and not ignores_events and event.get("status") == "running":
            # A saved, supported run (current or legacy) whose last tick is still
            # running never finished; show it interrupted rather than as a live
            # stage. Nothing is persisted and the parent rolls this derived error
            # up. An unsupported >current trace never reaches here: its events are
            # ignored entirely, so the derivation cannot invent an interpretation.
            event = {**event, "status": "error", "detail": "Run ended while this stage was running."}
            displayed_events = [*stage_events[:-1], event]
            display_override = "Interrupted"
        stage_started_at = float(stage_events[0]["elapsed_seconds"]) if stage_events else None
        label, css_class = stage_label(number, spec, event, stage_started_at, display_override=display_override)
        header_changes.append(gr.update(label=label, elem_classes=["evow-stage", css_class]))
        card_changes.append(gr.update(value=stage_html(spec, event, stage_started_at, displayed_events)))
        status = str(event.get("status", "waiting")) if event else "waiting"
        elapsed = 0.0 if event is None else float(event.get("stage_elapsed_seconds", max(0, float(event.get("elapsed_seconds", 0)) - (stage_started_at or 0))))
        stage_states.append((status, elapsed))
    group_changes = [
        super_stage_update(group.label, *rollup_stage_states([stage_states[positions[stage_id]] for stage_id in group.stage_ids]))
        for group in available_super_stages(flow.groups, known_ids)
    ]
    # Feed the run total the same (possibly emptied) event list the rows used, so
    # an unsupported version's uninterpretable elapsed time never leaks into the
    # displayed total; the notice still travels with the render.
    return tuple(group_changes + header_changes + card_changes + [run_total_update({**trace, "events": events}, notice=resolution.notice)])


def _method_flow_change(feature: str, mode: str) -> tuple[dict, ...]:
    """Refresh dormant grouped flow cards immediately when a methodology is selected."""
    return _updates(feature, {"mode": mode, "flow_version": CURRENT_FLOW_VERSION, "events": []})


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


def _page(feature: str, title: str, source_label: str, controls: Callable[[], tuple], handler: Callable[..., tuple], result_factory: Callable[[], gr.components.Component], is_table: bool = False) -> gr.Blocks:
    """Compose the shared source controls, setup column, output, and internals."""
    with gr.Blocks(css=APP_CSS, analytics_enabled=False) as page:
        gr.Markdown(f"# {title}")
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("## 1. Source " + help_icon("Searches a stationary-view upload for up to five non-overlapping four-second clips. Explicit 3D is an estimate from one fixed camera.", "Source limits"))
                source, sample, saved, load = source_and_saved_controls(
                    source_label, list_runs(ASSETS, feature), f"ASSETS/{feature}/runs/",
                    _sample_videos(feature), f"evow-{feature}-saved-runs",
                )
                inputs = controls()
                gr.Markdown("## 4. Generate")
                button = gr.Button("Generate" if title == "Future View" else "Search" if title == "Text Query" else "Apply", variant="primary")
            with gr.Column(scale=2):
                gr.Markdown("## 5. Output", elem_classes="evow-section-heading")
                output = result_factory()
        groups, headers, cards, run_total = _flow(feature)
        button.click(handler, [source, *inputs], [output, *groups, *headers, *cards, run_total], concurrency_limit=1, show_progress="hidden")
        sample.click(_sample, None, source)
        def load_saved(run_id):
            loaded_source, primary, rows, trace = _load_saved(feature, run_id)
            return loaded_source, rows if is_table else primary, *_updates(feature, trace, is_saved=True)
        load.change(load_saved, saved, [source, output, *groups, *headers, *cards, run_total])
    return page


FUTURE_EXPLICIT_MODES = frozenset({"explicit"})
# The implicit family shares the SVD rollout group, while only self-trained adds
# the LoRA-fitting group. Naming these sets lets the visibility handler read like
# the methodology invariant instead of an inline string comparison.
FUTURE_IMPLICIT_MODES = frozenset({"implicit", "self_trained"})
FUTURE_SELF_TRAINED_MODES = frozenset({"self_trained"})


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
    """Clear stale scene output while reserving the Explicit 3D result panel."""
    return gr.update(
        value=empty_splat_html(),
        visible=mode in FUTURE_EXPLICIT_MODES,
    )


def _future_config_visibility(mode: str) -> tuple[dict, ...]:
    """Show only the Configuration groups the selected Future View method reads.

    Each update sets ``visible`` alone, never ``value``, so switching methods
    cannot reset a knob the owner already tuned: a hidden group simply reappears
    with its value intact when its methodology is reselected. The tuple order
    matches the groups registered in ``build_future``.
    """
    return (
        gr.update(visible=True),  # History
        gr.update(visible=True),  # Forecast
        gr.update(visible=True),  # Reconstruction
        gr.update(visible=mode in FUTURE_EXPLICIT_MODES),  # Explicit view
        gr.update(visible=mode in FUTURE_IMPLICIT_MODES),  # Implicit rollout
        gr.update(visible=mode in FUTURE_SELF_TRAINED_MODES),  # Self-trained
    )


def _future_timing_update(trace: dict) -> dict:
    """Show the total once, outside stage cards whose clocks are local."""
    events = trace.get("events", [])
    elapsed = float(events[-1].get("elapsed_seconds", 0)) if events else 0
    return gr.update(value=f"**Run total:** {format_stopwatch(elapsed)}")


def _execute_future_stream(source: str, mode: str, request: object, settings):
    """Yield each persisted Future View stage so Gradio renders real progress."""
    if not source: raise gr.Error("Choose a stationary-view video first.")
    # Create the trace before reading so the source stage has a genuine start
    # boundary rather than appearing as an instantaneous completed event.
    trace = _new_feature_trace("future", mode, {**request.__dict__, "source_name": Path(source).name})
    active_stage = "Sample input tail"
    active_stage_id = "source_tail"
    try:
        trace.add(StageEvent(active_stage, "running", "Sampling the final stationary-camera history window.", stage_id=active_stage_id))
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
                                      "input_fps": fps, "input_segment": "tail end of source video"},
                             stage_id=active_stage_id))
        yield None, trace.record
        result = None
        for item in future.run(frames, fps, request, trace.artifacts, settings, source):
            if isinstance(item, StageEvent):
                active_stage = item.stage
                active_stage_id = item.stage_id or active_stage
                trace.add(item); yield None, trace.record
            else: result = item
        if result is None: raise RuntimeError("The Future View adapter returned no result.")
        # The replayable trace is written after the adapter finishes, so wrap the
        # real write in its declared terminal row. A feature that does not declare
        # ``write_trace`` would simply have this id ignored by the renderer.
        active_stage = "Write run trace"
        active_stage_id = "write_trace"
        trace.add(StageEvent(active_stage, "running", "Writing the replayable run trace.", stage_id=active_stage_id))
        yield None, trace.record
        trace.finish(result)
        trace.add(StageEvent(active_stage, "complete", "Wrote the replayable run trace for this forecast.", stage_id=active_stage_id))
        yield result, trace.record
    except Exception as error:
        # Replay records the stage that failed and a generic terminal event. The
        # former keeps the visible configured stage from being left running, but
        # only when that stage has not already terminated: a failure after the
        # active stage already completed (e.g. saving the clip) must not emit a
        # second terminal for it.
        if not _stage_has_terminal(trace, active_stage_id):
            trace.add(StageEvent(active_stage, "error", str(error), stage_id=active_stage_id))
        trace.add(StageEvent("Run failed", "error", str(error)))
        yield None, trace.record
        raise gr.Error(str(error)) from error
def _future_number(name, raw, integer):
    """Parse one editable knob at the boundary, or fail loudly under its field name.

    ``bool`` is refused before any numeric handling because ``True == 1`` would
    otherwise let a checkbox masquerade as a number. A numeric string is parsed so
    a value Gradio hands back as text still works, while ``"abc"`` names the field
    instead of raising a bare conversion error. Non-finite floats are refused so a
    NaN (whose comparisons are always false) cannot slip past the range check, and
    an integer knob demands a whole value rather than silently truncating ``1.9``.
    """
    if isinstance(raw, bool):
        raise ValueError(f"Future View {name} must be a number (got {raw!r}).")
    if isinstance(raw, str):
        try:
            raw = float(raw)
        except ValueError:
            raise ValueError(f"Future View {name} must be a number (got {raw!r}).") from None
    if not isinstance(raw, (int, float)):
        raise ValueError(f"Future View {name} must be a number (got {raw!r}).")
    if isinstance(raw, float) and not math.isfinite(raw):
        raise ValueError(f"Future View {name} must be a finite number (got {raw!r}).")
    if integer:
        if raw != int(raw):
            raise ValueError(f"Future View {name} must be a whole number (got {raw!r}).")
        return int(raw)
    return float(raw)


def _future_effective_settings(base, *values):
    """Validate every editable value and return one immutable per-run snapshot.

    Values arrive positionally in ``_EDITABLE_FUTURE_FIELDS`` order. Each is parsed
    to the base field's own type at the boundary (so an integer knob cannot keep
    Gradio's float representation), then range-checked by name, so a malformed or
    out-of-range control fails loudly before any model work starts.
    """
    if len(values) != len(_EDITABLE_FUTURE_FIELDS):
        raise ValueError(
            f"Future View configuration needs {len(_EDITABLE_FUTURE_FIELDS)} values "
            f"in {_EDITABLE_FUTURE_FIELDS} order (got {len(values)})."
        )
    overrides = {}
    for name, raw in zip(_EDITABLE_FUTURE_FIELDS, values):
        if raw is None:
            raise ValueError(f"Future View {name} must not be empty.")
        base_value = getattr(base, name)
        is_integer_field = isinstance(base_value, int) and not isinstance(base_value, bool)
        value = _future_number(name, raw, is_integer_field)
        low, high = _FUTURE_FIELD_RANGES[name]
        _require_future_range(name, value, low, high)
        overrides[name] = value
    return replace(base, **overrides)


def _require_future_range(name, value, low, high):
    """Reject an out-of-range field, naming it so the owner can find the control."""
    if not low <= value <= high:
        raise ValueError(f"Future View {name} must be between {low:g} and {high:g} (got {value!r}).")


# Every Future View knob a run may override. The order is the order the
# Configuration controls are built in and therefore the order ``handle`` receives
# their values; ``build_future`` and its tests both rely on this single list.
_EDITABLE_FUTURE_FIELDS = (
    "history_seconds", "history_fps", "observed_splat_timeline_fps",
    "output_seconds", "output_fps", "forecast_splat_timeline_fps",
    "chunk_frames", "motion_fit_frames",
    "render_yaw", "render_shift", "render_fov", "source_fov_degrees",
    "max_side", "training_max_side",
    "training_lora_steps", "training_lora_rank", "training_max_pairs",
    "svd_decode_chunk_size",
)

# Inclusive valid range per override. Kept beside the field order so a new knob
# cannot be added without also deciding what values the workers can honor.
_FUTURE_FIELD_RANGES = {
    "history_seconds": (0.5, 60), "history_fps": (1, 60),
    "observed_splat_timeline_fps": (1, 60),
    "output_seconds": (1, 120), "output_fps": (1, 30),
    "forecast_splat_timeline_fps": (1, 60),
    "chunk_frames": (1, 64), "motion_fit_frames": (1, 64),
    "render_yaw": (-180.0, 180.0), "render_shift": (-1.0, 1.0),
    "render_fov": (35.0, 110.0), "source_fov_degrees": (35.0, 110.0),
    "max_side": (64, 2160), "training_max_side": (64, 2160),
    "training_lora_steps": (1, 10000), "training_lora_rank": (1, 128),
    "training_max_pairs": (1, 1024),
    "svd_decode_chunk_size": (1, 8),
}


def future_configuration_rows(settings) -> list[list[str]]:
    """Return the effective, read-only Future View defaults for the UI."""
    return [
        ["Implicit model", settings.model_id],
        ["Observed history", f"{settings.history_seconds:g}s at {settings.history_fps} fps"],
        ["Generated future", f"{settings.output_seconds}s at {settings.output_fps} fps ({settings.output_frames} frames)"],
        ["SVD rollout chunk", f"{settings.chunk_frames} frames · decode batch {settings.svd_decode_chunk_size}"],
        ["Explicit motion fit", f"{settings.motion_fit_frames} observed frames"],
        ["Explicit render pose", f"yaw {settings.render_yaw:g}° · shift {settings.render_shift:g} · fov {settings.render_fov:g}° · source fov {settings.source_fov_degrees:g}°"],
        ["Self-trained LoRA", f"{settings.training_lora_steps} steps · rank {settings.training_lora_rank} · {settings.training_max_pairs} pairs at {settings.training_max_side}px"],
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
    with gr.Blocks(css=APP_CSS, js=SAVED_RUN_PLACEHOLDER_JS, analytics_enabled=False) as page:
        gr.Markdown("# Future View")
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("## 1. Source " + help_icon("Uses only the final stationary-camera history window. Shorter videos use all available history.", "Observed-history limits"))
                source, sample, saved, load = source_and_saved_controls("Observed camera history", list_runs(ASSETS, "future"), "ASSETS/future/runs/", _sample_videos("future"), "evow-future-saved-runs")
                gr.Markdown("## 2. Methodology " + help_icon("Choose Explicit 3D, the base Stable Video Diffusion prior, or self-trained Stable Video Diffusion. The self-trained method learns only from pre-holdout source windows before forecasting the final tail.", "Methodology"))
                mode = gr.Radio([("Explicit 3D", "explicit"), ("Implicit 3D [prior]", "implicit"), ("Implicit 3D [self-trained]", "self_trained")], value="explicit", show_label=False, elem_classes="evow-method-choice")
                with gr.Accordion("Configuration", elem_classes="evow-config-panel", open=False):
                    # One label/value row per effective setting keeps the full
                    # configuration scannable and makes its ⓘ explanation local.
                    # Rows come from the shared app_core builder, so the mode
                    # visibility columns below add no extra vertical padding.
                    with gr.Column(visible=True) as history_group:
                        history_seconds = flat_config_number("History (s)", settings.history_seconds, minimum=0.5, maximum=60, step=0.5, detail="Seconds of source-tail history sampled before the forecast.")
                        history_fps = flat_config_number("History FPS", settings.history_fps, minimum=1, maximum=60, detail="Sampling rate for the observed history sent into the selected forecasting method.")
                    with gr.Column(visible=True) as forecast_group:
                        output_seconds = flat_config_number("Output (s)", settings.output_seconds, minimum=1, maximum=120, detail="Duration of the generated future scenario; it is never observed camera footage.")
                        output_fps = flat_config_number("Output FPS", settings.output_fps, minimum=1, maximum=30, detail="Frame rate used when encoding the generated forecast video.")
                    with gr.Column(visible=True) as reconstruction_group:
                        max_side = flat_config_number("Max side", settings.max_side, minimum=64, maximum=2160, detail="Longest processing edge for observed and generated frames in both methods.")
                    with gr.Column(visible=True) as explicit_view_group:
                        observed_fps = flat_config_number("Observed FPS", settings.observed_splat_timeline_fps, minimum=1, maximum=60, detail="Timeline rate used for observed-history Gaussian splat frames.")
                        forecast_fps = flat_config_number("Forecast FPS", settings.forecast_splat_timeline_fps, minimum=1, maximum=60, detail="Timeline rate used for extrapolated Gaussian splat frames.")
                        motion_fit_frames = flat_config_number("Motion fit", settings.motion_fit_frames, minimum=1, maximum=64, detail="Observed frames used to estimate the constant-velocity motion prior for Explicit 3D.")
                        render_yaw = flat_config_slider("Yaw", settings.render_yaw, -180, 180, detail="Virtual-camera yaw in degrees for the Explicit 3D render.")
                        render_shift = flat_config_slider("Shift", settings.render_shift, -1, 1, step=0.05, detail="Virtual-camera sideways translation in scene units for Explicit 3D.")
                        render_fov = flat_config_slider("FOV", settings.render_fov, 35, 110, detail="Virtual-camera field of view in degrees for Explicit 3D output.")
                        source_fov = flat_config_slider("Source FOV", settings.source_fov_degrees, 35, 110, detail="Assumed fixed-camera field of view used when reconstructing source geometry.")
                    with gr.Column(visible=False) as implicit_rollout_group:
                        chunk_frames = flat_config_number("Chunk frames", settings.chunk_frames, minimum=1, maximum=64, detail="Frames generated per Stable Video Diffusion rollout chunk.")
                        svd_decode_chunk = flat_config_number("SVD chunk", settings.svd_decode_chunk_size, minimum=1, maximum=8, detail="VAE decode batch size within each Stable Video Diffusion rollout chunk.")
                        seed = flat_config_number("Seed", 0, minimum=0, maximum=2147483647, precision=0, detail="Random seed that makes an implicit forecast repeatable with the same model and inputs.")
                    with gr.Column(visible=False) as self_trained_group:
                        training_max_side = flat_config_number("Training side", settings.training_max_side, minimum=64, maximum=2160, detail="Longest edge used when sampling source frames for the self-trained LoRA.")
                        training_lora_steps = flat_config_number("LoRA steps", settings.training_lora_steps, minimum=1, maximum=10000, detail="Optimizer updates for the source-specific Stable Video Diffusion LoRA fit.")
                        training_lora_rank = flat_config_number("LoRA rank", settings.training_lora_rank, minimum=1, maximum=128, detail="Low-rank adapter capacity for the source-specific LoRA.")
                        training_max_pairs = flat_config_number("Training pairs", settings.training_max_pairs, minimum=1, maximum=1024, detail="Pre-holdout temporal training pairs sampled for self-trained forecasting.")
                gr.Markdown("## 3. Forecast")
                button = gr.Button("Forecast", variant="primary")
            with gr.Column(scale=2):
                gr.Markdown("## 4. Output", elem_classes="evow-section-heading")
                output = gr.Video(label="Source View", interactive=False, elem_classes=["evow-media", "evow-output-slot"], show_download_button=True)
                splat_viewer = gr.HTML(value=empty_splat_html(), label="3D View", show_label=True, container=True, padding=False, elem_classes=["evow-splat-viewer"], elem_id="future-splat-viewer-output")
        groups, headers, cards, run_total = _flow("future")
        def handle(video, selected_mode, selected_seed, *values):
            effective = _future_effective_settings(settings, *values)
            request = future.FutureRequest(selected_mode, int(selected_seed))
            for result, trace in _execute_future_stream(video, selected_mode, request, effective):
                viewer = _future_viewer_update(result.metadata, result.primary.parent.parent) if result and selected_mode in FUTURE_EXPLICIT_MODES else gr.update(visible=False) if result else gr.update()
                yield str(result.primary) if result else None, viewer, *_updates("future", trace)
        def load_saved(run_id):
            loaded_source, primary, _rows, trace = _load_saved("future", run_id)
            result = trace.get("result") or {}
            viewer = _future_viewer_update(result.get("metadata", {}), ASSETS / "future" / "runs" / run_id) if trace.get("mode") in FUTURE_EXPLICIT_MODES else gr.update(visible=False)
            return loaded_source, primary, viewer, *_updates("future", trace, is_saved=True)
        mode.change(lambda selected: _method_flow_change("future", selected), mode, [*headers, *cards, run_total])
        # Order must mirror ``_EDITABLE_FUTURE_FIELDS``: ``handle`` receives these
        # values positionally and hands them to ``_future_effective_settings``.
        config_inputs = [
            history_seconds, history_fps, observed_fps,
            output_seconds, output_fps, forecast_fps,
            chunk_frames, motion_fit_frames,
            render_yaw, render_shift, render_fov, source_fov,
            max_side, training_max_side,
            training_lora_steps, training_lora_rank, training_max_pairs,
            svd_decode_chunk,
        ]
        button.click(handle, [source, mode, seed, *config_inputs], [output, splat_viewer, *groups, *headers, *cards, run_total], concurrency_limit=1, show_progress="hidden")
        mode.change(_future_method_change, mode, splat_viewer)
        # Group order must match ``_future_config_visibility``'s tuple.
        mode.change(
            _future_config_visibility, mode,
            [history_group, forecast_group, reconstruction_group, explicit_view_group, implicit_rollout_group, self_trained_group],
        )
        sample.click(_sample, None, source)
        load.change(load_saved, saved, [source, output, splat_viewer, *groups, *headers, *cards, run_total])
    return page


def _text_number(name, raw, integer, feature: str = "Text Query"):
    """Parse one editable Text Query/Manipulation knob at the boundary, or fail under its field name.

    Mirrors ``_future_number`` deliberately: ``bool`` is refused before any numeric
    handling because ``True == 1`` would otherwise let a checkbox masquerade as a
    number; a numeric string is parsed so a value Gradio hands back as text still
    works while ``"abc"`` names the field instead of raising a bare conversion error;
    non-finite floats are refused so a NaN (whose comparisons are always false) cannot
    slip past the range check; and an integer knob demands a whole value rather than
    silently truncating ``1.9``. ``feature`` labels the owning page so a shared
    validation message never misnames a Text Manipulation error as Text Query.
    """
    if isinstance(raw, bool):
        raise ValueError(f"{feature} {name} must be a number (got {raw!r}).")
    if isinstance(raw, str):
        try:
            raw = float(raw)
        except ValueError:
            raise ValueError(f"{feature} {name} must be a number (got {raw!r}).") from None
    if not isinstance(raw, (int, float)):
        raise ValueError(f"{feature} {name} must be a number (got {raw!r}).")
    if isinstance(raw, float) and not math.isfinite(raw):
        raise ValueError(f"{feature} {name} must be a finite number (got {raw!r}).")
    if integer:
        if raw != int(raw):
            raise ValueError(f"{feature} {name} must be a whole number (got {raw!r}).")
        return int(raw)
    return float(raw)


def _require_text_range(name, value, low, high, feature: str = "Text Query"):
    """Reject an out-of-range field, naming its page and control so the owner can find it."""
    if not low <= value <= high:
        raise ValueError(f"{feature} {name} must be between {low:g} and {high:g} (got {value!r}).")


# Every Text Query knob a run may override. The order is the order the
# Configuration controls are built in and therefore the order ``handle`` receives
# their values; ``build_selection`` and its tests both rely on this single list.
_EDITABLE_TEXT_FIELDS = (
    "query_max_source_seconds", "query_sample_fps", "query_clip_seconds",
    "query_result_fps", "query_max_results", "query_explicit_candidate_pool",
    "grounding_threshold", "low_specificity_coverage",
    "source_fov_degrees", "scene_cache_version",
)

# Inclusive valid range per override. The single string knob (the scene cache
# epoch) is validated separately. Kept beside the field order so a new knob cannot
# be added without also deciding what values the pipeline can honor.
_TEXT_FIELD_RANGES = {
    "query_max_source_seconds": (1, 3600), "query_sample_fps": (1, 30),
    "query_clip_seconds": (1, 30), "query_result_fps": (1, 30),
    "query_max_results": (1, 50), "query_explicit_candidate_pool": (1, 64),
    "grounding_threshold": (0.0, 1.0), "low_specificity_coverage": (0.0, 1.0),
    "source_fov_degrees": (35.0, 110.0),
}


def _text_effective_settings(base, *values):
    """Validate every editable value and return one immutable per-run snapshot.

    Values arrive positionally in ``_EDITABLE_TEXT_FIELDS`` order. Each numeric
    value is parsed to the base field's own type at the boundary (so an integer knob
    cannot keep Gradio's float representation), then range-checked by name, so a
    malformed or out-of-range control fails loudly before any model work starts. The
    single string knob (the scene cache epoch) must be a non-empty string.
    """
    if len(values) != len(_EDITABLE_TEXT_FIELDS):
        raise ValueError(
            f"Text Query configuration needs {len(_EDITABLE_TEXT_FIELDS)} values "
            f"in {_EDITABLE_TEXT_FIELDS} order (got {len(values)})."
        )
    overrides = {}
    for name, raw in zip(_EDITABLE_TEXT_FIELDS, values):
        if raw is None:
            raise ValueError(f"Text Query {name} must not be empty.")
        base_value = getattr(base, name)
        if isinstance(base_value, str):
            if not isinstance(raw, str) or not raw.strip():
                raise ValueError(f"Text Query {name} must be a non-empty string (got {raw!r}).")
            overrides[name] = raw.strip()
            continue
        is_integer_field = isinstance(base_value, int) and not isinstance(base_value, bool)
        value = _text_number(name, raw, is_integer_field, feature="Text Query")
        low, high = _TEXT_FIELD_RANGES[name]
        _require_text_range(name, value, low, high, feature="Text Query")
        overrides[name] = value
    return replace(base, **overrides)


def build_selection() -> gr.Blocks:
    """Build Text Query with native paused players for non-cropped highlights."""
    from .text_settings import load_text_settings
    settings = load_text_settings()
    with gr.Blocks(css=APP_CSS, js=SAVED_RUN_PLACEHOLDER_JS, analytics_enabled=False) as page:
        gr.Markdown("# Text Query")
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("## 1. Source " + help_icon("Searches a stationary-view upload for up to five non-overlapping four-second clips. Explicit 3D is an estimate from one fixed camera.", "Source limits"))
                source, sample, saved, load = source_and_saved_controls("Source video (up to 5 minutes)", list_runs(ASSETS, "selection"), "ASSETS/selection/runs/", _sample_videos("selection"), "evow-selection-saved-runs")
                gr.Markdown("## 2. Query")
                query = gr.Textbox(label="Query", placeholder="dark clouds above moving tree branches", container=False)
                gr.Markdown("## 3. Methodology " + help_icon("Choose Explicit 3D to lift tracked masks through Video Depth Anything + gsplat, or Implicit 3D for direct video masks. Exact active models appear in the internal-flow stages.", "Methodology"))
                mode = gr.Radio([("Explicit 3D", "explicit"), ("Implicit 3D", "implicit")], value="explicit", show_label=False, elem_classes="evow-method-choice")
                with gr.Accordion("Configuration", elem_classes="evow-config-panel", open=False):
                    # Every effective setting has one visible label/value row;
                    # its ⓘ popover explains the model or pipeline consequence.
                    # The mode-toggle columns remain so methods can be gated, but
                    # the shared row builder keeps their spacing identical.
                    with gr.Column(visible=True) as source_group:
                        max_source = flat_config_number("Max source (s)", settings.query_max_source_seconds, minimum=1, maximum=3600, detail="Maximum source duration searched before SigLIP retrieval sampling stops.")
                        sample_fps = flat_config_number("Sample FPS", settings.query_sample_fps, minimum=1, maximum=30, detail="Rate at which source frames are sampled for SigLIP text/video retrieval.")
                        clip_seconds = flat_config_number("Clip (s)", settings.query_clip_seconds, minimum=1, maximum=30, detail="Duration represented by each candidate interval before grounding and reranking.")
                        trim_ui = flat_trim_range("Trim video", "selection-trim-range-value", settings.query_max_source_seconds, detail="Choose the original-video interval processed by this non-destructive query run.")
                        trim_range = gr.Textbox(value=f"0,{settings.query_max_source_seconds:.3f}", visible=False, elem_id="selection-trim-range-value")
                    with gr.Column(visible=True) as results_group:
                        result_fps = flat_config_number("Result FPS", settings.query_result_fps, minimum=1, maximum=30, detail="Decode and export rate for each returned source and highlighted clip.")
                        max_results = flat_config_number("Max results", settings.query_max_results, minimum=1, maximum=50, detail="Maximum final matching clips returned to the result gallery.")
                        candidate_pool = flat_config_number("Candidate pool", settings.query_explicit_candidate_pool, minimum=1, maximum=64, detail="Non-overlapping temporal candidates grounded and reranked before Explicit 3D selection.")
                    with gr.Column(visible=True) as matching_group:
                        grounding = flat_config_slider("Grounding", settings.grounding_threshold, 0, 1, step=0.05, detail="Grounding DINO confidence required to accept a text-referred region instead of rejecting it.")
                    with gr.Column(visible=True) as explicit_scene_group:
                        coverage = flat_config_slider("Coverage", settings.low_specificity_coverage, 0, 1, step=0.05, detail="Projected-mask coverage at which a grounded region is flagged as low-specificity.")
                        source_fov = flat_config_slider("Source FOV", settings.source_fov_degrees, 35, 110, detail="Assumed fixed-camera field of view used to lift masks through the Gaussian scene.")
                        cache_version = flat_config_textbox("Cache version", settings.scene_cache_version, detail="Scene-cache epoch; changing it deliberately invalidates cached Gaussian reconstructions.")
                gr.Markdown("## 4. Search")
                button = gr.Button("Search", variant="primary")
            with gr.Column(scale=2):
                gr.Markdown("## 5. Highlighted matching clips " + help_icon("Explicit mode shows selected 4D Gaussian structure in solid magenta; it is an orbitable structure viewer, not a projected video mask.", "3D structure viewer"), elem_classes="evow-section-heading")
                # Both methods begin with one empty selector. A completed search
                # fills it with results and shows just the chosen source-view clip.
                semantic_gallery = gr.HTML(value=_selection_player_html([]), visible=False, container=False, padding=False, elem_classes="evow-video-list-output")
                explicit_match = gr.Dropdown(label="Explicit match", choices=[], visible=True)
                explicit_video = gr.Video(label="Selected match", interactive=False, visible=False, elem_classes="evow-media")
                explicit_pairs = gr.State([])
                splat_viewer = gr.HTML(value=empty_splat_html(), label="3D View", show_label=True, container=True, padding=False, elem_classes=["evow-splat-viewer"], elem_id="selection-splat-viewer", visible=False)
        groups, headers, cards, run_total = _flow("selection")
        def handle(video, text, selected_mode, *values):
            # Direct Python callers may omit trim values; browser wiring always
            # supplies them before the configuration snapshot.
            if len(values) == len(_EDITABLE_TEXT_FIELDS):
                trim_start_seconds, trim_end_seconds, values = 0.0, None, values
            else:
                trim_range_value, *values = values
                trim_start_seconds, trim_end_seconds = _parse_trim_range(trim_range_value)
            # Parse every editable knob before any work so an invalid control fails
            # under its field name without half-starting a run.
            effective = _text_effective_settings(settings, *values)
            yield gr.update(value=_selection_player_html([]), visible=False), gr.update(choices=[], value=None, visible=True), gr.update(value=None, visible=False), [], gr.update(value=empty_splat_html(), visible=False), *_updates("selection", {"mode": selected_mode, "events": []})
            request = selection.SelectionRequest(selected_mode, text, float(trim_start_seconds), None if trim_end_seconds is None else float(trim_end_seconds))
            # The adapter seam is generic, so bind the effective settings into the
            # selection run without widening ``SelectionRequest`` (which the trace
            # persists as JSON and must stay serializable).
            adapter = partial(selection.run, settings=effective)
            for result, trace in _execute_source_adapter_stream("selection", selected_mode, video, request, adapter):
                pairs = _selection_pair_records(result.clips, explicit=selected_mode in SELECTION_EXPLICIT_MODES) if result else []
                if pairs:
                    match = gr.update(choices=[(pair["label"], str(index)) for index, pair in enumerate(pairs)], value="0", visible=True)
                    paired_video, viewer = _selection_pair_update("0", pairs)
                else:
                    match = gr.update(choices=[], value=None, visible=True)
                    paired_video = gr.update(value=None, visible=False)
                    viewer = gr.update(value=empty_splat_html(), visible=False)
                yield gr.update(value=_selection_player_html([]), visible=False), match, paired_video, pairs, viewer, *_updates("selection", trace)
        def load_saved(run_id):
            loaded_source, _primary, _rows, trace = _load_saved("selection", run_id); clips = ((trace.get("result") or {}).get("clips") or []); run_dir = ASSETS / "selection" / "runs" / run_id
            pairs = _selection_pair_records(clips, run_dir, explicit=trace.get("mode") in SELECTION_EXPLICIT_MODES)
            match = gr.update(choices=[(pair["label"], str(index)) for index, pair in enumerate(pairs)], value="0" if pairs else None, visible=True)
            paired_video, viewer = _selection_pair_update("0", pairs) if pairs else (gr.update(value=None, visible=False), gr.update(value=empty_splat_html(), visible=False))
            return loaded_source, gr.update(value=_selection_player_html([]), visible=False), match, paired_video, pairs, viewer, *_updates("selection", trace, is_saved=True)
        mode.change(_selection_method_change, mode, splat_viewer)
        # Group order must match ``_selection_config_visibility``'s tuple.
        mode.change(
            _selection_config_visibility, mode,
            [source_group, results_group, matching_group, explicit_scene_group],
        )
        mode.change(lambda selected: _method_flow_change("selection", selected), mode, [*headers, *cards, run_total])
        # Order must mirror ``_EDITABLE_TEXT_FIELDS``: ``handle`` receives these
        # values positionally and hands them to ``_text_effective_settings``.
        config_inputs = [
            max_source, sample_fps, clip_seconds,
            result_fps, max_results, candidate_pool,
            grounding, coverage,
            source_fov, cache_version,
        ]
        button.click(handle, [source, query, mode, trim_range, *config_inputs], [semantic_gallery, explicit_match, explicit_video, explicit_pairs, splat_viewer, *groups, *headers, *cards, run_total], concurrency_limit=1, show_progress="hidden")
        explicit_match.change(_selection_pair_update, [explicit_match, explicit_pairs], [explicit_video, splat_viewer], queue=False)
        source.change(lambda video: _trim_range_bounds(video, settings.query_max_source_seconds, "selection-trim-range-value"), source, [trim_ui, trim_range], queue=False)
        sample.click(_sample, None, source)
        load.change(load_saved, saved, [source, semantic_gallery, explicit_match, explicit_video, explicit_pairs, splat_viewer, *groups, *headers, *cards, run_total])
    return page


def _editing_result_caption(run_id: str, request: editing.EditingRequest, primary: Path) -> str:
    """Make the result's unique run and text conditioning visible beside playback."""
    method = "Explicit 3D" if request.mode == "explicit" else "Implicit 3D"
    return (f"**Generated result** · run `{escape(run_id)}` · {method} · seed `{request.seed}` · "
            f"prompt: “{escape(request.prompt.strip())}” · artifact `{escape(primary.name)}`")


# Every Text Manipulation knob a run may override, in the exact order the
# Configuration controls are built and therefore the order ``handle`` receives
# their values. ``build_editing`` and its tests both rely on this single list.
# ``seed`` stays on the request instead because the trace persists the request
# as JSON and the seed belongs to that serializable identity, not the settings.
_EDITABLE_EDITING_FIELDS = (
    "episode_fps", "episode_max_frames",
    "edit_max_side", "edit_max_height", "vace_steps",
    "vace_guidance_scale", "edit_mask_expand_px",
    "grounding_threshold",
    "source_fov_degrees", "scene_cache_version",
)

# Inclusive valid range per override. The single string knob (the scene cache
# epoch) is validated separately. ``episode_max_frames`` tops out at 81 because
# Wan VACE consumes 1 + 4k frames and 81 is the largest owned value; a smaller
# count is trimmed down to the previous valid 1 + 4k by ``vace_frame_count``.
_EDITING_FIELD_RANGES = {
    "episode_fps": (1, 60), "episode_max_frames": (5, 81),
    "edit_max_side": (64, 2160), "edit_max_height": (64, 2160),
    "vace_steps": (1, 200), "vace_guidance_scale": (0.0, 20.0),
    "edit_mask_expand_px": (0, 64), "grounding_threshold": (0.0, 1.0),
    "source_fov_degrees": (35.0, 110.0),
}


def _editing_effective_settings(base, *values):
    """Validate every editable value and return one immutable per-run snapshot.

    Values arrive positionally in ``_EDITABLE_EDITING_FIELDS`` order. Numeric
    parsing reuses Text Query's boundary helpers so the same bool/non-numeric/
    non-finite/fractional-integer rules apply; the single string knob (the scene
    cache epoch) must be a non-empty string. A malformed or out-of-range control
    fails loudly under its field name before any model work starts.
    """
    if len(values) != len(_EDITABLE_EDITING_FIELDS):
        raise ValueError(
            f"Text Manipulation configuration needs {len(_EDITABLE_EDITING_FIELDS)} values "
            f"in {_EDITABLE_EDITING_FIELDS} order (got {len(values)})."
        )
    overrides = {}
    for name, raw in zip(_EDITABLE_EDITING_FIELDS, values):
        if raw is None:
            raise ValueError(f"Text Manipulation {name} must not be empty.")
        base_value = getattr(base, name)
        if isinstance(base_value, str):
            if not isinstance(raw, str) or not raw.strip():
                raise ValueError(f"Text Manipulation {name} must be a non-empty string (got {raw!r}).")
            overrides[name] = raw.strip()
            continue
        is_integer_field = isinstance(base_value, int) and not isinstance(base_value, bool)
        value = _text_number(name, raw, is_integer_field, feature="Text Manipulation")
        low, high = _EDITING_FIELD_RANGES[name]
        _require_text_range(name, value, low, high, feature="Text Manipulation")
        overrides[name] = value
    return replace(base, **overrides)


def build_editing() -> gr.Blocks:
    """Build Text Manipulation as a single-pass text-conditioned edit."""
    from .text_settings import load_text_settings
    settings = load_text_settings()
    with gr.Blocks(css=APP_CSS, js=SAVED_RUN_PLACEHOLDER_JS, analytics_enabled=False) as page:
        gr.Markdown("# Text Manipulation")
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("## 1. Source " + help_icon("Upload one episode up to 6.75 seconds, or load a raw Text Query result. The yellow query overlay is never used as edit input; explicit geometry is estimated.", "Editing source limits"))
                source, sample, saved, load = source_and_saved_controls(
                    "Short source episode", list_runs(ASSETS, "editing"), "ASSETS/editing/runs/",
                    _sample_videos("editing"), "evow-editing-saved-runs",
                )
                gr.Markdown("## 2. Instruction")
                prompt = gr.Textbox(label="Instruction", placeholder="flood the forest while preserving the fixed camera view", container=False)
                gr.Markdown("## 3. Methodology " + help_icon("Choose Explicit 3D to project scope through Video Depth Anything + gsplat, or Implicit 3D for direct VACE conditioning. Exact active models appear in the internal-flow stages.", "Methodology"))
                mode = gr.Radio(
                    [("Explicit 3D", "explicit"),
                     ("Implicit 3D", "implicit")],
                    value="explicit", show_label=False, elem_classes="evow-method-choice",
                )
                with gr.Accordion("Configuration", elem_classes="evow-config-panel", open=False):
                    # Every effective setting has one visible label/value row;
                    # its ⓘ popover explains the model or pipeline consequence.
                    # The mode-toggle columns remain so methods can be gated, but
                    # the shared row builder keeps their spacing identical.
                    with gr.Column(visible=True) as episode_group:
                        episode_fps = flat_config_number("Episode FPS", settings.episode_fps, minimum=1, maximum=60, detail="Sampling rate for the bounded source episode supplied to Wan VACE or Explicit 3D reconstruction.")
                        episode_frames = flat_config_number("Episode frames", settings.episode_max_frames, minimum=5, maximum=81, detail="Maximum source frames; Wan VACE uses a 1 + 4k frame shape and trims to the previous valid count.")
                        trim_ui = flat_trim_range("Trim video", "editing-trim-range-value", settings.episode_seconds, detail="Choose the original-video interval processed by this non-destructive editing run.")
                        trim_range = gr.Textbox(value=f"0,{settings.episode_seconds:.3f}", visible=False, elem_id="editing-trim-range-value")
                    with gr.Column(visible=True) as edit_group:
                        max_side = flat_config_number("Max side", settings.edit_max_side, minimum=64, maximum=2160, detail="Maximum generated edit width/long edge processed by Wan VACE.")
                        max_height = flat_config_number("Max height", settings.edit_max_height, minimum=64, maximum=2160, detail="Maximum generated edit height processed by Wan VACE.")
                        vace_steps = flat_config_number("VACE steps", settings.vace_steps, minimum=1, maximum=200, detail="Wan VACE denoising iterations; more steps trade speed for refinement.")
                        guidance = flat_config_slider("Guidance", settings.vace_guidance_scale, 0, 20, step=0.5, detail="Wan VACE classifier-free guidance strength for adherence to the text instruction.")
                        mask_expand = flat_config_number("Mask expand", settings.edit_mask_expand_px, minimum=0, maximum=64, detail="Pixels used to dilate the resolved mask before conditioning, softening edit boundaries.")
                    with gr.Column(visible=True) as matching_group:
                        grounding = flat_config_slider("Grounding", settings.grounding_threshold, 0, 1, step=0.05, detail="Grounding DINO confidence needed to accept the instruction-referred target; otherwise a full-scene mask is used.")
                    with gr.Column(visible=True) as scene_group:
                        source_fov = flat_config_slider("Source FOV", settings.source_fov_degrees, 35, 110, detail="Assumed fixed-camera field of view for projecting masks through Explicit 3D Gaussians.")
                        cache_version = flat_config_textbox("Cache version", settings.scene_cache_version, detail="Scene-cache epoch; changing it invalidates cached dynamic Gaussian scenes.")
                    with gr.Column(visible=True) as run_group:
                        seed = flat_config_number("Seed", 0, minimum=0, maximum=2147483647, precision=0, detail="Random seed that makes the generated edit repeatable with identical inputs and models.")
                gr.Markdown("## 4. Modify")
                button = gr.Button("Modify", variant="primary")
            with gr.Column(scale=2):
                gr.Markdown("## 5. Output", elem_classes="evow-section-heading")
                output = gr.Video(label="Source View", interactive=False, elem_classes=["evow-media", "evow-output-slot"], show_download_button=True)
                result_caption = gr.Markdown("")
                splat_viewer = gr.HTML(value=empty_splat_html(), label="3D View", show_label=True, container=True, padding=False, elem_classes=["evow-splat-viewer"], elem_id="editing-splat-viewer-output", visible=True)
        groups, headers, cards, run_total = _flow("editing")

        def handle(video, instruction, selected_mode, selected_seed, *values):
            # Direct Python callers may omit trim values; browser wiring always
            # supplies them before the configuration snapshot.
            if len(values) == len(_EDITABLE_EDITING_FIELDS):
                trim_start_seconds, trim_end_seconds, values = 0.0, None, values
            else:
                trim_range_value, *values = values
                trim_start_seconds, trim_end_seconds = _parse_trim_range(trim_range_value)
            # Parse every editable knob before any work so an invalid control fails
            # under its field name without half-starting a run.
            effective = _editing_effective_settings(settings, *values)
            if not video:
                raise gr.Error("Choose a stationary-view video first.")
            request = editing.EditingRequest(selected_mode, instruction, int(selected_seed or 0), float(trim_start_seconds), None if trim_end_seconds is None else float(trim_end_seconds))
            trace = _new_feature_trace("editing", selected_mode, {**request.__dict__, "source_name": Path(video).name})
            # The pre-adapter work belongs to the flow's first declared stage, so
            # seed both its label and its stable id; an early failure then repaints
            # the real row instead of an unknown display-name identity.
            first_stage = _current_flow("editing").stages[0]
            active_stage = first_stage.spec.name
            active_stage_id = first_stage.stage_id
            try:
                source_copy = trace.artifacts.file("source.mp4")
                shutil.copy2(video, source_copy)
                trace.record["source_video"] = source_copy.name
                trace._save()
                # Clear the prior result surface before any potentially long scene work.
                yield None, gr.update(value=empty_splat_html(), visible=request.mode in EDITING_EXPLICIT_MODES), "", *_updates("editing", trace.record)
                result = None
                # The generated frame count is not duplicated into result
                # metadata; capture it from the run's own ``write_proposal``
                # stage metric so the publish summary stays truthful.
                generated_frames = None
                # The adapter seam binds the effective settings without widening
                # ``EditingRequest`` (which the trace persists as JSON and must
                # stay serializable); the seed remains part of that identity.
                adapter = partial(editing.run, settings=effective)
                for item in adapter(str(source_copy), request, trace.artifacts):
                    if isinstance(item, StageEvent):
                        active_stage = item.stage
                        active_stage_id = item.stage_id or active_stage
                        if "output_frames" in item.metrics:
                            generated_frames = item.metrics["output_frames"]
                        trace.add(item)
                        yield gr.update(), gr.update(), gr.update(), *_updates("editing", trace.record)
                    else:
                        result = item
                if result is None:
                    raise RuntimeError("Text Manipulation returned no generated result.")
                # The replayable trace is written after the adapter finishes, so
                # wrap the real write in its declared terminal row.
                active_stage = "Publish artifacts"
                active_stage_id = "publish_artifacts"
                trace.add(StageEvent(active_stage, "running", "Writing the replayable run trace.", stage_id=active_stage_id))
                yield gr.update(), gr.update(), gr.update(), *_updates("editing", trace.record)
                trace.finish(result)
                publish_metrics = {
                    "output_frames": generated_frames,
                    "scope": result.metadata.get("scope"),
                    "mode": request.mode,
                }
                trace.add(StageEvent(
                    active_stage, "complete", "Wrote the replayable run trace for this edit.",
                    metrics={key: value for key, value in publish_metrics.items() if value is not None},
                    stage_id=active_stage_id,
                ))
                viewer = _editing_viewer_update(result.metadata, trace.run_dir) if request.mode in EDITING_EXPLICIT_MODES else gr.update(value=empty_splat_html(), visible=False)
                yield str(result.primary), viewer, _editing_result_caption(trace.run_dir.name, request, result.primary), *_updates("editing", trace.record)
            except Exception as error:
                # A failure after the active stage already terminated emits only
                # the generic run-failure event, never a second terminal.
                if not _stage_has_terminal(trace, active_stage_id):
                    trace.add(StageEvent(active_stage, "error", str(error), stage_id=active_stage_id))
                trace.add(StageEvent("Run failed", "error", str(error)))
                yield gr.update(), gr.update(), gr.update(), *_updates("editing", trace.record)
                raise gr.Error(str(error)) from error

        def load_saved(run_id):
            loaded_source, primary, _rows, record = _load_saved("editing", run_id)
            result = record.get("result") or {}
            request_data = record.get("request") or {}
            request = editing.EditingRequest(str(record.get("mode", "")), str(request_data.get("prompt", "")), int(request_data.get("seed", 0)))
            run_dir = ASSETS / "editing" / "runs" / run_id
            viewer = _editing_viewer_update(result.get("metadata", {}), run_dir) if record.get("mode") in EDITING_EXPLICIT_MODES else gr.update(value=empty_splat_html(), visible=False)
            caption = _editing_result_caption(run_id, request, Path(primary)) if primary else ""
            return loaded_source, primary, viewer, caption, *_updates("editing", record, is_saved=True)
        mode.change(lambda selected: _method_flow_change("editing", selected), mode, [*headers, *cards, run_total])

        # Order must mirror ``_EDITABLE_EDITING_FIELDS``: ``handle`` receives these
        # values positionally and hands them to ``_editing_effective_settings``.
        config_inputs = [
            episode_fps, episode_frames,
            max_side, max_height, vace_steps, guidance, mask_expand,
            grounding,
            source_fov, cache_version,
        ]
        button.click(handle, [source, prompt, mode, seed, trim_range, *config_inputs], [output, splat_viewer, result_caption, *groups, *headers, *cards, run_total], concurrency_limit=1, show_progress="hidden")
        source.change(lambda video: _trim_range_bounds(video, settings.episode_seconds, "editing-trim-range-value"), source, [trim_ui, trim_range], queue=False)
        mode.change(_editing_method_change, mode, splat_viewer)
        # Group order must match ``_editing_config_visibility``'s tuple.
        mode.change(
            _editing_config_visibility, mode,
            [episode_group, edit_group, matching_group, scene_group, run_group],
        )
        sample.click(_sample, None, source)
        load.change(load_saved, saved, [source, output, splat_viewer, result_caption, *groups, *headers, *cards, run_total])
    return page
