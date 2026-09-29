"""Shared super-stage lifecycle presentation for every dashboard workflow."""

from dataclasses import dataclass

import gradio as gr

from .timing import format_stopwatch


@dataclass(frozen=True)
class SuperStage:
    """A stable named parent containing ordered child stage IDs.

    The parent owns ``stage_ids`` (never positional indexes): saved runs resolve
    against the flow version they were written with, and indexes would silently
    regroup a run whose profile changed underneath it.
    """

    label: str
    stage_ids: tuple[str, ...]


def available_super_stages(groups: tuple[SuperStage, ...], known_ids: set[str] | tuple[str, ...]) -> tuple[SuperStage, ...]:
    """Drop unknown stage IDs and empty parents while retaining declaration order."""
    known = set(known_ids)
    available = tuple(
        SuperStage(group.label, tuple(stage_id for stage_id in group.stage_ids if stage_id in known))
        for group in groups
    )
    return tuple(group for group in available if group.stage_ids)


def rollup_stage_states(states: list[tuple[str, float]]) -> tuple[str, float]:
    """Compute one honest lifecycle state and duration from atomic-stage states."""
    statuses = [status for status, _elapsed in states]
    elapsed = sum(value for _status, value in states)
    if any(status == "error" for status in statuses):
        return "error", elapsed
    if any(status == "running" for status in statuses):
        return "running", elapsed
    if statuses and all(status == "skipped" for status in statuses):
        return "skipped", elapsed
    if statuses and all(status in {"complete", "skipped"} for status in statuses):
        return "complete", elapsed
    return "waiting", elapsed


def super_stage_update(label: str, status: str, elapsed: float, *, visible: bool = True) -> dict:
    """Return the one shared Gradio update and CSS contract for a parent stage."""
    display = {"complete": "Completed", "running": "Running", "error": "Failed", "waiting": "Waiting", "skipped": "Skipped"}[status]
    elapsed_label = "—" if status in {"waiting", "skipped"} else format_stopwatch(elapsed)
    css_status = "failed" if status == "error" else status
    return gr.update(
        label=f"{label} · {display} · {elapsed_label}",
        elem_classes=["evow-flow-group", f"evow-flow-group-{css_status}"],
        visible=visible,
    )
