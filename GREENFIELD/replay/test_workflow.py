"""Regression coverage for Replay's atomic execution-flow presentation."""

from GREENFIELD.replay.workflow import MAX_FLOW_STAGES, workflow_card_html, workflow_header, workflow_visible


def _trace(mode="explicit", events=None):
    return {"workflow_version": 2, "mode": mode, "run_id": "missing", "events": events or []}


def _event(stage_id, status, elapsed=1.0, **extra):
    return {"stage_id": stage_id, "stage": stage_id, "status": status, "elapsed_seconds": elapsed, "stage_elapsed_seconds": elapsed, **extra}


def test_atomic_rows_start_waiting_and_preallocated_tail_is_hidden():
    trace = _trace()
    label, state = workflow_header(trace, 0)
    assert label.startswith("1. Validate request · Waiting")
    assert state == "evow-stage-waiting"
    assert workflow_visible(trace, 0)
    assert not workflow_visible(trace, MAX_FLOW_STAGES - 1)


def test_one_stage_event_never_changes_another_atomic_row():
    trace = _trace(events=[_event("fit_4d_scene", "running", 4.0)])
    sample_label, sample_state = workflow_header(trace, 1)
    fit_label, fit_state = workflow_header(trace, 8)
    assert "Sample episode · Waiting" in sample_label
    assert sample_state == "evow-stage-waiting"
    assert "Fit 4D scene · Running" in fit_label
    assert fit_state == "evow-stage-running"


def test_terminal_event_immediately_replaces_running_status():
    trace = _trace(events=[
        _event("fit_4d_scene", "running", 2.0),
        _event("fit_4d_scene", "complete", 8.0),
    ])
    label, state = workflow_header(trace, 8)
    assert "Completed" in label
    assert "8.000s" in label
    assert state == "evow-stage-complete"


def test_concurrent_label_requires_explicit_concurrent_execution_mode():
    trace = _trace(events=[_event("fit_4d_scene", "running", execution="concurrent")])
    label, _state = workflow_header(trace, 8)
    assert "Running (concurrent/async)" in label


def test_legacy_trace_stays_raw_instead_of_fabricating_atomic_states():
    trace = {"mode": "explicit", "events": [{"stage": "4D Gaussian fitting", "status": "running"}]}
    label, state = workflow_header(trace, 0)
    assert "Legacy trace" in label
    assert state == "evow-stage-waiting"
    assert "4D Gaussian fitting" in workflow_card_html(trace, 0)
    assert not workflow_visible(trace, 1)


def test_expanded_atomic_card_exposes_code_and_raw_event_evidence():
    trace = _trace(events=[_event("sample_episode", "complete", 3.0, metrics={"frames": 13})])
    card = workflow_card_html(trace, 1)
    assert "GREENFIELD/replay/clip.py" in card
    assert "raw event JSON" in card
    assert "&quot;stage_id&quot;: &quot;sample_episode&quot;" in card
    assert "frames" in card
