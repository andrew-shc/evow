"""Lifecycle invariants for the Replay V2 trace format."""

import pytest

from GREENFIELD.replay.trace import RunTrace


def test_trace_rejects_two_simultaneous_serial_stages(tmp_path):
    trace = RunTrace(tmp_path, "explicit", {})
    trace.add("first", "running", "first", stage_id="first")
    with pytest.raises(ValueError, match="still running"):
        trace.add("second", "running", "second", stage_id="second")
    trace.add("first", "complete", "done", stage_id="first")
    trace.add("second", "running", "second", stage_id="second")
    trace.add("second", "complete", "done", stage_id="second")
    assert [event["sequence"] for event in trace.record["events"]] == [1, 2, 3, 4]


def test_trace_allows_declared_concurrent_work(tmp_path):
    trace = RunTrace(tmp_path, "explicit", {})
    trace.add("first", "running", "first", stage_id="first", execution="concurrent")
    trace.add("second", "running", "second", stage_id="second", execution="concurrent")
    assert all(event["execution"] == "concurrent" for event in trace.record["events"])
