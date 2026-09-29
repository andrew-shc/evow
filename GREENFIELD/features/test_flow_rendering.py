"""Behavioral coverage for id-driven flow rendering in ``pages``.

These tests exercise the boundary between a persisted trace and the stage rows
``pages._updates`` repaints. They register temporary flows in ``pages._FLOWS``
so a single-group, two-stage profile keeps the update indices unambiguous: one
group, then one header per stage, then one card per stage, then the run total.
"""

from pathlib import Path

from GREENFIELD.app_core.contracts import StageSpec
from GREENFIELD.app_core.flow import CURRENT_FLOW_VERSION, LEGACY_FLOW_VERSION, FeatureFlow, FlowStage
from GREENFIELD.app_core.super_stages import SuperStage
from GREENFIELD.app_core.trace import load_trace
from GREENFIELD.features import pages


def _two_stage_flow(
    *stages: FlowStage,
    legacy_stage_ids: tuple[str, ...] = (),
    legacy_profiles: dict[int, tuple[str, ...]] | None = None,
) -> FeatureFlow:
    return FeatureFlow(
        "future",
        CURRENT_FLOW_VERSION,
        stages,
        (SuperStage("All", tuple(stage.stage_id for stage in stages)),),
        legacy_stage_ids=legacy_stage_ids,
        legacy_profiles=legacy_profiles or {},
    )


def _header(updates: tuple[dict, ...], count: int, index: int) -> dict:
    """Return the header update for stage ``index`` in a one-group flow."""
    return updates[1 + index]


def _card(updates: tuple[dict, ...], count: int, index: int) -> dict:
    """Return the card update for stage ``index`` in a one-group flow."""
    return updates[1 + count + index]


def test_new_feature_trace_persists_current_flow_identity(monkeypatch, tmp_path: Path) -> None:
    """A newly started run carries its current flow version and readable profile id."""
    monkeypatch.setattr(pages, "ASSETS", tmp_path / "ASSETS")

    trace = pages._new_feature_trace("future", "self_trained", {"seed": 1})
    record = load_trace(trace.run_dir)

    assert record["flow_version"] == CURRENT_FLOW_VERSION
    assert record["profile_id"] == f"future.self_trained.v{CURRENT_FLOW_VERSION}"
    assert record["profile_id"]


def test_updates_match_events_by_stage_id_even_when_the_display_name_differs(monkeypatch) -> None:
    """A persisted stage_id drives the row, not the label that happened to be shown."""
    flow = _two_stage_flow(
        FlowStage("source_tail", StageSpec("Renamed display", "purpose")),
        FlowStage("save_forecast", StageSpec("Save forecast", "purpose")),
    )
    monkeypatch.setitem(pages._FLOWS, "future", lambda: flow)

    updates = pages._updates("future", {
        "mode": "explicit",
        "flow_version": CURRENT_FLOW_VERSION,
        "events": [
            {"stage": "Totally different name", "stage_id": "source_tail", "status": "complete", "detail": "done", "elapsed_seconds": 1},
        ],
    })

    assert "evow-stage-complete" in _header(updates, 2, 0)["elem_classes"]
    assert "Skipped" not in _header(updates, 2, 1)["label"]


def test_latest_flow_falls_back_to_the_name_for_id_less_legacy_events() -> None:
    """A v1 event without a stage_id resolves onto the current row by its label."""
    updates = pages._updates("future", {
        "mode": "implicit",
        "flow_version": LEGACY_FLOW_VERSION,
        "events": [
            {"stage": "Sample input tail", "status": "complete", "detail": "done", "elapsed_seconds": 1},
        ],
    })

    # Future View's real flow has four groups: Input covers ``source_tail``.
    assert "evow-stage-complete" in updates[4]["elem_classes"]
    assert "1. Sample input tail · Completed" in updates[4]["label"]


def test_v1_trace_maps_name_events_and_marks_new_rows_skipped(monkeypatch) -> None:
    """A row the v1 run could not have recorded is skipped, not left waiting."""
    flow = _two_stage_flow(
        FlowStage("source_tail", StageSpec("Sample input tail", "purpose")),
        FlowStage("new_stage", StageSpec("Brand new stage", "purpose")),
        legacy_stage_ids=("Sample input tail",),
    )
    monkeypatch.setitem(pages._FLOWS, "future", lambda: flow)

    updates = pages._updates("future", {
        "mode": "implicit",
        "flow_version": LEGACY_FLOW_VERSION,
        "events": [
            {"stage": "Sample input tail", "status": "complete", "detail": "done", "elapsed_seconds": 1},
        ],
    })

    assert "evow-stage-complete" in _header(updates, 2, 0)["elem_classes"]
    assert "evow-stage-skipped" in _header(updates, 2, 1)["elem_classes"]
    assert "Not recorded by this legacy run." in _card(updates, 2, 1)["value"]


def test_v2_trace_maps_by_display_name_and_marks_new_rows_skipped(monkeypatch) -> None:
    """A v2 event whose retired id no longer exists still maps by display name."""
    flow = _two_stage_flow(
        FlowStage("train_lora", StageSpec("Train selected method", "purpose")),
        FlowStage("new_stage", StageSpec("Brand new stage", "purpose")),
        legacy_stage_ids=("Train selected method",),
    )
    monkeypatch.setitem(pages._FLOWS, "future", lambda: flow)

    updates = pages._updates("future", {
        "mode": "self_trained",
        "flow_version": 2,
        "events": [
            # A retired v2 id (``train_method``), an unchanged display name.
            {"stage": "Train selected method", "stage_id": "train_method", "status": "complete", "detail": "done", "elapsed_seconds": 1},
        ],
    })

    assert "evow-stage-complete" in _header(updates, 2, 0)["elem_classes"]
    assert "evow-stage-skipped" in _header(updates, 2, 1)["elem_classes"]
    # A v2 trace is named as legacy in the run-total notice.
    assert "Legacy workflow version v2" in updates[-1]["value"]


def test_v2_trace_keeps_a_known_stage_id_over_its_display_name(monkeypatch) -> None:
    """A legacy event with a still-valid id resolves by that id, not the label."""
    flow = _two_stage_flow(
        FlowStage("alpha", StageSpec("Alpha", "purpose")),
        FlowStage("beta", StageSpec("Beta", "purpose")),
        legacy_stage_ids=("Alpha", "Beta"),
    )
    monkeypatch.setitem(pages._FLOWS, "future", lambda: flow)

    updates = pages._updates("future", {
        "mode": "explicit",
        "flow_version": 2,
        "events": [
            {"stage": "Beta", "stage_id": "alpha", "status": "complete", "elapsed_seconds": 1},
        ],
    })

    assert "evow-stage-complete" in _header(updates, 2, 0)["elem_classes"]
    assert "Waiting" in _header(updates, 2, 1)["label"]


def test_unknown_flow_version_flags_and_interprets_no_events(monkeypatch) -> None:
    """An unknown future version leaves every row waiting and surfaces a notice."""
    flow = _two_stage_flow(
        FlowStage("source_tail", StageSpec("Sample input tail", "purpose")),
        FlowStage("save_forecast", StageSpec("Save forecast", "purpose")),
    )
    monkeypatch.setitem(pages._FLOWS, "future", lambda: flow)

    updates = pages._updates("future", {
        "mode": "explicit",
        "flow_version": CURRENT_FLOW_VERSION + 1,
        "events": [
            {"stage": "Sample input tail", "stage_id": "source_tail", "status": "complete", "elapsed_seconds": 1},
        ],
    })

    assert "Waiting" in _header(updates, 2, 0)["label"]
    assert "Unsupported workflow version" in updates[-1]["value"]


def test_numeric_looking_stage_id_is_not_treated_as_an_index(monkeypatch) -> None:
    """A decimal event id must not silently select a stage by position."""
    flow = _two_stage_flow(
        FlowStage("alpha", StageSpec("Alpha", "purpose")),
        FlowStage("beta", StageSpec("Beta", "purpose")),
    )
    monkeypatch.setitem(pages._FLOWS, "future", lambda: flow)

    updates = pages._updates("future", {
        "mode": "explicit",
        "flow_version": CURRENT_FLOW_VERSION,
        "events": [
            {"stage": "Position zero?", "stage_id": "0", "status": "complete", "elapsed_seconds": 1},
        ],
    })

    # "0" is an unknown id, so neither the first stage nor any other becomes complete.
    assert "Waiting" in _header(updates, 2, 0)["label"]
    assert "Waiting" in _header(updates, 2, 1)["label"]


def test_a_saved_current_run_ending_in_running_renders_interrupted_and_rolls_up_error(monkeypatch) -> None:
    """A saved run whose last tick never finished is an interrupted stage, not live."""
    flow = _two_stage_flow(
        FlowStage("source_tail", StageSpec("Sample input tail", "purpose")),
        FlowStage("save_forecast", StageSpec("Save forecast", "purpose")),
    )
    monkeypatch.setitem(pages._FLOWS, "future", lambda: flow)

    updates = pages._updates("future", {
        "mode": "explicit",
        "flow_version": CURRENT_FLOW_VERSION,
        "profile_id": f"future.explicit.v{CURRENT_FLOW_VERSION}",
        "result": {"primary": "forecast.mp4"},
        "events": [
            {"stage": "Save forecast", "stage_id": "save_forecast", "status": "running", "detail": "encoding", "elapsed_seconds": 5},
        ],
    }, is_saved=True)

    assert "Interrupted" in _header(updates, 2, 1)["label"]
    assert "evow-stage-failed" in _header(updates, 2, 1)["elem_classes"]
    assert "Run ended while this stage was running." in _card(updates, 2, 1)["value"]
    assert "evow-flow-group-failed" in updates[0]["elem_classes"]


def test_a_failed_saved_run_with_no_result_still_derives_interrupted(monkeypatch) -> None:
    """A saved run whose ``result`` is absent is interrupted, not live.

    ``_load_saved``/``list_runs`` allow a failed or partial saved trace whose
    ``result`` is ``None``. An explicit ``is_saved`` flag keeps that from looking
    like an in-flight live stream, so its dangling final ``running`` tick is
    derived as interrupted and rolls up a group error.
    """
    flow = _two_stage_flow(
        FlowStage("source_tail", StageSpec("Sample input tail", "purpose")),
        FlowStage("save_forecast", StageSpec("Save forecast", "purpose")),
    )
    monkeypatch.setitem(pages._FLOWS, "future", lambda: flow)

    trace = {
        "mode": "explicit",
        "flow_version": CURRENT_FLOW_VERSION,
        "profile_id": f"future.explicit.v{CURRENT_FLOW_VERSION}",
        "result": None,
        "events": [
            {"stage": "Save forecast", "stage_id": "save_forecast", "status": "running", "detail": "encoding", "elapsed_seconds": 5},
        ],
    }

    saved = pages._updates("future", trace, is_saved=True)
    assert "Interrupted" in _header(saved, 2, 1)["label"]
    assert "evow-stage-failed" in _header(saved, 2, 1)["elem_classes"]
    assert "Run ended while this stage was running." in _card(saved, 2, 1)["value"]
    assert "evow-flow-group-failed" in saved[0]["elem_classes"]

    # The same record streamed live must still read as Running: an in-flight
    # tick is only interrupted once the trace is known to be a saved run.
    live = pages._updates("future", trace)
    assert "Running" in _header(live, 2, 1)["label"]
    assert "evow-stage-running" in _header(live, 2, 1)["elem_classes"]
    assert "evow-flow-group-failed" not in live[0]["elem_classes"]


def test_v2_trace_maps_retired_ids_by_alias_over_a_changed_label(monkeypatch) -> None:
    """A v2 retired id resolves onto its renamed current row, id-to-id.

    The labels here deliberately match no current display name, so only the
    ``RETIRED_STAGE_ALIASES`` step can put each event on its row.
    """
    flow = _two_stage_flow(
        FlowStage("train_lora", StageSpec("Train the source LoRA", "purpose")),
        FlowStage("svd_load", StageSpec("Load the frozen prior", "purpose")),
        FlowStage("svd_generate", StageSpec("Roll out a plausible future", "purpose")),
        legacy_profiles={2: ("train_lora", "svd_load", "svd_generate")},
    )
    monkeypatch.setitem(pages._FLOWS, "future", lambda: flow)

    updates = pages._updates("future", {
        "mode": "self_trained",
        "flow_version": 2,
        "events": [
            {"stage": "Old train label", "stage_id": "train_method", "status": "complete", "elapsed_seconds": 1},
            {"stage": "Old prepare label", "stage_id": "prepare_method", "status": "complete", "elapsed_seconds": 2},
            {"stage": "Old generate label", "stage_id": "generate_future", "status": "complete", "elapsed_seconds": 3},
        ],
    })

    assert "evow-stage-complete" in _header(updates, 3, 0)["elem_classes"]
    assert "evow-stage-complete" in _header(updates, 3, 1)["elem_classes"]
    assert "evow-stage-complete" in _header(updates, 3, 2)["elem_classes"]


def test_a_v2_recorded_row_is_not_marked_not_recorded_while_a_v1_row_is(monkeypatch) -> None:
    """A row the trace's own version recorded is never called "not recorded"."""
    flow = _two_stage_flow(
        FlowStage("old_row", StageSpec("Old row", "purpose")),
        FlowStage("new_row", StageSpec("New row", "purpose")),
        legacy_stage_ids=("Old row", "New row"),
        # v1 lacked new_row; v2 recorded it.
        legacy_profiles={1: ("old_row",), 2: ("old_row", "new_row")},
    )
    monkeypatch.setitem(pages._FLOWS, "future", lambda: flow)

    v2 = pages._updates("future", {"mode": "implicit", "flow_version": 2, "events": []})
    assert "evow-stage-skipped" not in _header(v2, 2, 1)["elem_classes"]
    assert "Not recorded by this legacy run." not in _card(v2, 2, 1)["value"]

    v1 = pages._updates("future", {"mode": "implicit", "flow_version": 1, "events": []})
    assert "evow-stage-skipped" in _header(v1, 2, 1)["elem_classes"]
    assert "Not recorded by this legacy run." in _card(v1, 2, 1)["value"]


def test_a_saved_v2_run_ending_in_running_renders_interrupted_and_rolls_up_error(monkeypatch) -> None:
    """A saved legacy run whose last tick never finished is interrupted, not live."""
    flow = _two_stage_flow(
        FlowStage("source_tail", StageSpec("Sample input tail", "purpose")),
        FlowStage("save_forecast", StageSpec("Save forecast", "purpose")),
        legacy_profiles={2: ("source_tail", "save_forecast")},
    )
    monkeypatch.setitem(pages._FLOWS, "future", lambda: flow)

    updates = pages._updates("future", {
        "mode": "explicit",
        "flow_version": 2,
        "profile_id": "future.explicit.v2",
        "result": {"primary": "forecast.mp4"},
        "events": [
            {"stage": "Save forecast", "stage_id": "save_forecast", "status": "running", "detail": "encoding", "elapsed_seconds": 5},
        ],
    }, is_saved=True)

    assert "Interrupted" in _header(updates, 2, 1)["label"]
    assert "evow-stage-failed" in _header(updates, 2, 1)["elem_classes"]
    assert "Run ended while this stage was running." in _card(updates, 2, 1)["value"]
    assert "evow-flow-group-failed" in updates[0]["elem_classes"]


def test_a_v2_idless_running_tick_still_repairs_when_a_later_stage_begins(monkeypatch) -> None:
    """A legacy (v2) trace keeps the serial visual repair; it is not interrupted."""
    flow = _two_stage_flow(
        FlowStage("prepare", StageSpec("Prepare", "purpose")),
        FlowStage("generate", StageSpec("Generate", "purpose")),
        legacy_stage_ids=("Prepare", "Generate"),
    )
    monkeypatch.setitem(pages._FLOWS, "future", lambda: flow)

    updates = pages._updates("future", {
        "mode": "implicit",
        "flow_version": 2,
        "events": [
            {"stage": "Prepare", "status": "running", "detail": "old tick", "elapsed_seconds": 1},
            {"stage": "Generate", "status": "running", "detail": "current tick", "elapsed_seconds": 2},
        ],
    })

    assert "evow-stage-complete" in _header(updates, 2, 0)["elem_classes"]
    assert "evow-stage-running" in _header(updates, 2, 1)["elem_classes"]


def test_mode_inapplicable_stage_renders_skipped(monkeypatch) -> None:
    """A stage that only applies to another method is skipped, never waiting."""
    flow = _two_stage_flow(
        FlowStage("source_tail", StageSpec("Sample input tail", "purpose"), modes=("explicit",)),
        FlowStage("save_forecast", StageSpec("Save forecast", "purpose")),
    )
    monkeypatch.setitem(pages._FLOWS, "future", lambda: flow)

    updates = pages._updates("future", {
        "mode": "implicit",
        "flow_version": CURRENT_FLOW_VERSION,
        "events": [],
    })

    assert "evow-stage-skipped" in _header(updates, 2, 0)["elem_classes"]
    assert "Not used by this method." in _card(updates, 2, 0)["value"]


def test_unknown_non_empty_id_is_ignored_even_when_its_label_matches_a_stage(monkeypatch) -> None:
    """A v2 event with an unknown id never falls back to matching by display name."""
    flow = _two_stage_flow(
        FlowStage("source_tail", StageSpec("Sample input tail", "purpose")),
        FlowStage("save_forecast", StageSpec("Save forecast", "purpose")),
    )
    monkeypatch.setitem(pages._FLOWS, "future", lambda: flow)

    updates = pages._updates("future", {
        "mode": "explicit",
        "flow_version": CURRENT_FLOW_VERSION,
        "events": [
            {"stage": "Sample input tail", "stage_id": "ghost_id", "status": "complete", "elapsed_seconds": 1},
        ],
    })

    # The label is a known name, but the non-empty unknown id wins and matches nothing.
    assert "Waiting" in _header(updates, 2, 0)["label"]
    assert "Waiting" in _header(updates, 2, 1)["label"]


def test_a_non_empty_id_wins_over_a_conflicting_display_label(monkeypatch) -> None:
    """The id namespace is authoritative; a stray label cannot pull an event elsewhere."""
    flow = _two_stage_flow(
        FlowStage("alpha", StageSpec("Alpha", "purpose")),
        FlowStage("beta", StageSpec("Beta", "purpose")),
    )
    monkeypatch.setitem(pages._FLOWS, "future", lambda: flow)

    updates = pages._updates("future", {
        "mode": "explicit",
        "flow_version": CURRENT_FLOW_VERSION,
        "events": [
            {"stage": "Beta", "stage_id": "alpha", "status": "complete", "elapsed_seconds": 1},
        ],
    })

    assert "evow-stage-complete" in _header(updates, 2, 0)["elem_classes"]
    assert "Waiting" in _header(updates, 2, 1)["label"]


def test_id_less_label_equal_to_a_stage_id_does_not_misalign(monkeypatch) -> None:
    """A v1 label that spells a stage id is ignored, not silently matched to it.

    The label "foo" is a stage id but not a display name, so the legacy name map
    refuses it. Were the old id-or-name fallback kept, it would have selected the
    ``foo`` stage by accident.
    """
    flow = _two_stage_flow(
        FlowStage("foo", StageSpec("Bar", "purpose")),
        FlowStage("baz", StageSpec("Baz", "purpose")),
        legacy_stage_ids=("Bar", "Baz"),
    )
    monkeypatch.setitem(pages._FLOWS, "future", lambda: flow)

    updates = pages._updates("future", {
        "mode": "explicit",
        "flow_version": LEGACY_FLOW_VERSION,
        "events": [
            {"stage": "foo", "status": "complete", "elapsed_seconds": 1},
        ],
    })

    assert "Waiting" in _header(updates, 2, 0)["label"]
    assert "Waiting" in _header(updates, 2, 1)["label"]


def test_id_less_legacy_event_still_maps_by_its_unique_name(monkeypatch) -> None:
    """A true v1 event without an id resolves onto the current row by display name."""
    flow = _two_stage_flow(
        FlowStage("source_tail", StageSpec("Sample input tail", "purpose")),
        FlowStage("save_forecast", StageSpec("Save forecast", "purpose")),
        legacy_stage_ids=("Sample input tail", "Save forecast"),
    )
    monkeypatch.setitem(pages._FLOWS, "future", lambda: flow)

    updates = pages._updates("future", {
        "mode": "explicit",
        "flow_version": LEGACY_FLOW_VERSION,
        "events": [
            {"stage": "Save forecast", "status": "complete", "elapsed_seconds": 9},
        ],
    })

    assert "Waiting" in _header(updates, 2, 0)["label"]
    assert "evow-stage-complete" in _header(updates, 2, 1)["elem_classes"]


def test_editing_retired_prepare_method_maps_to_its_own_row_not_the_global_future_alias() -> None:
    """Text Manipulation's retired ``prepare_method`` is never rewritten to Future's alias.

    The global ``RETIRED_STAGE_ALIASES`` maps ``prepare_method -> svd_load``, but
    Text Manipulation scopes that same retired identity to its own
    ``prepare_mask``. A saved editing trace must land on the editing row, never
    on a Future View stage it never had.
    """
    from GREENFIELD.features import editing

    flow = editing.flow()
    position = {stage.stage_id: number for number, stage in enumerate(flow.stages)}
    updates = pages._updates("editing", {
        "mode": "explicit",
        "flow_version": 4,
        "events": [
            {"stage": "Old prepare label", "stage_id": "prepare_method", "status": "complete", "elapsed_seconds": 1},
        ],
    })
    groups = len(flow.groups)
    prepare_mask = updates[groups + position["prepare_mask"]]
    assert "Prepare implicit mask" in prepare_mask["label"]
    assert "Completed" in prepare_mask["label"]
    assert "evow-stage-complete" in prepare_mask["elem_classes"]
    # A Future View stage that the global alias would have selected is untouched.
    assert "svd_load" not in position


def test_a_current_editing_id_wins_over_any_feature_scoped_alias(monkeypatch) -> None:
    """An exact current editing id is authoritative over every alias step."""
    flow = FeatureFlow(
        "editing",
        CURRENT_FLOW_VERSION,
        (
            FlowStage("alpha", StageSpec("Alpha", "purpose")),
            FlowStage("beta", StageSpec("Beta", "purpose")),
        ),
        (SuperStage("All", ("alpha", "beta")),),
        legacy_profiles={1: ("alpha",)},
    )
    monkeypatch.setitem(pages._FLOWS, "editing", lambda: flow)
    # Deliberately alias the exact current id ``alpha`` onto ``beta``; the exact
    # current-id match must win before that alias is ever consulted.
    monkeypatch.setitem(pages.RETIRED_STAGE_ALIASES_BY_FEATURE, "editing", {"alpha": "beta"})

    updates = pages._updates("editing", {
        "mode": "implicit",
        "flow_version": 1,
        "events": [
            {"stage": "whatever", "stage_id": "alpha", "status": "complete", "elapsed_seconds": 1},
        ],
    })

    assert "evow-stage-complete" in updates[1]["elem_classes"]
    # Had the alias been consulted, ``beta`` would have absorbed the completion.
    assert "evow-stage-complete" not in updates[2]["elem_classes"]


def test_unsupported_version_run_total_shows_no_derived_elapsed(monkeypatch) -> None:
    """An unsupported trace's events cannot leak elapsed time into the run total."""
    flow = _two_stage_flow(
        FlowStage("source_tail", StageSpec("Sample input tail", "purpose")),
        FlowStage("save_forecast", StageSpec("Save forecast", "purpose")),
    )
    monkeypatch.setitem(pages._FLOWS, "future", lambda: flow)

    updates = pages._updates("future", {
        "mode": "explicit",
        "flow_version": CURRENT_FLOW_VERSION + 1,
        "events": [
            {"stage": "Sample input tail", "stage_id": "source_tail", "status": "complete", "elapsed_seconds": 42},
        ],
    })

    total = updates[-1]["value"]
    assert "Unsupported workflow version" in total
    assert "**Run total:** —" in total
    assert "42" not in total
