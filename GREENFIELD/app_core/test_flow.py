"""Pure unit coverage for the stage-identity flow schema and version resolution."""

import pytest

from GREENFIELD.app_core.contracts import StageSpec
from GREENFIELD.app_core.flow import (
    CURRENT_FLOW_VERSION,
    LEGACY_FLOW_VERSION,
    FeatureFlow,
    FlowStage,
    empty_flow,
    normalize_stage_id,
    profile_id,
    resolve_flow,
    stage_applies,
)
from GREENFIELD.app_core.super_stages import SuperStage


def _synthetic_flow() -> FeatureFlow:
    """A two-stage current flow with a distinct recorded set per older version."""
    return FeatureFlow(
        "future",
        CURRENT_FLOW_VERSION,
        (
            FlowStage("source_tail", StageSpec("Sample input tail", "purpose")),
            FlowStage("new_stage", StageSpec("Brand new stage", "purpose")),
        ),
        (SuperStage("Input", ("source_tail", "new_stage")),),
        legacy_stage_ids=("Sample input tail",),
        # v1 recorded only the first stage (name-only profile); v2 recorded both.
        legacy_profiles={1: ("source_tail",), 2: ("source_tail", "new_stage")},
    )


_REGISTRY = {"future": _synthetic_flow}


def test_normalize_stage_id_falls_back_to_name_and_prefers_explicit_id() -> None:
    """Legacy events carry no id, so the name is the only stable identity."""
    assert normalize_stage_id("", "Sample input tail") == "Sample input tail"
    assert normalize_stage_id("source_tail", "Sample input tail") == "source_tail"


def test_numeric_looking_id_is_never_reinterpreted_as_a_position() -> None:
    """A real id that happens to look like a number stays exactly that id."""
    assert normalize_stage_id("123", "whatever") == "123"


def test_stage_applies_treats_empty_modes_as_every_mode() -> None:
    every_mode = FlowStage("source_tail", StageSpec("Sample input tail", "purpose"))
    explicit_only = FlowStage("scene_cache", StageSpec("Scene cache", "purpose"), modes=("explicit",))

    assert stage_applies(every_mode, "implicit")
    assert stage_applies(every_mode, "explicit")
    assert stage_applies(explicit_only, "explicit")
    assert not stage_applies(explicit_only, "implicit")


def test_empty_flow_has_no_stages_or_groups() -> None:
    assert empty_flow("selection") == FeatureFlow("selection", LEGACY_FLOW_VERSION, (), ())


def test_profile_id_encodes_feature_mode_and_flow_version() -> None:
    assert profile_id("future", "self_trained", CURRENT_FLOW_VERSION) == f"future.self_trained.v{CURRENT_FLOW_VERSION}"
    assert profile_id("selection", "implicit", LEGACY_FLOW_VERSION) == "selection.implicit.v1"


def test_resolve_flow_v1_exposes_only_the_stages_v1_could_record() -> None:
    """A v1 trace resolves on the current flow and names the legacy version."""
    resolution = resolve_flow("future", "implicit", LEGACY_FLOW_VERSION, "", _REGISTRY)

    assert resolution.flow.stages[0].stage_id == "source_tail"
    # The exposed set is the current-id set v1 could have recorded, so the
    # renderer marks ``new_stage`` (which v1 lacked) as not recorded.
    assert resolution.legacy_ids == frozenset({"source_tail"})
    assert "Legacy workflow version v1" in resolution.notice


def test_resolve_flow_v2_exposes_exactly_what_v2_recorded() -> None:
    """A v2 trace exposes the v2 recorded set, not v1's frozen smaller set."""
    resolution = resolve_flow("future", "implicit", 2, "future.implicit.v2", _REGISTRY)

    assert resolution.legacy_ids == frozenset({"source_tail", "new_stage"})
    assert "Legacy workflow version v2" in resolution.notice


def test_resolve_flow_unknown_legacy_version_exposes_no_recorded_set() -> None:
    """A version below current with no profile entry exposes no legacy ids."""
    resolution = resolve_flow("future", "implicit", 0, "", _REGISTRY)

    assert resolution.legacy_ids == frozenset()
    assert "Legacy workflow version v0" in resolution.notice


def test_resolve_flow_current_exposes_every_current_id() -> None:
    resolution = resolve_flow("future", "implicit", CURRENT_FLOW_VERSION, "", _REGISTRY)

    assert resolution.legacy_ids == frozenset({"source_tail", "new_stage"})
    assert resolution.notice == ""


def test_resolve_flow_unknown_version_flags_and_exposes_no_legacy_ids() -> None:
    """A future version is uninterpretable: flag it and interpret no events."""
    resolution = resolve_flow("future", "implicit", CURRENT_FLOW_VERSION + 1, "", _REGISTRY)

    assert f"Unsupported workflow version v{CURRENT_FLOW_VERSION + 1}" in resolution.notice
    assert resolution.flow.flow_version == CURRENT_FLOW_VERSION
    assert resolution.legacy_ids == frozenset()


def test_resolve_flow_flags_a_mismatched_profile_without_changing_resolution() -> None:
    resolution = resolve_flow("future", "implicit", CURRENT_FLOW_VERSION, "future.explicit.v2", _REGISTRY)

    assert "future.explicit.v2" in resolution.notice
    assert resolution.legacy_ids == frozenset({"source_tail", "new_stage"})


def test_resolve_flow_accepts_an_empty_or_matching_profile() -> None:
    current = profile_id("future", "implicit", CURRENT_FLOW_VERSION)

    assert resolve_flow("future", "implicit", CURRENT_FLOW_VERSION, "", _REGISTRY).notice == ""
    assert resolve_flow("future", "implicit", CURRENT_FLOW_VERSION, current, _REGISTRY).notice == ""


def test_feature_flow_rejects_two_stages_with_the_same_stage_id() -> None:
    """Duplicate ids would make the id namespace ambiguous, so construction fails."""
    with pytest.raises(ValueError, match="stage ids must be unique"):
        FeatureFlow(
            "future",
            CURRENT_FLOW_VERSION,
            (
                FlowStage("source_tail", StageSpec("One", "purpose")),
                FlowStage("source_tail", StageSpec("Two", "purpose")),
            ),
            (),
        )


def test_feature_flow_rejects_two_stages_with_the_same_name() -> None:
    """Duplicate display names would make the legacy name map ambiguous."""
    with pytest.raises(ValueError, match="stage names must be unique"):
        FeatureFlow(
            "future",
            CURRENT_FLOW_VERSION,
            (
                FlowStage("source_tail", StageSpec("Same label", "purpose")),
                FlowStage("save_forecast", StageSpec("Same label", "purpose")),
            ),
            (),
        )


def test_feature_flow_rejects_an_id_equal_to_another_stages_name() -> None:
    """An id/name crossing would let a legacy name-only event land on two rows."""
    with pytest.raises(ValueError, match="stage ids and names must not cross"):
        FeatureFlow(
            "future",
            CURRENT_FLOW_VERSION,
            (
                FlowStage("source_tail", StageSpec("Sample input tail", "purpose")),
                FlowStage("save_forecast", StageSpec("source_tail", "purpose")),
            ),
            (),
        )


def test_feature_flow_allows_an_id_that_equals_its_own_name() -> None:
    """A feature that never adopted snake_case ids keeps working unchanged."""
    flow = FeatureFlow(
        "future",
        CURRENT_FLOW_VERSION,
        (FlowStage("Sample input tail", StageSpec("Sample input tail", "purpose")),),
        (),
    )

    assert flow.stages[0].stage_id == flow.stages[0].spec.name == "Sample input tail"
