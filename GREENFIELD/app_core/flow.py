"""Pure stage-identity model and version resolution for feature traces.

This is the schema seam between a feature's declarative stage specs and the
versioned profile a trace persists. It intentionally imports only ``contracts``
and ``super_stages`` (never a feature or ``trace``), so it stays a leaf that
``trace`` can import without creating a cycle.
"""

from dataclasses import dataclass, field
from typing import Mapping

from .contracts import StageSpec
from .super_stages import SuperStage


# A trace with no explicit version predates named stage IDs; its events were
# keyed by name alone. Bump CURRENT_FLOW_VERSION whenever a feature's ordered
# stage set changes, so a saved run resolves against the profile that wrote it.
# v4 atomized Text Query's five coarse bars into a 22-row union; v5 atomizes
# Text Manipulation's five coarse bars into a 19-row union. Every previous
# profile is retained per feature so its saved traces still resolve.
CURRENT_FLOW_VERSION = 5
LEGACY_FLOW_VERSION = 1

# Retired identities a later profile renamed, mapped onto their current
# identity. Keys are either a retired ``stage_id`` or a retired display *name*:
# a v1 trace carried no id at all, so its only mapping handle is the label it
# persisted, while a v2/v3 trace carried the old id. The renderer tries this
# alias step before ever falling back to a current display-name match. This is
# deliberately global: a retired identity names exactly one current stage across
# every feature (a feature that still uses it as a current id/name matches it
# directly and never reaches the alias step).
RETIRED_STAGE_ALIASES: Mapping[str, str] = {
    # Future View v1/v2 renames (id-only; the display names were unchanged).
    "train_method": "train_lora",
    "prepare_method": "svd_load",
    "generate_future": "svd_generate",
    # Text Query's coarse v1/v2/v3 retrieval, grounding, and save bars split into
    # atomic rows. The old ids and the old display names both map onto the atomic
    # stage that now owns the operation, so an id-less v1 trace and an id-bearing
    # v2 trace land on the same truthful row.
    "retrieve_intervals": "window_rank",
    "Retrieve relevant intervals": "window_rank",
    "ground_track": "mask_validate",
    "Ground and track binary target": "mask_validate",
    "save_clips": "write_clips",
    "Save highlighted clips": "write_clips",
}

# Feature-scoped retired aliases, consulted *before* the global map. A retired
# identity can be reused by a different feature as its own current id, in which
# case a single global mapping would silently misroute it. The canonical example
# is ``prepare_method``: Future View retired it onto ``svd_load``, but Text
# Manipulation used it as a current id until v4 and now retires it onto its own
# ``prepare_mask``. Scoping the alias lets each feature's saved trace land on the
# row that feature actually owns, while the global map stays the fallback for
# every identity no feature has scoped.
RETIRED_STAGE_ALIASES_BY_FEATURE: Mapping[str, Mapping[str, str]] = {
    "editing": {
        # Text Manipulation's five coarse v1-v4 bars split into the 19-row union;
        # both the old ids and the old display names map onto the atomic stage
        # that now owns the operation, so a v1 name-only trace and a v4 id-bearing
        # trace land on the same truthful row.
        "inspect_episode": "episode_decode",
        "Inspect source episode": "episode_decode",
        "resolve_scope": "scope_classify",
        "Resolve edit scope": "scope_classify",
        "prepare_method": "prepare_mask",
        "Prepare selected method": "prepare_mask",
        "generate_edit": "vace_prepare",
        "Generate edit proposal": "vace_prepare",
        "save_result": "publish_artifacts",
        "Save and render result": "publish_artifacts",
    },
}


@dataclass(frozen=True)
class FlowStage:
    """One atomic stage identity plus the modes it applies to.

    Empty ``modes`` means the stage applies to every mode. Keeping applicability
    on the flow (not the renderer) lets a feature mark a branch inapplicable
    without a page hardcoding method strings.
    """

    stage_id: str
    spec: StageSpec
    modes: tuple[str, ...] = ()


def _validate_stage_namespace(stages: tuple[FlowStage, ...]) -> None:
    """Reject a profile whose stage ids and display names are not one-to-one.

    A v1 event carried no id, so its display name is resolved through the
    ``name -> stage_id`` map. That fallback is only unambiguous when every id is
    unique, every name is unique, and the id and name namespaces never cross.
    Otherwise two different stages could claim one event, or a legacy label could
    silently select a stage that happens to spell the same string. We fail fast at
    construction so a saved run can never misalign onto the wrong row.

    A stage whose *own* id equals its *own* name is the supported "never adopted
    snake_case" case and is deliberately allowed; only a collision across stages
    is rejected.
    """
    by_id: dict[str, str] = {}
    by_name: dict[str, str] = {}
    for stage in stages:
        stage_id, name = stage.stage_id, stage.spec.name
        if stage_id in by_id:
            raise ValueError(
                f"Flow stage id {stage_id!r} is shared by stages "
                f"{by_id[stage_id]!r} and {name!r}; stage ids must be unique."
            )
        if name in by_name:
            raise ValueError(
                f"Flow stage name {name!r} is shared by stages "
                f"{by_name[name]!r} and {stage_id!r}; stage names must be unique."
            )
        by_id[stage_id] = name
        by_name[name] = stage_id
    # Cross-namespace collision: an id equal to another stage's display name.
    # ``by_name`` maps that name to its owning stage id; if the owner is a
    # different stage, a legacy name-only event could land on the wrong one.
    for stage in stages:
        owner_id = by_name.get(stage.stage_id)
        if owner_id is not None and owner_id != stage.stage_id:
            raise ValueError(
                f"Flow stage id {stage.stage_id!r} collides with the display name "
                f"of stage {owner_id!r}; stage ids and names must not cross."
            )


@dataclass(frozen=True)
class FeatureFlow:
    """One feature's ordered stage profile and its named parent groups.

    ``flow_version`` is persisted beside the trace so a saved run is resolved
    against the exact profile it was produced with. ``groups`` own ``stage_ids``
    rather than positional indexes, so reordering ``stages`` can never silently
    regroup a saved run.

    ``legacy_stage_ids`` is the frozen set of *display names* that existed when
    the v1 name-only profile was written. It stays literal for back-compat and
    as the last-resort name map for a version this build does not otherwise know.

    ``legacy_profiles`` maps each pre-current ``flow_version`` to the ordered
    *current* ``stage_id`` set that exact version could have recorded. A v1
    trace carried no id at all, so its six original labels map onto current ids;
    a v2 trace already carried ids and recorded a different (often larger) union.
    Keying by exact version is what lets the renderer mark a genuinely new row as
    not recorded without mislabeling a row the trace's own version did record.
    An unknown version below current has no entry and therefore exposes no legacy
    ids, so the caller falls back to the v1 name set.
    """

    feature: str
    flow_version: int
    stages: tuple[FlowStage, ...]
    groups: tuple[SuperStage, ...]
    legacy_stage_ids: tuple[str, ...] = ()
    legacy_profiles: Mapping[int, tuple[str, ...]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        # Every construction path -- feature ``flow()``, ``empty_flow``, and tests
        # alike -- must satisfy the one-to-one id/name namespace. Failing here
        # makes an ambiguous flow impossible to build rather than deferring the
        # surprise to a saved run.
        _validate_stage_namespace(self.stages)


@dataclass(frozen=True)
class FlowResolution:
    """How one saved trace's persisted identity maps onto a rendered flow.

    ``flow`` is always the current profile whose rows are rendered. ``legacy_ids``
    is the set of current stage ids the trace's *exact* pre-current version could
    have recorded, so the renderer can mark a row the run legitimately could not
    have recorded; it is empty when the version is unknown below current, in
    which case the caller falls back to the flow's frozen v1 name set. The caller
    maps legacy events onto current rows by id, retired-id alias, or display
    name. ``notice`` is a human-readable warning (empty when the trace resolves
    cleanly).
    """

    flow: FeatureFlow
    legacy_ids: frozenset[str]
    notice: str


def normalize_stage_id(stage_id: str, stage_name: str) -> str:
    """Return a stable stage identity, falling back to the legacy name.

    Legacy specs and events carried no id, so their ``name`` was the only stable
    handle available. Normalizing here keeps every downstream lookup keyed the
    same way without rewriting old traces. A numeric-looking id is a real id and
    is never reinterpreted as a position.
    """
    return stage_id or stage_name


def stage_applies(stage: FlowStage, mode: str) -> bool:
    """Return whether ``stage`` runs in ``mode`` (empty modes means every mode)."""
    return not stage.modes or mode in stage.modes


def empty_flow(feature: str) -> FeatureFlow:
    """Return a feature's zero-stage legacy flow, for tests and placeholders."""
    return FeatureFlow(feature, LEGACY_FLOW_VERSION, (), ())


def _profile_identifier(feature: str, mode: str, flow_version: int) -> str:
    """Return the stable ``feature.mode.vN`` identifier persisted with a run."""
    return f"{feature}.{mode}.v{flow_version}"


def profile_id(feature: str, mode: str, flow_version: int) -> str:
    """Return the stable ``feature.mode.vN`` identifier persisted with a run."""
    return _profile_identifier(feature, mode, flow_version)


def resolve_flow(
    feature: str,
    mode: str,
    flow_version: int,
    profile_id: str,
    registry,
) -> FlowResolution:
    """Resolve a trace's persisted identity onto the registry's current flow.

    Rules, in order:
    * A version newer than this build's ``CURRENT_FLOW_VERSION`` is uninterpretable:
      show the current layout, flag it, and expose no legacy ids so the caller
      ignores events.
    * A version older than this build is mapped best-effort onto the current
      rows: an event resolves to the current row by an exact current-id match,
      then a known retired-id alias, then a unique display-name match. A display
      name is the only identity that survives a stage-set change, so this keeps a
      v2 event whose retired id no longer exists — e.g. an old ``train_method``
      aliased onto ``train_lora`` — on the right row. Unmappable events are
      ignored and the legacy version is named in the notice. The exposed
      ``legacy_ids`` is the current-id set that *that exact version* could have
      recorded, so a caller can mark a row the version genuinely lacked; it is
      empty for an unknown version below current.
    * The current version renders as-is; a non-empty ``profile_id`` that names a
      different profile is flagged but never changes the resolution.
    """
    flow = registry[feature]()
    if flow_version > CURRENT_FLOW_VERSION:
        notice = (
            f"Unsupported workflow version v{flow_version}; "
            "showing the current layout without interpreting events."
        )
        return FlowResolution(flow, frozenset(), notice)
    if flow_version < CURRENT_FLOW_VERSION:
        notice = (
            f"Legacy workflow version v{flow_version}; "
            "mapping its events onto the current layout."
        )
        # ``legacy_ids`` is the current-id set the trace's *exact* version
        # recorded, never "every current id". The renderer uses it to mark a row
        # that did not yet exist; an unknown sub-version exposes nothing and the
        # caller falls back to the frozen v1 name set.
        recorded_ids = flow.legacy_profiles.get(flow_version, ())
        return FlowResolution(flow, frozenset(recorded_ids), notice)
    expected = _profile_identifier(feature, mode, CURRENT_FLOW_VERSION)
    notice = "" if not profile_id or profile_id == expected else (
        f"Saved run was written by profile '{profile_id}', not '{expected}'; showing the current layout."
    )
    return FlowResolution(flow, frozenset(stage.stage_id for stage in flow.stages), notice)
