"""Typed contracts crossing dashboard, trace, and feature-adapter boundaries."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal


StageStatus = Literal["waiting", "running", "complete", "error", "skipped"]


@dataclass(frozen=True)
class StageSpec:
    """Stable execution-stage metadata rendered with Replay's program contract.

    ``name`` is the human-facing label drawn on the stage card. ``stage_id`` is
    the stable identity persisted on events and referenced by a feature flow's
    parent groups; when it is empty it falls back to ``name`` (via
    ``app_core.flow.normalize_stage_id``), which keeps legacy features that
    declared only names working unchanged. It stays last so positional
    construction that predates stage IDs keeps working.
    """

    name: str
    purpose: str
    source_file: str = ""
    source_location: str = ""
    function_chain: str = ""
    inputs: str = ""
    outputs: str = ""
    controls: str = ""
    # Static model provenance is kept with the stage contract so the UI never
    # guesses which implementation produced a live execution tick.
    model_refs: tuple["ModelReference", ...] = ()
    # Stable persisted identity, independent of the display label. Empty means
    # "use ``name``"; see ``app_core.flow.normalize_stage_id``.
    stage_id: str = ""


@dataclass(frozen=True)
class ModelReference:
    """One selected model dependency with its primary project URL."""

    name: str
    identifier: str
    role: str
    url: str


@dataclass(frozen=True)
class StageEvent:
    """A serializable update emitted by an adapter while a run is executing.

    ``stage`` is the human-facing label. ``stage_id`` is the stable identity the
    trace persists for that label; adapters may leave it empty to fall back to
    ``stage`` (via ``app_core.flow.normalize_stage_id``). It stays last so
    existing positional construction keeps working.
    """

    stage: str
    status: StageStatus
    detail: str
    preview: Path | None = None
    metrics: dict[str, Any] = field(default_factory=dict)
    # Stable persisted identity; empty means "use ``stage``".
    stage_id: str = ""


@dataclass(frozen=True)
class RunArtifacts:
    """The only writable locations supplied to a feature adapter."""

    run_dir: Path

    def file(self, relative_name: str) -> Path:
        """Return a safe output path directly beneath this invocation's directory."""
        candidate = (self.run_dir / relative_name).resolve()
        if self.run_dir.resolve() not in candidate.parents:
            raise ValueError("Artifacts must remain inside the current run directory.")
        candidate.parent.mkdir(parents=True, exist_ok=True)
        return candidate


@dataclass(frozen=True)
class ClipArtifact:
    """One source interval and its non-destructively highlighted video.

    ``source`` stays distinct from ``highlighted`` so a Text Query result can
    become a Text Manipulation input without carrying yellow query pixels into
    the edit model.
    """

    source: Path
    highlighted: Path
    start_seconds: float
    end_seconds: float
    score: float
    method: str
    # Explicit Text Query retains the directly grounded 2D result above as its
    # reference. This optional sibling is the deliberately coarser projection
    # through dynamic Gaussian primitives, never an implicit replacement.
    projected_highlighted: Path | None = None
    grounding_confidence: float | None = None
    semantic_mask_coverage: float | None = None
    projected_mask_coverage: float | None = None
    projected_low_specificity: bool = False
    selected_splats: tuple[Path, ...] = ()


@dataclass(frozen=True)
class FeatureResult:
    """Final artifact locations and structured UI data from one feature run."""

    primary: Path | None = None
    secondary: Path | None = None
    rows: list[list[Any]] = field(default_factory=list)
    clips: list[ClipArtifact] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
