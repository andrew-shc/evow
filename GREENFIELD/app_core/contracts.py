"""Typed contracts crossing dashboard, trace, and feature-adapter boundaries."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal


StageStatus = Literal["waiting", "running", "complete", "error"]


@dataclass(frozen=True)
class StageSpec:
    """Stable execution-stage metadata rendered with Replay's program contract."""

    name: str
    purpose: str
    source_file: str = ""
    source_location: str = ""
    function_chain: str = ""
    inputs: str = ""
    outputs: str = ""
    controls: str = ""


@dataclass(frozen=True)
class StageEvent:
    """A serializable update emitted by an adapter while a run is executing."""

    stage: str
    status: StageStatus
    detail: str
    preview: Path | None = None
    metrics: dict[str, Any] = field(default_factory=dict)


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
class FeatureResult:
    """Final artifact locations and structured UI data from one feature run."""

    primary: Path | None = None
    secondary: Path | None = None
    rows: list[list[Any]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
