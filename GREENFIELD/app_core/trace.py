"""Versioned, atomic feature-run manifests shared by newly migrated pages."""

from datetime import datetime, timezone
import json
from pathlib import Path
import time
from typing import Any
from uuid import uuid4

from .contracts import FeatureResult, RunArtifacts, StageEvent


MANIFEST_VERSION = 1


class FeatureTrace:
    """Persist one feature invocation without exposing partially written JSON."""

    def __init__(self, root: Path, feature: str, mode: str, request: dict[str, Any]):
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        self.run_dir = root / feature / "runs" / f"{stamp}_{uuid4().hex[:8]}"
        self.run_dir.mkdir(parents=True, exist_ok=False)
        self.started = time.monotonic()
        # Keep per-stage clocks separate from the total run clock. A feature can
        # emit several live updates for one stage, and the dashboard must not
        # make each of those stages look as old as the whole job.
        self._stage_started: dict[str, float] = {}
        self.record: dict[str, Any] = {
            "manifest_version": MANIFEST_VERSION,
            "run_id": self.run_dir.name,
            "feature": feature,
            "mode": mode,
            "request": request,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "events": [],
            "result": None,
        }
        self._save()

    @property
    def artifacts(self) -> RunArtifacts:
        return RunArtifacts(self.run_dir)

    def add(self, event: StageEvent) -> dict[str, Any]:
        """Append a stage update and return its serialized representation."""
        now = time.monotonic()
        stage_started = self._stage_started.setdefault(event.stage, now)
        item: dict[str, Any] = {
            "stage": event.stage,
            "status": event.status,
            "detail": event.detail,
            # Retain the total for existing saved-run consumers, but persist
            # the stage-local value so renderers do not need to infer it.
            "elapsed_seconds": round(now - self.started, 2),
            "stage_elapsed_seconds": round(now - stage_started, 2),
        }
        if event.preview:
            item["preview"] = str(event.preview.relative_to(self.run_dir))
        if event.metrics:
            item["metrics"] = event.metrics
        self.record["events"].append(item)
        self._save()
        return item

    def finish(self, result: FeatureResult) -> None:
        """Save final artifact references relative to the run directory."""
        def relative(path: Path | None) -> str | None:
            return str(path.relative_to(self.run_dir)) if path else None

        self.record["result"] = {
            "primary": relative(result.primary),
            "secondary": relative(result.secondary),
            "rows": result.rows,
            "metadata": result.metadata,
        }
        self._save()

    def _save(self) -> None:
        target = self.run_dir / "trace.json"
        temporary = self.run_dir / "trace.json.tmp"
        temporary.write_text(json.dumps(self.record, indent=2))
        temporary.replace(target)


def load_trace(run_dir: Path) -> dict[str, Any]:
    """Load new manifests and Replay's legacy trace shape without mutating either."""
    record = json.loads((run_dir / "trace.json").read_text())
    record.setdefault("manifest_version", 0)
    record.setdefault("feature", "replay")
    record.setdefault("result", None)
    return record


def list_runs(root: Path, feature: str) -> list[tuple[str, str]]:
    """Return newest saved runs as Gradio dropdown choices."""
    directory = root / feature / "runs"
    if not directory.is_dir():
        return []
    choices = []
    for trace_path in sorted(directory.glob("*/trace.json"), reverse=True):
        record = load_trace(trace_path.parent)
        label = f"{record.get('mode', 'unknown')} · {trace_path.parent.name}"
        choices.append((label, trace_path.parent.name))
    return choices
