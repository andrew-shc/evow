"""Persist ordered, truthful execution records for Replay runs."""

from datetime import datetime, timezone
import json
from pathlib import Path
import time
from typing import Any

from GREENFIELD.app_core.timing import round_stopwatch


class RunTrace:
    """Persist source-of-truth stage lifecycle events for one run."""

    def __init__(self, run_dir: Path, mode: str, request: dict[str, Any]) -> None:
        self.run_dir = run_dir
        self._stage_started: dict[str, float] = {}
        self._stage_status: dict[str, str] = {}
        self._active_serial_stage: str | None = None
        self.started = time.monotonic()
        self.record: dict[str, Any] = {
            "run_id": run_dir.name,
            # V1 used broad labels for several unrelated operations. Never
            # manufacture a granular state for those records after the fact.
            "workflow_version": 2,
            "mode": mode,
            "request": request,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "events": [],
        }
        self._save()

    def update_request(self, values: dict[str, Any]) -> None:
        """Persist validation metadata alongside the user-requested parameters."""
        self.record["request"].update(values)
        self._save()

    def add(
        self,
        stage: str,
        status: str,
        detail: str,
        preview: Path | None = None,
        metrics: dict[str, Any] | None = None,
        *,
        stage_id: str | None = None,
        execution: str = "serial",
        log_path: Path | None = None,
    ) -> dict[str, Any]:
        """Add one producer-owned lifecycle event.

        A serial stage may publish repeated running ticks, but a second serial
        stage cannot begin until the first one is terminal. That invariant keeps
        the execution flow truthful without presentation-layer guesses.
        """
        stage_id = stage_id or stage
        if status not in {"running", "complete", "error", "skipped"}:
            raise ValueError(f"Unsupported stage status: {status!r}")
        if execution not in {"serial", "concurrent"}:
            raise ValueError(f"Unsupported execution mode: {execution!r}")
        previous = self._stage_status.get(stage_id)
        if previous in {"complete", "error", "skipped"}:
            raise ValueError(f"Stage {stage_id!r} is already terminal.")
        if status == "running" and execution == "serial":
            if self._active_serial_stage not in {None, stage_id}:
                raise ValueError(
                    f"Cannot start serial stage {stage_id!r} while "
                    f"{self._active_serial_stage!r} is still running."
                )
            self._active_serial_stage = stage_id
        if status in {"complete", "error", "skipped"}:
            if previous is None:
                raise ValueError(f"Terminal stage {stage_id!r} was never started.")
            if self._active_serial_stage == stage_id:
                self._active_serial_stage = None
        self._stage_status[stage_id] = status

        now = time.monotonic()
        stage_started = self._stage_started.setdefault(stage_id, now)
        event: dict[str, Any] = {
            "sequence": len(self.record["events"]) + 1,
            "stage": stage,
            "stage_id": stage_id,
            "status": status,
            "execution": execution,
            "detail": detail,
            "elapsed_seconds": round_stopwatch(now - self.started),
            "stage_elapsed_seconds": round_stopwatch(now - stage_started),
        }
        if preview is not None:
            event["preview"] = str(preview.relative_to(self.run_dir))
        if metrics:
            event["metrics"] = metrics
        if log_path is not None:
            event["log_path"] = str(log_path.relative_to(self.run_dir))
        self.record["events"].append(event)
        self._save()
        return event

    def _save(self) -> None:
        target = self.run_dir / "trace.json"
        temporary = self.run_dir / "trace.json.tmp"
        temporary.write_text(json.dumps(self.record, indent=2))
        temporary.replace(target)
