"""Persist the visual stages of a replay run for later interactive playback."""

from datetime import datetime, timezone
import json
from pathlib import Path
import time
from typing import Any


class RunTrace:
    def __init__(self, run_dir: Path, mode: str, request: dict[str, Any]) -> None:
        self.run_dir = run_dir
        self._stage_started: dict[str, float] = {}
        self.started = time.monotonic()
        self.record: dict[str, Any] = {
            "run_id": run_dir.name,
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
    ) -> dict[str, Any]:
        now = time.monotonic()
        stage_started = self._stage_started.setdefault(stage, now)
        event: dict[str, Any] = {
            "stage": stage,
            "status": status,
            "detail": detail,
            "elapsed_seconds": round(now - self.started, 2), "stage_elapsed_seconds": round(now - stage_started, 2),
        }
        if preview is not None:
            event["preview"] = str(preview.relative_to(self.run_dir))
        if metrics:
            event["metrics"] = metrics
        self.record["events"].append(event)
        self._save()
        return event

    def _save(self) -> None:
        target = self.run_dir / "trace.json"
        temporary = self.run_dir / "trace.json.tmp"
        temporary.write_text(json.dumps(self.record, indent=2))
        temporary.replace(target)
