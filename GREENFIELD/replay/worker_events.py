"""Atomic, replayable progress events shared by isolated Replay workers.

Workers run outside the Gradio process so CUDA allocations are released between
stages.  Stdout is useful for human diagnostics but is not a reliable progress
protocol: a buffered line can disappear when a process exits.  This module
writes the complete ordered event list atomically, allowing the coordinator to
read every transition, including transitions produced just before exit.
"""

import json
from pathlib import Path
from typing import Any, get_args

from GREENFIELD.app_core.contracts import StageStatus


class WorkerEventWriter:
    """Publish ordered worker events without exposing a partially written file."""

    def __init__(self, path: Path | None) -> None:
        self.path = path
        self.events: list[dict[str, Any]] = []

    def emit(
        self,
        stage_id: str,
        status: str,
        detail: str,
        *,
        metrics: dict[str, Any] | None = None,
        preview: Path | None = None,
        execution: str = "serial",
    ) -> None:
        """Append one lifecycle transition and atomically replace the sidecar."""
        if self.path is None:
            return
        event: dict[str, Any] = {
            "sequence": len(self.events) + 1,
            "stage_id": stage_id,
            "status": status,
            "detail": detail,
            "execution": execution,
        }
        if metrics:
            event["metrics"] = metrics
        if preview is not None:
            event["preview"] = str(preview)
        self.events.append(event)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(self.path.suffix + ".tmp")
        temporary.write_text(json.dumps(self.events))
        temporary.replace(self.path)


# The only statuses a well-formed lifecycle event may carry. Derived from the
# shared contract so a new status cannot be added there without this reader
# accepting it.
_ALLOWED_STATUSES = frozenset(get_args(StageStatus))


def _salvage_leading_records(text: str) -> list[Any]:
    """Recover the complete leading records of a truncated JSON array.

    The atomic writer never publishes a partial file, so this only guards a
    hand-edited or disk-truncated sidecar. Decoding one value at a time means a
    cut-off tail cannot discard the records already written before it.
    """
    decoder = json.JSONDecoder()
    index = text.find("[")
    if index == -1:
        return []
    index += 1
    records: list[Any] = []
    while True:
        while index < len(text) and text[index] in " \t\r\n,":
            index += 1
        if index >= len(text):
            return records
        try:
            value, index = decoder.raw_decode(text, index)
        except json.JSONDecodeError:
            return records
        records.append(value)


def _decode_events(text: str) -> list[Any]:
    """Decode the sidecar, salvaging complete leading records from a truncation."""
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return _salvage_leading_records(text)
    return payload if isinstance(payload, list) else []


def _is_valid_event(event: Any) -> bool:
    """Whether one persisted record is safe to hand to the relay.

    The relay coerces ``sequence`` to an int and treats ``metrics`` as a mapping,
    so a malformed record must be dropped here rather than raising later in the
    middle of a live run. A record is valid only when its sequence is
    int-coercible, its status is a known lifecycle state, its ``stage_id`` and
    ``detail`` are strings, and any ``metrics`` is a real object.
    """
    if not isinstance(event, dict):
        return False
    if not isinstance(event.get("stage_id"), str) or not isinstance(event.get("detail"), str):
        return False
    if event.get("status") not in _ALLOWED_STATUSES:
        return False
    metrics = event.get("metrics")
    if metrics is not None and not isinstance(metrics, dict):
        return False
    try:
        int(event["sequence"])
    except (KeyError, TypeError, ValueError):
        return False
    return True


def _read_sidecar_text(path: Path) -> str | None:
    """Read the sidecar, tolerating both I/O and text-decoding failures.

    The atomic writer always publishes valid UTF-8, but the sidecar still lives
    on disk where a partial or externally corrupted file is possible. Returning
    ``None`` for a file that cannot be opened and decoding a non-UTF8 file
    leniently lets the caller salvage every complete record written before the
    bad byte instead of aborting the relay on the first one.
    """
    try:
        return path.read_text()
    except UnicodeDecodeError:
        # Replace the corrupt bytes so ``_salvage_leading_records`` can still
        # recover the valid prefix; returning nothing would drop the whole file.
        try:
            return path.read_bytes().decode(errors="replace")
        except OSError:
            return None
    except OSError:
        return None


def read_events(path: Path, after_sequence: int) -> list[dict[str, Any]]:
    """Return fully persisted, well-formed events newer than ``after_sequence``.

    Atomic replacement means JSON decoding never observes a half-written list.
    A worker can still be between its first instruction and first publish, so a
    missing sidecar is normal and simply has no events yet. A malformed record
    is skipped rather than raising, and an unreadable, non-UTF8, or truncated
    file yields whatever valid records it did contain so one bad byte cannot
    abort a run.
    """
    if not path.is_file():
        return []
    text = _read_sidecar_text(path)
    if text is None:
        return []
    return [
        event for event in _decode_events(text)
        if _is_valid_event(event) and int(event["sequence"]) > after_sequence
    ]
