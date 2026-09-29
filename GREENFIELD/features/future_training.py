"""Source-bound Future View training data, worker orchestration, and cache manifests."""
from __future__ import annotations
from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path
import subprocess
import sys
import time

import cv2
import numpy as np

from GREENFIELD.app_core.contracts import StageEvent


@dataclass(frozen=True)
class PreparedImplicitPairs:
    """One bounded pre-holdout sampling pass and its honest metrics.

    ``count`` is the number of temporal pairs, ``samples`` the total number of
    source frames read (one condition plus one target chunk per pair), and
    ``endpoint`` the last eligible pair start in seconds. Keeping these together
    lets the training stages report real numbers without re-opening the video.
    """

    count: int
    samples: int
    endpoint: float


def source_digest(source: str) -> str:
    """Hash bytes rather than trusting an upload path or browser filename."""
    digest = sha256()
    with Path(source).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _checkpoint_revision(checkpoint: Path) -> str | None:
    """Read the owner installer's tiny manifest; never touch the weight files themselves."""
    try:
        manifest = json.loads((checkpoint / "model_manifest.json").read_text())
    except (OSError, json.JSONDecodeError, TypeError):
        return None
    revision = manifest.get("revision")
    return str(revision) if revision else None


def checkpoint_identity(checkpoint) -> list:
    """Fingerprint the training checkpoint cheaply enough to run on every request.

    The adapter is fitted from ``settings.checkpoint``, so replacing or updating
    that snapshot must change ``training_key``; otherwise a later run could
    silently reuse an adapter learned from different base weights. Hashing
    multi-gigabyte weights on every request would be prohibitive, so instead we
    combine the resolved path, the checkpoint's filesystem metadata (size and
    nanosecond mtime), and the installer's recorded Hugging Face revision when
    one exists. The revision is the only genuinely content-stable identifier,
    while size/mtime also cover an ad-hoc local checkpoint that has no manifest.
    A missing or unreadable checkpoint yields a stable marker rather than
    crashing key computation before the checkpoint is ever needed.
    """
    if not checkpoint:
        return [None, None, None, None]
    path = Path(checkpoint)
    try:
        resolved = str(path.resolve())
        info = path.stat()
    except OSError:
        return [str(path), None, None, None]
    return [resolved, info.st_size, info.st_mtime_ns, _checkpoint_revision(path)]


def training_key(source: str, mode: str, settings) -> str:
    """Identify only artifacts that can safely affect this exact request."""
    values = {
        "source_sha256": source_digest(source), "mode": mode,
        # The adapter is fitted from ``settings.checkpoint``, so a replaced or
        # updated checkpoint must invalidate the cached artifact. See
        # ``checkpoint_identity`` for why path/size/mtime plus revision suffices.
        "checkpoint_identity": checkpoint_identity(getattr(settings, "checkpoint", None)),
        "history_seconds": settings.history_seconds,
        # ``history_fps`` is deliberately absent. It shapes the inference history
        # window (``pages._execute_future_stream`` -> ``read_recent_window``) and
        # the explicit-3D motion/timeline path, but training samples pre-holdout
        # pairs at the video's own FPS (``prepare_implicit_pairs`` -> ``_frame``)
        # and drives the worker with ``output_fps``. Changing it cannot change the
        # adapter, so hashing it would only cause needless retraining.
        "chunk_frames": settings.chunk_frames, "output_fps": settings.output_fps,
        # Training adapts at min(forecast cap, training ceiling), so only that
        # effective resolution can change what the adapter learns. Hashing the
        # forecast ``max_side`` on its own would needlessly retrain whenever it
        # rises above the ceiling without touching the training pass.
        "training_resolution": min(settings.max_side, getattr(settings, "training_max_side", settings.max_side)),
        # Every remaining knob that changes what the adapter learns must be part of
        # the key; otherwise a different step count, rank, or pair count could
        # silently reuse a stale artifact.
        "training_lora_steps": settings.training_lora_steps,
        "training_lora_rank": settings.training_lora_rank,
        "training_max_pairs": getattr(settings, "training_max_pairs", 0),
    }
    return sha256(json.dumps(values, sort_keys=True).encode()).hexdigest()[:20]


def training_dir(source: str, mode: str, settings) -> Path:
    return settings.root / "ASSETS" / "future" / "training" / training_key(source, mode, settings)


def training_manifest(source: str | None, mode: str, settings) -> dict | None:
    if not source:
        return None
    path = training_dir(source, mode, settings) / "manifest.json"
    try:
        record = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    return record if record.get("complete") and record.get("mode") == mode else None


def artifact_file(source: str | None, mode: str, settings) -> Path | None:
    """Return a complete, mode-specific artifact and never a partial worker output."""
    record = training_manifest(source, mode, settings)
    if not record:
        return None
    path = training_dir(source, mode, settings) / str(record.get("model_artifact", ""))
    return path if path.is_file() or path.is_dir() else None


def _video_info(source: str) -> tuple[cv2.VideoCapture, float, int]:
    capture = cv2.VideoCapture(source)
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 0)
    count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    if not capture.isOpened() or fps <= 0 or count < 1:
        capture.release()
        raise ValueError("The selected video contains no readable frames.")
    return capture, fps, count


def _frame(capture: cv2.VideoCapture, source_fps: float, count: int, seconds: float, max_side: int) -> np.ndarray:
    capture.set(cv2.CAP_PROP_POS_FRAMES, min(count - 1, round(seconds * source_fps)))
    ok, frame = capture.read()
    if not ok:
        raise ValueError("The selected video ended while training frames were sampled.")
    height, width = frame.shape[:2]
    scale = min(1.0, max_side / max(height, width))
    size = (max(2, round(width * scale / 2) * 2), max(2, round(height * scale / 2) * 2))
    return cv2.cvtColor(cv2.resize(frame, size), cv2.COLOR_BGR2RGB)


def _eligible_end(source: str, settings, target_seconds: float) -> tuple[cv2.VideoCapture, float, int, float]:
    capture, source_fps, count = _video_info(source)
    # A target must fully precede the final inference tail; it can never leak held-out frames.
    last_condition = count / source_fps - settings.history_seconds - target_seconds
    if last_condition <= 0:
        capture.release()
        raise ValueError("Training needs video longer than the held-out forecast history plus one SVD target chunk.")
    return capture, source_fps, count, last_condition


def prepare_implicit_pairs(source: str, settings, destination: Path) -> PreparedImplicitPairs:
    """Create one-frame SVD conditions and future chunks at inference cadence."""
    capture, source_fps, count, end = _eligible_end(source, settings, settings.chunk_frames / settings.output_fps)
    times = np.linspace(0, end, min(settings.training_max_pairs, max(1, int(end * settings.output_fps) + 1)))
    # Adaptation has its own lower spatial ceiling so a full backward pass fits the
    # owner GPU. The held-out forecast still uses the normal Future View resolution.
    training_max_side = min(settings.max_side, getattr(settings, "training_max_side", settings.max_side))
    conditions, targets = [], []
    try:
        for start in times:
            conditions.append(_frame(capture, source_fps, count, float(start), training_max_side))
            targets.append(np.stack([_frame(capture, source_fps, count, float(start) + (offset + 1) / settings.output_fps, training_max_side) for offset in range(settings.chunk_frames)]))
    finally:
        capture.release()
    np.savez_compressed(destination, conditions=np.stack(conditions), targets=np.stack(targets), times=times)
    # Each pair reads one condition frame plus one target chunk; the count is
    # therefore len(times) * (1 + chunk_frames) and endpoint is the last start.
    return PreparedImplicitPairs(len(times), len(times) * (1 + settings.chunk_frames), float(times[-1]))


def _read_progress(path: Path) -> dict:
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return {}


def _run_worker(command: list[str], root: Path, stage: str, detail: str, *, stage_id: str = ""):
    """Stream JSON progress emitted by a child so Gradio never blocks silently."""
    log = root / "worker.log"
    progress = root / "progress.json"
    with log.open("w") as output:
        process = subprocess.Popen(command, stdout=output, stderr=subprocess.STDOUT)
        last = None
        while process.poll() is None:
            time.sleep(0.2)
            record = _read_progress(progress)
            marker = json.dumps(record, sort_keys=True) if record else ""
            if marker != last:
                last = marker
                completed, total = record.get("step"), record.get("total")
                suffix = f" · {completed}/{total}" if completed and total else ""
                yield StageEvent(stage, "running", detail + suffix, metrics={**record, "log_file": log.name}, stage_id=stage_id)
            else:
                yield StageEvent(stage, "running", detail, metrics={"log_file": log.name}, stage_id=stage_id)
    if process.returncode:
        tail = "\n".join(log.read_text(errors="replace").splitlines()[-20:])
        raise RuntimeError(f"{stage} failed. Log: {log}\n{tail}")


def train(source: str, mode: str, settings):
    """Build data, run the chosen local optimizer, then atomically publish compatibility metadata."""
    if mode != "implicit":
        raise ValueError("Only Implicit 3D [self-trained] has a trainable Future View adapter.")
    root = training_dir(source, mode, settings)
    cached = training_manifest(source, mode, settings)
    # A manifest is only half the promise: it is written after the artifact, but
    # a pruned or manually deleted artifact dir must not be reported as a cache
    # hit. Resolve the artifact first, so a missing file falls through to the
    # full retrain path rather than a misleading training-complete trace.
    if cached and artifact_file(source, mode, settings):
        # A cache hit does no work: report the check as a single terminal (no
        # running tick) and mark the reused stages complete from what is on disk,
        # so no training row is left at Waiting.
        yield StageEvent("Check training cache", "complete",
                         "Reused the matching source-specific training artifact.",
                         metrics={"cache": "hit", "artifact": cached.get("model_artifact"), "settings_key": cached.get("settings_key")},
                         stage_id="training_cache")
        yield StageEvent("Prepare training data", "skipped",
                         "Reusing the existing source-specific training data; no pre-holdout sampling was needed.",
                         metrics={"cache": "hit"}, stage_id="training_pairs")
        yield StageEvent("Train selected method", "complete",
                         "Reusing the matching completed source-specific training artifact.",
                         metrics=cached, stage_id="train_lora")
        yield StageEvent("Save training artifact", "complete",
                         "Reused the cached source-specific training artifact.",
                         metrics=cached, stage_id="train_artifact")
        return
    root.mkdir(parents=True, exist_ok=True)
    trace_path = root / "training_trace.json"
    events = []
    def emit(event):
        events.append({"stage": event.stage, "status": event.status, "detail": event.detail, "metrics": event.metrics})
        temporary = trace_path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(events, indent=2)); temporary.replace(trace_path)
        return event
    yield emit(StageEvent("Check training cache", "running", "Checking for a reusable source-specific training artifact.", stage_id="training_cache"))
    yield emit(StageEvent("Check training cache", "complete", "No reusable artifact; training this source.", metrics={"cache": "miss"}, stage_id="training_cache"))
    yield emit(StageEvent("Prepare training data", "running", "Sampling only pre-holdout frames; the final inference history remains unseen.", stage_id="training_pairs"))
    prepared = prepare_implicit_pairs(source, settings, root / "pairs.npz")
    training_max_side = min(settings.max_side, getattr(settings, "training_max_side", settings.max_side))
    yield emit(StageEvent("Prepare training data", "complete",
                          f"Prepared {prepared.count} bounded pre-holdout observations.",
                          metrics={"pairs": prepared.count, "samples": prepared.samples, "fps": settings.output_fps,
                                   "endpoint": prepared.endpoint, "training_side": training_max_side},
                          stage_id="training_pairs"))
    model_artifact = "lora"
    command = [sys.executable, "-m", "GREENFIELD.features.future_implicit_train", "--pairs", str(root / "pairs.npz"), "--checkpoint", str(settings.checkpoint), "--out", str(root / model_artifact), "--progress", str(root / "progress.json"), "--steps", str(settings.training_lora_steps), "--rank", str(settings.training_lora_rank), "--fps", str(settings.output_fps), "--max-side", str(training_max_side)]
    detail = "Optimizing the source-specific Stable Video Diffusion LoRA adapter"
    for event in _run_worker(command, root, "Train selected method", detail, stage_id="train_lora"):
        yield emit(event)
    artifact = root / model_artifact
    if not (artifact.is_file() or artifact.is_dir()):
        raise RuntimeError("Training completed without its required reusable model artifact.")
    # The worker's last progress tick is the honest optimizer result; read it
    # after exit rather than inventing a loss when training wrote nothing.
    progress = _read_progress(root / "progress.json")
    yield emit(StageEvent("Train selected method", "complete",
                          "Fitted the source-specific LoRA adapter.",
                          metrics={"steps": progress.get("step", settings.training_lora_steps),
                                   "rank": getattr(settings, "training_lora_rank", None),
                                   "loss": progress.get("loss")},
                          stage_id="train_lora"))
    record = {"complete": True, "mode": mode, "source_sha256": source_digest(source), "samples": prepared.count, "model_artifact": model_artifact, "settings_key": training_key(source, mode, settings), "training_max_side": training_max_side}
    temporary = root / "manifest.json.tmp"
    temporary.write_text(json.dumps(record, indent=2))
    temporary.replace(root / "manifest.json")
    # Persisting the manifest is its own declared "Save training artifact" stage,
    # so the artifact is never reported as saved before it exists on disk.
    yield emit(StageEvent("Save training artifact", "complete",
                          "Saved a source-specific training artifact for matching Future View forecasts.",
                          metrics={"artifact": model_artifact, "manifest": "manifest.json", "samples": prepared.count},
                          stage_id="train_artifact"))
