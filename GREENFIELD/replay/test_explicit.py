"""Regression coverage for native Gaussian worker progress streaming."""

import json

from GREENFIELD.replay.explicit import _worker


class _FinishedAfterOnePoll:
    """Expose one running poll so the wrapper reads one progress record."""

    def __init__(self) -> None:
        self._polls = iter((None, 0))
        self.returncode = 0

    def poll(self) -> int | None:
        return next(self._polls)


def _stream_one_progress_record(monkeypatch, tmp_path, record: dict) -> list[dict]:
    """Run the subprocess wrapper through exactly one saved progress record."""
    from GREENFIELD.replay import explicit

    progress_path = tmp_path / "progress.json"
    progress_path.write_text(json.dumps(record))
    monkeypatch.setattr(explicit.subprocess, "Popen", lambda *args, **kwargs: _FinishedAfterOnePoll())
    monkeypatch.setattr(explicit.time, "sleep", lambda _seconds: None)
    return list(_worker(["gaussian-worker"], tmp_path / "worker.log", "4D Gaussian fitting",
                        "Optimizing persistent 3D Gaussians.", progress_path))


def test_worker_streams_training_progress_with_metrics(monkeypatch, tmp_path) -> None:
    """Optimizer records retain their loss and step update in the live trace."""
    updates = _stream_one_progress_record(monkeypatch, tmp_path, {
        "phase": "fit_gaussians", "step": 12, "total": 260,
        "loss": 0.12345, "gaussians": 384,
    })

    assert updates[-1]["metrics"] == {
        "step": 12, "image_loss": 0.12345, "gaussians": 384,
    }
    assert "step 12/260" in updates[-1]["detail"]


def test_worker_ignores_step_less_splat_export_progress(monkeypatch, tmp_path) -> None:
    """Export records must not be parsed as optimizer records after fitting."""
    updates = _stream_one_progress_record(monkeypatch, tmp_path, {
        "phase": "export_observed_keyframes", "completed": 1,
        "total": 13, "gaussians": 384,
    })

    assert len(updates) == 2
    assert updates[-1]["status"] == "running"
    assert updates[-1]["detail"].startswith("Optimizing persistent 3D Gaussians.")
