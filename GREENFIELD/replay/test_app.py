"""Regression coverage for dashboard file-serving authorization."""

from pathlib import Path

from GREENFIELD.replay.app import _allowed_paths, settings


def test_dashboard_allows_future_run_artifacts_without_broad_asset_access() -> None:
    allowed = {Path(path) for path in _allowed_paths()}
    assert settings.run_root in allowed
    assert settings.assets / "future" / "runs" in allowed
    assert settings.assets / "selection" / "runs" in allowed
    assert settings.assets / "editing" / "runs" in allowed
    assert settings.assets / "scenes" in allowed
    assert settings.assets / "checkpoints" not in allowed
