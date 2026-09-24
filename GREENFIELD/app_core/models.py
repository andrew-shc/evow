"""Explicit model availability checks; dashboard requests never download weights."""

from importlib.util import find_spec
from pathlib import Path


def require_package(package: str, extra: str) -> None:
    """Raise an actionable error instead of silently falling back to a heuristic."""
    if find_spec(package) is None:
        raise RuntimeError(f"{package} is required for this mode. Install the '{extra}' feature dependencies with: conda run -n evow python -m pip install -r CONFIGS/replay_requirements.txt")


def require_checkpoint(path: Path, label: str, setup_command: str = "conda run -n evow python -m GREENFIELD.features.future_setup") -> None:
    """Make model setup an owner action before GPU work begins."""
    if not path.is_dir() and not path.is_file():
        raise RuntimeError(f"{label} weights are not installed at {path}. Install them with: {setup_command}")
