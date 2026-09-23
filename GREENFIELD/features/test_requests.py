"""Unit tests for feature request validation and deterministic UI contracts."""

import pytest

from GREENFIELD.features.editing import EditingRequest
from GREENFIELD.features.future import FutureRequest
from GREENFIELD.features.selection import SelectionRequest


@pytest.mark.parametrize("invalid", [
    FutureRequest("other", 0),
    SelectionRequest("implicit", " "),
    EditingRequest("other", "make it bright", 0),
])
def test_invalid_requests_are_rejected(invalid) -> None:
    with pytest.raises(ValueError):
        invalid.validate()


def test_feature_pages_build_with_distinct_output_types() -> None:
    from GREENFIELD.features.pages import build_editing, build_future, build_selection
    assert len(build_future().blocks) > 0
    assert len(build_selection().blocks) > 0
    assert len(build_editing().blocks) > 0


def test_future_viewer_visibility_tracks_explicit_method_family() -> None:
    from GREENFIELD.features.pages import _future_method_change
    assert _future_method_change("explicit")["visible"] is True
    assert _future_method_change("implicit")["visible"] is False
