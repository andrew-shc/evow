import numpy as np
from GREENFIELD.features.future_explicit import extrapolate_motion, ordered_future_splats
from GREENFIELD.replay.splat import timeline_keyframe_durations, timeline_keyframe_indices
from GREENFIELD.app_core.splat_viewer import (
    MAX_SCENES_PER_VIEWER,
    _boundary_percent,
    _scene_groups,
    splat_html,
)
from GREENFIELD.features.future_settings import load_future_settings


def test_future_settings_have_fixed_30_second_output():
    settings = load_future_settings()
    assert settings.output_seconds == 30
    assert settings.output_fps == 6
    assert settings.output_frames == 180
    assert settings.observed_splat_timeline_fps == 1
    assert settings.forecast_splat_timeline_fps == 6


def test_constant_velocity_extrapolation_uses_recent_motion():
    motion = np.arange(18, dtype=np.float32).reshape(6, 1, 3)
    future = extrapolate_motion(motion, history_fps=12, output_fps=6, output_frames=3, fit_frames=6)
    assert future.shape == (3, 1, 3)
    assert np.all(future[1] > future[0])


def test_future_splats_are_ordered_and_complete(tmp_path):
    for name in (
        "splat_000.splat",
        "splat_001.splat",
        "forecast_splat_000.splat",
        "forecast_splat_001.splat",
        "forecast_splat_002.splat",
    ):
        (tmp_path / name).touch()
    paths = ordered_future_splats(tmp_path, observed_indices=(0, 1), forecast_indices=(0, 1, 2))
    assert [path.name for path in paths] == [
        "splat_000.splat",
        "splat_001.splat",
        "forecast_splat_000.splat",
        "forecast_splat_001.splat",
        "forecast_splat_002.splat",
    ]


def test_future_splat_timeline_requires_every_frame(tmp_path):
    (tmp_path / "splat_000.splat").touch()
    try:
        ordered_future_splats(tmp_path, observed_indices=(0,), forecast_indices=(0,))
    except RuntimeError as error:
        assert "complete" in str(error)
    else:
        raise AssertionError("Missing forecast splat should reject the explicit result.")


def test_splat_viewer_preloads_full_mixed_rate_timeline_without_control_clutter():
    paths = [f"frame_{index:03d}.splat" for index in range(MAX_SCENES_PER_VIEWER + 3)]
    html = splat_html(paths, [1 / 12, 1 / 6] + [1 / 6] * (len(paths) - 2), observed_frame_count=1)
    assert "const sourceGroups" in html
    assert "Promise.all(sourceGroups.map(preloadGroup))" in html
    assert "addSplatScenes(paths.map" in html
    assert "dynamicScene: false" in html
    assert "sources.slice(1)" not in html
    assert "viewer.splatMesh.scenes.forEach" not in html
    assert "updateTransforms" not in html
    assert "time.textContent" not in html
    assert "loadedScenes" not in html
    assert "#context" not in html
    assert "function frameIndexAt" in html
    assert "function cameraState" in html
    assert "function applyCameraState" in html
    assert "previous.viewer.stop()" in html
    assert "next.viewer.start()" in html
    assert "step=&quot;0.01&quot;" in html
    assert "observedBoundaryPercent = " in html
    assert "Reset view" not in html
    assert "document.getElementById('reset')" not in html
    assert "controls.reset()" not in html
    assert "ResizeObserver" in html
    assert "redrawForSize" in html
    assert "window.dispatchEvent(new Event(&#x27;resize&#x27;))" in html
    assert "#8e6a95" not in html


def test_full_timeline_groups_keep_every_frame_within_shader_limit():
    paths = [f"frame_{index:03d}.splat" for index in range(189)]
    groups = _scene_groups(paths)
    assert len(groups) == 6
    assert all(1 <= len(group) <= MAX_SCENES_PER_VIEWER for group in groups)
    assert [path for group in groups for path in group] == paths


def test_elapsed_time_boundary_uses_mixed_frame_durations():
    assert _boundary_percent([1 / 12, 1 / 6, 1 / 6], observed_frame_count=1) == 20.0
    assert _boundary_percent([1 / 12, 1 / 6], observed_frame_count=None) is None


def test_future_page_forwards_every_saved_splat_to_viewer(monkeypatch, tmp_path):
    from GREENFIELD.features import pages

    relative_paths = [f"explicit/{index:03d}.splat" for index in range(240)]
    for relative_path in relative_paths:
        path = tmp_path / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
    captured = {}

    def fake_splat_html(paths, durations, observed_frame_count):
        captured["paths"] = paths
        captured["durations"] = durations
        captured["observed_frame_count"] = observed_frame_count
        return "viewer"

    monkeypatch.setattr(pages, "splat_html", fake_splat_html)
    pages._future_viewer_update(
        {
            "viewer": {
                "splat_paths": relative_paths,
                "observed_frames": 72,
                "forecast_frames": 168,
                "observed_fps": 12,
                "forecast_fps": 6,
                "splat_durations": [1 / 6] * len(relative_paths),
            }
        },
        tmp_path,
    )

    assert len(captured["paths"]) == len(relative_paths)
    assert captured["durations"] == [1 / 6] * len(relative_paths)
    assert captured["observed_frame_count"] == 72


def test_explicit_result_persists_relative_viewer_metadata(monkeypatch, tmp_path):
    from types import SimpleNamespace
    from GREENFIELD.app_core.contracts import RunArtifacts
    from GREENFIELD.features import future as future_module
    from GREENFIELD.features.future_explicit import ExplicitFutureArtifacts

    settings = SimpleNamespace(
        output_seconds=30, output_frames=2, output_fps=6, observed_splat_timeline_fps=1, forecast_splat_timeline_fps=6
    )
    forecast = tmp_path / "explicit" / "forecast.mp4"
    splats = (
        tmp_path / "explicit" / "splat_000.splat",
        tmp_path / "explicit" / "forecast_splat_000.splat",
        tmp_path / "explicit" / "forecast_splat_001.splat",
    )
    monkeypatch.setattr(future_module, "load_future_settings", lambda: settings)

    def fake_explicit(frames, artifacts, settings):
        if False:
            yield None
        return ExplicitFutureArtifacts(forecast, splats, (0,), (0, 1), (1 / 12,), (1 / 6, 1 / 6))

    monkeypatch.setattr(future_module, "run_explicit", fake_explicit)

    events = list(
        future_module.run(
            [np.zeros((2, 2, 3), dtype=np.uint8)],
            12,
            future_module.FutureRequest("explicit", 0),
            RunArtifacts(tmp_path),
        )
    )
    result = events[-1]
    assert result.metadata["viewer"]["splat_paths"] == [
        "explicit/splat_000.splat",
        "explicit/forecast_splat_000.splat",
        "explicit/forecast_splat_001.splat",
    ]
    assert result.metadata["viewer"]["observed_fps"] == 1
    assert result.metadata["viewer"]["forecast_fps"] == 6
    assert result.metadata["viewer"]["keyframe_fps"] == 6


def test_future_viewer_keyframes_include_each_segment_end():
    observed = timeline_keyframe_indices(90, fps=12, timeline_fps=1)
    forecast = timeline_keyframe_indices(180, fps=6, timeline_fps=1)
    assert observed == (0, 12, 24, 36, 48, 60, 72, 84, 89)
    assert forecast[0] == 0 and forecast[-1] == 179
    assert len(observed) + len(forecast) == 40
    assert sum(timeline_keyframe_durations(observed, 12)) == 7.5
    assert sum(timeline_keyframe_durations(forecast, 6)) == 30
