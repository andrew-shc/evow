import numpy as np
from GREENFIELD.features.future_explicit import extrapolate_motion, ordered_future_splats
from GREENFIELD.replay.splat import timeline_keyframe_durations, timeline_keyframe_indices
from GREENFIELD.replay.splat_viewer import splat_html
from GREENFIELD.features.future_settings import load_future_settings
from GREENFIELD.features.pages import _bounded_future_timeline

def test_future_settings_have_fixed_30_second_output():
    settings = load_future_settings()
    assert settings.output_seconds == 30
    assert settings.output_fps == 6
    assert settings.output_frames == 180

def test_constant_velocity_extrapolation_uses_recent_motion():
    motion = np.arange(18, dtype=np.float32).reshape(6, 1, 3)
    future = extrapolate_motion(motion, history_fps=12, output_fps=6, output_frames=3, fit_frames=6)
    assert future.shape == (3, 1, 3)
    assert np.all(future[1] > future[0])


def test_future_splats_are_ordered_and_complete(tmp_path):
    for name in ("splat_000.splat", "splat_001.splat", "forecast_splat_000.splat", "forecast_splat_001.splat", "forecast_splat_002.splat"):
        (tmp_path / name).touch()
    paths = ordered_future_splats(tmp_path, observed_indices=(0, 1), forecast_indices=(0, 1, 2))
    assert [path.name for path in paths] == ["splat_000.splat", "splat_001.splat", "forecast_splat_000.splat", "forecast_splat_001.splat", "forecast_splat_002.splat"]


def test_future_splat_timeline_requires_every_frame(tmp_path):
    (tmp_path / "splat_000.splat").touch()
    try:
        ordered_future_splats(tmp_path, observed_indices=(0,), forecast_indices=(0,))
    except RuntimeError as error:
        assert "complete" in str(error)
    else:
        raise AssertionError("Missing forecast splat should reject the explicit result.")


def test_splat_viewer_supports_mixed_rate_future_timeline():
    html = splat_html(["observed.splat", "future.splat"], [1 / 12, 1 / 6],
                      "Observed reconstruction · Generated future (not observed footage)")
    assert "const frameDurations" in html
    assert "setTimeout(advance" in html
    assert "addSplatScene(sources[0]" in html
    assert "loadedScenes = 1" in html
    assert "loading selected 3D timeline" in html
    assert "addSplatScenes(sources.slice(1)" in html
    assert "viewer.removeSplatScenes" not in html
    assert "showLoadingUI: false" in html
    assert "not observed footage" in html


def test_explicit_result_persists_relative_viewer_metadata(monkeypatch, tmp_path):
    from types import SimpleNamespace
    from GREENFIELD.app_core.contracts import RunArtifacts
    from GREENFIELD.features import future as future_module
    from GREENFIELD.features.future_explicit import ExplicitFutureArtifacts

    settings = SimpleNamespace(output_seconds=30, output_frames=2, output_fps=6, splat_timeline_fps=1)
    forecast = tmp_path / "explicit" / "forecast.mp4"
    splats = (tmp_path / "explicit" / "splat_000.splat",
              tmp_path / "explicit" / "forecast_splat_000.splat",
              tmp_path / "explicit" / "forecast_splat_001.splat")
    monkeypatch.setattr(future_module, "load_future_settings", lambda: settings)
    def fake_explicit(frames, artifacts, settings):
        if False:
            yield None
        return ExplicitFutureArtifacts(forecast, splats, (0,), (0, 1), (1 / 12,), (1 / 6, 1 / 6))

    monkeypatch.setattr(future_module, "run_explicit", fake_explicit)

    events = list(future_module.run([np.zeros((2, 2, 3), dtype=np.uint8)], 12,
                                    future_module.FutureRequest("explicit", 0), RunArtifacts(tmp_path)))
    result = events[-1]
    assert result.metadata["viewer"]["splat_paths"] == ["explicit/splat_000.splat",
                                                         "explicit/forecast_splat_000.splat",
                                                         "explicit/forecast_splat_001.splat"]
    assert result.metadata["viewer"]["observed_fps"] == 1
    assert result.metadata["viewer"]["keyframe_fps"] == 1


def test_future_viewer_keyframes_include_each_segment_end():
    observed = timeline_keyframe_indices(90, fps=12, timeline_fps=1)
    forecast = timeline_keyframe_indices(180, fps=6, timeline_fps=1)
    assert observed == (0, 12, 24, 36, 48, 60, 72, 84, 89)
    assert forecast[0] == 0 and forecast[-1] == 179
    assert len(observed) + len(forecast) == 40
    assert sum(timeline_keyframe_durations(observed, 12)) == 7.5
    assert sum(timeline_keyframe_durations(forecast, 6)) == 30


def test_future_viewer_downsamples_dense_legacy_timeline_without_losing_duration(tmp_path):
    paths = [tmp_path / f"{index}.splat" for index in range(252)]
    durations = [1 / 6] * len(paths)
    selected, selected_durations = _bounded_future_timeline(paths, durations, observed_frames=72)
    assert len(selected) == 13
    assert selected[0] == paths[0]
    assert paths[71] in selected and paths[72] in selected
    assert selected[-1] == paths[-1]
    assert abs(sum(selected_durations) - sum(durations)) < 1e-9
