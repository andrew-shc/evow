import pytest
import numpy as np
import gradio as gr
from GREENFIELD.features.future_explicit import extrapolate_motion, ordered_future_splats
from GREENFIELD.replay.splat import timeline_keyframe_durations, timeline_keyframe_indices
from GREENFIELD.app_core.splat_viewer import (
    MAX_SCENES_PER_VIEWER,
    _boundary_percent,
    _scene_groups,
    empty_splat_html,
    splat_html,
)
from GREENFIELD.features.future_settings import load_future_settings


def test_future_settings_have_fixed_30_second_output():
    settings = load_future_settings()
    assert settings.output_seconds == 30
    assert settings.output_fps == 6
    assert settings.output_frames == 180
    assert settings.training_max_side == 512
    assert settings.observed_splat_timeline_fps == 1
    assert settings.forecast_splat_timeline_fps == 6
    # The newly editable knobs default to the values the workers previously
    # hardcoded, so an untouched run stays byte-for-byte equivalent.
    assert settings.render_yaw == 0.0
    assert settings.render_shift == 0.0
    assert settings.render_fov == 70.0
    assert settings.source_fov_degrees == 70.0
    assert settings.svd_decode_chunk_size == 1
    assert settings.training_max_pairs == 64
    assert settings.training_lora_steps == 500
    assert settings.training_lora_rank == 8


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


def test_empty_splat_viewer_keeps_its_pending_message_high_contrast():
    """An Explicit 3D placeholder must stay legible against its reserved frame."""
    html = empty_splat_html()

    assert "background:#101b27" in html
    assert "color:#ffffff" in html
    assert "An Explicit 3D scene will appear here after the run completes." in html


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


def test_training_key_is_source_and_settings_bound(tmp_path):
    from types import SimpleNamespace
    from GREENFIELD.features.future_training import training_key

    source = tmp_path / "source.mp4"; source.write_bytes(b"first")
    settings = SimpleNamespace(history_seconds=7.5, history_fps=12, chunk_frames=14, output_fps=6, max_side=1024, training_max_side=512, training_max_pairs=64, training_lora_steps=500, training_lora_rank=8)
    first = training_key(str(source), "implicit", settings)
    source.write_bytes(b"second")
    assert first != training_key(str(source), "implicit", settings)
    assert training_key(str(source), "implicit", settings) != training_key(str(source), "explicit", settings)
    before_resolution_change = training_key(str(source), "implicit", settings)
    settings.training_max_side = 448
    assert before_resolution_change != training_key(str(source), "implicit", settings)


def test_implicit_training_pairs_never_read_held_out_tail(monkeypatch, tmp_path):
    from types import SimpleNamespace
    from GREENFIELD.features import future_training

    class Capture:
        def release(self): pass
    settings = SimpleNamespace(history_seconds=7.5, chunk_frames=2, output_fps=2, training_max_pairs=3, max_side=8)
    sampled = []
    monkeypatch.setattr(future_training, "_eligible_end", lambda source, settings, target: (Capture(), 12, 999, 10.0))
    def frame(_capture, _fps, _count, seconds, _side):
        sampled.append(seconds); return np.zeros((2, 2, 3), dtype=np.uint8)
    monkeypatch.setattr(future_training, "_frame", frame)
    prepared = future_training.prepare_implicit_pairs("unused", settings, tmp_path / "pairs.npz")
    assert prepared.count == 3
    assert prepared.samples == 3 * (1 + settings.chunk_frames)
    assert prepared.endpoint == 10.0
    # end=10 and the two target frames span precisely one second; no sample can enter the held-out tail.
    assert max(sampled) <= 11.0


def test_only_complete_matching_training_artifacts_are_reused(tmp_path):
    from types import SimpleNamespace
    from GREENFIELD.features.future_training import artifact_file, training_dir

    source = tmp_path / "source.mp4"; source.write_bytes(b"source")
    settings = SimpleNamespace(root=tmp_path, history_seconds=7.5, history_fps=12, chunk_frames=14, output_fps=6, max_side=1024, training_lora_steps=500, training_lora_rank=8)
    root = training_dir(str(source), "implicit", settings); root.mkdir(parents=True)
    (root / "lora").mkdir()
    (root / "manifest.json").write_text('{"complete": false, "mode": "implicit", "model_artifact": "lora"}')
    assert artifact_file(str(source), "implicit", settings) is None
    (root / "manifest.json").write_text('{"complete": true, "mode": "implicit", "model_artifact": "lora"}')
    assert artifact_file(str(source), "implicit", settings) == root / "lora"


def test_train_reuses_cached_artifact_and_marks_data_preparation_skipped(monkeypatch, tmp_path):
    """A cache hit must not leave any training card at Waiting."""
    from types import SimpleNamespace
    from GREENFIELD.features import future_training

    cached = {"complete": True, "mode": "implicit", "model_artifact": "lora", "settings_key": "abc"}
    # Both lookups are stubbed so the cache-hit branch returns before any file or worker work.
    monkeypatch.setattr(future_training, "training_dir", lambda *_: tmp_path / "training")
    monkeypatch.setattr(future_training, "training_manifest", lambda *_: cached)
    monkeypatch.setattr(future_training, "artifact_file", lambda *_: tmp_path / "training" / "lora")

    events = list(future_training.train("source.mp4", "implicit", SimpleNamespace()))

    assert [event.stage for event in events] == [
        "Check training cache", "Prepare training data", "Train selected method", "Save training artifact",
    ]
    # The check completes without ever running; the reused stages are terminal.
    assert [event.status for event in events] == ["complete", "skipped", "complete", "complete"]
    assert events[0].metrics == {"cache": "hit", "artifact": "lora", "settings_key": "abc"}
    assert events[1].metrics == {"cache": "hit"}
    assert events[2].metrics == cached
    assert events[3].metrics == cached


def test_train_retrains_when_the_manifest_exists_but_the_artifact_is_missing(monkeypatch, tmp_path):
    """A pruned artifact is a cache miss, not a misleading training-complete trace."""
    from types import SimpleNamespace
    from GREENFIELD.app_core.contracts import StageEvent
    from GREENFIELD.features import future_training

    cached = {"complete": True, "mode": "implicit", "model_artifact": "lora", "settings_key": "abc"}
    monkeypatch.setattr(future_training, "training_dir", lambda *_: tmp_path / "training")
    monkeypatch.setattr(future_training, "training_manifest", lambda *_: cached)
    # The manifest is intact but the artifact file is gone: the real resolver
    # must report a miss so the run retrains instead of announcing completion.
    monkeypatch.setattr(future_training, "prepare_implicit_pairs",
                        lambda *_: future_training.PreparedImplicitPairs(count=1, samples=2, endpoint=1.0))

    def fake_worker(command, root, stage, detail, **_kwargs):
        (root / "lora").mkdir(parents=True, exist_ok=True)
        yield StageEvent(stage, "running", detail)

    monkeypatch.setattr(future_training, "_run_worker", fake_worker)
    settings = SimpleNamespace(
        root=tmp_path, history_seconds=7.5, history_fps=12, chunk_frames=1, output_fps=1,
        max_side=8, training_max_side=8, training_max_pairs=2, training_lora_steps=1,
        training_lora_rank=1, checkpoint=None,
    )
    source = tmp_path / "source.mp4"
    source.write_bytes(b"source")

    events = list(future_training.train(str(source), "implicit", settings))

    # It genuinely retrained: a cache-miss tick, not the reused-artifact terminal.
    cache_check = next(event for event in events if event.stage == "Check training cache" and event.status == "complete")
    assert cache_check.metrics == {"cache": "miss"}
    assert any(event.stage == "Train selected method" and event.status == "running" for event in events)
    assert events[-1].stage == "Save training artifact" and events[-1].status == "complete"
    assert not any(event.metrics.get("cache") == "hit" for event in events)


def test_future_training_emits_only_declared_stage_names(monkeypatch, tmp_path):
    """The artifact save must fold into a declared stage, never orphan a name."""
    from types import SimpleNamespace
    from GREENFIELD.app_core.contracts import StageEvent
    from GREENFIELD.features import future, future_training

    declared = {spec.name for spec in future.STAGES}
    source = tmp_path / "source.mp4"
    source.write_bytes(b"source")
    settings = SimpleNamespace(
        root=tmp_path, history_seconds=7.5, history_fps=12, chunk_frames=1, output_fps=1,
        max_side=8, training_max_side=8, training_max_pairs=2, training_lora_steps=1,
        training_lora_rank=1, checkpoint=None,
    )
    monkeypatch.setattr(future_training, "training_dir", lambda *_: tmp_path / "training")
    monkeypatch.setattr(future_training, "training_manifest", lambda *_: None)
    monkeypatch.setattr(
        future_training, "prepare_implicit_pairs",
        lambda *_: future_training.PreparedImplicitPairs(count=2, samples=4, endpoint=1.0),
    )

    def fake_worker(command, root, stage, detail, **_kwargs):
        (root / "lora").mkdir(parents=True, exist_ok=True)
        yield StageEvent(stage, "running", detail)

    monkeypatch.setattr(future_training, "_run_worker", fake_worker)

    events = list(future_training.train(str(source), "implicit", settings))

    assert {event.stage for event in events} <= declared
    # The saved artifact is its own declared terminal stage, after the manifest
    # is on disk and never before.
    assert events[-1].stage == "Save training artifact"
    assert events[-1].status == "complete"
    assert events[-1].metrics["artifact"] == "lora"


def test_self_trained_implicit_method_trains_then_uses_its_adapter(monkeypatch, tmp_path):
    from types import SimpleNamespace
    from GREENFIELD.app_core.contracts import RunArtifacts, StageEvent
    from GREENFIELD.features import future as future_module

    settings = SimpleNamespace(root=tmp_path, output_seconds=1, output_frames=1, output_fps=1)
    adapter = tmp_path / "lora"; adapter.mkdir()
    captured = {}

    def fake_train(source, mode, effective):
        captured["train"] = (source, mode, effective)
        yield StageEvent("Prepare training data", "complete", "prepared")
        yield StageEvent("Train selected method", "complete", "trained")

    def fake_steps(frame, count, seed, loaded_adapter, settings=None):
        captured["adapter"] = loaded_adapter
        captured["settings"] = settings
        yield 0, None
        yield 1, [np.zeros((2, 2, 3), dtype=np.uint8)]

    monkeypatch.setattr(future_module, "train", fake_train)
    monkeypatch.setattr(future_module, "artifact_file", lambda *_: adapter)
    monkeypatch.setattr(future_module, "training_dir", lambda *_: adapter.parent)
    monkeypatch.setattr(future_module, "generate_continuation_steps", fake_steps)
    monkeypatch.setattr(future_module, "write_video", lambda frames, path, fps: path)

    events = list(future_module.run([np.zeros((2, 2, 3), dtype=np.uint8)], 1, future_module.FutureRequest("self_trained", 3), RunArtifacts(tmp_path), settings, "source.mp4"))
    assert captured["train"][:2] == ("source.mp4", "implicit")
    assert captured["adapter"] == str(adapter)
    assert events[-1].metadata["mode"] == "self_trained"
    assert events[-1].metadata["training_artifact"] == str(adapter.parent)


def test_implicit_prior_marks_both_training_stages_skipped_before_running(monkeypatch, tmp_path):
    """Method-inapplicable training stages must be terminal, not left at Waiting."""
    from types import SimpleNamespace
    from GREENFIELD.app_core.contracts import RunArtifacts
    from GREENFIELD.features import future as future_module

    settings = SimpleNamespace(root=tmp_path, output_seconds=1, output_frames=1, output_fps=1)

    def fake_steps(frame, count, seed, loaded_adapter, settings=None):
        yield 0, None
        yield 1, [np.zeros((2, 2, 3), dtype=np.uint8)]

    monkeypatch.setattr(future_module, "generate_continuation_steps", fake_steps)
    monkeypatch.setattr(future_module, "write_video", lambda frames, path, fps: path)

    events = list(future_module.run(
        [np.zeros((2, 2, 3), dtype=np.uint8)], 1, future_module.FutureRequest("implicit", 3), RunArtifacts(tmp_path), settings
    ))
    skipped = [event for event in events if getattr(event, "status", None) == "skipped"]
    assert [(event.stage, event.status) for event in skipped] == [
        ("Prepare training data", "skipped"),
        ("Train selected method", "skipped"),
    ]
    first_running = next(index for index, event in enumerate(events) if getattr(event, "status", None) == "running")
    last_skipped = max(index for index, event in enumerate(events) if getattr(event, "status", None) == "skipped")
    assert last_skipped < first_running


def test_self_trained_mode_is_a_valid_future_request():
    from GREENFIELD.features.future import FutureRequest
    FutureRequest("self_trained", 0).validate()


def test_internal_flow_has_one_running_stage_for_a_serial_trace(monkeypatch):
    from GREENFIELD.app_core.contracts import StageSpec
    from GREENFIELD.app_core.flow import LEGACY_FLOW_VERSION, FeatureFlow, FlowStage
    from GREENFIELD.app_core.super_stages import SuperStage
    from GREENFIELD.features import pages

    def serial_flow() -> FeatureFlow:
        return FeatureFlow(
            "future",
            LEGACY_FLOW_VERSION,
            (
                FlowStage("prepare", StageSpec("Prepare", "first")),
                FlowStage("generate", StageSpec("Generate", "second")),
            ),
            (SuperStage("Prepare group", ("prepare",)), SuperStage("Generate group", ("generate",))),
            legacy_stage_ids=("Prepare", "Generate"),
        )

    monkeypatch.setitem(pages._FLOWS, "future", serial_flow)
    updates = pages._updates("future", {
        "mode": "implicit",
        "flow_version": LEGACY_FLOW_VERSION,
        # A v1 trace recorded display names only; a non-empty stage_id here would
        # be an unknown id under the collision-safe rule and get ignored.
        "events": [
            {"stage": "Prepare", "status": "running", "detail": "old tick", "elapsed_seconds": 1},
            {"stage": "Generate", "status": "running", "detail": "current tick", "elapsed_seconds": 2},
        ],
    })
    # Parent groups are emitted before their child stage updates. The earlier
    # running tick is completed visually once the serial flow reaches Generate.
    assert "evow-flow-group-complete" in updates[0]["elem_classes"]
    assert "evow-flow-group-running" in updates[1]["elem_classes"]
    assert "evow-stage-complete" in updates[2]["elem_classes"]
    assert "evow-stage-running" in updates[3]["elem_classes"]


def test_future_stream_yields_red_terminal_stage_before_gradio_error(monkeypatch, tmp_path):
    from types import SimpleNamespace
    import gradio as gr
    from GREENFIELD.app_core.contracts import StageEvent
    from GREENFIELD.features import pages
    from GREENFIELD.features.future import FutureRequest

    source = tmp_path / "source.mp4"
    source.write_bytes(b"video")
    monkeypatch.setattr(pages, "ASSETS", tmp_path / "ASSETS")
    monkeypatch.setattr(pages, "read_recent_window", lambda *_: ([np.zeros((2, 2, 3), dtype=np.uint8)], 12))

    def failing_run(*_args):
        yield StageEvent("Train selected method", "running", "worker started")
        raise RuntimeError("worker failed")

    monkeypatch.setattr(pages.future, "run", failing_run)
    settings = SimpleNamespace(history_seconds=7.5, history_fps=12, max_side=1024)
    stream = pages._execute_future_stream(str(source), "self_trained", FutureRequest("self_trained", 0), settings)
    next(stream)  # source starts
    next(stream)  # source completes
    next(stream)  # worker starts
    _result, failed_trace = next(stream)
    assert failed_trace["events"][-2]["stage"] == "Train selected method"
    assert failed_trace["events"][-2]["status"] == "error"
    with pytest.raises(gr.Error):
        next(stream)


def test_error_stage_uses_the_dashboard_failed_class():
    from GREENFIELD.app_core.contracts import StageSpec
    from GREENFIELD.app_core.ui import stage_label

    label, css_class = stage_label(0, StageSpec("Train", "purpose"), {"status": "error", "elapsed_seconds": 1})
    assert "Error" in label
    assert css_class == "evow-stage-failed"


def test_skipped_stage_uses_the_dashboard_skipped_class_without_duration():
    from GREENFIELD.app_core.contracts import StageSpec
    from GREENFIELD.app_core.ui import stage_label

    label, css_class = stage_label(0, StageSpec("Train", "purpose"), {"status": "skipped", "elapsed_seconds": 1})
    assert "Skipped" in label
    assert "—" in label
    assert css_class == "evow-stage-skipped"


def test_lora_training_conditions_clip_with_pil_and_vae_with_processed_pixels():
    import torch
    from PIL import Image
    from GREENFIELD.features.future_implicit_train import _conditioning_inputs, _target_pixels

    class VideoProcessor:
        def __init__(self): self.calls = []
        def preprocess(self, images):
            self.calls.append(images)
            count = len(images) if isinstance(images, list) else 1
            assert all(isinstance(image, Image.Image) for image in images) if isinstance(images, list) else isinstance(images, Image.Image)
            return torch.zeros((count, 3, 8, 8))

    class Pipe:
        def __init__(self): self.video_processor = VideoProcessor(); self.clip_input = None; self.vae_input = None
        def _encode_image(self, image, *_args):
            self.clip_input = image
            assert isinstance(image, Image.Image)
            return torch.zeros((1, 1, 4))
        def _encode_vae_image(self, pixels, *_args):
            self.vae_input = pixels
            return torch.zeros((1, 4, 1, 1))

    pipe = Pipe()
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    embeddings, latents = _conditioning_inputs(pipe, frame, "cpu", torch.float32)
    targets = _target_pixels(pipe, np.stack([frame, frame]), "cpu", torch.float32)
    assert isinstance(pipe.clip_input, Image.Image)
    assert pipe.vae_input.shape == (1, 3, 8, 8)
    assert embeddings.shape == (1, 1, 4) and latents.shape == (1, 4, 1, 1)
    assert targets.shape == (1, 2, 3, 8, 8)
    from GREENFIELD.features.future_implicit_train import _training_image
    assert _training_image(frame, 512).size == (512, 384)


def test_lora_training_initializes_and_samples_svd_continuous_timesteps():
    import torch
    from diffusers import EulerDiscreteScheduler
    from GREENFIELD.features.future_implicit_train import (
        _initialize_training_scheduler,
        _sample_training_timestep,
        _scale_training_input,
        _training_target,
    )

    scheduler = EulerDiscreteScheduler(
        num_train_timesteps=32,
        prediction_type="v_prediction",
        timestep_type="continuous",
        use_karras_sigmas=True,
    )
    _initialize_training_scheduler(scheduler, "cpu")
    timestep = _sample_training_timestep(scheduler, "cpu")
    latents = torch.zeros((1, 2, 4, 2, 2))
    noise = torch.ones_like(latents)
    noisy = scheduler.add_noise(latents, noise, timestep)
    prediction_target = _training_target(scheduler, latents, noise, timestep)
    model_input = _scale_training_input(scheduler, noisy, timestep)

    assert timestep.dtype.is_floating_point
    assert any(torch.equal(timestep, candidate.reshape(1)) for candidate in scheduler.timesteps)
    assert prediction_target.shape == latents.shape
    assert torch.isfinite(model_input).all()


def test_lora_wrapper_owns_fresh_hooks_instead_of_proxying_the_base_unet(monkeypatch):
    """Reproduce Accelerate's real ``PeftModel._hf_hook`` AttributeError and prove the fix avoids it.

    The production failure needs real objects: ``PeftModel.__getattr__`` proxies an unknown
    ``_hf_hook`` to the wrapped UNet, while Accelerate's ``remove_hook_from_module`` deletes the
    attribute from the wrapper itself. A fake wrapper would never exercise that proxy/``delattr``
    mismatch, so this test builds the smallest possible hooked UNet that real PEFT can wrap.
    """
    import torch
    import peft
    from accelerate.hooks import remove_hook_from_module
    from peft import LoraConfig, get_peft_model
    from GREENFIELD.features import model_adapters

    class StubHook:  # Stand-in for Accelerate's CPU-offload hook; only detach_hook is called.
        def detach_hook(self, module):
            pass

    class TinyUnet(torch.nn.Module):  # LoRA target "to_q" must exist.
        def __init__(self):
            super().__init__()
            self.to_q = torch.nn.Linear(4, 4)

        def forward(self, hidden):
            return self.to_q(hidden)

    config = LoraConfig(r=2, lora_alpha=4, target_modules=["to_q"])

    # 1) The production failure: wrapping an already-hooked UNet makes the wrapper proxy
    #    _hf_hook, so Accelerate's delattr raises instead of removing a hook it does not own.
    hooked_base = TinyUnet()
    hooked_base._hf_hook = StubHook()
    proxied = get_peft_model(hooked_base, config)
    assert "_hf_hook" not in proxied.__dict__ and hasattr(proxied, "_hf_hook")
    with pytest.raises(AttributeError, match="_hf_hook"):
        remove_hook_from_module(proxied, recurse=True)

    class Pipe:
        """Fake SVD pipeline mirroring Diffusers' hook bookkeeping.

        Diffusers' real ``remove_all_hooks`` walks ``components`` and deletes the hook from each
        module that owns one; ``enable_model_cpu_offload`` calls ``remove_all_hooks`` and then
        attaches a fresh hook, which is why its call log contains an extra "remove".
        """

        def __init__(self, unet):
            self.unet = unet
            self._all_hooks = [unet._hf_hook]
            self.calls = []

        @property
        def components(self):
            return {"unet": self.unet}

        def remove_all_hooks(self):
            self.calls.append("remove")
            for module in self.components.values():
                if hasattr(module, "_hf_hook"):
                    remove_hook_from_module(module, recurse=True)
            self._all_hooks = []

        def enable_model_cpu_offload(self):
            self.calls.append("enable")
            self.remove_all_hooks()
            self.unet._hf_hook = StubHook()
            self._all_hooks = [self.unet._hf_hook]

    # 2) The production helper detaches the base hook BEFORE wrapping, so the wrapper owns its
    #    own hook and Accelerate can remove it cleanly.
    base = TinyUnet()
    # Deterministic, strictly non-zero base weights make the restore check below
    # discriminating: a base layer left adapter-contaminated cannot coincidentally match.
    base.to_q.weight.data.copy_(torch.arange(16, dtype=torch.float32).reshape(4, 4) + 1.0)
    base.to_q.bias.data.copy_(torch.arange(4, dtype=torch.float32) + 1.0)
    base._hf_hook = StubHook()
    pipe = Pipe(base)
    # PeftModel is imported lazily inside the production helper, so patch the class it imports.
    monkeypatch.setattr(
        peft.PeftModel, "from_pretrained",
        staticmethod(lambda wrapped_base, _adapter: get_peft_model(wrapped_base, config)),
    )

    returned_base, offload_enabled = model_adapters._load_lora_with_offload_hooks(pipe, "adapter")
    assert returned_base is base and offload_enabled is True
    assert "_hf_hook" not in base.__dict__  # Base hook detached before the wrap.
    wrapper = pipe.unet
    assert wrapper is not base
    assert "_hf_hook" in wrapper.__dict__  # Wrapper owns the rebuilt hook.
    assert pipe.calls == ["remove", "enable", "remove"]
    remove_hook_from_module(wrapper, recurse=True)  # Must NOT raise (regression guard).

    # Record the true base weights and make the installed LoRA delta non-zero. PEFT
    # zero-inits the B matrix, so a merge-instead-of-unload "restore" would otherwise be
    # indistinguishable from a genuine unload; a non-zero delta exposes it as corrupted.
    original_weight = base.to_q.weight.detach().clone()
    original_bias = base.to_q.bias.detach().clone()
    lora_layer = wrapper.base_model.model.to_q
    for name in lora_layer.lora_A:
        lora_layer.lora_A[name].weight.data.fill_(1.0)
    for name in lora_layer.lora_B:
        lora_layer.lora_B[name].weight.data.fill_(1.0)

    model_adapters._restore_unet_with_offload_hooks(pipe, returned_base, offload_enabled)
    assert pipe.unet is base
    assert pipe.calls == ["remove", "enable", "remove", "remove", "enable", "remove"]
    assert "_hf_hook" in base.__dict__  # Restored base regains its offload hook.
    # Restoring must genuinely unload PEFT's in-place mutation. PeftModel.from_pretrained
    # replaced base.to_q with a LoRA layer; after restore it must be the original base layer
    # again, or the cached pipeline silently keeps adapter-contaminated weights.
    assert type(base.to_q) is torch.nn.Linear
    assert not isinstance(base.to_q, peft.tuners.lora.layer.Linear)
    assert not hasattr(base.to_q, "lora_A")

    # The restored layer must be the untouched base projection, not the non-zero LoRA delta
    # merged into its weights: identical parameters and a plain functional.linear response.
    assert torch.equal(base.to_q.weight, original_weight)
    assert torch.equal(base.to_q.bias, original_bias)
    probe = torch.arange(1, 5, dtype=torch.float32).reshape(1, 4)
    assert torch.equal(base(probe), torch.nn.functional.linear(probe, base.to_q.weight, base.to_q.bias))


def test_generation_holds_model_lifecycle_lock_while_suspended(monkeypatch):
    """The queue=False Clear-GPU button must wait for an in-flight generation.

    ``generate_continuation_steps`` acquires ``_MODEL_LIFECYCLE_LOCK`` before it touches
    the pipeline and releases it only in its outer ``finally``. Gradio 5.50.0 advances
    sync generators on arbitrary threadpool threads and closes them from the event-loop
    thread, so this test reproduces that split: worker thread A advances the generator
    and suspends at its first yield, then the main thread closes it. This FAILS against
    the old thread-owned ``RLock`` (the cross-thread outer-finally release raises
    ``RuntimeError: cannot release un-acquired lock``) and PASSES with the ownership-free
    semaphore.
    """
    import threading
    from GREENFIELD.features import model_adapters

    class FakePipe:
        def __call__(self, *args, **kwargs):
            raise AssertionError("generation must stay suspended at its first yield")

    monkeypatch.setattr(model_adapters, "stable_video_pipeline", lambda: FakePipe())
    # Teardown must not pull in the text/video model caches for this lock-only check.
    monkeypatch.setattr(model_adapters, "_clear_gpu_memory_unlocked", lambda: {})

    steps = model_adapters.generate_continuation_steps(
        np.zeros((2, 2, 3), dtype=np.uint8), count=1, seed=0
    )
    reached_yield = threading.Event()
    release_worker = threading.Event()
    worker_failures = []

    def advance_on_worker():
        try:
            assert next(steps) == (0, None)
        except BaseException as error:  # Surface worker-thread failures to the assertion.
            worker_failures.append(error)
        finally:
            reached_yield.set()
            release_worker.wait(timeout=5)

    worker = threading.Thread(target=advance_on_worker)
    worker.start()
    try:
        assert reached_yield.wait(timeout=5), "generator never reached its first yield"
        assert worker_failures == []
        # Main thread != worker thread, so a non-blocking acquire proves real exclusion.
        acquired_while_suspended = model_adapters._MODEL_LIFECYCLE_LOCK.acquire(blocking=False)
        if acquired_while_suspended:
            model_adapters._MODEL_LIFECYCLE_LOCK.release()
        assert not acquired_while_suspended, "lock must be held while the generator is suspended"

        # Close from the main thread while A still waits: the outer ``finally`` therefore
        # runs on a different thread than the one that acquired the lock.
        steps.close()
        assert model_adapters._MODEL_LIFECYCLE_LOCK.acquire(blocking=False), (
            "closing on a different thread must release the ownership-free semaphore"
        )
        model_adapters._MODEL_LIFECYCLE_LOCK.release()
    finally:
        release_worker.set()
        worker.join(timeout=5)

    assert not worker.is_alive()
    assert worker_failures == []


def test_generator_resumes_across_threads_while_holding_lock(monkeypatch):
    """The ownership-free guard lets a generator acquired on A resume on the main thread.

    Gradio's threadpool may run each ``next()`` on a different thread; a semaphore does
    not care which thread advances or releases it, so resuming while the guard is held
    must work.
    """
    import threading
    from PIL import Image
    from GREENFIELD.features import model_adapters

    class Frames:
        def __init__(self, images):
            self.frames = [images]

    class FakePipe:
        def __call__(self, *args, **kwargs):
            # One chunk of ``future_svd_chunk_frames`` (14) images satisfies count=1.
            return Frames([Image.new("RGB", (2, 2)) for _ in range(14)])

    monkeypatch.setattr(model_adapters, "stable_video_pipeline", lambda: FakePipe())
    monkeypatch.setattr(model_adapters, "_clear_gpu_memory_unlocked", lambda: {})

    steps = model_adapters.generate_continuation_steps(
        np.zeros((2, 2, 3), dtype=np.uint8), count=1, seed=0
    )
    reached_yield = threading.Event()
    release_worker = threading.Event()
    worker_failures = []

    def advance_on_worker():
        try:
            assert next(steps) == (0, None)
        except BaseException as error:
            worker_failures.append(error)
        finally:
            reached_yield.set()
            release_worker.wait(timeout=5)

    worker = threading.Thread(target=advance_on_worker)
    worker.start()
    try:
        assert reached_yield.wait(timeout=5), "generator never reached its first yield"
        assert worker_failures == []
        # Advancing here runs the generator body while the semaphore is held; thread
        # ownership would not allow this with an RLock-style guard. ``FakePipe``
        # exposes no honest step total, so the first item is a coarse tick.
        completed, frames = _next_chunk(steps)
        assert completed == 1 and len(frames) == 14
        # Suspended at its second yield, the generator still holds the guard.
        assert not model_adapters._MODEL_LIFECYCLE_LOCK.acquire(blocking=False)
    finally:
        release_worker.set()
        worker.join(timeout=5)

    assert not worker.is_alive()
    assert worker_failures == []
    steps.close()  # Release the still-held guard so later tests are not blocked.
    assert model_adapters._MODEL_LIFECYCLE_LOCK.acquire(blocking=False)
    model_adapters._MODEL_LIFECYCLE_LOCK.release()


@pytest.mark.parametrize("teardown_name", ["_restore_unet_with_offload_hooks", "clear_gpu_memory"])
def test_teardown_failure_does_not_mask_an_inference_error(monkeypatch, teardown_name):
    """A hook or allocator failure during teardown must not replace the real error."""
    from GREENFIELD.features import model_adapters

    class FailingPipe:
        def __call__(self, *args, **kwargs):
            raise ValueError("inference exploded")

    def fail(*args, **kwargs):
        raise RuntimeError(f"{teardown_name} exploded")

    def no_op(*args, **kwargs):
        return None

    monkeypatch.setattr(model_adapters, "stable_video_pipeline", lambda: FailingPipe())
    monkeypatch.setattr(model_adapters, "_load_lora_with_offload_hooks", lambda pipe, adapter: (object(), False))
    monkeypatch.setattr(
        model_adapters, "_restore_unet_with_offload_hooks",
        fail if teardown_name == "_restore_unet_with_offload_hooks" else no_op,
    )
    monkeypatch.setattr(
        model_adapters, "_clear_gpu_memory_unlocked",
        fail if teardown_name == "clear_gpu_memory" else no_op,
    )

    steps = model_adapters.generate_continuation_steps(np.zeros((2, 2, 3), dtype=np.uint8), 1, 0, adapter="adapter")
    assert next(steps) == (0, None)
    with pytest.raises(ValueError, match="inference exploded"):
        next(steps)


@pytest.mark.parametrize("teardown_name", ["_restore_unet_with_offload_hooks", "clear_gpu_memory"])
def test_teardown_failure_propagates_after_clean_generation(monkeypatch, teardown_name):
    """With no inference error in flight, a teardown failure must still surface."""
    from PIL import Image
    from GREENFIELD.features import model_adapters

    class Frames:
        def __init__(self, images):
            self.frames = [images]

    class FakePipe:
        def __call__(self, *args, **kwargs):
            return Frames([Image.new("RGB", (2, 2))])

    def fail(*args, **kwargs):
        raise RuntimeError(f"{teardown_name} exploded")

    def no_op(*args, **kwargs):
        return None

    monkeypatch.setattr(model_adapters, "stable_video_pipeline", lambda: FakePipe())
    monkeypatch.setattr(model_adapters, "_load_lora_with_offload_hooks", lambda pipe, adapter: (object(), False))
    monkeypatch.setattr(
        model_adapters, "_restore_unet_with_offload_hooks",
        fail if teardown_name == "_restore_unet_with_offload_hooks" else no_op,
    )
    monkeypatch.setattr(
        model_adapters, "_clear_gpu_memory_unlocked",
        fail if teardown_name == "clear_gpu_memory" else no_op,
    )

    steps = model_adapters.generate_continuation_steps(np.zeros((2, 2, 3), dtype=np.uint8), 1, 0, adapter="adapter")
    assert next(steps) == (0, None)
    # ``FakePipe`` accepts the callback but exposes no step total, so the chunk is
    # preceded by one coarse tick rather than per-step ticks.
    completed, frames = _next_chunk(steps)
    assert completed == 1 and len(frames) == 1
    with pytest.raises(RuntimeError, match=f"{teardown_name} exploded"):
        next(steps)


def test_model_load_failure_propagates_without_masking(monkeypatch):
    """A pipeline-load failure must surface unchanged and still release the lifecycle lock.

    ``pipe = stable_video_pipeline()`` sits *outside* the inner ``try`` whose ``finally``
    does ``del pipe``, so a load failure never enters that teardown block and cannot
    produce an ``UnboundLocalError``. The outer ``finally`` must still release the
    semaphore; otherwise every later generation and Clear-GPU call deadlocks.
    """
    from GREENFIELD.features import model_adapters

    def fail_to_load():
        raise RuntimeError("model load failed")

    # No real model, no GPU: the loader is the only thing that fails.
    monkeypatch.setattr(model_adapters, "stable_video_pipeline", fail_to_load)

    steps = model_adapters.generate_continuation_steps(np.zeros((2, 2, 3), dtype=np.uint8), count=1, seed=0)
    with pytest.raises(RuntimeError, match="model load failed") as excinfo:
        next(steps)
    # Bind the exact type so a masking UnboundLocalError/NameError could never satisfy this.
    assert type(excinfo.value) is RuntimeError
    assert not isinstance(excinfo.value, UnboundLocalError)

    # The outer ``finally`` ran: a non-blocking acquire must succeed, proving the
    # semaphore was released even though the inner teardown never executed.
    acquired = model_adapters._MODEL_LIFECYCLE_LOCK.acquire(blocking=False)
    try:
        assert acquired, "outer finally must release the lifecycle semaphore on load failure"
    finally:
        if acquired:
            model_adapters._MODEL_LIFECYCLE_LOCK.release()


# --- Exhaustive editable Future View configuration ---------------------------
#
# The Configuration panel mirrors ``pages._EDITABLE_FUTURE_FIELDS`` exactly. These
# tests pin that the panel exposes every knob with the tracked default, that each
# control carries a short description plus the shared config-field class, that
# defaults reproduce the base settings, that out-of-range values fail by field
# name, and that the render pose, decode batch, and LoRA fingerprint all receive
# the effective per-run values.

# Panel label -> FutureSettings field name, grouped by the Configuration panel's
# visibility sections. "Seed" is intentionally absent: it rides on the request,
# not the editable settings, and stays visible with the implicit rollout group.
_FUTURE_PANEL_DEFAULTS = {
    # History (always visible)
    "History (s)": "history_seconds",
    "History FPS": "history_fps",
    # Forecast (always visible)
    "Output (s)": "output_seconds",
    "Output FPS": "output_fps",
    # Reconstruction (always visible)
    "Max side": "max_side",
    # Explicit view (explicit-only)
    "Observed FPS": "observed_splat_timeline_fps",
    "Forecast FPS": "forecast_splat_timeline_fps",
    "Motion fit": "motion_fit_frames",
    "Yaw": "render_yaw",
    "Shift": "render_shift",
    "FOV": "render_fov",
    "Source FOV": "source_fov_degrees",
    # Implicit rollout (implicit + self-trained)
    "Chunk frames": "chunk_frames",
    "SVD chunk": "svd_decode_chunk_size",
    # Self-trained (self-trained-only)
    "Training side": "training_max_side",
    "LoRA steps": "training_lora_steps",
    "LoRA rank": "training_lora_rank",
    "Training pairs": "training_max_pairs",
}


def _in_configuration_panel(component) -> bool:
    """Walk up the layout tree so page-wide controls (e.g. capture seconds) are excluded."""
    ancestor = getattr(component, "parent", None)
    while ancestor is not None:
        if isinstance(ancestor, gr.Accordion) and ancestor.label == "Configuration":
            return True
        ancestor = getattr(ancestor, "parent", None)
    return False


def _future_config_controls(page) -> dict:
    return {
        component.label: component
        for component in page.blocks.values()
        if isinstance(component, (gr.Number, gr.Slider)) and _in_configuration_panel(component)
    }


def test_future_panel_exposes_every_editable_default():
    from GREENFIELD.features import pages

    settings = load_future_settings()
    controls = _future_config_controls(pages.build_future())
    assert set(_FUTURE_PANEL_DEFAULTS.values()) == set(pages._EDITABLE_FUTURE_FIELDS)
    assert set(_FUTURE_PANEL_DEFAULTS) <= set(controls)
    for label, field in _FUTURE_PANEL_DEFAULTS.items():
        assert controls[label].value == getattr(settings, field), label


def test_future_panel_controls_expose_a_description_and_field_class():
    from GREENFIELD.features import pages

    controls = _future_config_controls(pages.build_future())
    assert controls
    for component in controls.values():
        # The visible Markdown sibling owns the label and ⓘ popover; the bare
        # Gradio control remains directly editable in the right-hand cell.
        assert "evow-config-control" in component.elem_classes, component.label
        assert component.container is False
        assert component.show_label is False
        assert "evow-config-row" in component.parent.elem_classes


def test_future_config_visibility_gates_groups_by_methodology():
    """Future View reveals only the groups its selected method actually reads.

    Group order: History, Forecast, Reconstruction, Explicit view, Implicit
    rollout, Self-trained. The three shared groups stay visible in every mode;
    Explicit view is exclusive to Explicit 3D, the SVD rollout is shared by the
    implicit family, and the LoRA-fit group exists only for self-trained.
    """
    from GREENFIELD.features import pages

    assert [update["visible"] for update in pages._future_config_visibility("explicit")] == [True, True, True, True, False, False]
    assert [update["visible"] for update in pages._future_config_visibility("implicit")] == [True, True, True, False, True, False]
    assert [update["visible"] for update in pages._future_config_visibility("self_trained")] == [True, True, True, False, True, True]


def test_future_config_visibility_never_rewrites_control_values():
    """A visibility update must carry no ``value``, or a mode switch would reset knobs."""
    from GREENFIELD.features import pages

    for mode in ("explicit", "implicit", "self_trained"):
        updates = pages._future_config_visibility(mode)
        assert len(updates) == 6
        assert all("value" not in update for update in updates), mode


def test_future_page_starts_with_only_explicit_groups_visible():
    """The built page's default Explicit 3D mode hides the implicit/self-trained groups."""
    from GREENFIELD.features import pages

    page = pages.build_future()
    registration = next(
        fn for fn in page.fns.values()
        if getattr(fn.fn, "__name__", "") == "_future_config_visibility"
    )
    assert [component.visible for component in registration.outputs] == [True, True, True, True, False, False]


def test_future_effective_settings_reproduce_tracked_defaults():
    """Sending every untouched panel value back reproduces the base snapshot."""
    from GREENFIELD.features import pages

    settings = load_future_settings()
    values = [getattr(settings, field) for field in pages._EDITABLE_FUTURE_FIELDS]
    assert pages._future_effective_settings(settings, *values) == settings


@pytest.mark.parametrize(
    "field, out_of_range",
    [
        ("history_seconds", 0.1), ("history_fps", 61), ("output_seconds", 0),
        ("output_fps", 31), ("observed_splat_timeline_fps", 61),
        ("forecast_splat_timeline_fps", 0), ("chunk_frames", 65),
        ("motion_fit_frames", 0), ("max_side", 32), ("training_max_side", 5000),
        ("training_max_pairs", 0), ("training_lora_steps", 10001),
        ("training_lora_rank", 129), ("render_yaw", 181), ("render_shift", 1.5),
        ("render_fov", 34), ("source_fov_degrees", 111), ("svd_decode_chunk_size", 9),
    ],
)
def test_future_effective_settings_reject_out_of_range_by_name(field, out_of_range):
    from GREENFIELD.features import pages

    settings = load_future_settings()
    values = [getattr(settings, name) for name in pages._EDITABLE_FUTURE_FIELDS]
    values[pages._EDITABLE_FUTURE_FIELDS.index(field)] = out_of_range
    with pytest.raises(ValueError, match=field):
        pages._future_effective_settings(settings, *values)


def test_future_effective_settings_reject_bool_for_numeric_fields():
    """A bool must never pass as 1 / 1.0 for either an int or a float knob."""
    from GREENFIELD.features import pages

    settings = load_future_settings()
    for field in ("history_fps", "history_seconds", "max_side"):
        values = [getattr(settings, name) for name in pages._EDITABLE_FUTURE_FIELDS]
        values[pages._EDITABLE_FUTURE_FIELDS.index(field)] = True
        with pytest.raises(ValueError, match=field):
            pages._future_effective_settings(settings, *values)


def test_future_effective_settings_reject_non_numeric_with_field_name():
    """A bare conversion error must become a named failure the owner can locate."""
    from GREENFIELD.features import pages

    settings = load_future_settings()
    values = [getattr(settings, name) for name in pages._EDITABLE_FUTURE_FIELDS]
    values[pages._EDITABLE_FUTURE_FIELDS.index("history_fps")] = "abc"
    with pytest.raises(ValueError, match="history_fps"):
        pages._future_effective_settings(settings, *values)


@pytest.mark.parametrize("non_finite", [float("nan"), float("inf"), float("-inf")])
def test_future_effective_settings_reject_non_finite_floats(non_finite):
    """NaN/inf must be refused before the range check can be fooled by NaN."""
    from GREENFIELD.features import pages

    settings = load_future_settings()
    values = [getattr(settings, name) for name in pages._EDITABLE_FUTURE_FIELDS]
    values[pages._EDITABLE_FUTURE_FIELDS.index("history_seconds")] = non_finite
    with pytest.raises(ValueError, match="history_seconds"):
        pages._future_effective_settings(settings, *values)


def test_future_effective_settings_accept_numeric_string_for_int_field():
    """Gradio can hand an integer knob back as text; ``"30"`` must become a strict int."""
    from GREENFIELD.features import pages

    settings = load_future_settings()
    values = [getattr(settings, name) for name in pages._EDITABLE_FUTURE_FIELDS]
    values[pages._EDITABLE_FUTURE_FIELDS.index("history_fps")] = "30"
    effective = pages._future_effective_settings(settings, *values)
    assert effective.history_fps == 30
    assert type(effective.history_fps) is int


@pytest.mark.parametrize("text", ["nan", "inf", "-inf"])
def test_future_effective_settings_reject_non_finite_numeric_strings(text):
    """A string spelling NaN/inf parses to a float and must still fail by field name."""
    from GREENFIELD.features import pages

    settings = load_future_settings()
    values = [getattr(settings, name) for name in pages._EDITABLE_FUTURE_FIELDS]
    values[pages._EDITABLE_FUTURE_FIELDS.index("history_seconds")] = text
    with pytest.raises(ValueError, match="history_seconds"):
        pages._future_effective_settings(settings, *values)


def test_future_effective_settings_non_finite_error_names_the_field_exactly():
    """Pin the full message so the owner sees the exact control and value."""
    from GREENFIELD.features import pages

    settings = load_future_settings()
    values = [getattr(settings, name) for name in pages._EDITABLE_FUTURE_FIELDS]
    values[pages._EDITABLE_FUTURE_FIELDS.index("history_seconds")] = "nan"
    with pytest.raises(ValueError) as excinfo:
        pages._future_effective_settings(settings, *values)
    assert str(excinfo.value) == "Future View history_seconds must be a finite number (got nan)."


def test_future_effective_settings_reject_fractional_integer():
    """1.9 for an int knob must fail by name instead of being truncated to 1."""
    from GREENFIELD.features import pages

    settings = load_future_settings()
    values = [getattr(settings, name) for name in pages._EDITABLE_FUTURE_FIELDS]
    values[pages._EDITABLE_FUTURE_FIELDS.index("history_fps")] = 1.9
    with pytest.raises(ValueError, match="history_fps"):
        pages._future_effective_settings(settings, *values)


def test_future_effective_settings_normalize_integral_float_to_int():
    """An integral float is a valid whole number and must be stored as a strict int."""
    from GREENFIELD.features import pages

    settings = load_future_settings()
    values = [getattr(settings, name) for name in pages._EDITABLE_FUTURE_FIELDS]
    values[pages._EDITABLE_FUTURE_FIELDS.index("history_fps")] = 30.0
    effective = pages._future_effective_settings(settings, *values)
    assert effective.history_fps == 30
    assert type(effective.history_fps) is int


def test_future_effective_settings_keep_fractional_floats():
    """Float knobs must still accept genuinely fractional values."""
    from GREENFIELD.features import pages

    settings = load_future_settings()
    values = [getattr(settings, name) for name in pages._EDITABLE_FUTURE_FIELDS]
    values[pages._EDITABLE_FUTURE_FIELDS.index("history_seconds")] = 7.25
    effective = pages._future_effective_settings(settings, *values)
    assert effective.history_seconds == 7.25
    assert type(effective.history_seconds) is float


def test_future_render_pose_reaches_the_explicit_worker(tmp_path):
    """The editable virtual camera pose must appear in the Gaussian worker command."""
    from types import SimpleNamespace
    from GREENFIELD.features.future_explicit import gaussian_worker_command

    settings = SimpleNamespace(
        render_yaw=12.0, render_shift=-0.25, render_fov=80.0, source_fov_degrees=65.0,
        output_frames=8, output_fps=2, motion_fit_frames=3, history_fps=12,
        observed_splat_timeline_fps=1, forecast_splat_timeline_fps=2,
    )
    command = gaussian_worker_command(tmp_path, tmp_path / "out", tmp_path / "depth.npz", settings)

    def argument(flag):
        return command[command.index(flag) + 1]

    assert argument("--render-yaw") == "12.0"
    assert argument("--render-shift") == "-0.25"
    assert argument("--render-fov") == "80.0"
    assert argument("--source-fov") == "65.0"


def test_future_decode_chunk_size_reaches_the_svd_pipeline(monkeypatch):
    """The effective VAE decode batch must reach the pipeline call, not its old constant."""
    from types import SimpleNamespace
    from PIL import Image
    from GREENFIELD.features import model_adapters

    captured = {}

    class Frames:
        def __init__(self, images):
            self.frames = [images]

    class FakePipe:
        def __call__(self, *args, **kwargs):
            captured.update(kwargs)
            return Frames([Image.new("RGB", (2, 2))])

    monkeypatch.setattr(model_adapters, "stable_video_pipeline", lambda: FakePipe())
    monkeypatch.setattr(model_adapters, "_clear_gpu_memory_unlocked", lambda: {})

    settings = SimpleNamespace(chunk_frames=14, svd_decode_chunk_size=4)
    steps = model_adapters.generate_continuation_steps(
        np.zeros((2, 2, 3), dtype=np.uint8), count=1, seed=0, settings=settings
    )
    assert next(steps) == (0, None)
    next(steps)
    assert captured["decode_chunk_size"] == 4
    steps.close()


# --- Real per-denoise-step SVD ticks -----------------------------------------
#
# ``generate_continuation_steps`` forwards Diffusers' ``callback_on_step_end`` as
# streamed ``svd_generate`` ticks. These fakes invoke the callback for real, so
# the tests prove the emitted ``step/total/timestep/chunk/latent_shape`` are the
# pipeline's own values, that ticks arrive before the completed chunk, that a
# worker failure propagates after the thread join, and that a pipeline without
# the callback degrades to one coarse tick instead of crashing.


class _FakeLatents:
    def __init__(self, shape):
        self.shape = tuple(shape)

class _Frames:
    def __init__(self, images):
        self.frames = [images]


def _fake_frames(num_frames):
    from PIL import Image

    return _Frames([Image.new("RGB", (2, 2)) for _ in range(num_frames)])


def _svd_settings(chunk_frames, svd_decode_chunk_size=1):
    from types import SimpleNamespace

    return SimpleNamespace(chunk_frames=chunk_frames, svd_decode_chunk_size=svd_decode_chunk_size)


def _next_chunk(steps):
    """Advance to the next ``(completed, frames)`` chunk, skipping progress ticks."""
    from GREENFIELD.app_core.contracts import StageEvent

    while True:
        item = next(steps)
        if not isinstance(item, StageEvent):
            return item


class StepCallbackPipe:
    """Fake SVD pipeline that really calls the Diffusers step callback."""

    def __init__(self, total_steps=4, fail_at=None):
        self.num_timesteps = total_steps
        self.fail_at = fail_at
        self.callback_tensor_inputs = None

    def __call__(self, initial_image, num_frames, generator, decode_chunk_size, callback_on_step_end, callback_on_step_end_tensor_inputs):
        self.callback_tensor_inputs = callback_on_step_end_tensor_inputs
        latents = _FakeLatents((1, 4, num_frames, 8, 8))
        for step in range(self.num_timesteps):
            if self.fail_at is not None and step == self.fail_at:
                raise RuntimeError("denoise exploded")
            callback_on_step_end(self, step, 1000 - step, {"latents": latents})
        return _fake_frames(num_frames)


class BlockingStepPipe(StepCallbackPipe):
    """Emit one tick, then block inside the pipeline until the test releases it."""

    def __init__(self):
        import threading

        super().__init__(total_steps=2)
        self.entered = threading.Event()
        self.release = threading.Event()

    def __call__(self, initial_image, num_frames, generator, decode_chunk_size, callback_on_step_end, callback_on_step_end_tensor_inputs):
        callback_on_step_end(self, 0, 999.0, {"latents": _FakeLatents((1, 4, num_frames, 8, 8))})
        self.entered.set()
        self.release.wait(timeout=5)
        return _fake_frames(num_frames)


class NoCallbackPipe:
    """An older pipeline whose ``__call__`` accepts no step callback at all."""

    def __call__(self, initial_image, num_frames, generator, decode_chunk_size):
        return _fake_frames(num_frames)


def test_generate_steps_streams_real_denoise_ticks_before_the_chunk(monkeypatch):
    from GREENFIELD.app_core.contracts import StageEvent
    from GREENFIELD.features import model_adapters

    pipe = StepCallbackPipe(total_steps=4)
    monkeypatch.setattr(model_adapters, "stable_video_pipeline", lambda: pipe)
    monkeypatch.setattr(model_adapters, "_clear_gpu_memory_unlocked", lambda: {})

    steps = model_adapters.generate_continuation_steps(
        np.zeros((2, 2, 3), dtype=np.uint8), count=2, seed=0, settings=_svd_settings(chunk_frames=2)
    )
    assert next(steps) == (0, None)
    ticks = []
    while True:
        item = next(steps)
        if isinstance(item, StageEvent):
            ticks.append(item)  # Ticks arrive before the completed chunk.
        else:
            chunk_result = item
            break

    assert pipe.callback_tensor_inputs == ["latents"]
    assert [(tick.metrics["step"], tick.metrics["total"]) for tick in ticks] == [(0, 4), (1, 4), (2, 4), (3, 4)]
    assert all(tick.metrics["chunk"] == 1 for tick in ticks)
    assert all(tick.metrics["latent_shape"] == [1, 4, 2, 8, 8] for tick in ticks)
    assert [tick.metrics["timestep"] for tick in ticks] == [1000.0, 999.0, 998.0, 997.0]
    assert all(tick.stage_id == "svd_generate" and tick.status == "running" for tick in ticks)
    assert chunk_result[0] == 2 and len(chunk_result[1]) == 2
    steps.close()


def test_ticks_are_truly_live_while_the_pipeline_still_runs(monkeypatch):
    """The first tick must be observable while the pipeline thread is still blocked."""
    from GREENFIELD.app_core.contracts import StageEvent
    from GREENFIELD.features import model_adapters

    pipe = BlockingStepPipe()
    monkeypatch.setattr(model_adapters, "stable_video_pipeline", lambda: pipe)
    monkeypatch.setattr(model_adapters, "_clear_gpu_memory_unlocked", lambda: {})

    steps = model_adapters.generate_continuation_steps(
        np.zeros((2, 2, 3), dtype=np.uint8), count=2, seed=0, settings=_svd_settings(chunk_frames=2)
    )
    assert next(steps) == (0, None)
    tick = next(steps)
    assert isinstance(tick, StageEvent) and tick.metrics["step"] == 0
    assert pipe.entered.wait(timeout=5)
    assert not pipe.release.is_set(), "tick arrived before the pipeline returned, so it streamed live"

    pipe.release.set()
    completed, frames = next(steps)
    assert completed == 2 and len(frames) == 2
    steps.close()


def test_on_step_callback_observes_every_reported_step(monkeypatch):
    from GREENFIELD.features import model_adapters

    pipe = StepCallbackPipe(total_steps=3)
    monkeypatch.setattr(model_adapters, "stable_video_pipeline", lambda: pipe)
    monkeypatch.setattr(model_adapters, "_clear_gpu_memory_unlocked", lambda: {})

    observed = []
    steps = model_adapters.generate_continuation_steps(
        np.zeros((2, 2, 3), dtype=np.uint8), count=2, seed=0, settings=_svd_settings(chunk_frames=2), on_step=observed.append
    )
    assert next(steps) == (0, None)
    for _ in steps:
        pass

    assert [record["step"] for record in observed] == [0, 1, 2]
    assert all(record["total"] == 3 and record["chunk"] == 1 for record in observed)
    assert observed[0]["latent_shape"] == [1, 4, 2, 8, 8]


def test_step_ticks_throttle_but_keep_real_step_and_total(monkeypatch):
    from GREENFIELD.app_core.contracts import StageEvent
    from GREENFIELD.features import model_adapters

    pipe = StepCallbackPipe(total_steps=25)
    monkeypatch.setattr(model_adapters, "stable_video_pipeline", lambda: pipe)
    monkeypatch.setattr(model_adapters, "_clear_gpu_memory_unlocked", lambda: {})

    steps = model_adapters.generate_continuation_steps(
        np.zeros((2, 2, 3), dtype=np.uint8), count=2, seed=0, settings=_svd_settings(chunk_frames=2)
    )
    assert next(steps) == (0, None)
    ticks = []
    while True:
        item = next(steps)
        if isinstance(item, StageEvent):
            ticks.append(item)
        else:
            break

    steps_seen = [tick.metrics["step"] for tick in ticks]
    assert len(ticks) <= model_adapters._MAX_STEP_TICKS_PER_CHUNK
    assert steps_seen[0] == 0 and steps_seen[-1] == 24
    # Thinning keeps the true indices: total stays 25 for every emitted tick.
    assert all(tick.metrics["total"] == 25 for tick in ticks)
    steps.close()


def test_worker_failure_propagates_after_joining_the_denoise_thread(monkeypatch):
    import threading
    from GREENFIELD.features import model_adapters

    pipe = StepCallbackPipe(total_steps=4, fail_at=2)
    monkeypatch.setattr(model_adapters, "stable_video_pipeline", lambda: pipe)
    monkeypatch.setattr(model_adapters, "_clear_gpu_memory_unlocked", lambda: {})

    steps = model_adapters.generate_continuation_steps(
        np.zeros((2, 2, 3), dtype=np.uint8), count=2, seed=0, settings=_svd_settings(chunk_frames=2)
    )
    assert next(steps) == (0, None)
    with pytest.raises(RuntimeError, match="denoise exploded"):
        while True:
            next(steps)

    # The generator's ``finally`` joined the worker, so no denoise thread survives.
    assert [thread for thread in threading.enumerate() if thread.name == "svd-denoise"] == []
    steps.close()


def test_pipeline_without_step_callback_degrades_to_one_coarse_tick(monkeypatch):
    from GREENFIELD.app_core.contracts import StageEvent
    from GREENFIELD.features import model_adapters

    monkeypatch.setattr(model_adapters, "stable_video_pipeline", lambda: NoCallbackPipe())
    monkeypatch.setattr(model_adapters, "_clear_gpu_memory_unlocked", lambda: {})

    steps = model_adapters.generate_continuation_steps(
        np.zeros((2, 2, 3), dtype=np.uint8), count=2, seed=0, settings=_svd_settings(chunk_frames=2)
    )
    assert next(steps) == (0, None)
    items = list(steps)
    ticks = [item for item in items if isinstance(item, StageEvent)]

    assert [tick.metrics for tick in ticks] == [
        {"chunk": 1, "total_chunks": 1, "generated_frames": 2, "total_frames": 2},
    ]
    assert items[-1][0] == 2 and len(items[-1][1]) == 2


def test_pipeline_without_a_real_step_total_streams_one_coarse_tick(monkeypatch):
    """A callback-capable pipeline with no honest total must not fabricate one."""
    from GREENFIELD.app_core.contracts import StageEvent
    from GREENFIELD.features import model_adapters

    # ``num_timesteps`` is falsy, so ``_total_denoise_steps`` honestly returns None
    # even though the pipeline accepts the step callback.
    pipe = StepCallbackPipe(total_steps=0)
    monkeypatch.setattr(model_adapters, "stable_video_pipeline", lambda: pipe)
    monkeypatch.setattr(model_adapters, "_clear_gpu_memory_unlocked", lambda: {})

    observed = []
    steps = model_adapters.generate_continuation_steps(
        np.zeros((2, 2, 3), dtype=np.uint8), count=2, seed=0,
        settings=_svd_settings(chunk_frames=2), on_step=observed.append,
    )
    assert next(steps) == (0, None)
    items = list(steps)
    ticks = [item for item in items if isinstance(item, StageEvent)]

    # One coarse chunk tick, never a per-step tick with a fabricated total.
    assert [tick.metrics for tick in ticks] == [
        {"chunk": 1, "total_chunks": 1, "generated_frames": 2, "total_frames": 2},
    ]
    assert observed == []
    assert all("total" not in tick.metrics for tick in ticks)
    assert items[-1][0] == 2 and len(items[-1][1]) == 2
    steps.close()


def test_implicit_run_streams_ticks_and_emits_one_terminal_complete(monkeypatch, tmp_path):
    """Real streaming through ``future.run`` still terminates ``svd_generate`` once."""
    from types import SimpleNamespace
    from GREENFIELD.app_core.contracts import RunArtifacts, StageEvent
    from GREENFIELD.features import model_adapters
    from GREENFIELD.features import future as future_module

    pipe = StepCallbackPipe(total_steps=4)
    monkeypatch.setattr(model_adapters, "stable_video_pipeline", lambda: pipe)
    monkeypatch.setattr(model_adapters, "_clear_gpu_memory_unlocked", lambda: {})
    monkeypatch.setattr(future_module, "uses_cpu_offload", lambda: False)
    monkeypatch.setattr(future_module, "write_video", lambda frames, path, fps: path)
    settings = SimpleNamespace(
        checkpoint=None, output_seconds=1, output_frames=2, output_fps=1, chunk_frames=2, svd_decode_chunk_size=1,
    )

    events = list(future_module.run(
        [np.zeros((2, 2, 3), dtype=np.uint8)], 1, future_module.FutureRequest("implicit", 0),
        RunArtifacts(tmp_path), settings,
    ))
    generated = [event for event in events if isinstance(event, StageEvent) and event.stage_id == "svd_generate"]

    assert sum(1 for event in generated if event.status == "complete") == 1
    assert generated[-1].status == "complete"
    running = [event for event in generated if event.status == "running"]
    assert any("step" in event.metrics for event in running)
    # A chunk that streamed real step ticks must not also emit the coarse fallback.
    assert all("generated_frames" not in event.metrics for event in running)


def test_lora_fingerprint_covers_every_training_knob(tmp_path):
    """Steps, rank, pair count, and training side must all change the artifact key."""
    from types import SimpleNamespace
    from GREENFIELD.features.future_training import training_key

    source = tmp_path / "source.mp4"
    source.write_bytes(b"source")
    base = SimpleNamespace(
        history_seconds=7.5, history_fps=12, chunk_frames=14, output_fps=6,
        max_side=1024, training_max_side=512, training_max_pairs=64,
        training_lora_steps=500, training_lora_rank=8,
    )
    stable = training_key(str(source), "implicit", base)
    # An identical copy is stable: no accidental nondeterminism in the key.
    assert training_key(str(source), "implicit", SimpleNamespace(**vars(base))) == stable
    for field, value in (
        ("training_lora_steps", 501), ("training_lora_rank", 16),
        ("training_max_pairs", 65), ("training_max_side", 448),
    ):
        changed = SimpleNamespace(**{**vars(base), field: value})
        assert training_key(str(source), "implicit", changed) != stable, field


def test_training_key_tracks_the_training_checkpoint_identity(tmp_path):
    """Replacing or updating the base checkpoint must never reuse a stale adapter."""
    from types import SimpleNamespace
    from GREENFIELD.features.future_training import training_key

    source = tmp_path / "source.mp4"
    source.write_bytes(b"source")
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    manifest = checkpoint / "model_manifest.json"
    manifest.write_text('{"revision": "sha-one"}')
    base = SimpleNamespace(
        history_seconds=7.5, history_fps=12, chunk_frames=14, output_fps=6,
        max_side=1024, training_max_side=512, training_max_pairs=64,
        training_lora_steps=500, training_lora_rank=8, checkpoint=checkpoint,
    )
    stable = training_key(str(source), "implicit", base)
    # An identical descriptor (same path and metadata) reproduces the same key.
    assert training_key(str(source), "implicit", SimpleNamespace(**vars(base))) == stable
    # Reinstalling the same path at a new revision must invalidate the adapter.
    manifest.write_text('{"revision": "sha-two"}')
    assert training_key(str(source), "implicit", base) != stable
    # Pointing at a different checkpoint directory must also invalidate it.
    other = tmp_path / "other-checkpoint"
    other.mkdir()
    changed = SimpleNamespace(**{**vars(base), "checkpoint": other})
    assert training_key(str(source), "implicit", changed) != stable


def test_training_key_ignores_history_fps(tmp_path):
    """Training samples at the video's own FPS, so the key must not hash history_fps."""
    from types import SimpleNamespace
    from GREENFIELD.features.future_training import training_key

    source = tmp_path / "source.mp4"
    source.write_bytes(b"source")
    base = SimpleNamespace(
        history_seconds=7.5, history_fps=12, chunk_frames=14, output_fps=6,
        max_side=1024, training_max_side=512, training_max_pairs=64,
        training_lora_steps=500, training_lora_rank=8,
    )
    stable = training_key(str(source), "implicit", base)
    changed = SimpleNamespace(**{**vars(base), "history_fps": 24})
    assert training_key(str(source), "implicit", changed) == stable


def test_training_key_tracks_effective_training_resolution_not_forecast_cap(tmp_path):
    """Changing the forecast cap above the training ceiling must not invalidate an adapter.

    Training adapts at ``min(max_side, training_max_side)``, so only that effective
    resolution can change what the adapter learns. Hashing the forecast ``max_side``
    independently would needlessly retrain when it rises above the ceiling.
    """
    from types import SimpleNamespace
    from GREENFIELD.features.future_training import training_key

    source = tmp_path / "source.mp4"
    source.write_bytes(b"source")
    base = SimpleNamespace(
        history_seconds=7.5, history_fps=12, chunk_frames=14, output_fps=6,
        max_side=1024, training_max_side=512, training_max_pairs=64,
        training_lora_steps=500, training_lora_rank=8,
    )
    stable = training_key(str(source), "implicit", base)
    # Raising the forecast cap above the 512px ceiling leaves the training pass
    # identical, so the cached adapter stays reusable.
    raised = SimpleNamespace(**{**vars(base), "max_side": 2048})
    assert training_key(str(source), "implicit", raised) == stable
    # Dropping the forecast cap below the ceiling lowers the effective training
    # resolution, so the key must change.
    lowered = SimpleNamespace(**{**vars(base), "max_side": 256})
    assert training_key(str(source), "implicit", lowered) != stable
    for field, value in (
        ("training_max_side", 448), ("training_lora_steps", 501),
        ("training_lora_rank", 16), ("training_max_pairs", 65),
    ):
        changed = SimpleNamespace(**{**vars(base), field: value})
        assert training_key(str(source), "implicit", changed) != stable, field
