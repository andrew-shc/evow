"""Boundary validation and propagation coverage for editable Text Manipulation settings.

These tests exercise the page-boundary parser (``_editing_effective_settings``),
then prove the resulting effective ``TextSettings`` actually reaches Grounding
DINO, the mask expander, Wan VACE, and the reusable Gaussian scene cache. They
also pin the VACE 1 + 4k frame rule and the panel's tracked defaults.
"""

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from GREENFIELD.app_core.contracts import RunArtifacts
from GREENFIELD.features import editing, pages, scene_cache, text_segmentation, video_edit
from GREENFIELD.features.scene_cache import CachedScene
from GREENFIELD.features.scene_masks import PrimitiveMask
from GREENFIELD.features.text_models import GroundedBox
from GREENFIELD.features.text_segmentation import MaskTrack
from GREENFIELD.features.text_settings import load_text_settings


def _values_with(base, **overrides):
    """Return parser-order values with selected knobs replaced."""
    values = [getattr(base, name) for name in pages._EDITABLE_EDITING_FIELDS]
    for name, value in overrides.items():
        values[pages._EDITABLE_EDITING_FIELDS.index(name)] = value
    return values


def _tiny_frames(count: int = 5) -> list[np.ndarray]:
    """Make a tiny stationary RGB episode for deterministic adapter tests."""
    return [np.zeros((16, 20, 3), dtype=np.uint8) for _ in range(count)]


def _scene(root, name: str, frame_count: int = 5) -> CachedScene:
    """Build tiny file-backed scene metadata for orchestration tests only."""
    scene_dir = root / name
    scene_dir.mkdir(parents=True)
    checkpoint = scene_dir / "scene.pt"
    rendered = scene_dir / "rendered.mp4"
    checkpoint.touch()
    rendered.write_bytes(b"rendered")
    splats = []
    for index in range(frame_count):
        path = scene_dir / f"splat_{index:03d}.splat"
        path.write_bytes(b"splat")
        splats.append(path)
    return CachedScene(name, scene_dir.parent, scene_dir, rendered, checkpoint, tuple(splats), 20, 16, 12, 70)


def test_editing_effective_settings_default_parity() -> None:
    """Untouched controls rebuild the tracked settings exactly, including the key."""
    base = load_text_settings()
    effective = pages._editing_effective_settings(base, *_values_with(base))
    assert effective == base
    assert scene_cache.scene_cache_key("a" * 64, 0, 8, 12, effective) == scene_cache.scene_cache_key(
        "a" * 64, 0, 8, 12, base
    )


def test_editing_effective_settings_accepts_in_range_overrides() -> None:
    """Every knob is replaceable and keeps the base field's own type."""
    base = load_text_settings()
    effective = pages._editing_effective_settings(
        base,
        *_values_with(
            base,
            episode_fps=24, episode_max_frames=41,
            edit_max_side=512, edit_max_height=256, vace_steps=12,
            vace_guidance_scale=7.5, edit_mask_expand_px=11,
            grounding_threshold=0.45, source_fov_degrees=90,
            scene_cache_version="epoch-2",
        ),
    )
    assert effective.episode_fps == 24
    assert effective.episode_max_frames == 41
    assert effective.edit_max_side == 512
    assert effective.edit_max_height == 256
    assert effective.vace_steps == 12
    assert effective.vace_guidance_scale == 7.5
    assert effective.edit_mask_expand_px == 11
    assert effective.grounding_threshold == 0.45
    assert effective.source_fov_degrees == 90
    assert effective.scene_cache_version == "epoch-2"
    assert isinstance(effective.edit_mask_expand_px, int)
    assert isinstance(effective.vace_guidance_scale, float)


def test_editing_effective_settings_rejects_bad_values_by_field_name() -> None:
    """Out-of-range, bool, non-numeric, NaN, fractional, empty, and wrong-count inputs fail loudly."""
    base = load_text_settings()

    with pytest.raises(ValueError, match="Text Manipulation episode_fps"):
        pages._editing_effective_settings(base, *_values_with(base, episode_fps=0))
    with pytest.raises(ValueError, match="Text Manipulation episode_max_frames"):
        pages._editing_effective_settings(base, *_values_with(base, episode_max_frames=4))
    with pytest.raises(ValueError, match="Text Manipulation episode_max_frames"):
        pages._editing_effective_settings(base, *_values_with(base, episode_max_frames=82))
    with pytest.raises(ValueError, match="Text Manipulation edit_max_side"):
        pages._editing_effective_settings(base, *_values_with(base, edit_max_side=63))
    with pytest.raises(ValueError, match="Text Manipulation edit_max_height"):
        pages._editing_effective_settings(base, *_values_with(base, edit_max_height=2161))
    with pytest.raises(ValueError, match="Text Manipulation vace_steps"):
        pages._editing_effective_settings(base, *_values_with(base, vace_steps=201))
    with pytest.raises(ValueError, match="Text Manipulation vace_guidance_scale"):
        pages._editing_effective_settings(base, *_values_with(base, vace_guidance_scale=20.5))
    with pytest.raises(ValueError, match="Text Manipulation edit_mask_expand_px"):
        pages._editing_effective_settings(base, *_values_with(base, edit_mask_expand_px=65))
    with pytest.raises(ValueError, match="Text Manipulation grounding_threshold"):
        pages._editing_effective_settings(base, *_values_with(base, grounding_threshold=1.5))
    with pytest.raises(ValueError, match="Text Manipulation source_fov_degrees"):
        pages._editing_effective_settings(base, *_values_with(base, source_fov_degrees=120))

    with pytest.raises(ValueError, match="Text Manipulation episode_fps must be a number"):
        pages._editing_effective_settings(base, *_values_with(base, episode_fps=True))
    with pytest.raises(ValueError, match="Text Manipulation vace_steps must be a number"):
        pages._editing_effective_settings(base, *_values_with(base, vace_steps="abc"))
    with pytest.raises(ValueError, match="Text Manipulation vace_guidance_scale must be a finite number"):
        pages._editing_effective_settings(base, *_values_with(base, vace_guidance_scale=float("nan")))
    with pytest.raises(ValueError, match="Text Manipulation edit_mask_expand_px must be a whole number"):
        pages._editing_effective_settings(base, *_values_with(base, edit_mask_expand_px=2.5))

    with pytest.raises(ValueError, match="scene_cache_version must be a non-empty string"):
        pages._editing_effective_settings(base, *_values_with(base, scene_cache_version="   "))
    with pytest.raises(ValueError, match="scene_cache_version must be a non-empty string"):
        pages._editing_effective_settings(base, *_values_with(base, scene_cache_version=5))
    with pytest.raises(ValueError, match="edit_max_side must not be empty"):
        pages._editing_effective_settings(base, *_values_with(base, edit_max_side=None))
    with pytest.raises(ValueError, match="10 values"):
        pages._editing_effective_settings(base)


def test_effective_values_reach_segmentation_and_video_edit(monkeypatch, tmp_path) -> None:
    """Grounding, mask dilation, and the VACE settings come from the effective snapshot."""
    base = load_text_settings()
    effective = pages._editing_effective_settings(
        base,
        *_values_with(
            base,
            grounding_threshold=0.6, edit_mask_expand_px=9,
            vace_steps=7, vace_guidance_scale=12.5,
            edit_max_side=512, edit_max_height=256,
        ),
    )
    frames = _tiny_frames()
    captured: dict = {}
    monkeypatch.setattr(editing, "read_editing_episode", lambda *_args: (frames, 12.0))

    def fake_ground_steps(_frames, _prompt, anchor_index=None, threshold=0.25, *, candidate_index=1, total_candidates=1, stage_ids):
        captured["grounding_threshold"] = threshold
        if False:
            yield
        return None

    monkeypatch.setattr(editing, "ground_and_track_steps", fake_ground_steps)
    monkeypatch.setattr(editing, "source_digest", lambda _path: "source-hash")
    monkeypatch.setattr(editing, "expand_masks", lambda masks, **kwargs: (
        captured.update(mask_radius=kwargs["radius"]) or masks
    ))

    def fake_vace_steps(source, _mask, _prompt, seed, settings=None, *, stage_ids=None):
        captured.update(edit_settings=settings, seed=seed)
        if False:
            yield
        return source

    monkeypatch.setattr(editing, "edit_video_steps", fake_vace_steps)
    monkeypatch.setattr(editing, "write_video", lambda _frames, path, _fps: (path.write_bytes(b"video"), path)[1])

    list(editing.run(
        "unused.mp4", editing.EditingRequest("implicit", "flood the forest", 3),
        RunArtifacts(tmp_path / "prop"), effective,
    ))

    assert captured["grounding_threshold"] == 0.6
    assert captured["mask_radius"] == 9
    assert captured["edit_settings"] is effective
    assert captured["edit_settings"].vace_steps == 7
    assert captured["edit_settings"].vace_guidance_scale == 12.5
    assert captured["seed"] == 3


def test_effective_scene_settings_reach_the_scene_cache(monkeypatch, tmp_path) -> None:
    """Explicit mode hands the effective FOV/epoch/rate/length to the reusable scene build."""
    base = load_text_settings()
    effective = pages._editing_effective_settings(
        base,
        *_values_with(base, source_fov_degrees=88, scene_cache_version="epoch-99", episode_fps=24, episode_max_frames=41),
    )
    frames = _tiny_frames()
    masks = np.ones((len(frames), 16, 20), dtype=bool)
    captured: dict = {}
    monkeypatch.setattr(editing, "read_editing_episode", lambda *_args: (frames, 12.0))

    def fake_ground_steps(_frames, _prompt, anchor_index=None, threshold=0.25, *, candidate_index=1, total_candidates=1, stage_ids):
        if False:
            yield
        return MaskTrack(masks, GroundedBox((0, 0, 20, 16), 0.9), 2)

    monkeypatch.setattr(editing, "ground_and_track_steps", fake_ground_steps)
    monkeypatch.setattr(editing, "source_digest", lambda _path: "source-hash")
    monkeypatch.setattr(editing, "write_video", lambda _frames, path, _fps: (path.write_bytes(b"video"), path)[1])

    def fake_vace_steps(source, *_args, **_kwargs):
        if False:
            yield
        return source

    monkeypatch.setattr(editing, "edit_video_steps", fake_vace_steps)

    scenes = iter((_scene(tmp_path, "base"), _scene(tmp_path, "edited")))

    def fake_scene(_frames, _fps, _digest, _stage, settings):
        captured.setdefault("settings", []).append(settings)
        if False:
            yield  # pragma: no cover - makes this a generator function
        return next(scenes)

    monkeypatch.setattr(editing, "_consume_scene", fake_scene)
    monkeypatch.setattr(editing, "lift_masks", lambda *_args: PrimitiveMask(np.asarray([True]), 1.0))
    monkeypatch.setattr(editing, "project_primitive_mask", lambda *_args: masks)

    list(editing.run(
        "unused.mp4", editing.EditingRequest("explicit", "make the sky red", 0),
        RunArtifacts(tmp_path / "scene"), effective,
    ))

    assert captured["settings"] and all(settings is effective for settings in captured["settings"])
    assert captured["settings"][0].source_fov_degrees == 88
    assert captured["settings"][0].scene_cache_version == "epoch-99"
    assert captured["settings"][0].episode_fps == 24
    assert captured["settings"][0].episode_max_frames == 41


def test_scene_cache_key_reacts_to_source_fov_and_cache_version() -> None:
    """Changing either the FOV or the cache epoch yields a fresh scene key."""
    base = load_text_settings()
    baseline = scene_cache.scene_cache_key("a" * 64, 0, 8, 12, base)
    assert scene_cache.scene_cache_key("a" * 64, 0, 8, 12, replace(base, source_fov_degrees=75)) != baseline
    assert scene_cache.scene_cache_key("a" * 64, 0, 8, 12, replace(base, scene_cache_version="next-epoch")) != baseline

    tuned = pages._editing_effective_settings(
        base, *_values_with(base, source_fov_degrees=95, scene_cache_version="epoch-2"),
    )
    assert scene_cache.scene_cache_key("a" * 64, 0, 8, 12, tuned) != baseline


def test_video_edit_consumes_effective_resolution_steps_and_guidance(monkeypatch) -> None:
    """Wan VACE reads its resolution ceilings, steps, and guidance from the effective settings."""
    base = load_text_settings()
    effective = replace(base, vace_steps=7, vace_guidance_scale=12.5, edit_max_side=16, edit_max_height=16)
    captured: dict = {}

    class FakePipeline:
        def __call__(self, **kwargs):
            captured.update(kwargs)
            return SimpleNamespace(frames=[kwargs["video"]])

    monkeypatch.setattr(video_edit, "vace_pipeline", lambda: FakePipeline())
    monkeypatch.setattr(video_edit, "device", lambda: "cpu")
    frames = _tiny_frames(5)
    masks = np.zeros((5, 16, 20), dtype=bool)
    masks[:, 4:12, 5:15] = True

    result = video_edit.edit_video(frames, masks, "make it rainy", 9, effective)

    assert captured["num_inference_steps"] == 7
    assert captured["guidance_scale"] == 12.5
    assert captured["height"] == 16
    assert captured["width"] == 16
    assert len(result) == 5


def test_ground_and_track_forwards_the_effective_threshold(monkeypatch) -> None:
    """The editing path's grounding override reaches Grounding DINO unchanged."""
    seen: dict = {}
    monkeypatch.setattr(text_segmentation, "ground_query", lambda _frame, _query, threshold=0.25: (
        seen.update(threshold=threshold) or None
    ))
    assert text_segmentation.ground_and_track(_tiny_frames(), "tree", threshold=0.6) is None
    assert seen["threshold"] == 0.6


def test_vace_frame_rule_is_honored_by_the_episode_cap() -> None:
    """The 1 + 4k rule trims an incomplete tail and the cap never exceeds 81 frames."""
    assert video_edit.vace_frame_count(81) == 81
    assert video_edit.vace_frame_count(82) == 81
    assert video_edit.vace_frame_count(80) == 77
    assert video_edit.vace_frame_count(5) == 5
    with pytest.raises(ValueError, match="at least five"):
        video_edit.vace_frame_count(4)
    assert pages._EDITING_FIELD_RANGES["episode_max_frames"] == (5, 81)


def test_editing_panel_defaults_equal_tracked_settings_with_descriptions() -> None:
    """The built panel exposes tracked defaults under the compact row contract."""
    import gradio as gr

    settings = load_text_settings()
    page = pages.build_editing()

    def _in_configuration(component):
        ancestor = getattr(component, "parent", None)
        while ancestor is not None:
            if isinstance(ancestor, gr.Accordion) and ancestor.label == "Configuration":
                return True
            ancestor = getattr(ancestor, "parent", None)
        return False

    controls = {
        component.label: component
        for component in page.blocks.values()
        if isinstance(component, (gr.Number, gr.Slider, gr.Textbox)) and component.label is not None and _in_configuration(component)
    }
    expected = {
        "Episode FPS": settings.episode_fps,
        "Episode frames": settings.episode_max_frames,
        "Max side": settings.edit_max_side,
        "Max height": settings.edit_max_height,
        "VACE steps": settings.vace_steps,
        "Guidance": settings.vace_guidance_scale,
        "Mask expand": settings.edit_mask_expand_px,
        "Grounding": settings.grounding_threshold,
        "Source FOV": settings.source_fov_degrees,
        "Cache version": settings.scene_cache_version,
        "Seed": 0,
    }
    assert set(controls) == set(expected)
    for label, value in expected.items():
        assert controls[label].value == value
        # The compact row keeps the control bare (the adjacent Markdown label
        # owns the name and ⓘ description) while staying a self-contained block.
        assert "evow-config-control" in controls[label].elem_classes, label
        assert controls[label].container is False
        assert controls[label].show_label is False
        assert "evow-config-row" in controls[label].parent.elem_classes
        assert any(
            isinstance(sibling, gr.Markdown)
            and "evow-config-label" in (sibling.elem_classes or [])
            and sibling.value.strip()
            for sibling in controls[label].parent.children
        ), label



def test_editing_config_visibility_gates_the_scene_group() -> None:
    """Text Manipulation hides the Scene group only for Implicit 3D runs.

    Group order: Episode, Edit, Matching, Scene, Run. Only Scene is
    methodology-gated; the rest stay visible so shared controls never disappear.
    """
    explicit = pages._editing_config_visibility("explicit")
    implicit = pages._editing_config_visibility("implicit")
    assert [update["visible"] for update in explicit] == [True, True, True, True, True]
    assert [update["visible"] for update in implicit] == [True, True, True, False, True]


def test_editing_config_visibility_never_rewrites_control_values() -> None:
    """A visibility update must carry no ``value``, or a mode switch would reset knobs."""
    for mode in ("explicit", "implicit"):
        updates = pages._editing_config_visibility(mode)
        assert len(updates) == 5
        assert all("value" not in update for update in updates), mode


def test_editing_page_starts_with_the_scene_group_visible() -> None:
    """The built page's default Explicit 3D mode shows every configuration group."""
    page = pages.build_editing()
    registration = next(
        fn for fn in page.fns.values()
        if getattr(fn.fn, "__name__", "") == "_editing_config_visibility"
    )
    assert [component.visible for component in registration.outputs] == [True, True, True, True, True]
