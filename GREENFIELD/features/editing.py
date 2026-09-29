"""Text-only implicit and explicit 3D manipulation of a short source episode.

The declared execution flow is an atomic, ID-stable union of 19 rows across five
parent groups (Input / Resolve scope / Prepare method / Generate / Save). Every
row is emitted with its stable ``stage_id``; the per-methodology rows carry
``modes`` on the flow so the renderer derives "Skipped" instead of waiting, and
the producers never hand-emit a duplicate skip.

Mask expansion is mode-scoped into two rows because the two methods expand at
different points: ``mask_expand`` dilates the 2D tracked mask *before*
``prepare_mask`` for Implicit 3D, while ``mask_expand_projected`` dilates the
mask *after* ``mask_project`` for Explicit 3D. Explicit must lift the
**unexpanded** tracked mask (so ``base_primitive_mask.npy`` is the raw primitive
selection) and only dilate the projected result for ``edit_mask.npy``; a single
shared expand row could not honestly sit on both sides of ``mask_project`` under
"declared order == execution order".

Two rows are aggregates with a caller-owned terminal: ``dino_detect`` and
``sam2_propagate`` (fed by ``text_segmentation.ground_and_track_steps``) and
``vace_denoise`` (fed by ``video_edit.edit_video_steps``). ``scene_cache`` and
``edited_scene`` relay the tick-only ``ensure_scene_steps`` and own exactly one
terminal each, reading ``CachedScene.cache_hit`` to report reuse truthfully.
"""

from dataclasses import dataclass
from hashlib import sha256
import shutil
import re
from typing import Iterator, Mapping

import numpy as np

from GREENFIELD.app_core.contracts import FeatureResult, RunArtifacts, StageEvent, StageSpec
from GREENFIELD.app_core.flow import CURRENT_FLOW_VERSION, LEGACY_FLOW_VERSION, FeatureFlow, FlowStage
from GREENFIELD.app_core.media import write_video
from GREENFIELD.app_core.model_catalog import GROUNDING_DINO, GSPLAT, SAM2, VIDEO_DEPTH_ANYTHING, WAN_VACE
from GREENFIELD.app_core.super_stages import SuperStage
from .scene_cache import ensure_scene_steps
from .scene_masks import lift_masks, project_primitive_mask
from .text_media import read_editing_episode, source_digest, trim_source
from .text_segmentation import expand_masks, full_scene_masks, ground_and_track_steps
from .text_settings import load_text_settings
from .video_edit import edit_video_steps, vace_frame_count


# The declared union flow, in execution order. Each row keeps one stable
# ``stage_id`` plus the real implementation metadata the expanded card shows.
STAGES = (
    StageSpec(
        "Decode source episode", "Validate and decode one bounded editing episode from an uploaded or queried clip.",
        "GREENFIELD/features/text_media.py", "read_editing_episode",
        "read_editing_episode → bounded RGB frames · fps",
        "source video", "bounded RGB frames · fps", "Episode FPS · Episode frames",
        stage_id="episode_decode",
    ),
    StageSpec(
        "Trim to VACE frames", "Trim the episode to Wan VACE's 1 + 4k temporal VAE frame shape.",
        "GREENFIELD/features/video_edit.py", "vace_frame_count",
        "vace_frame_count → frames[:1 + 4k]",
        "decoded RGB frames", "VACE-compatible RGB frames", "Episode frames",
        stage_id="vace_trim",
    ),
    StageSpec(
        "Ground target with Grounding DINO", "Detect the instruction-referred region in the episode anchor frame.",
        "GREENFIELD/features/text_segmentation.py", "ground_and_track_steps",
        "ground_query → Grounding DINO",
        "anchor frame · instruction", "grounded box or no match", "Instruction · Grounding",
        model_refs=(GROUNDING_DINO,), stage_id="dino_detect",
    ),
    StageSpec(
        "Propagate SAM2 masks", "Propagate the grounded region forward and backward through the episode.",
        "GREENFIELD/features/text_segmentation.py", "track_box_steps",
        "propagate_in_video_iterator (reverse=False, reverse=True)",
        "SAM2 session", "per-frame binary masks", "Grounding",
        model_refs=(SAM2,), stage_id="sam2_propagate",
    ),
    StageSpec(
        "Classify edit scope", "Classify the instruction as a targeted edit or a scene-wide environmental/style change.",
        "GREENFIELD/features/editing.py", "def _scene_wide_token",
        "scene-wide token set → scope",
        "instruction", "targeted or scene-wide scope", "Instruction",
        stage_id="scope_classify",
    ),
    StageSpec(
        "Select edit mask", "Pick the grounded target mask or a deliberate full-scene generation mask.",
        "GREENFIELD/features/editing.py", "def run",
        "grounded target vs full_scene_masks",
        "tracked masks · instruction", "selected binary generation mask", "Instruction · Grounding",
        model_refs=(SAM2, GROUNDING_DINO), stage_id="mask_select",
    ),
    StageSpec(
        "Expand edit mask", "Dilate the selected binary mask so Wan VACE can blend edit boundaries.",
        "GREENFIELD/features/text_segmentation.py", "expand_masks",
        "expand_masks(radius=edit_mask_expand_px)",
        "selected binary mask", "expanded binary mask", "Mask expand",
        stage_id="mask_expand",
    ),
    StageSpec(
        "Prepare implicit mask", "Use the tracked 2D video mask directly for Implicit 3D VACE conditioning.",
        "GREENFIELD/features/editing.py", "def run",
        "edit_mask.npy",
        "expanded binary mask", "Implicit 2D conditioning mask", "Methodology",
        stage_id="prepare_mask",
    ),
    StageSpec(
        "Check scene cache", "Check for a complete cached episode-scale Gaussian scene for this episode.",
        "GREENFIELD/features/scene_cache.py", "ensure_scene_steps",
        "scene_cache_key → manifest validity",
        "source hash · episode · settings", "cache hit or miss", "Cache version",
        model_refs=(VIDEO_DEPTH_ANYTHING, GSPLAT), stage_id="scene_cache",
    ),
    StageSpec(
        "Project primitive mask", "Lift the unexpanded region onto Gaussian primitives and render it back to source-view masks.",
        "GREENFIELD/features/scene_masks.py", "project_primitive_mask",
        "lift_masks → project_primitive_mask",
        "cached scene · unexpanded mask", "projected binary mask or 3D fallback", "Source FOV",
        model_refs=(VIDEO_DEPTH_ANYTHING, GSPLAT), stage_id="mask_project",
    ),
    StageSpec(
        "Expand projected mask", "Dilate the projected source-view mask so Wan VACE can blend explicit edit boundaries.",
        "GREENFIELD/features/text_segmentation.py", "expand_masks",
        "expand_masks(radius=edit_mask_expand_px)",
        "projected binary mask", "expanded binary mask", "Mask expand",
        stage_id="mask_expand_projected",
    ),
    StageSpec(
        "Prepare VACE conditioning", "Resize frames and masks into Wan VACE's envelope and build the 0/255 conditioning mask.",
        "GREENFIELD/features/video_edit.py", "edit_video_steps",
        "cv2.resize → generation mask · prompt conditioning",
        "frames · conditioning mask · instruction", "ready VACE conditioning tensors", "Max side · Max height",
        model_refs=(WAN_VACE,), stage_id="vace_prepare",
    ),
    StageSpec(
        "Denoise edit proposal", "Run Wan VACE text-conditioned denoising while preserving masked-off source context.",
        "GREENFIELD/features/video_edit.py", "edit_video_steps",
        "WanVACEPipeline callback_on_step_end",
        "VACE conditioning · seed", "denoised latent proposal", "VACE steps · Guidance",
        model_refs=(WAN_VACE,), stage_id="vace_denoise",
    ),
    StageSpec(
        "Decode edit proposal", "VAE-decode the proposal and Lanczos-upscale it back to source resolution.",
        "GREENFIELD/features/video_edit.py", "edit_video_steps",
        "vae.decode → cv2.INTER_LANCZOS4",
        "denoised latent proposal", "generated RGB frames", "Methodology",
        model_refs=(WAN_VACE,), stage_id="vace_decode",
    ),
    StageSpec(
        "Write edit proposal", "Encode Wan VACE's generated proposal; it is not observed camera footage.",
        "GREENFIELD/app_core/media.py", "def write_video",
        "write_video",
        "generated RGB frames", "vace_proposal.mp4", "Methodology",
        stage_id="write_proposal",
    ),
    StageSpec(
        "Save implicit edit", "Write the generated Implicit 3D source-view edit.",
        "GREENFIELD/app_core/media.py", "def write_video",
        "write_video",
        "generated RGB frames", "edited.mp4", "Apply",
        stage_id="finalize",
    ),
    StageSpec(
        "Render explicit edit", "Fit the generated frames into a dynamic Gaussian scene and copy its source-view render.",
        "GREENFIELD/features/scene_cache.py", "ensure_scene_steps",
        "gaussian_worker → edited_explicit.mp4",
        "generated RGB frames", "edited_explicit.mp4", "Cache version",
        model_refs=(VIDEO_DEPTH_ANYTHING, GSPLAT), stage_id="edited_scene",
    ),
    StageSpec(
        "Copy explicit splats", "Copy the edited scene's splat timeline into the run so the saved viewer survives a cache prune.",
        "GREENFIELD/features/editing.py", "def _copy_scene_splats",
        "copy2 → explicit_splats/splat_###.splat",
        "edited Gaussian scene", "run-owned splat timeline", "Apply",
        model_refs=(GSPLAT,), stage_id="splat_copy",
    ),
    StageSpec(
        "Publish artifacts", "Write the replayable run trace once the edit is complete.",
        "GREENFIELD/app_core/trace.py", "def finish",
        "FeatureTrace.finish",
        "generated edit · events", "trace.json", "None",
        stage_id="publish_artifacts",
    ),
)

# The named parents own explicit stage IDs, so reordering ``STAGES`` can never
# silently regroup a saved run.
FLOW_GROUPS = (
    SuperStage("Input", ("episode_decode", "vace_trim")),
    SuperStage("Resolve scope", ("dino_detect", "sam2_propagate", "scope_classify", "mask_select", "mask_expand")),
    SuperStage("Prepare method", ("prepare_mask", "scene_cache", "mask_project", "mask_expand_projected")),
    SuperStage("Generate", ("vace_prepare", "vace_denoise", "vace_decode", "write_proposal")),
    SuperStage("Save", ("finalize", "edited_scene", "splat_copy", "publish_artifacts")),
)

# Frozen v1 identities (the original display names; v1 had no stage ids).
LEGACY_STAGE_IDS = (
    "Inspect source episode",
    "Resolve edit scope",
    "Prepare selected method",
    "Generate edit proposal",
    "Save and render result",
)

# The five coarse bars every pre-v5 profile recorded. v1-v4 kept the same names
# and ids, so all four versions map onto this exact set; the atomized 19-row
# union is new in v5, so a saved v4 trace must never claim it recorded a union
# row. The retired ids/names are additionally aliased per feature in
# ``app_core.flow`` (``prepare_method`` here is Text Manipulation's own, not
# Future View's global ``svd_load`` alias).
_OLD_STAGE_IDS = (
    "inspect_episode", "resolve_scope", "prepare_method", "generate_edit", "save_result",
)
LEGACY_PROFILES = {
    LEGACY_FLOW_VERSION: _OLD_STAGE_IDS,
    2: _OLD_STAGE_IDS,
    3: _OLD_STAGE_IDS,
    # v4 was the same five coarse bars; Text Manipulation's atomization landed
    # with the v5 global bump, so a v4 trace resolves against its own set.
    4: _OLD_STAGE_IDS,
}

# Per-row methodology applicability. Empty means every mode.
_IMPLICIT_MODES = ("implicit",)
_EXPLICIT_MODES = ("explicit",)
_STAGE_MODES = {
    "episode_decode": (),
    "vace_trim": (),
    "dino_detect": (),
    "sam2_propagate": (),
    "scope_classify": (),
    "mask_select": (),
    "mask_expand": _IMPLICIT_MODES,
    "prepare_mask": _IMPLICIT_MODES,
    "scene_cache": _EXPLICIT_MODES,
    "mask_project": _EXPLICIT_MODES,
    "mask_expand_projected": _EXPLICIT_MODES,
    "vace_prepare": (),
    "vace_denoise": (),
    "vace_decode": (),
    "write_proposal": (),
    "finalize": _IMPLICIT_MODES,
    "edited_scene": _EXPLICIT_MODES,
    "splat_copy": _EXPLICIT_MODES,
    "publish_artifacts": (),
}

# The grounding/tracking helper owns its own labels but takes the caller's ids.
# Editing declares only ``dino_detect`` and ``sam2_propagate``; the helper's
# ``sam2_setup``/``mask_validate`` rows map to blank ids and are dropped.
_GROUND_STAGE_IDS: Mapping[str, str] = {
    "dino_detect": "dino_detect",
    "sam2_setup": "",
    "sam2_propagate": "sam2_propagate",
    "mask_validate": "",
}

# The three Wan VACE rows streamed by ``video_edit.edit_video_steps``.
_VACE_STAGE_IDS: Mapping[str, str] = {
    "vace_prepare": "vace_prepare",
    "vace_denoise": "vace_denoise",
    "vace_decode": "vace_decode",
}

# Display label per stage id, derived once from the declaration so an emission
# can never drift from the label the page rendered.
_LABELS = {spec.stage_id: spec.name for spec in STAGES}

# The scene-cache rows carry distinct display labels into the shared adapter.
_SCENE_LABELS = {"scene_cache": "Check scene cache", "edited_scene": "Render explicit edit"}
_SCENE_HIT_DETAIL = {
    "scene_cache": "Reused a complete cached Gaussian scene for this episode.",
    "edited_scene": "Reused a complete cached Gaussian scene for the generated edit.",
}
_SCENE_MISS_DETAIL = {
    "scene_cache": "Built a reusable Gaussian scene for this episode.",
    "edited_scene": "Built a Gaussian scene for the generated edit.",
}


def flow() -> FeatureFlow:
    """Return Text Manipulation's current ordered stage profile."""
    stages = tuple(FlowStage(spec.stage_id, spec, modes=_STAGE_MODES[spec.stage_id]) for spec in STAGES)
    return FeatureFlow(
        "editing", CURRENT_FLOW_VERSION, stages, FLOW_GROUPS,
        legacy_stage_ids=LEGACY_STAGE_IDS, legacy_profiles=LEGACY_PROFILES,
    )


@dataclass(frozen=True)
class EditingRequest:
    """Text Manipulation's public request retains only user-controlled values."""

    mode: str
    prompt: str
    seed: int
    trim_start_seconds: float = 0.0
    trim_end_seconds: float | None = None

    def validate(self) -> None:
        if self.mode not in {"implicit", "explicit"}:
            raise ValueError("Choose Implicit 3D or Explicit 3D.")
        if not self.prompt.strip():
            raise ValueError("Enter an editing instruction.")


SCENE_WIDE_TOKENS = frozenset({
    "flood", "flooded", "rain", "rainy", "snow", "snowy", "fog", "foggy",
    "storm", "weather", "season", "winter", "night", "sunset", "sunrise",
    "underwater", "wildfire", "watercolor", "anime", "style",
})


def _scene_wide_token(prompt: str) -> str | None:
    """Return the matching scene-wide token, or ``None`` for a targeted edit.

    The token (not just a boolean) is recorded so the scope row can show *why*
    a full-scene mask was chosen instead of merely that one was.
    """
    matched = set(re.findall(r"[a-z]+", prompt.lower())) & SCENE_WIDE_TOKENS
    return sorted(matched)[0] if matched else None


def _is_scene_wide_instruction(prompt: str) -> bool:
    """Prefer a full-scene edit for clear environmental or style intent."""
    return _scene_wide_token(prompt) is not None


def _event(stage_id: str, status: str, detail: str, metrics: dict | None = None, preview=None) -> StageEvent:
    """Build one typed event for a declared stage id, with its declared label."""
    return StageEvent(_LABELS[stage_id], status, detail, preview=preview, metrics=metrics or {}, stage_id=stage_id)


def _consume_scene(frames: list[np.ndarray], fps: float, digest: str, stage_id: str, settings):
    """Forward a reusable scene build's tick-only stream and return its entry.

    ``stage_id`` is the owning row (``scene_cache`` or ``edited_scene``); the
    shared ``ensure_scene_steps`` never emits a terminal, so the caller owns the
    single one and reads ``CachedScene.cache_hit`` to report reuse truthfully.
    """
    steps = ensure_scene_steps(
        frames, fps, digest, 0.0, settings, _SCENE_LABELS[stage_id], stage_id=stage_id,
    )
    while True:
        try:
            yield next(steps)
        except StopIteration as finished:
            return finished.value


def _relay_scene_ticks(scene_id: str, stream):
    """Forward a scene stream's running ticks and return ``(scene, cache_hit)``.

    A cache hit is tick-free and a miss relays the build's progress; either way
    the single terminal belongs to the caller, which can then report any
    post-work (e.g. copying the rendered video) before terminating the row.
    """
    try:
        first = next(stream)
    except StopIteration as finished:
        scene = finished.value
        return scene, True
    yield first
    while True:
        try:
            yield next(stream)
        except StopIteration as finished:
            return finished.value, False


def _scene_terminal(scene_id: str, scene, hit: bool) -> StageEvent:
    """The one caller-owned terminal for a scene-cache row, reporting real reuse."""
    return _event(
        scene_id, "complete",
        _SCENE_HIT_DETAIL[scene_id] if hit else _SCENE_MISS_DETAIL[scene_id],
        metrics={"cache": "hit" if hit else "miss", "scene_key": scene.key},
    )


def _denoise_terminal(last_metrics: dict | None) -> StageEvent:
    """Emit the one ``vace_denoise`` terminal, reporting only a real step total."""
    metrics = {"steps": last_metrics["total"]} if last_metrics and "total" in last_metrics else {}
    return _event("vace_denoise", "complete", "Wan VACE finished the text-conditioned denoising pass.", metrics=metrics)


def _consume_vace(frames: list[np.ndarray], masks: np.ndarray, request: "EditingRequest", settings):
    """Relay Wan VACE's prepare/denoise/decode stream, returning generated frames.

    ``vace_denoise`` is an aggregate row: this generator forwards its running
    ticks and emits the single terminal just before the decode row begins, so the
    row terminal always precedes the next row in declared order.
    """
    steps = edit_video_steps(
        frames, masks, request.prompt, request.seed, settings=settings, stage_ids=_VACE_STAGE_IDS,
    )
    generated = None
    denoise_done = False
    last_denoise: dict | None = None
    while True:
        try:
            item = next(steps)
        except StopIteration as finished:
            generated = finished.value
            break
        if item.stage_id == "vace_decode" and not denoise_done:
            yield _denoise_terminal(last_denoise)
            denoise_done = True
        if item.stage_id == "vace_denoise":
            last_denoise = dict(item.metrics)
        yield item
    if not denoise_done:
        yield _denoise_terminal(last_denoise)
    return generated


def _ground_scope(frames: list[np.ndarray], prompt: str, threshold: float):
    """Relay the grounding/tracking helper into editing's two declared rows.

    The shared helper interleaves its own ``sam2_setup``/``mask_validate`` rows;
    those map to blank ids and are dropped here. The helper's ticks are running
    only, so this generator emits the single ``dino_detect`` and
    ``sam2_propagate`` terminals, keeping declared order serial (the DINO
    terminal precedes the first propagation tick).
    """
    track = None
    dino_metrics: dict = {"matched": False}
    dino_seen = False
    dino_done = False
    sam2_seen = False
    sam2_metrics: dict = {"reversed": False, "propagated_frames": 0, "total_frames": len(frames)}
    steps = ground_and_track_steps(frames, prompt, threshold=threshold, stage_ids=_GROUND_STAGE_IDS)
    while True:
        try:
            tick = next(steps)
        except StopIteration as finished:
            track = finished.value
            break
        if tick.stage_id == "dino_detect":
            dino_seen = True
            dino_metrics = dict(tick.metrics)
            yield tick
        elif tick.stage_id == "sam2_propagate":
            if not dino_done:
                yield _event("dino_detect", "complete", _dino_detail(dino_metrics), metrics=_dino_metrics(dino_metrics))
                dino_done = True
            sam2_seen = True
            sam2_metrics = dict(tick.metrics)
            yield tick
        # A blank-id ``sam2_setup``/``mask_validate`` tick belongs to no declared row.
    if not dino_done:
        if not dino_seen:
            yield _event("dino_detect", "running", "No frames were available to ground a target.")
        yield _event("dino_detect", "complete", _dino_detail(dino_metrics), metrics=_dino_metrics(dino_metrics))
    if not sam2_seen:
        yield _event("sam2_propagate", "running", "No grounded target reached SAM2 propagation.")
    yield _event(
        "sam2_propagate", "complete",
        f"Propagated {sam2_metrics.get('propagated_frames', 0)} frames through the grounded target.",
        metrics={
            "reversed": sam2_metrics.get("reversed", False),
            "propagated_frames": sam2_metrics.get("propagated_frames", 0),
            "total_frames": sam2_metrics.get("total_frames", len(frames)),
        },
    )
    return track


def _dino_metrics(tick_metrics: dict) -> dict:
    """Report only the honest DINO outcome: a real confidence or an explicit no-match."""
    if "confidence" in tick_metrics:
        return {"confidence": tick_metrics["confidence"]}
    return {"matched": False}


def _dino_detail(tick_metrics: dict) -> str:
    if "confidence" in tick_metrics:
        return "Grounding DINO matched the instruction-referred target."
    return "No reliable named visual target was grounded."


def _copy_scene_splats(scene, artifacts: RunArtifacts) -> list[str]:
    """Keep explicit saved runs replayable even if the shared cache is later pruned."""
    relative_paths: list[str] = []
    for index, source in enumerate(scene.splats):
        target = artifacts.file(f"explicit_splats/splat_{index:03d}.splat")
        shutil.copy2(source, target)
        relative_paths.append(str(target.relative_to(artifacts.run_dir)))
    return relative_paths


def run(source: str, request: EditingRequest, artifacts: RunArtifacts, settings=None) -> Iterator[StageEvent | FeatureResult]:
    """Resolve the edit scope, generate the proposal, and save the result in one pass.

    ``settings`` is the validated effective per-run configuration; when omitted
    the tracked defaults are loaded so existing callers keep their behavior.
    """
    request.validate()
    settings = settings or load_text_settings()

    # Input: decode then trim to the VAE's 1 + 4k shape.
    yield _event("episode_decode", "running", "Opening the source and validating a bounded VACE-compatible episode.")
    if request.trim_end_seconds is None and request.trim_start_seconds == 0:
        # Existing direct callers already supply a bounded source episode.
        trim_start, trim_end = 0.0, None
        frames, fps = read_editing_episode(source, settings.episode_fps, settings.episode_max_frames)
    else:
        trimmed_source, _trim_info, trim_start, trim_end = trim_source(
            source, artifacts.file("trimmed_source.mp4"), request.trim_start_seconds,
            request.trim_end_seconds, settings.episode_max_frames / settings.episode_fps,
        )
        frames, fps = read_editing_episode(str(trimmed_source), settings.episode_fps, settings.episode_max_frames)
    yield _event(
        "episode_decode", "complete",
        f"Decoded {len(frames)} stationary-view frames ({len(frames) / fps:.2f}s) for text manipulation.",
        metrics={"frames": len(frames), "fps": fps, "episode_seconds": round(len(frames) / fps, 2), "trim_start_seconds": trim_start, "trim_end_seconds": trim_end},
    )
    yield _event("vace_trim", "running", "Trimming the episode to Wan VACE's 1 + 4k temporal frame shape.")
    total_frames = len(frames)
    frames = frames[:vace_frame_count(total_frames)]
    yield _event(
        "vace_trim", "complete",
        f"Kept {len(frames)} of {total_frames} frames for the temporal VAE.",
        metrics={"kept_frames": len(frames), "dropped_frames": total_frames - len(frames), "total_frames": total_frames},
    )

    # Resolve scope: ground, track, classify, select, expand.
    track = yield from _ground_scope(frames, request.prompt, settings.grounding_threshold)
    scope_token = _scene_wide_token(request.prompt)
    scene_wide = scope_token is not None
    yield _event("scope_classify", "running", "Classifying the instruction as targeted or scene-wide.")
    yield _event(
        "scope_classify", "complete",
        (
            f"The instruction names a scene-wide condition ({scope_token}); a full-scene edit is appropriate."
            if scene_wide else
            "The instruction is treated as a targeted edit of one grounded region."
        ),
        metrics={"scope_token": scope_token, "scene_wide": scene_wide},
    )

    yield _event("mask_select", "running", "Selecting the grounded target or a deliberate full-scene mask.")
    if scene_wide:
        direct_masks = full_scene_masks(frames)
        scope = "full_scene_environmental"
        select_detail = "The instruction describes a scene-wide environmental/style change; using a full-scene generation mask."
        select_metrics = {"scope": scope, "mask_coverage": 1.0}
    elif track is None:
        direct_masks = full_scene_masks(frames)
        scope = "full_scene_fallback"
        select_detail = "No reliable named visual target was grounded; using a full-scene generation mask for this environmental/style edit."
        select_metrics = {"scope": scope, "mask_coverage": 1.0}
    else:
        direct_masks = track.masks
        scope = "grounded_target"
        select_detail = "Grounded and tracked the instruction-referred visual target through the source episode."
        select_metrics = {
            "scope": scope,
            "grounding_confidence": round(track.box.confidence, 4),
            "mask_coverage": round(float(direct_masks.mean()), 4),
        }
    yield _event("mask_select", "complete", select_detail, metrics=select_metrics)

    # Prepare method: implicit dilates the 2D tracked mask before conditioning;
    # explicit lifts the UNEXPANDED mask through the cached Gaussian scene, then
    # dilates the projected result. The two expansions are separate declared rows
    # because they execute on opposite sides of ``mask_project``.
    source_hash = source_digest(source)
    if request.mode == "implicit":
        yield _event("mask_expand", "running", f"Expanding the edit mask by {settings.edit_mask_expand_px}px to soften boundaries.")
        direct_masks = expand_masks(direct_masks, radius=settings.edit_mask_expand_px)
        yield _event(
            "mask_expand", "complete",
            f"Dilated the selected mask by {settings.edit_mask_expand_px}px.",
            metrics={"radius": settings.edit_mask_expand_px, "mask_coverage": round(float(direct_masks.mean()), 4)},
        )
        yield _event("prepare_mask", "running", "Preparing the tracked 2D video mask for Implicit 3D VACE conditioning.")
        conditioning_masks = direct_masks
        np.save(artifacts.file("edit_mask.npy"), conditioning_masks.astype(np.uint8))
        yield _event(
            "prepare_mask", "complete",
            "Prepared the tracked 2D video mask for Implicit 3D VACE conditioning.",
            metrics={"scope": scope, "mask_coverage": round(float(conditioning_masks.mean()), 4)},
        )
    else:
        base_scene, base_hit = yield from _relay_scene_ticks(
            "scene_cache", _consume_scene(frames, fps, source_hash, "scene_cache", settings),
        )
        yield _scene_terminal("scene_cache", base_scene, base_hit)
        yield _event("mask_project", "running", "Projecting the unexpanded edit scope through source-scene Gaussian primitives.")
        # Lift the pre-expansion tracked mask so the saved primitive membership
        # stays the raw selection; expansion happens only after projection.
        primitive_mask = lift_masks(base_scene, direct_masks)
        projected = project_primitive_mask(base_scene, primitive_mask)
        np.save(artifacts.file("base_primitive_mask.npy"), primitive_mask.selected)
        if not projected.any():
            conditioning_masks = full_scene_masks(frames)
            scope = "full_scene_3d_fallback"
            project_detail = "Projection produced no visible primitive footprints; using a full-scene generation mask."
            project_metrics = {"scope": scope}
        else:
            conditioning_masks = projected
            project_detail = "Projected the unexpanded edit scope through source-scene Gaussian primitives."
            project_metrics = {
                "primitive_support": round(primitive_mask.support, 4),
                "projected_mask_coverage": round(float(projected.mean()), 4),
                "scope": scope,
            }
        yield _event("mask_project", "complete", project_detail, metrics=project_metrics)

        yield _event("mask_expand_projected", "running", f"Expanding the projected mask by {settings.edit_mask_expand_px}px to soften boundaries.")
        conditioning_masks = expand_masks(conditioning_masks, radius=settings.edit_mask_expand_px)
        np.save(artifacts.file("edit_mask.npy"), conditioning_masks.astype(np.uint8))
        yield _event(
            "mask_expand_projected", "complete",
            f"Dilated the projected mask by {settings.edit_mask_expand_px}px.",
            metrics={"radius": settings.edit_mask_expand_px, "mask_coverage": round(float(conditioning_masks.mean()), 4)},
        )

    # Generate: VACE prepare/denoise/decode, then encode the proposal.
    generated = yield from _consume_vace(frames, conditioning_masks, request, settings)
    yield _event("write_proposal", "running", "Encoding Wan VACE's generated proposal; it is not observed camera footage.")
    proposal_path = write_video(generated, artifacts.file("vace_proposal.mp4"), fps)
    yield _event(
        "write_proposal", "complete",
        "Wan VACE produced a temporally conditioned generated proposal.",
        preview=proposal_path,
        # ``output_frames`` is the run's honest generated-frame count; it is the
        # single source the page's ``publish_artifacts`` summary reads, so the
        # count never has to be duplicated into ``FeatureResult.metadata``.
        metrics={"output_frames": len(generated), "fps": fps},
    )

    coverage = float(conditioning_masks.mean())
    broad_warning = scope == "grounded_target" and coverage > 0.75
    # The metadata contract stays fixed (the viewer adds its own keys for
    # explicit runs); the generated frame count lives on the ``write_proposal``
    # stage metric instead of being duplicated here.
    metadata = {
        "generated": True, "mode": request.mode, "seed": request.seed, "scope": scope,
        "mask_coverage": coverage, "broad_target_warning": broad_warning,
        "proposal": proposal_path.name,
    }

    # Save: implicit writes the edited video; explicit renders and copies splats.
    if request.mode == "implicit":
        yield _event("finalize", "running", "Encoding the generated Implicit 3D source-view edit.")
        output = write_video(generated, artifacts.file("edited.mp4"), fps)
        yield _event(
            "finalize", "complete", "Saved the generated Implicit 3D source-view edit.",
            metrics={"output_frames": len(generated), "output_fps": fps},
        )
    else:
        edited_hash = sha256(f"{source_hash}:{request.prompt}:{request.seed}:vace".encode()).hexdigest()
        edited_scene, edited_hit = yield from _relay_scene_ticks(
            "edited_scene", _consume_scene(generated, fps, edited_hash, "edited_scene", settings),
        )
        output = artifacts.file("edited_explicit.mp4")
        shutil.copy2(edited_scene.rendered, output)
        yield _scene_terminal("edited_scene", edited_scene, edited_hit)
        yield _event("splat_copy", "running", "Copying the edited splat timeline into the run.")
        splat_paths = _copy_scene_splats(edited_scene, artifacts)
        yield _event(
            "splat_copy", "complete",
            f"Copied {len(splat_paths)} splat frames out of the shared scene cache.",
            metrics={"splats": len(splat_paths)},
        )
        metadata["viewer"] = {"splat_paths": splat_paths, "durations": [1 / fps] * len(splat_paths)}
    yield FeatureResult(primary=output, secondary=proposal_path, metadata=metadata)
