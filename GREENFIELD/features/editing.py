"""Text-only implicit and explicit 3D manipulation of a short source episode."""

from dataclasses import dataclass
from hashlib import sha256
import shutil
import re
from typing import Iterator

import numpy as np

from GREENFIELD.app_core.contracts import FeatureResult, RunArtifacts, StageEvent, StageSpec
from GREENFIELD.app_core.media import write_video
from GREENFIELD.app_core.model_catalog import GROUNDING_DINO, GSPLAT, SAM2, VIDEO_DEPTH_ANYTHING, WAN_VACE
from .scene_cache import ensure_scene_steps
from .scene_masks import lift_masks, project_primitive_mask
from .text_media import read_editing_episode, source_digest
from .text_segmentation import expand_masks, full_scene_masks, ground_and_track
from .text_settings import load_text_settings
from .video_edit import edit_video, vace_frame_count


STAGES = (
    StageSpec("Inspect source episode", "Validate and decode one bounded editing episode from an uploaded or queried clip.", "GREENFIELD/features/text_media.py", "read_editing_episode", "source video", "bounded RGB frames · fps", "Source"),
    StageSpec("Resolve edit scope", "Ground with Grounding DINO and track with SAM2 when possible; otherwise use a deliberate full-scene edit mask.", "GREENFIELD/features/text_segmentation.py", "ground_and_track", "frames · instruction", "tracked target mask or full-scene mask", "Instruction", model_refs=(GROUNDING_DINO, SAM2)),
    StageSpec("Prepare selected method", "Use the video mask directly for Wan VACE or project it through cached dynamic 4D Gaussian primitives for Explicit 3D.", "GREENFIELD/features/scene_masks.py", "project_primitive_mask", "source frames · masks · methodology", "VACE conditioning mask", "Methodology", model_refs=(VIDEO_DEPTH_ANYTHING, GSPLAT)),
    StageSpec("Generate edit proposal", "Apply the instruction with local Wan VACE while preserving fixed-camera source context.", "GREENFIELD/features/video_edit.py", "edit_video", "frames · conditioning mask · instruction · seed", "generated RGB frames", "Instruction · Seed", model_refs=(WAN_VACE,)),
    StageSpec("Save and render result", "Save generated video or fit and render an edited dynamic Gaussian scene.", "GREENFIELD/features/scene_cache.py", "ensure_scene_steps", "generated frames · methodology", "edited.mp4 · splats · trace.json", "Apply"),
)


@dataclass(frozen=True)
class EditingRequest:
    """Text Manipulation's public request retains only user-controlled values."""

    mode: str
    prompt: str
    seed: int

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


def _is_scene_wide_instruction(prompt: str) -> bool:
    """Prefer a full-scene edit for clear environmental or style intent."""
    tokens = set(re.findall(r"[a-z]+", prompt.lower()))
    return bool(tokens & SCENE_WIDE_TOKENS)


def _consume_scene(frames: list[np.ndarray], fps: float, digest: str, stage: str):
    """Forward scene cache progress into the feature trace and return the result."""
    steps = ensure_scene_steps(frames, fps, digest, 0.0, load_text_settings(), stage)
    while True:
        try:
            yield next(steps)
        except StopIteration as finished:
            return finished.value


def _copy_scene_splats(scene, artifacts: RunArtifacts) -> list[str]:
    """Keep explicit saved runs replayable even if the shared cache is later pruned."""
    relative_paths: list[str] = []
    for index, source in enumerate(scene.splats):
        target = artifacts.file(f"explicit_splats/splat_{index:03d}.splat")
        shutil.copy2(source, target)
        relative_paths.append(str(target.relative_to(artifacts.run_dir)))
    return relative_paths


def run(source: str, request: EditingRequest, artifacts: RunArtifacts) -> Iterator[StageEvent | FeatureResult]:
    """Resolve the edit scope, generate the proposal, and save the result in one pass."""
    request.validate()
    settings = load_text_settings()
    yield StageEvent("Inspect source episode", "running", "Opening the source and validating a bounded VACE-compatible episode.")

    frames, fps = read_editing_episode(source, settings.episode_fps, settings.episode_max_frames)
    frames = frames[:vace_frame_count(len(frames))]
    yield StageEvent(
        "Inspect source episode", "complete",
        f"Decoded {len(frames)} stationary-view frames ({len(frames) / fps:.2f}s) for text manipulation.",
        metrics={"frames": len(frames), "fps": fps, "episode_seconds": round(len(frames) / fps, 2)},
    )
    # Grounding and temporal tracking can be expensive, so record its active
    # state before invoking the local models.
    yield StageEvent("Resolve edit scope", "running", "Grounding the requested target and tracking it through the episode.")
    track = ground_and_track(frames, request.prompt)
    scene_wide = _is_scene_wide_instruction(request.prompt)
    if track is None or scene_wide:
        direct_masks = full_scene_masks(frames)
        scope = "full_scene_environmental" if scene_wide else "full_scene_fallback"
        detail = (
            "The instruction describes a scene-wide environmental/style change; using a full-scene generation mask."
            if scene_wide else
            "No reliable named visual target was grounded; using a full-scene generation mask for this environmental/style edit."
        )
        yield StageEvent("Resolve edit scope", "complete", detail, metrics={"scope": scope, "mask_coverage": 1.0})
    else:
        direct_masks = track.masks
        scope = "grounded_target"
        yield StageEvent(
            "Resolve edit scope", "complete",
            "Grounded and tracked the instruction-referred visual target through the source episode.",
            metrics={"scope": scope, "grounding_confidence": round(track.box.confidence, 4), "mask_coverage": round(float(direct_masks.mean()), 4)},
        )
    source_hash = source_digest(source)
    conditioning_masks = direct_masks
    if request.mode == "explicit":
        base_stream = _consume_scene(frames, fps, source_hash, "Prepare selected method")
        while True:
            try:
                yield next(base_stream)
            except StopIteration as finished:
                base_scene = finished.value
                break
        primitive_mask = lift_masks(base_scene, direct_masks)
        conditioning_masks = project_primitive_mask(base_scene, primitive_mask)
        if not conditioning_masks.any():
            conditioning_masks = full_scene_masks(frames)
            scope = "full_scene_3d_fallback"
        np.save(artifacts.file("base_primitive_mask.npy"), primitive_mask.selected)
        yield StageEvent(
            "Prepare selected method", "complete",
            "Projected the edit scope through source-scene Gaussian primitives.",
            metrics={"scope": scope, "primitive_support": round(primitive_mask.support, 4), "scene_key": base_scene.key},
        )
    else:
        yield StageEvent(
            "Prepare selected method", "complete",
            "Prepared the tracked 2D video mask for Implicit 3D VACE conditioning.",
            metrics={"scope": scope},
        )
    conditioning_masks = expand_masks(conditioning_masks)
    np.save(artifacts.file("edit_mask.npy"), conditioning_masks.astype(np.uint8))
    coverage = float(conditioning_masks.mean())
    broad_warning = scope == "grounded_target" and coverage > 0.75

    yield StageEvent("Generate edit proposal", "running", "Loading local Wan VACE and generating the text-conditioned source-view proposal.")
    generated = edit_video(frames, conditioning_masks, request.prompt, request.seed)
    proposal_path = write_video(generated, artifacts.file("vace_proposal.mp4"), fps)
    yield StageEvent(
        "Generate edit proposal", "complete",
        "Wan VACE produced a temporally conditioned generated proposal; it is not observed camera footage.",
        preview=proposal_path,
        metrics={"generated_frames": len(generated), "scope": scope},
    )
    metadata = {"generated": True, "mode": request.mode, "seed": request.seed, "scope": scope,
                "mask_coverage": coverage, "broad_target_warning": broad_warning,
                "proposal": proposal_path.name}
    if request.mode == "implicit":
        output = write_video(generated, artifacts.file("edited.mp4"), fps)
        yield StageEvent(
            "Save and render result", "complete", "Saved the generated Implicit 3D source-view edit.",
            metrics={"output_frames": len(generated), "output_fps": fps},
        )
    else:
        source_hash = source_digest(source)
        edited_hash = sha256(f"{source_hash}:{request.prompt}:{request.seed}:vace".encode()).hexdigest()
        edited_stream = _consume_scene(generated, fps, edited_hash, "Save and render result")
        while True:
            try:
                yield next(edited_stream)
            except StopIteration as finished:
                edited_scene = finished.value
                break
        output = artifacts.file("edited_explicit.mp4")
        shutil.copy2(edited_scene.rendered, output)
        splat_paths = _copy_scene_splats(edited_scene, artifacts)
        yield StageEvent(
            "Save and render result", "complete",
            "Saved the generated edited scene's source-camera render and replayable Gaussian splat timeline.",
            metrics={"output_frames": len(generated), "splats": len(splat_paths), "scene_key": edited_scene.key},
        )
        metadata["viewer"] = {"splat_paths": splat_paths, "durations": [1 / fps] * len(splat_paths)}
    yield FeatureResult(primary=output, secondary=proposal_path, metadata=metadata)
