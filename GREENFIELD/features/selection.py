"""Two text-query methods that return highlighted, playable source intervals."""

from dataclasses import dataclass
from typing import Iterator

import imageio.v2 as imageio
import numpy as np

from GREENFIELD.app_core.contracts import ClipArtifact, FeatureResult, RunArtifacts, StageEvent, StageSpec
from GREENFIELD.app_core.media import write_video
from GREENFIELD.app_core.model_catalog import GROUNDING_DINO, GSPLAT, SAM2, SIGLIP, VIDEO_DEPTH_ANYTHING
from .scene_cache import ensure_scene_steps
from .scene_masks import export_selected_splats, lift_masks, project_primitive_mask
from .text_media import read_interval, read_uniform_samples, source_digest, timestamp_label
from .text_models import semantic_scores_steps
from .text_retrieval import ranked_windows
from .text_segmentation import ground_and_track, highlighted_frames
from .text_settings import load_text_settings


TEXT_QUERY_PIPELINE_REVISION = "text-query-v2"
LOW_SPECIFICITY_COVERAGE = 0.65


STAGES = (
    StageSpec("Inspect source video", "Validate the bounded source video and sample retrieval frames.", "GREENFIELD/features/text_media.py", "read_uniform_samples", "source video", "sparse RGB frames · timestamps", "Source"),
    StageSpec("Retrieve relevant intervals", "Embed sampled video frames and query text with local SigLIP, then rank temporal windows.", "GREENFIELD/features/text_models.py", "semantic_scores_steps", "frames · query", "ranked non-overlapping source windows", "Text · Methodology", model_refs=(SIGLIP,)),
    StageSpec("Ground and track binary target", "Use Grounding DINO and SAM2 to produce a binary mask through each candidate clip.", "GREENFIELD/features/text_segmentation.py", "ground_and_track", "candidate frames · query", "tracked binary masks", "Text", model_refs=(GROUNDING_DINO, SAM2)),
    StageSpec("Resolve selected method", "Use direct video masks for Implicit 3D or lift them through cached 4D Gaussian primitives for Explicit 3D.", "GREENFIELD/features/scene_masks.py", "lift_masks", "masks · selected methodology", "2D or projected 3D binary masks", "Methodology", model_refs=(VIDEO_DEPTH_ANYTHING, GSPLAT)),
    StageSpec("Save highlighted clips", "Encode raw and highlighted source clips without any spatial crop.", "GREENFIELD/app_core/media.py", "write_video", "source frames · binary masks", "source.mp4 · highlighted.mp4 · trace.json", "Search"),
)


@dataclass(frozen=True)
class SelectionRequest:
    """Text Query's intentionally small, replayable public request."""

    mode: str
    query: str

    def validate(self) -> None:
        if self.mode not in {"implicit", "explicit"}:
            raise ValueError("Choose Implicit 3D or Explicit 3D.")
        if not self.query.strip():
            raise ValueError("Enter a text query.")


def _consume_scores(frames: list[np.ndarray], query: str) -> Iterator[StageEvent]:
    """Forward SigLIP's streamed progress, then return aligned frame scores."""
    steps = semantic_scores_steps(frames, query)
    while True:
        try:
            completed, total = next(steps)
            yield StageEvent(
                "Retrieve relevant intervals", "running", f"Embedded {completed}/{total} retrieval frames.",
                metrics={"embedded_frames": completed, "total_frames": total},
            )
        except StopIteration as finished:
            return finished.value


def _consume_scene(frames: list[np.ndarray], fps: float, digest: str, start_seconds: float, stage: str):
    """Forward a reusable scene build stream and return its completed cache entry."""
    steps = ensure_scene_steps(frames, fps, digest, start_seconds, load_text_settings(), stage)
    while True:
        try:
            yield next(steps)
        except StopIteration as finished:
            return finished.value


def run(source: str, request: SelectionRequest, artifacts: RunArtifacts) -> Iterator[StageEvent | FeatureResult]:
    """Search one source video and save up to five non-cropped highlighted clips."""
    request.validate()
    settings = load_text_settings()
    yield StageEvent("Inspect source video", "running", "Opening the bounded source video and scheduling uniform retrieval samples.")

    sparse_frames, timestamps, info = read_uniform_samples(source, settings.query_sample_fps, settings.query_max_source_seconds)
    yield StageEvent(
        "Inspect source video", "complete",
        f"Sampled {len(sparse_frames)} retrieval frames across {info.duration_seconds:.1f}s of source video.",
        metrics={"source_seconds": round(info.duration_seconds, 2), "sample_frames": len(sparse_frames), "sample_fps": settings.query_sample_fps},
    )
    score_stream = _consume_scores(sparse_frames, request.query)
    yield StageEvent("Retrieve relevant intervals", "running", "Loading local SigLIP and embedding the query.")
    while True:
        try:
            yield next(score_stream)
        except StopIteration as finished:
            scores = finished.value
            break
    np.save(artifacts.file("semantic_scores.npy"), scores)
    candidates = ranked_windows(
        scores, timestamps, info.duration_seconds, settings.query_clip_seconds, settings.query_explicit_candidate_pool
    )
    yield StageEvent(
        "Retrieve relevant intervals", "complete",
        f"Ranked {len(candidates)} non-overlapping candidate intervals for: {request.query}",
        metrics={"candidate_pool": len(candidates)},
    )
    # Hash every copied source so result provenance proves it is fresh.
    digest = source_digest(source)
    clips: list[ClipArtifact] = []
    for candidate in candidates:
        if request.mode == "implicit" and len(clips) >= settings.query_max_results:
            break
        # Decode is part of the candidate-level grounding operation. Report it
        # first so this stage remains active throughout the whole operation.
        yield StageEvent(
            "Ground and track binary target", "running",
            f"Decoding and grounding candidate {len(clips) + 1} at {timestamp_label(candidate.start_seconds)}.",
        )
        clip_frames, clip_fps = read_interval(
            source, candidate.start_seconds, candidate.end_seconds - candidate.start_seconds, settings.query_result_fps
        )
        yield StageEvent(
            "Ground and track binary target", "running",
            f"Grounding and tracking candidate {len(clips) + 1} at {timestamp_label(candidate.start_seconds)}.",
        )
        track = ground_and_track(clip_frames, request.query)
        if track is None:
            continue
        semantic_masks = track.masks
        semantic_coverage = float(semantic_masks.mean())
        projected_masks: np.ndarray | None = None
        projected_coverage: float | None = None
        projected_low_specificity = False
        selected_splats: tuple = ()
        method_score = candidate.score
        if request.mode == "explicit":
            scene_stream = _consume_scene(clip_frames, clip_fps, digest, candidate.start_seconds, "Resolve selected method")
            while True:
                try:
                    yield next(scene_stream)
                except StopIteration as finished:
                    scene = finished.value
                    break
            primitive_mask = lift_masks(scene, semantic_masks)
            projected_masks = project_primitive_mask(scene, primitive_mask)
            if not projected_masks.any():
                continue
            method_score += primitive_mask.support * 0.05
            projected_coverage = float(projected_masks.mean())
            projected_low_specificity = projected_coverage >= LOW_SPECIFICITY_COVERAGE or primitive_mask.support >= LOW_SPECIFICITY_COVERAGE
            np.save(artifacts.file(f"clips/{len(clips):02d}_primitive_mask.npy"), primitive_mask.selected)
            selected_splats = export_selected_splats(scene, primitive_mask, artifacts.file(f"selected_splats/{len(clips):02d}/.marker").parent) if hasattr(scene, "checkpoint") else ()
            yield StageEvent(
                "Resolve selected method", "complete",
                "Projected the grounded target through persistent Gaussian primitives.",
                metrics={"primitive_support": round(primitive_mask.support, 4), "scene_key": scene.key, "projected_mask_coverage": round(projected_coverage, 4), "low_spatial_specificity": projected_low_specificity},
            )
        else:
            yield StageEvent(
                "Resolve selected method", "complete",
                "Used the tracked video mask directly for Implicit 3D highlighting.",
                metrics={"mask_coverage": round(semantic_coverage, 4)},
            )
        clip_index = len(clips)
        raw_path = artifacts.file(f"clips/{clip_index:02d}_source.mp4")
        semantic_path = artifacts.file(f"clips/{clip_index:02d}_semantic_highlighted.mp4")
        write_video(clip_frames, raw_path, clip_fps)
        semantic_frames = highlighted_frames(clip_frames, semantic_masks)
        write_video(semantic_frames, semantic_path, clip_fps)
        np.save(artifacts.file(f"clips/{clip_index:02d}_semantic_mask.npy"), semantic_masks.astype(np.uint8))
        projected_path = None
        if projected_masks is not None:
            projected_path = artifacts.file(f"clips/{clip_index:02d}_projected_highlighted.mp4")
            np.save(artifacts.file(f"clips/{clip_index:02d}_projected_mask.npy"), projected_masks.astype(np.uint8))
            write_video(highlighted_frames(clip_frames, projected_masks), projected_path, clip_fps)
        preview_path = artifacts.file(f"previews/{clip_index:02d}_semantic_highlighted.png")
        imageio.imwrite(preview_path, semantic_frames[len(semantic_frames) // 2])
        clip = ClipArtifact(raw_path, semantic_path, candidate.start_seconds, candidate.end_seconds, float(method_score), request.mode, projected_path, track.box.confidence, semantic_coverage, projected_coverage, projected_low_specificity, selected_splats)
        clips.append(clip)
        yield StageEvent(
            "Save highlighted clips", "running",
            f"Saved highlighted rerank candidate {len(clips)}/{len(candidates)} without spatial cropping.",
            preview=preview_path,
            metrics={"saved_clips": len(clips), "semantic_mask_coverage": round(semantic_coverage, 4), "projected_mask_coverage": round(projected_coverage, 4) if projected_coverage is not None else None, "low_spatial_specificity": projected_low_specificity},
        )
    if not clips:
        raise ValueError("No reliable text-grounded regions were found. Try a more concrete visual query.")
    # Explicit mode deliberately evaluates all eight candidates before its 3D support
    # rerank; only the best five are exposed as final user-facing clips.
    clips.sort(key=lambda clip: (-clip.score, clip.start_seconds))
    clips = clips[:settings.query_max_results]
    rows = [[timestamp_label(clip.start_seconds), timestamp_label(clip.end_seconds),
             round(clip.score, 3), clip.method] for clip in clips]
    yield StageEvent(
        "Ground and track binary target", "complete", f"Tracked binary regions in {len(clips)} matching source clips.",
        metrics={"matched_clips": len(clips)},
    )
    yield StageEvent(
        "Save highlighted clips", "complete", f"Saved {len(clips)} playable highlighted source clips; no result was spatially cropped.",
        metrics={"saved_clips": len(clips)},
    )
    yield FeatureResult(clips=clips, rows=rows, metadata={
        "query": request.query.strip(), "mode": request.mode, "source_seconds": info.duration_seconds,
        "clip_seconds": settings.query_clip_seconds, "generated": False,
        "provenance": {"pipeline_revision": TEXT_QUERY_PIPELINE_REVISION, "source_sha256": digest, "semantic_model": getattr(settings, "semantic_model_id", "test-model"), "grounding_model": getattr(settings, "grounding_model_id", "test-model"), "segmentation_model": getattr(settings, "segmentation_model_id", "test-model")},
        "viewer": {"splat_paths": [str(path.relative_to(artifacts.run_dir)) for path in clips[0].selected_splats], "durations": [1 / info.fps] * len(clips[0].selected_splats)} if request.mode == "explicit" and clips[0].selected_splats else None,
    })
