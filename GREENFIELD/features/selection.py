"""Two text-query methods that return highlighted, playable source intervals.

The declared execution flow is an atomic, ID-stable union of 22 rows across five
parent groups (Input / Retrieve / Ground + track / Resolve 3D method / Save).
Every row is emitted with its stable ``stage_id``; the per-methodology rows carry
``modes`` on the flow so the renderer derives "Skipped" instead of waiting.
"""

from dataclasses import dataclass
from typing import Iterator

import imageio.v2 as imageio
import numpy as np

from GREENFIELD.app_core.contracts import ClipArtifact, FeatureResult, RunArtifacts, StageEvent, StageSpec
from GREENFIELD.app_core.flow import CURRENT_FLOW_VERSION, LEGACY_FLOW_VERSION, FeatureFlow, FlowStage
from GREENFIELD.app_core.media import write_video
from GREENFIELD.app_core.model_catalog import FAISS, GROUNDING_DINO, GSPLAT, SAM2, SIGLIP, VIDEO_DEPTH_ANYTHING
from GREENFIELD.app_core.super_stages import SuperStage
from .scene_cache import ensure_scene_steps
from .scene_masks import export_selected_splats, lift_masks, project_primitive_mask
from .text_media import read_interval, read_uniform_samples, source_digest, timestamp_label, trim_source
from .text_models import semantic_scores_steps
from .text_retrieval import ranked_windows
from .text_segmentation import ground_and_track_steps, highlighted_frames
from .text_settings import load_text_settings


TEXT_QUERY_PIPELINE_REVISION = "text-query-v2"

# Methodology applicability. Empty modes means every mode.
_IMPLICIT_MODES = ("implicit",)
_EXPLICIT_MODES = ("explicit",)


# The declared union flow, in execution order. Each row keeps one stable
# ``stage_id`` plus the real implementation metadata the expanded card shows.
STAGES = (
    StageSpec(
        "Inspect source video", "Validate the bounded source video and sample retrieval frames.",
        "GREENFIELD/features/text_media.py", "read_uniform_samples",
        "read_uniform_samples → sparse RGB frames · timestamps",
        "source video", "sparse RGB frames · timestamps", "Source",
        stage_id="inspect_source",
    ),
    StageSpec(
        "Load local SigLIP", "Load the owner-installed SigLIP processor and weights from local snapshots only.",
        "GREENFIELD/features/text_models.py", "def siglip_model",
        "AutoProcessor.from_pretrained → AutoModel.from_pretrained",
        "local SigLIP snapshot", "ready SigLIP processor and model", "Text",
        model_refs=(SIGLIP,), stage_id="siglip_load",
    ),
    StageSpec(
        "Embed query text", "Encode the query text into one SigLIP embedding.",
        "GREENFIELD/features/text_models.py", "semantic_scores_steps",
        "get_text_features",
        "query text", "text embedding [1, D]", "Text",
        model_refs=(SIGLIP,), stage_id="text_embed",
    ),
    StageSpec(
        "Embed source frames", "Encode every sampled source frame into a SigLIP image embedding.",
        "GREENFIELD/features/text_models.py", "semantic_scores_steps",
        "get_image_features (per frame)",
        "sampled source frames", "image embeddings [N, D]", "Text",
        model_refs=(SIGLIP,), stage_id="frame_embed",
    ),
    StageSpec(
        "Normalize embeddings", "L2-normalize the image and text embeddings for cosine scoring.",
        "GREENFIELD/features/text_models.py", "semantic_scores_steps",
        "L2 normalization",
        "image embeddings · text embedding", "unit image/text embeddings", "Text",
        model_refs=(SIGLIP,), stage_id="normalize",
    ),
    StageSpec(
        "Build local FAISS index", "Build a local inner-product index over the source frame embeddings.",
        "GREENFIELD/features/text_models.py", "semantic_scores_steps",
        "faiss.IndexFlatIP",
        "unit image embeddings", "local FAISS index", "Text",
        model_refs=(FAISS,), stage_id="faiss_index",
    ),
    StageSpec(
        "Search FAISS index", "Score every frame against the query embedding through the local index.",
        "GREENFIELD/features/text_models.py", "semantic_scores_steps",
        "IndexFlatIP.search",
        "FAISS index · text embedding", "one similarity score per frame", "Text",
        model_refs=(FAISS,), stage_id="faiss_search",
    ),
    StageSpec(
        "Rank temporal windows", "Smooth frame scores across each candidate window, then sort, clamp, and suppress overlaps.",
        "GREENFIELD/features/text_retrieval.py", "def ranked_windows",
        "smoothed_window_scores → temporal NMS",
        "frame scores · timestamps", "ranked non-overlapping source windows", "Text",
        stage_id="window_rank",
    ),
    StageSpec(
        "Decode candidate clips", "Decode each ranked candidate interval without spatial crop or resize.",
        "GREENFIELD/features/text_media.py", "read_interval",
        "read_interval",
        "source video · candidate interval", "candidate RGB frames · fps", "Clip (s) · Result FPS",
        stage_id="candidate_decode",
    ),
    StageSpec(
        "Ground target with Grounding DINO", "Detect the text-referred region inside the candidate's anchor frame.",
        "GREENFIELD/features/text_segmentation.py", "ground_and_track_steps",
        "ground_query → Grounding DINO",
        "anchor frame · query", "grounded box or no match", "Grounding",
        model_refs=(GROUNDING_DINO,), stage_id="dino_detect",
    ),
    StageSpec(
        "Prepare SAM2 tracking", "Initialize a SAM2 video session seeded from the grounded box.",
        "GREENFIELD/features/text_segmentation.py", "track_box_steps",
        "Sam2VideoProcessor.init_video_session",
        "candidate frames · grounded box", "ready SAM2 video session", "Grounding",
        model_refs=(SAM2,), stage_id="sam2_setup",
    ),
    StageSpec(
        "Propagate SAM2 masks", "Propagate the region forward and backward through the candidate clip.",
        "GREENFIELD/features/text_segmentation.py", "track_box_steps",
        "propagate_in_video_iterator (reverse=False, reverse=True)",
        "SAM2 session", "per-frame binary masks", "Grounding",
        model_refs=(SAM2,), stage_id="sam2_propagate",
    ),
    StageSpec(
        "Validate tracked masks", "Reject an empty or degenerate tracked region instead of fabricating a highlight.",
        "GREENFIELD/features/text_segmentation.py", "ground_and_track_steps",
        "mean < 0.0001 rejection",
        "per-frame binary masks", "validated tracked masks or no match", "Grounding",
        model_refs=(SAM2,), stage_id="mask_validate",
    ),
    StageSpec(
        "Resolve selected method", "Use the directly tracked video masks for Implicit 3D highlighting.",
        "GREENFIELD/features/selection.py", "def run",
        "tracked masks → highlighted clips",
        "tracked binary masks", "2D binary masks", "Methodology",
        stage_id="resolve_method",
    ),
    StageSpec(
        "Check scene cache", "Check for a complete cached episode-scale Gaussian scene for this candidate.",
        "GREENFIELD/features/scene_cache.py", "ensure_scene_steps",
        "scene_cache_key → manifest validity",
        "source hash · interval · settings", "cache hit or miss", "Cache version",
        model_refs=(VIDEO_DEPTH_ANYTHING, GSPLAT), stage_id="scene_cache",
    ),
    StageSpec(
        "Reconstruct Gaussian scene", "Build and cache the dynamic Gaussian scene on a cache miss.",
        "GREENFIELD/features/scene_cache.py", "ensure_scene_steps",
        "depth_worker → gaussian_worker → atomic manifest",
        "candidate frames", "reusable dynamic Gaussian scene", "Cache version",
        model_refs=(VIDEO_DEPTH_ANYTHING, GSPLAT), stage_id="scene_reconstruct",
    ),
    StageSpec(
        "Lift masks to primitives", "Vote the tracked 2D region onto persistent Gaussian primitive identities.",
        "GREENFIELD/features/scene_masks.py", "lift_masks",
        "lift_masks",
        "cached scene · tracked masks", "selected primitive membership", "Source FOV",
        model_refs=(GSPLAT,), stage_id="mask_lift",
    ),
    StageSpec(
        "Project primitive masks", "Render the selected primitives back into source-view binary masks.",
        "GREENFIELD/features/scene_masks.py", "project_primitive_mask",
        "project_primitive_mask",
        "cached scene · primitive membership", "projected binary masks", "Source FOV",
        model_refs=(GSPLAT,), stage_id="mask_project",
    ),
    StageSpec(
        "Export selected splats", "Export the full 4D scene timeline with the selected primitives recolored.",
        "GREENFIELD/features/scene_masks.py", "export_selected_splats",
        "export_selected_splats → save_splat",
        "cached scene · primitive membership", "selected splat timeline", "Source FOV",
        model_refs=(GSPLAT,), stage_id="splat_export",
    ),
    StageSpec(
        "Render highlighted frames", "Overlay the accepted binary mask while retaining every nonmasked source pixel.",
        "GREENFIELD/features/text_segmentation.py", "highlighted_frames",
        "highlighted_frames",
        "candidate frames · binary masks", "highlighted RGB frames", "Text",
        stage_id="overlay",
    ),
    StageSpec(
        "Save highlighted clips", "Encode raw and highlighted clips plus mask arrays and previews, without spatial crop.",
        "GREENFIELD/app_core/media.py", "def write_video",
        "write_video → trace clip records",
        "source frames · binary masks", "source.mp4 · highlighted.mp4 · mask npy", "Search",
        stage_id="write_clips",
    ),
    StageSpec(
        "Rerank final clips", "Sort accepted clips by score then start time, and keep at most the result limit.",
        "GREENFIELD/features/selection.py", "def run",
        "sorted((-score, start_seconds))[:query_max_results]",
        "accepted clips", "ranked clips (≤5)", "Max results",
        stage_id="final_rerank",
    ),
)

# The named parents own explicit stage IDs, so reordering ``STAGES`` can never
# silently regroup a saved run.
FLOW_GROUPS = (
    SuperStage("Input", ("inspect_source",)),
    SuperStage("Retrieve", ("siglip_load", "text_embed", "frame_embed", "normalize", "faiss_index", "faiss_search", "window_rank")),
    SuperStage("Ground + track", ("candidate_decode", "dino_detect", "sam2_setup", "sam2_propagate", "mask_validate")),
    SuperStage("Resolve 3D method", ("resolve_method", "scene_cache", "scene_reconstruct", "mask_lift", "mask_project", "splat_export")),
    SuperStage("Save", ("overlay", "write_clips", "final_rerank")),
)

# Frozen v1 identities (the original display names; v1 had no stage ids). They
# stay literal so an unknown pre-current version still has a name fallback, and
# the three renamed bars are additionally aliased in ``app_core.flow``.
LEGACY_STAGE_IDS = (
    "Inspect source video",
    "Retrieve relevant intervals",
    "Ground and track binary target",
    "Resolve selected method",
    "Save highlighted clips",
)

# The exact current-id sets each pre-current Text Query profile could record.
# Every old version wrote the same five coarse bars, which map onto the atomic
# rows that now own each operation: the retrieval bar became ``window_rank``,
# the grounding bar became ``mask_validate``, and the save bar became
# ``write_clips``. ``resolve_method`` is a current id and stays one, so a legacy
# explicit run's projected 3D step remains representable on that row even though
# new explicit runs skip it as implicit-only.
LEGACY_PROFILES = {
    LEGACY_FLOW_VERSION: (
        "inspect_source", "window_rank", "mask_validate", "resolve_method", "write_clips",
    ),
    2: (
        "inspect_source", "window_rank", "mask_validate", "resolve_method", "write_clips",
    ),
    # v3 also wrote the five coarse bars; it predates the v4 atomization, so its
    # persisted traces must resolve against this exact set rather than falling
    # through to the v1 name fallback and mislabeling valid rows "Not recorded".
    3: (
        "inspect_source", "window_rank", "mask_validate", "resolve_method", "write_clips",
    ),
    # v4 was the atomized 22-row union; the v5 bump (Text Manipulation's
    # atomization) left Text Query untouched, so its saved traces recorded every
    # current row and must not be marked "not recorded".
    4: tuple(spec.stage_id for spec in STAGES),
}

# Per-row methodology applicability. Empty means every mode.
_STAGE_MODES = {
    "inspect_source": (),
    "siglip_load": (),
    "text_embed": (),
    "frame_embed": (),
    "normalize": (),
    "faiss_index": (),
    "faiss_search": (),
    "window_rank": (),
    "candidate_decode": (),
    "dino_detect": (),
    "sam2_setup": (),
    "sam2_propagate": (),
    "mask_validate": (),
    "resolve_method": _IMPLICIT_MODES,
    "scene_cache": _EXPLICIT_MODES,
    "scene_reconstruct": _EXPLICIT_MODES,
    "mask_lift": _EXPLICIT_MODES,
    "mask_project": _EXPLICIT_MODES,
    "splat_export": _EXPLICIT_MODES,
    "overlay": (),
    "write_clips": (),
    "final_rerank": (),
}

# The six atomic retrieval sub-stages, handed to ``semantic_scores_steps`` so
# that adapter stays feature-agnostic while the caller's flow owns the ids.
_RETRIEVAL_STAGE_IDS = {
    "siglip_load": "siglip_load",
    "text_embed": "text_embed",
    "frame_embed": "frame_embed",
    "normalize": "normalize",
    "faiss_index": "faiss_index",
    "faiss_search": "faiss_search",
}

# The four atomic grounding/tracking sub-stages.
_GROUND_STAGE_IDS = {
    "dino_detect": "dino_detect",
    "sam2_setup": "sam2_setup",
    "sam2_propagate": "sam2_propagate",
    "mask_validate": "mask_validate",
}

# Display label per stage id, derived once from the declaration so an emission
# can never drift from the label the page rendered.
_LABELS = {spec.stage_id: spec.name for spec in STAGES}


def flow() -> FeatureFlow:
    """Return Text Query's current ordered stage profile."""
    stages = tuple(FlowStage(spec.stage_id, spec, modes=_STAGE_MODES[spec.stage_id]) for spec in STAGES)
    return FeatureFlow(
        "selection", CURRENT_FLOW_VERSION, stages, FLOW_GROUPS,
        legacy_stage_ids=LEGACY_STAGE_IDS, legacy_profiles=LEGACY_PROFILES,
    )


@dataclass(frozen=True)
class SelectionRequest:
    """Text Query's intentionally small, replayable public request."""

    mode: str
    query: str
    trim_start_seconds: float = 0.0
    trim_end_seconds: float | None = None

    def validate(self) -> None:
        if self.mode not in {"implicit", "explicit"}:
            raise ValueError("Choose Implicit 3D or Explicit 3D.")
        if not self.query.strip():
            raise ValueError("Enter a text query.")


def _event(stage_id: str, status: str, detail: str, metrics: dict | None = None, preview=None) -> StageEvent:
    """Build one typed event for a declared stage id, with its declared label."""
    return StageEvent(_LABELS[stage_id], status, detail, preview=preview, metrics=metrics or {}, stage_id=stage_id)


def _consume_retrieval(frames: list[np.ndarray], query: str) -> Iterator[StageEvent]:
    """Forward SigLIP's atomic retrieval events, then return aligned frame scores."""
    steps = semantic_scores_steps(frames, query, stage_ids=_RETRIEVAL_STAGE_IDS)
    while True:
        try:
            yield next(steps)
        except StopIteration as finished:
            return finished.value


def _consume_scene(frames: list[np.ndarray], fps: float, digest: str, start_seconds: float, settings):
    """Forward a reusable scene build's tick-only stream and return its cache entry.

    The scene cache no longer emits a terminal, so this caller owns the single
    terminal for ``scene_reconstruct``; ``scene.cache_hit`` reports whether the
    entry was reused or freshly built.
    """
    steps = ensure_scene_steps(
        frames, fps, digest, start_seconds, settings,
        "Reconstruct Gaussian scene", stage_id="scene_reconstruct",
    )
    while True:
        try:
            yield next(steps)
        except StopIteration as finished:
            return finished.value


def run(source: str, request: SelectionRequest, artifacts: RunArtifacts, settings=None) -> Iterator[StageEvent | FeatureResult]:
    """Search one source video and save up to five non-cropped highlighted clips.

    ``settings`` is the validated effective per-run configuration; when omitted
    the tracked defaults are loaded so existing callers keep their behavior.
    """
    request.validate()
    settings = settings or load_text_settings()
    yield _event("inspect_source", "running", "Opening the bounded source video and scheduling uniform retrieval samples.")
    if request.trim_end_seconds is None and request.trim_start_seconds == 0:
        # Preserve the public adapter's historical no-trim call contract.
        source_offset = 0.0
        sparse_frames, timestamps, info = read_uniform_samples(source, settings.query_sample_fps, settings.query_max_source_seconds)
        trim_end = info.duration_seconds
    else:
        trimmed_source, _trim_info, source_offset, trim_end = trim_source(
            source, artifacts.file("trimmed_source.mp4"), request.trim_start_seconds,
            request.trim_end_seconds, settings.query_max_source_seconds,
        )
        source = str(trimmed_source)
        sparse_frames, timestamps, info = read_uniform_samples(source, settings.query_sample_fps, settings.query_max_source_seconds)
    yield _event(
        "inspect_source", "complete",
        f"Trimmed {source_offset:.1f}s–{trim_end:.1f}s and sampled {len(sparse_frames)} retrieval frames.",
        metrics={
            "source_seconds": round(info.duration_seconds, 2),
            "duration_seconds": round(info.duration_seconds, 2),
            "sample_frames": len(sparse_frames),
            "sample_fps": settings.query_sample_fps,
            "trim_start_seconds": source_offset, "trim_end_seconds": trim_end,
        },
    )
    score_stream = _consume_retrieval(sparse_frames, request.query)
    while True:
        try:
            yield next(score_stream)
        except StopIteration as finished:
            scores = finished.value
            break
    np.save(artifacts.file("semantic_scores.npy"), scores)
    yield _event("window_rank", "running", "Smoothing frame scores and applying temporal non-maximum suppression.")
    # A zero-frame sample yields no scores, and ``ranked_windows`` rejects an empty
    # input. Treat that as an empty candidate pool here instead of letting the
    # error escape mid-flow: every downstream aggregate row (and the resolve rows)
    # must still reach its single terminal before the user-facing "no matches"
    # error, or the browser would retain a dangling running stage.
    candidates = (
        []
        if len(scores) == 0
        else ranked_windows(
            scores, timestamps, info.duration_seconds, settings.query_clip_seconds, settings.query_explicit_candidate_pool
        )
    )
    yield _event(
        "window_rank", "complete",
        f"Ranked {len(candidates)} non-overlapping candidate intervals for: {request.query}",
        metrics={"candidate_pool": len(candidates)},
    )
    total_candidates = len(candidates)
    # Hash every copied source so result provenance proves it is fresh.
    digest = source_digest(source)
    clips: list[ClipArtifact] = []
    # Per-row aggregate counters. Each looping row emits one ``running`` per
    # iteration and exactly one owner-emitted terminal after the loop.
    processed = 0
    dino_matched = 0
    accepted = 0
    resolved = 0
    lifted = 0
    projected = 0
    empty_projections = 0
    exported = 0
    highlighted_total = 0
    scene_hits = 0
    scene_misses = 0
    last_scene_key = None
    last_scene_splats = 0
    # Splat count of the most recent *freshly built* scene, kept separate from
    # ``last_scene_splats`` so a mixed hit/miss run never reports a reused scene
    # as if it were the reconstruction.
    reconstructed_splats = 0
    last_mask_coverage = None

    for index, candidate in enumerate(candidates, start=1):
        if request.mode == "implicit" and len(clips) >= settings.query_max_results:
            break
        clip_frames, clip_fps = read_interval(
            source, candidate.start_seconds, candidate.end_seconds - candidate.start_seconds, settings.query_result_fps
        )
        processed += 1
        yield _event(
            "candidate_decode", "running",
            f"Decoded candidate {index}/{total_candidates} at {timestamp_label(candidate.start_seconds)}.",
            metrics={"candidate_index": index, "total_candidates": total_candidates, "frames": len(clip_frames)},
        )
        track_stream = ground_and_track_steps(
            clip_frames, request.query, threshold=settings.grounding_threshold,
            candidate_index=index, total_candidates=total_candidates, stage_ids=_GROUND_STAGE_IDS,
        )
        while True:
            try:
                tick = next(track_stream)
            except StopIteration as finished:
                track = finished.value
                break
            # A detection tick carries ``confidence`` only when Grounding DINO
            # actually matched, so the terminal's match count is a measured fact
            # rather than an inference from the later mask acceptance.
            if tick.stage_id == "dino_detect" and "confidence" in tick.metrics:
                dino_matched += 1
            yield tick
        if track is None:
            continue
        accepted += 1
        semantic_masks = track.masks
        semantic_coverage = float(semantic_masks.mean())
        projected_masks: np.ndarray | None = None
        projected_coverage: float | None = None
        projected_low_specificity = False
        selected_splats: tuple = ()
        method_score = candidate.score
        if request.mode == "explicit":
            scene_stream = _consume_scene(clip_frames, clip_fps, digest, candidate.start_seconds, settings)
            while True:
                try:
                    yield next(scene_stream)
                except StopIteration as finished:
                    scene = finished.value
                    break
            last_scene_key = scene.key
            # ``splats`` belongs to the real CachedScene; a lightweight test
            # double may omit it, so count defensively rather than fabricate.
            last_scene_splats = len(getattr(scene, "splats", ()))
            if getattr(scene, "cache_hit", False):
                scene_hits += 1
            else:
                scene_misses += 1
                reconstructed_splats = len(getattr(scene, "splats", ()))
                yield _event(
                    "scene_cache", "running", "No complete cached scene for this candidate; building one.",
                    metrics={"cache": "miss", "scene_key": scene.key},
                )
            primitive_mask = lift_masks(scene, semantic_masks)
            yield _event(
                "mask_lift", "running",
                f"Voted the tracked region onto {int(primitive_mask.selected.sum())} persistent Gaussian primitives.",
                metrics={
                    "candidate_index": index,
                    "primitive_support": round(primitive_mask.support, 4),
                    "selected_primitives": int(primitive_mask.selected.sum()),
                },
            )
            lifted += 1
            projected_masks = project_primitive_mask(scene, primitive_mask)
            if not projected_masks.any():
                # Record the empty projection explicitly rather than going silent.
                empty_projections += 1
                yield _event(
                    "mask_project", "running",
                    "Projection produced no visible primitive footprints for this candidate.",
                    metrics={"candidate_index": index, "projected": "empty"},
                )
                continue
            projected_coverage = float(projected_masks.mean())
            projected_low_specificity = projected_coverage >= settings.low_specificity_coverage or primitive_mask.support >= settings.low_specificity_coverage
            yield _event(
                "mask_project", "running",
                "Projected the selected primitives back into source-view binary masks.",
                metrics={
                    "candidate_index": index,
                    "projected_mask_coverage": round(projected_coverage, 4),
                    "projected_low_specificity": projected_low_specificity,
                },
            )
            projected += 1
            method_score += primitive_mask.support * 0.05
            np.save(artifacts.file(f"clips/{len(clips):02d}_primitive_mask.npy"), primitive_mask.selected)
            selected_splats = export_selected_splats(scene, primitive_mask, artifacts.file(f"selected_splats/{len(clips):02d}/.marker").parent) if hasattr(scene, "checkpoint") else ()
            yield _event(
                "splat_export", "running",
                f"Exported {len(selected_splats)} selected splat frames for this candidate.",
                metrics={"candidate_index": index, "selected_splats": len(selected_splats)},
            )
            exported += 1
        else:
            last_mask_coverage = semantic_coverage
            resolved += 1
            yield _event(
                "resolve_method", "running",
                "Using the tracked video mask directly for Implicit 3D highlighting.",
                metrics={"candidate_index": index, "mask_coverage": round(semantic_coverage, 4)},
            )
        clip_index = len(clips)
        raw_path = artifacts.file(f"clips/{clip_index:02d}_source.mp4")
        semantic_path = artifacts.file(f"clips/{clip_index:02d}_semantic_highlighted.mp4")
        semantic_frames = highlighted_frames(clip_frames, semantic_masks)
        yield _event(
            "overlay", "running",
            f"Rendered {len(semantic_frames)} highlighted frames for candidate {index}/{total_candidates}.",
            metrics={"candidate_index": index, "highlighted_frames": len(semantic_frames)},
        )
        write_video(clip_frames, raw_path, clip_fps)
        write_video(semantic_frames, semantic_path, clip_fps)
        np.save(artifacts.file(f"clips/{clip_index:02d}_semantic_mask.npy"), semantic_masks.astype(np.uint8))
        projected_path = None
        if projected_masks is not None:
            projected_path = artifacts.file(f"clips/{clip_index:02d}_projected_highlighted.mp4")
            np.save(artifacts.file(f"clips/{clip_index:02d}_projected_mask.npy"), projected_masks.astype(np.uint8))
            write_video(highlighted_frames(clip_frames, projected_masks), projected_path, clip_fps)
        preview_path = artifacts.file(f"previews/{clip_index:02d}_semantic_highlighted.png")
        imageio.imwrite(preview_path, semantic_frames[len(semantic_frames) // 2])
        clip = ClipArtifact(raw_path, semantic_path, candidate.start_seconds + source_offset, candidate.end_seconds + source_offset, float(method_score), request.mode, projected_path, track.box.confidence, semantic_coverage, projected_coverage, projected_low_specificity, selected_splats)
        clips.append(clip)
        highlighted_total += len(semantic_frames)
        yield _event(
            "write_clips", "running",
            f"Saved highlighted rerank candidate {len(clips)}/{total_candidates} without spatial cropping.",
            preview=preview_path,
            metrics={
                "saved_clips": len(clips),
                "semantic_mask_coverage": round(semantic_coverage, 4),
                "projected_mask_coverage": round(projected_coverage, 4) if projected_coverage is not None else None,
                "low_spatial_specificity": projected_low_specificity,
            },
        )

    # ``candidate_decode`` is the ranked-candidate loop's own row; a zero-candidate
    # run still completes it (and the propagation row) with a real zero count.
    yield _event("candidate_decode", "complete", f"Decoded {processed} of {total_candidates} ranked candidates.", metrics={"decoded_candidates": processed, "total_candidates": total_candidates})
    yield _event("dino_detect", "complete", f"Grounding DINO matched {dino_matched} of {total_candidates} candidates.", metrics={"matched_candidates": dino_matched, "total_candidates": total_candidates})
    yield _event("sam2_setup", "complete", f"Prepared SAM2 for {dino_matched} of {total_candidates} candidates.", metrics={"tracked_candidates": dino_matched, "total_candidates": total_candidates})
    yield _event("sam2_propagate", "complete", f"Propagated SAM2 masks for {dino_matched} of {total_candidates} candidates.", metrics={"tracked_candidates": dino_matched, "total_candidates": total_candidates})
    yield _event("mask_validate", "complete", f"Validated {accepted} of {dino_matched} tracked candidates.", metrics={"accepted_candidates": accepted, "total_candidates": total_candidates})
    if request.mode == "implicit":
        yield _event(
            "resolve_method", "complete", f"Used tracked video masks directly for {resolved} candidates.",
            metrics={"tracked_candidates": resolved, "mask_coverage": round(last_mask_coverage, 4) if last_mask_coverage is not None else None},
        )
    else:
        # Aggregate the per-candidate cache outcome explicitly. A run where no
        # candidate reached scene caching is "none", not a misleading "miss", and
        # one that both reused and rebuilt is "mixed" rather than a single bucket.
        # ``checked_candidates`` is the real denominator for the two counts.
        checked_candidates = scene_hits + scene_misses
        if scene_hits and scene_misses:
            cache_status = "mixed"
        elif scene_hits:
            cache_status = "hit"
        elif scene_misses:
            cache_status = "miss"
        else:
            cache_status = "none"
        yield _event(
            "scene_cache", "complete", f"Scene cache: {scene_hits} hits, {scene_misses} misses.",
            metrics={
                "cache": cache_status,
                "checked_candidates": checked_candidates,
                "hit_candidates": scene_hits,
                "miss_candidates": scene_misses,
                "scene_key": last_scene_key,
            },
        )
        # Only claim a reconstruction when a miss actually rebuilt a scene. A run
        # with no checked candidate reports no scene at all rather than a
        # fabricated zero-splat one; a pure cache hit reports the reused scene.
        reconstruct_metrics = {"scene_key": last_scene_key}
        if scene_misses:
            reconstruct_metrics["cache"] = "miss"
            reconstruct_metrics["splats"] = reconstructed_splats
            reconstruct_detail = f"Scene ready with {reconstructed_splats} splat frames."
        elif scene_hits:
            reconstruct_metrics["cache"] = "hit"
            reconstruct_metrics["splats"] = last_scene_splats
            reconstruct_detail = f"Scene ready with {last_scene_splats} splat frames."
        else:
            reconstruct_detail = "No candidate reached scene caching; no scene was built."
        yield _event("scene_reconstruct", "complete", reconstruct_detail, metrics=reconstruct_metrics)
        yield _event("mask_lift", "complete", f"Lifted masks for {lifted} candidates.", metrics={"lifted_candidates": lifted})
        yield _event("mask_project", "complete", f"Projected {projected} candidates; {empty_projections} produced an empty projection.", metrics={"projected_candidates": projected, "empty_projections": empty_projections})
        yield _event("splat_export", "complete", f"Exported selected splats for {exported} candidates.", metrics={"exported_candidates": exported})
    yield _event("overlay", "complete", f"Rendered {highlighted_total} highlighted frames across saved clips.", metrics={"highlighted_frames": highlighted_total})
    yield _event("write_clips", "complete", f"Saved {len(clips)} playable highlighted source clips; no result was spatially cropped.", metrics={"saved_clips": len(clips)})

    # Explicit mode deliberately evaluates every candidate before its 3D support
    # rerank; only the best five are exposed as final user-facing clips.
    ranked_clips = len(clips)
    clips.sort(key=lambda clip: (-clip.score, clip.start_seconds))
    clips = clips[:settings.query_max_results]
    yield _event("final_rerank", "running", "Sorting accepted clips by score and start time and truncating to the result limit.", metrics={"ranked_clips": ranked_clips, "kept": len(clips)})
    yield _event("final_rerank", "complete", f"Kept {len(clips)} of {ranked_clips} accepted clips.", metrics={"ranked_clips": ranked_clips, "kept": len(clips)})
    if not clips:
        raise ValueError("No reliable text-grounded regions were found. Try a more concrete visual query.")
    rows = [[timestamp_label(clip.start_seconds), timestamp_label(clip.end_seconds),
             round(clip.score, 3), clip.method] for clip in clips]
    yield FeatureResult(clips=clips, rows=rows, metadata={
        "query": request.query.strip(), "mode": request.mode, "source_seconds": info.duration_seconds,
        "trim_start_seconds": source_offset, "trim_end_seconds": trim_end,
        "clip_seconds": settings.query_clip_seconds, "generated": False,
        "provenance": {"pipeline_revision": TEXT_QUERY_PIPELINE_REVISION, "source_sha256": digest, "semantic_model": getattr(settings, "semantic_model_id", "test-model"), "grounding_model": getattr(settings, "grounding_model_id", "test-model"), "segmentation_model": getattr(settings, "segmentation_model_id", "test-model")},
        "viewer": None,
    })
