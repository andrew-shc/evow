"""Text-seeded binary video masks and non-destructive highlight rendering."""

from dataclasses import dataclass
from typing import Iterator, Mapping

import cv2
import numpy as np

from GREENFIELD.app_core.contracts import StageEvent
from .text_models import GroundedBox, device, ground_query, sam2_video_model


@dataclass(frozen=True)
class MaskTrack:
    """A grounded box and its binary per-frame tracked masks."""

    masks: np.ndarray
    box: GroundedBox
    anchor_index: int


def _output_mask(prediction, width: int, height: int) -> np.ndarray:
    """Convert a SAM2 logit tensor into one source-resolution binary mask."""
    logits = prediction.pred_masks.detach().float().cpu().numpy()
    while logits.ndim > 2:
        logits = logits[0]
    mask = (logits > 0).astype(np.uint8)
    return cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST).astype(bool)


def track_box_steps(
    frames: list[np.ndarray],
    box: GroundedBox,
    anchor_index: int,
    *,
    candidate_index: int = 1,
    total_candidates: int = 1,
    stage_ids: Mapping[str, str],
) -> Iterator[StageEvent]:
    """Yield SAM2 setup/propagation ticks and return per-frame masks.

    The two propagation directions (``reverse=False`` then ``reverse=True``) are
    relayed as one aggregate row: every tick carries the candidate position, the
    ``reversed`` flag, and the running ``propagated_frames`` count, and the caller
    owns the single terminal.
    """
    if not frames:
        raise ValueError("Video tracking needs at least one frame.")
    if not 0 <= anchor_index < len(frames):
        raise ValueError("The segmentation anchor must be inside the clip.")
    yield StageEvent(
        "Prepare SAM2 tracking", "running",
        f"Initializing a SAM2 video session for candidate {candidate_index}/{total_candidates}.",
        metrics={"candidate_index": candidate_index, "frames": len(frames)}, stage_id=stage_ids["sam2_setup"],
    )
    processor, model = sam2_video_model()
    height, width = frames[0].shape[:2]
    # The processor cached video tensors must use the same dtype as SAM2
    # parameters. This is deliberately derived from the loaded model rather
    # than assuming CUDA always means float16.
    dtype = next(model.parameters()).dtype
    session = processor.init_video_session(
        video=frames,
        inference_device=device(),
        inference_state_device=device(),
        processing_device=device(),
        video_storage_device="cpu",
        dtype=dtype,
    )
    processor.add_inputs_to_inference_session(
        session,
        frame_idx=anchor_index,
        obj_ids=1,
        input_boxes=[[list(box.xyxy)]],
    )
    masks = np.zeros((len(frames), height, width), dtype=bool)
    for reverse in (False, True):
        propagated = 0
        for prediction in model.propagate_in_video_iterator(
            session, start_frame_idx=anchor_index, max_frame_num_to_track=len(frames), reverse=reverse
        ):
            masks[int(prediction.frame_idx)] |= _output_mask(prediction, width, height)
            propagated += 1
            yield StageEvent(
                "Propagate SAM2 masks", "running",
                f"Propagating {'backward' if reverse else 'forward'} through candidate {candidate_index}/{total_candidates}.",
                metrics={
                    "candidate_index": candidate_index,
                    "reversed": reverse,
                    "propagated_frames": propagated,
                    "total_frames": len(frames),
                },
                stage_id=stage_ids["sam2_propagate"],
            )
    return masks


def track_box(frames: list[np.ndarray], box: GroundedBox, anchor_index: int) -> np.ndarray:
    """Seed SAM2 at one grounded frame and propagate the region both directions."""
    steps = track_box_steps(frames, box, anchor_index, stage_ids=_NO_STAGE_IDS)
    try:
        while True:
            next(steps)
    except StopIteration as finished:
        return finished.value


def ground_and_track_steps(
    frames: list[np.ndarray],
    query: str,
    anchor_index: int | None = None,
    threshold: float = 0.25,
    *,
    candidate_index: int = 1,
    total_candidates: int = 1,
    stage_ids: Mapping[str, str],
) -> Iterator[StageEvent]:
    """Yield Text Query's atomic grounding/tracking ticks, returning a mask or None.

    The detector, the SAM2 session, both propagation directions, and the
    degenerate-mask check are each a tick on their own declared row; the caller
    owns each row's single terminal. A candidate whose detector finds nothing
    records ``matched: false`` rather than fabricating a confidence.
    """
    if not frames:
        return None
    anchor = len(frames) // 2 if anchor_index is None else anchor_index
    yield StageEvent(
        "Ground target with Grounding DINO", "running",
        f"Detecting the query in candidate {candidate_index}/{total_candidates}.",
        metrics={"candidate_index": candidate_index, "total_candidates": total_candidates},
        stage_id=stage_ids["dino_detect"],
    )
    grounded = ground_query(frames[anchor], query, threshold)
    if grounded is None:
        yield StageEvent(
            "Ground target with Grounding DINO", "running",
            f"No text-grounded target found in candidate {candidate_index}/{total_candidates}.",
            metrics={"candidate_index": candidate_index, "total_candidates": total_candidates, "matched": False},
            stage_id=stage_ids["dino_detect"],
        )
        return None
    yield StageEvent(
        "Ground target with Grounding DINO", "running",
        f"Grounding DINO found the target in candidate {candidate_index}/{total_candidates}.",
        metrics={"candidate_index": candidate_index, "total_candidates": total_candidates, "confidence": round(float(grounded.confidence), 4)},
        stage_id=stage_ids["dino_detect"],
    )
    masks = yield from track_box_steps(
        frames, grounded, anchor,
        candidate_index=candidate_index, total_candidates=total_candidates, stage_ids=stage_ids,
    )
    coverage = float(masks.mean())
    yield StageEvent(
        "Validate tracked masks", "running",
        f"Checking candidate {candidate_index}/{total_candidates} for a degenerate tracked region.",
        metrics={"candidate_index": candidate_index, "mask_coverage": round(coverage, 4)},
        stage_id=stage_ids["mask_validate"],
    )
    # A tracker occasionally returns an empty/degenerate region after a missed
    # target. Do not turn that into a misleading full-frame highlight.
    if not masks.any() or coverage < 0.0001:
        return None
    return MaskTrack(masks=masks, box=grounded, anchor_index=anchor)


def ground_and_track(frames: list[np.ndarray], query: str, anchor_index: int | None = None, threshold: float = 0.25) -> MaskTrack | None:
    """Create a reliable text-grounded binary video region or return no match.

    ``threshold`` is forwarded to Grounding DINO so Text Query can tune its
    detector confidence per run; the 0.25 default is the previous hardcoded value.
    """
    steps = ground_and_track_steps(frames, query, anchor_index, threshold, stage_ids=_NO_STAGE_IDS)
    try:
        while True:
            next(steps)
    except StopIteration as finished:
        return finished.value


# Editing consumes the plain ``ground_and_track`` wrapper (it has no per-row
# requirement), so the atomic steps emit under blank ids that no trace renders.
_NO_STAGE_IDS: Mapping[str, str] = {
    "dino_detect": "", "sam2_setup": "", "sam2_propagate": "", "mask_validate": "",
}


def full_scene_masks(frames: list[np.ndarray]) -> np.ndarray:
    """Return VACE's all-white generation mask for environmental instructions."""
    if not frames:
        raise ValueError("A full-scene edit needs at least one source frame.")
    height, width = frames[0].shape[:2]
    return np.ones((len(frames), height, width), dtype=bool)


def expand_masks(masks: np.ndarray, radius: int = 5) -> np.ndarray:
    """Expand a binary edit mask slightly so VACE can blend target boundaries."""
    if masks.ndim != 3:
        raise ValueError("Expected one binary mask per video frame.")
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (radius * 2 + 1, radius * 2 + 1))
    return np.asarray([cv2.dilate(mask.astype(np.uint8), kernel).astype(bool) for mask in masks])


def highlighted_frames(frames: list[np.ndarray], masks: np.ndarray) -> list[np.ndarray]:
    """Overlay yellow fill and contour while retaining every nonmasked source pixel."""
    if len(frames) != len(masks):
        raise ValueError("Highlight frames and masks must have the same length.")
    color = np.array((255, 215, 0), dtype=np.float32)
    output: list[np.ndarray] = []
    for frame, mask in zip(frames, masks):
        if frame.shape[:2] != mask.shape:
            raise ValueError("Binary mask resolution must match its source frame.")
        rendered = np.asarray(frame, dtype=np.uint8).copy()
        selected = mask.astype(bool)
        if selected.any():
            blended = rendered.astype(np.float32)
            blended[selected] = blended[selected] * 0.55 + color * 0.45
            rendered = blended.clip(0, 255).astype(np.uint8)
            # Keep the contour inside the selected region so pixels outside
            # the binary mask remain byte-for-byte source pixels.
            interior = cv2.erode(selected.astype(np.uint8), np.ones((3, 3), dtype=np.uint8)).astype(bool)
            outline = selected & ~interior
            rendered[outline] = np.array((255, 245, 80), dtype=np.uint8)
        output.append(rendered)
    return output
