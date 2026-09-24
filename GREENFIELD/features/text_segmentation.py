"""Text-seeded binary video masks and non-destructive highlight rendering."""

from dataclasses import dataclass

import cv2
import numpy as np

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


def track_box(frames: list[np.ndarray], box: GroundedBox, anchor_index: int) -> np.ndarray:
    """Seed SAM2 at one grounded frame and propagate the region both directions."""
    if not frames:
        raise ValueError("Video tracking needs at least one frame.")
    if not 0 <= anchor_index < len(frames):
        raise ValueError("The segmentation anchor must be inside the clip.")
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
        for prediction in model.propagate_in_video_iterator(
            session, start_frame_idx=anchor_index, max_frame_num_to_track=len(frames), reverse=reverse
        ):
            masks[int(prediction.frame_idx)] |= _output_mask(prediction, width, height)
    return masks


def ground_and_track(frames: list[np.ndarray], query: str, anchor_index: int | None = None) -> MaskTrack | None:
    """Create a reliable text-grounded binary video region or return no match."""
    if not frames:
        return None
    anchor = len(frames) // 2 if anchor_index is None else anchor_index
    grounded = ground_query(frames[anchor], query)
    if grounded is None:
        return None
    masks = track_box(frames, grounded, anchor)
    # A tracker occasionally returns an empty/degenerate region after a missed
    # target. Do not turn that into a misleading full-frame highlight.
    if not masks.any() or float(masks.mean()) < 0.0001:
        return None
    return MaskTrack(masks=masks, box=grounded, anchor_index=anchor)


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
