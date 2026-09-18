"""Reject clips whose dominant image motion indicates a moving physical camera."""

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class StaticCameraMetrics:
    """Global motion measured from background-consensus feature tracks."""

    valid_pairs: int
    median_translation_px: float
    peak_translation_px: float
    cumulative_translation_px: float
    median_rotation_degrees: float
    cumulative_rotation_degrees: float

    def as_dict(self) -> dict[str, float | int | bool]:
        return {
            "static_camera_verified": True,
            "motion_pairs": self.valid_pairs,
            "median_translation_px": round(self.median_translation_px, 3),
            "peak_translation_px": round(self.peak_translation_px, 3),
            "cumulative_translation_px": round(self.cumulative_translation_px, 3),
            "median_rotation_degrees": round(self.median_rotation_degrees, 3),
            "cumulative_rotation_degrees": round(self.cumulative_rotation_degrees, 3),
        }


def require_static_camera(
    frames: np.ndarray,
    *,
    min_pairs: int,
    min_tracked_features: int,
    max_median_translation_px: float,
    max_peak_translation_px: float,
    max_cumulative_translation_px: float,
    max_median_rotation_degrees: float,
    max_cumulative_rotation_degrees: float,
) -> StaticCameraMetrics:
    """Verify that RANSAC background motion stays compatible with a fixed camera.

    Trees, clouds, and other moving objects are allowed: only a transform
    supported by the dominant set of tracked features counts as camera motion.
    """
    grayscale = [
        cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        for frame in np.asarray(frames, dtype=np.uint8)
    ]
    translations: list[float] = []
    rotations: list[float] = []
    for previous, current in zip(grayscale, grayscale[1:]):
        features = cv2.goodFeaturesToTrack(
            previous, maxCorners=600, qualityLevel=0.01, minDistance=8, blockSize=7,
        )
        if features is None:
            continue
        next_features, tracked, _ = cv2.calcOpticalFlowPyrLK(
            previous, current, features, None,
            winSize=(21, 21), maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
        )
        if next_features is None or tracked is None:
            continue
        source = features[tracked.ravel() == 1].reshape(-1, 2)
        target = next_features[tracked.ravel() == 1].reshape(-1, 2)
        if len(source) < min_tracked_features:
            continue
        affine, inliers = cv2.estimateAffinePartial2D(
            source, target, method=cv2.RANSAC, ransacReprojThreshold=1.5,
        )
        if affine is None or inliers is None or int(inliers.sum()) < min_tracked_features:
            continue
        translations.append(float(np.hypot(affine[0, 2], affine[1, 2])))
        rotations.append(float(abs(np.degrees(np.arctan2(affine[1, 0], affine[0, 0])))))
    if len(translations) < min_pairs:
        raise ValueError(
            "Could not verify a fixed camera from this clip. Keep the camera locked "
            "on a view with stable background detail, then try again."
        )

    metrics = StaticCameraMetrics(
        valid_pairs=len(translations),
        median_translation_px=float(np.median(translations)),
        peak_translation_px=float(np.max(translations)),
        cumulative_translation_px=float(np.sum(translations)),
        median_rotation_degrees=float(np.median(rotations)),
        cumulative_rotation_degrees=float(np.sum(rotations)),
    )
    moving = (
        metrics.median_translation_px > max_median_translation_px
        or metrics.peak_translation_px > max_peak_translation_px
        or metrics.cumulative_translation_px > max_cumulative_translation_px
        or metrics.median_rotation_degrees > max_median_rotation_degrees
        or metrics.cumulative_rotation_degrees > max_cumulative_rotation_degrees
    )
    if moving:
        raise ValueError(
            "This clip appears to move the physical camera "
            f"(median background motion {metrics.median_translation_px:.2f}px; "
            f"cumulative {metrics.cumulative_translation_px:.2f}px). "
            "Use a tripod or a camera fixed on one view of the world."
        )
    return metrics
