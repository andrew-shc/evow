"""Normalize monocular inverse depth into a stable 3D initialization prior."""

import numpy as np


def scene_depths(relative_depths: np.ndarray) -> np.ndarray:
    # Video Depth Anything produces relative inverse-depth-like values. Gaussian
    # optimization starts with arbitrary scene units; these are not metric distances.
    finite = relative_depths[np.isfinite(relative_depths)]
    if finite.size == 0:
        raise ValueError("The depth model returned no finite values.")
    low, high = np.percentile(finite, [2, 98])
    if high - low < 1e-6:
        return np.full_like(relative_depths, 4.0, dtype=np.float32)
    normalized = np.clip((relative_depths - low) / (high - low), 0.0, 1.0)
    return (1.0 + 7.0 * (1.0 - normalized)).astype(np.float32)
