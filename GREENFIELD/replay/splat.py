"""Export a native 3D Gaussian frame in the standard 32-byte .splat format."""

from pathlib import Path

import numpy as np


def timeline_keyframe_indices(frame_count: int, fps: float, timeline_fps: float) -> tuple[int, ...]:
    """Select sparse timeline samples while always retaining the final frame."""
    if frame_count < 1 or fps <= 0 or timeline_fps <= 0:
        raise ValueError("Timeline sampling requires positive frame count and rates.")
    sample_count = max(1, int(np.ceil(frame_count * timeline_fps / fps)))
    indices = [min(frame_count - 1, round(sample * fps / timeline_fps)) for sample in range(sample_count)]
    if indices[-1] != frame_count - 1:
        indices.append(frame_count - 1)
    return tuple(dict.fromkeys(indices))


def timeline_keyframe_durations(indices: tuple[int, ...], fps: float) -> tuple[float, ...]:
    """Give each sparse scene its real display interval, including the endpoint."""
    if not indices or fps <= 0:
        raise ValueError("Timeline durations require indices and a positive frame rate.")
    durations = [max((following - current) / fps, 1 / fps) for current, following in zip(indices, indices[1:])]
    return tuple(durations + [1 / fps])


def save_splat(
    path: Path,
    centers: np.ndarray,
    scales: np.ndarray,
    colors: np.ndarray,
    opacities: np.ndarray,
    quaternions: np.ndarray,
) -> None:
    """Write sorted Gaussian primitives for browser splat viewers.

    Each record stores xyz, xyz scale, RGBA, then a wxyz quaternion. Front to back
    sorting helps streaming viewers display the initial camera view promptly.
    """
    count = len(centers)
    if not all(len(values) == count for values in (scales, colors, opacities, quaternions)):
        raise ValueError("Gaussian attribute counts disagree.")
    primitive = np.dtype([
        ("position", "<f4", (3,)),
        ("scale", "<f4", (3,)),
        ("rgba", "u1", (4,)),
        ("rotation", "u1", (4,)),
    ])
    records = np.empty(count, dtype=primitive)
    records["position"] = centers.astype(np.float32)
    records["scale"] = scales.astype(np.float32)
    records["rgba"][:, :3] = np.clip(colors * 255, 0, 255).astype(np.uint8)
    records["rgba"][:, 3] = np.clip(opacities * 255, 0, 255).astype(np.uint8)
    normalized = quaternions / np.maximum(np.linalg.norm(quaternions, axis=1, keepdims=True), 1e-8)
    records["rotation"] = np.clip(normalized * 128 + 128, 0, 255).astype(np.uint8)
    order = np.argsort(-centers[:, 2])
    path.write_bytes(records[order].tobytes())
