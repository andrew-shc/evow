"""Export a native 3D Gaussian frame in the standard 32-byte .splat format."""

from pathlib import Path

import numpy as np


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
