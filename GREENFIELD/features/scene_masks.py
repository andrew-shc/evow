"""Lift tracked 2D masks into cached Gaussian primitives and project them back."""

from dataclasses import dataclass
import math
from pathlib import Path

import cv2
import numpy as np

from .scene_cache import CachedScene
from GREENFIELD.replay.splat import save_splat


@dataclass(frozen=True)
class PrimitiveMask:
    """Per-Gaussian binary membership derived from visible tracked masks."""

    selected: np.ndarray
    support: float


def _scene_arrays(scene: CachedScene) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load the small geometry subset needed for source-view mask projection."""
    import torch

    state = torch.load(scene.checkpoint, map_location="cpu", weights_only=True)
    means = state["means"].detach().cpu().numpy().astype(np.float32)
    motion = state["motion"].detach().cpu().numpy().astype(np.float32)
    scales = state["log_scales"].detach().cpu().numpy().astype(np.float32)
    return means, motion, np.exp(scales)


def _project(points: np.ndarray, width: int, height: int, fov_degrees: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Project source-camera Gaussian centers without claiming metric calibration."""
    focal = 0.5 * width / math.tan(math.radians(fov_degrees) / 2)
    depth = points[:, 2]
    valid = depth > 1e-5
    x = np.rint(focal * points[:, 0] / np.maximum(depth, 1e-5) + width / 2).astype(np.int32)
    y = np.rint(focal * points[:, 1] / np.maximum(depth, 1e-5) + height / 2).astype(np.int32)
    valid &= (x >= 0) & (x < width) & (y >= 0) & (y < height)
    return x, y, valid


def lift_masks(scene: CachedScene, masks: np.ndarray, threshold: float = 0.30) -> PrimitiveMask:
    """Vote a tracked binary region onto persistent Gaussian primitive identities."""
    if masks.ndim != 3 or len(masks) != len(scene.splats):
        raise ValueError("Scene masks must contain one source-resolution frame per cached scene frame.")
    means, motion, _scales = _scene_arrays(scene)
    votes = np.zeros(len(means), dtype=np.float32)
    observations = np.zeros(len(means), dtype=np.float32)
    for index, mask in enumerate(masks):
        x, y, visible = _project(means + motion[index], scene.width, scene.height, scene.source_fov_degrees)
        observations[visible] += 1
        votes[visible] += mask[y[visible], x[visible]].astype(np.float32)
    fractions = votes / np.maximum(observations, 1)
    selected = fractions >= threshold
    if not selected.any() and votes.any():
        # Preserve the strongest observed primitive rather than claiming a broad
        # region if the tracker only saw a tiny valid area.
        selected[int(fractions.argmax())] = True
    return PrimitiveMask(selected=selected, support=float(selected.mean()))


def project_primitive_mask(scene: CachedScene, primitive_mask: PrimitiveMask) -> np.ndarray:
    """Render selected Gaussian footprints as binary source-view masks on CPU."""
    means, motion, scales = _scene_arrays(scene)
    selected = primitive_mask.selected.astype(bool)
    if len(selected) != len(means):
        raise ValueError("Primitive membership does not match the cached scene.")
    focal = 0.5 * scene.width / math.tan(math.radians(scene.source_fov_degrees) / 2)
    output = np.zeros((len(scene.splats), scene.height, scene.width), dtype=np.uint8)
    for index in range(len(output)):
        points = means + motion[index]
        x, y, visible = _project(points, scene.width, scene.height, scene.source_fov_degrees)
        active = np.flatnonzero(selected & visible)
        # Drawing far-to-near preserves the scene's coarse footprint ordering;
        # the result remains binary because it is only a conditioning mask.
        for primitive in active[np.argsort(points[active, 2])[::-1]]:
            radius = max(1, int(round(focal * float(max(scales[primitive, :2])) / max(points[primitive, 2], 1e-5))))
            cv2.circle(output[index], (int(x[primitive]), int(y[primitive])), radius, 1, thickness=-1)
    return output.astype(bool)


def export_selected_splats(scene: CachedScene, primitive_mask: PrimitiveMask, destination: Path) -> tuple[Path, ...]:
    """Export the complete 4D scene, coloring only text-selected primitives magenta."""
    import torch

    state = torch.load(scene.checkpoint, map_location="cpu", weights_only=True)
    selected = primitive_mask.selected.astype(bool)
    if not selected.any():
        return ()
    destination.mkdir(parents=True, exist_ok=True)
    means = state["means"].detach().cpu().numpy().astype(np.float32)
    motion = state["motion"].detach().cpu().numpy().astype(np.float32)
    scales = np.exp(state["log_scales"].detach().cpu().numpy().astype(np.float32))
    opacity_logits = state["opacity_logits"].detach().cpu().numpy()
    opacity = 1 / (1 + np.exp(-opacity_logits))
    quaternions = state["quaternions"].detach().cpu().numpy().astype(np.float32)
    base_color_logits = state["color_logits"].detach().cpu().numpy().astype(np.float32)
    temporal_color = state["temporal_color"].detach().cpu().numpy().astype(np.float32)
    paths = []
    for index in range(len(scene.splats)):
        # Keep the whole reconstructed scene for context. The same mask vote
        # used by the query recolors only its persistent primitive identities.
        colors = 1 / (1 + np.exp(-(base_color_logits + temporal_color[index])))
        colors[selected] = np.asarray((1.0, 0.0, 0.8), dtype=np.float32)
        path = destination / f"selected_{index:03d}.splat"
        save_splat(path, means + motion[index], scales, colors, opacity, quaternions)
        paths.append(path)
    return tuple(paths)
