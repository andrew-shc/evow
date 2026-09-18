"""Shared virtual camera geometry for explicit and implicit replay."""

from dataclasses import dataclass
import math

import numpy as np


@dataclass(frozen=True)
class ViewRequest:
    yaw_degrees: float
    lateral_shift: float
    fov_degrees: float

    def validate(self) -> None:
        if not -15.0 <= self.yaw_degrees <= 15.0:
            raise ValueError("Yaw must be between -15 and 15 degrees.")
        if not -0.5 <= self.lateral_shift <= 0.5:
            raise ValueError("Lateral shift must be between -0.5 and 0.5 scene units.")
        if not 35.0 <= self.fov_degrees <= 110.0:
            raise ValueError("Field of view must be between 35 and 110 degrees.")


def intrinsics(height: int, width: int, fov_degrees: float) -> np.ndarray:
    focal = 0.5 * width / math.tan(math.radians(fov_degrees) / 2.0)
    return np.array(
        [[focal, 0.0, (width - 1) / 2.0],
         [0.0, focal, (height - 1) / 2.0],
         [0.0, 0.0, 1.0]],
        dtype=np.float32,
    )


def target_cam_to_world(request: ViewRequest) -> np.ndarray:
    yaw = math.radians(request.yaw_degrees)
    cosine, sine = math.cos(yaw), math.sin(yaw)
    pose = np.eye(4, dtype=np.float32)
    pose[:3, :3] = np.array(
        [[cosine, 0.0, sine],
         [0.0, 1.0, 0.0],
         [-sine, 0.0, cosine]],
        dtype=np.float32,
    )
    pose[0, 3] = request.lateral_shift
    return pose
