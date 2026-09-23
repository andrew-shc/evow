"""Instruction-guided video editing adapter with motion-aware stabilization."""

from dataclasses import dataclass
from typing import Iterator

import cv2
import numpy as np

from GREENFIELD.app_core.contracts import FeatureResult, RunArtifacts, StageEvent, StageSpec
from GREENFIELD.app_core.media import write_video
from .model_adapters import edit_frame


STAGES = (
    StageSpec("Decode source", "Read the supplied recording into a bounded editing episode.", "GREENFIELD/features/editing.py", "def run(", "run() → app_core.media.read_video()", "source video", "bounded RGB frames · fps", "Source"),
    StageSpec("Apply instruction", "Apply the text-directed image-edit model to the episode.", "GREENFIELD/features/editing.py", "def run(", "run() → model_adapters.edit_frame() → InstructPix2Pix", "frames · instruction · seed", "edited RGB frames", "Instruction · Methodology · Seed"),
    StageSpec("Stabilize and render", "Use motion-aware blending to reduce flicker and save the generated video.", "GREENFIELD/features/editing.py", "def _stabilize(", "_stabilize() → Farneback optical flow → write_video()", "edited frames · fps", "edited.mp4 · trace.json", "Apply"),
)


@dataclass(frozen=True)
class EditingRequest:
    mode: str
    prompt: str
    seed: int

    def validate(self) -> None:
        if self.mode not in {"implicit", "explicit"}:
            raise ValueError("Choose Implicit video or Explicit 3D.")
        if not self.prompt.strip():
            raise ValueError("Enter an editing instruction.")


def run(frames: list[np.ndarray], fps: float, request: EditingRequest, artifacts: RunArtifacts) -> Iterator[StageEvent | FeatureResult]:
    """Use the configured Diffusers edit worker and preserve temporal continuity."""
    request.validate()
    yield StageEvent("Decode source", "complete", f"Loaded {len(frames)} observed RGB frames.", metrics={"frames": len(frames), "fps": fps})
    yield StageEvent("Apply instruction", "running", "Loading local instruction-edit model.")
    edited = []
    for completed, frame in enumerate(frames, 1):
        edited.append(edit_frame(frame, request.prompt, request.seed + completed - 1))
        yield StageEvent("Apply instruction", "running", "Edited {}/{} observed frames.".format(completed, len(frames)), metrics={"edited_frames": completed, "total_frames": len(frames)})
    yield StageEvent("Apply instruction", "complete", "Diffusers instruction-edit adapter prepared the edited episode.", metrics={"edited_frames": len(edited)})
    yield StageEvent("Stabilize and render", "running", "Stabilizing edited frames and encoding the result.")
    result = write_video(_stabilize(edited), artifacts.file("edited.mp4"), fps)
    yield StageEvent("Stabilize and render", "complete", "Saved generated edit with motion-aware temporal blending.", metrics={"output_frames": len(edited), "output_fps": fps})
    yield FeatureResult(primary=result, metadata={"generated": True, "mode": request.mode, "seed": request.seed})


def _stabilize(frames: list[np.ndarray]) -> list[np.ndarray]:
    """Blend each model frame with a Farneback-warped predecessor."""
    if not frames:
        return []
    output = [frames[0]]
    for current in frames[1:]:
        previous = output[-1]
        flow = cv2.calcOpticalFlowFarneback(cv2.cvtColor(previous, cv2.COLOR_RGB2GRAY), cv2.cvtColor(current, cv2.COLOR_RGB2GRAY), None, .5, 3, 15, 3, 5, 1.2, 0)
        height, width = current.shape[:2]
        grid_x, grid_y = np.meshgrid(np.arange(width), np.arange(height))
        warped = cv2.remap(previous, (grid_x + flow[..., 0]).astype(np.float32), (grid_y + flow[..., 1]).astype(np.float32), cv2.INTER_LINEAR)
        output.append(cv2.addWeighted(current, .8, warped, .2, 0))
    return output
