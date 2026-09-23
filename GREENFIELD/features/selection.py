"""SigLIP/FAISS-oriented semantic archive query adapter."""

from dataclasses import dataclass
from typing import Iterator

import numpy as np

from GREENFIELD.app_core.contracts import FeatureResult, RunArtifacts, StageEvent, StageSpec
from .model_adapters import rank_semantic_frame_steps


STAGES = (
    StageSpec("Decode archive window", "Read the selected video window into searchable samples.", "GREENFIELD/features/selection.py", "def run(", "run() → app_core.media.read_video()", "archive video", "bounded RGB frames · fps", "Source"),
    StageSpec("Embed video and query", "Embed frames and free-form text in a shared vision-language space.", "GREENFIELD/features/selection.py", "def run(", "run() → model_adapters.rank_semantic_frames() → SigLIP", "frames · query", "FAISS similarity scores · ranked frame IDs", "Text · Methodology"),
    StageSpec("Rank intervals", "Search the local vector index and return playable matching intervals.", "GREENFIELD/features/selection.py", "def run(", "rank_semantic_frames() → interval rows", "ranked frames · fps", "start/end intervals · scores", "Search"),
)


@dataclass(frozen=True)
class SelectionRequest:
    mode: str
    query: str

    def validate(self) -> None:
        if self.mode not in {"implicit", "explicit"}:
            raise ValueError("Choose Implicit video or Explicit 3D.")
        if not self.query.strip():
            raise ValueError("Enter a text query.")


def run(frames: list[np.ndarray], fps: float, request: SelectionRequest, artifacts: RunArtifacts) -> Iterator[StageEvent | FeatureResult]:
    """Create a persisted local embedding matrix, then return interval matches."""
    request.validate()
    yield StageEvent("Decode archive window", "complete", f"Decoded {len(frames)} searchable frames.", metrics={"frames": len(frames), "fps": fps})
    yield StageEvent("Embed video and query", "running", "Loading local SigLIP and embedding the text query.")
    steps = rank_semantic_frame_steps(frames, request.query)
    while True:
        try:
            completed, total = next(steps)
            yield StageEvent("Embed video and query", "running", "Embedded {}/{} searchable frames.".format(completed, total), metrics={"embedded_frames": completed, "total_frames": total})
        except StopIteration as finished:
            scores, ranked = finished.value
            break
    np.save(artifacts.file("ranked_scores.npy"), scores)
    yield StageEvent("Embed video and query", "complete", "Persisted local SigLIP/FAISS search results for this run.", metrics={"ranked_frames": len(ranked)})
    rows = [[f"{index / fps:.1f}s", f"{min((index + 1) / fps, len(frames) / fps):.1f}s", round(float(scores[position]), 3), request.mode] for position, index in enumerate(ranked)]
    yield StageEvent("Rank intervals", "complete", f"Returned {len(rows)} semantic interval candidates for: {request.query}")
    yield FeatureResult(rows=rows, metadata={"query": request.query, "mode": request.mode, "index": "per-run"})
