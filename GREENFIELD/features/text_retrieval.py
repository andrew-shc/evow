"""Temporal interval ranking on top of sparse video-language similarity scores."""

from dataclasses import dataclass

import numpy as np

from .text_media import interval_start


@dataclass(frozen=True)
class CandidateWindow:
    """One non-overlapping source interval selected from semantic frame scores."""

    anchor_index: int
    start_seconds: float
    end_seconds: float
    score: float


def smoothed_window_scores(scores: np.ndarray, timestamps: list[float], clip_seconds: float) -> np.ndarray:
    """Average each retrieval score across the clip it would represent."""
    if len(scores) != len(timestamps) or not len(scores):
        raise ValueError("Retrieval scores and timestamps must be non-empty and aligned.")
    if clip_seconds <= 0:
        raise ValueError("Candidate clip duration must be positive.")
    times = np.asarray(timestamps, dtype=np.float32)
    values = np.asarray(scores, dtype=np.float32)
    half = clip_seconds / 2
    return np.asarray([
        float(values[np.abs(times - timestamp) <= half].mean())
        for timestamp in times
    ], dtype=np.float32)


def ranked_windows(scores: np.ndarray, timestamps: list[float], source_seconds: float, clip_seconds: float, limit: int) -> list[CandidateWindow]:
    """Return high-score source windows after temporal non-maximum suppression."""
    if source_seconds <= 0 or limit < 1:
        raise ValueError("Source duration and result limit must be positive.")
    smoothed = smoothed_window_scores(scores, timestamps, clip_seconds)
    results: list[CandidateWindow] = []
    for index in np.argsort(-smoothed):
        start = interval_start(float(timestamps[int(index)]), clip_seconds, source_seconds)
        end = min(source_seconds, start + clip_seconds)
        # Candidate intervals must remain separately playable rather than five
        # copies of the same moment with nearly identical scores.
        overlaps = any(start < existing.end_seconds and end > existing.start_seconds for existing in results)
        if overlaps:
            continue
        results.append(CandidateWindow(int(index), start, end, float(smoothed[int(index)])))
        if len(results) == limit:
            break
    return results
