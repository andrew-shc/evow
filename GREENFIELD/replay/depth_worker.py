"""Estimate temporally aligned video depth in a short-lived GPU process."""

import argparse
from pathlib import Path
import sys

import numpy as np
import torch

from .overrides import apply_settings_json
from .worker_events import WorkerEventWriter


def build_parser() -> argparse.ArgumentParser:
    """Expose the worker CLI for direct and coordinator-owned invocations."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--events", type=Path, default=None)
    parser.add_argument("--settings-json", default=None)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    events = WorkerEventWriter(args.events)
    settings = apply_settings_json(args.settings_json)

    events.emit("load_depth_model", "running", "Loading Video Depth Anything.")
    if not settings.depth_checkpoint.is_file():
        raise FileNotFoundError(settings.depth_checkpoint)
    sys.path.insert(0, str(settings.depth_repo))
    from video_depth_anything.video_depth import VideoDepthAnything

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = VideoDepthAnything(
        encoder="vits", features=64, out_channels=[48, 96, 192, 384],
    )
    state = torch.load(settings.depth_checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=True)
    model = model.to(device).eval()
    events.emit("load_depth_model", "complete", "Depth model is ready.", metrics={"device": device})

    frames = np.load(args.frames)
    events.emit("infer_temporal_depth", "running", "Estimating temporally aligned depth.", metrics={"frames": len(frames)})
    depths, _ = model.infer_video_depth(
        frames, settings.fps, input_size=settings.depth_input_size,
        device=device, fp32=(device == "cpu"),
    )
    events.emit("infer_temporal_depth", "complete", "Depth inference finished.", metrics={"frames": len(frames)})

    events.emit("write_depth_prior", "running", "Writing compressed depth prior.")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, depths=depths.astype(np.float32))
    events.emit("write_depth_prior", "complete", "Saved depth prior.", metrics={"path": args.out.name})
    print(f"Depth saved to {args.out}", flush=True)


if __name__ == "__main__":
    main()
