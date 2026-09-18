"""Estimate temporally aligned video depth in a short-lived GPU process."""

import argparse
from pathlib import Path
import sys

import numpy as np
import torch

from .settings import load_settings


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    settings = load_settings()
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
    frames = np.load(args.frames)
    print(f"Estimating depth for {len(frames)} frames on {device}", flush=True)
    depths, _ = model.infer_video_depth(
        frames, settings.fps, input_size=settings.depth_input_size,
        device=device, fp32=(device == "cpu"),
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, depths=depths.astype(np.float32))
    print(f"Depth saved to {args.out}", flush=True)


if __name__ == "__main__":
    main()
