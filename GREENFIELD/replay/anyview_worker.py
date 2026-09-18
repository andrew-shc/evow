"""Run the external AnyView weights in a separate GPU process."""

import argparse
import importlib.util
import json
from pathlib import Path
import sys

import torch

from .settings import load_settings


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episode", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    settings = load_settings()
    if not torch.cuda.is_available():
        raise RuntimeError("AnyView requires a CUDA GPU.")
    spec = importlib.util.spec_from_file_location(
        "anyview_infer", settings.anyview_repo / "scripts" / "infer.py"
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("Cannot load the AnyView inference module.")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    sys.path.insert(0, str(settings.anyview_repo))
    from anyview.cameras import resolve_scale_factor
    from anyview.config import AnyViewConfig
    from anyview.pipe import load_pipeline
    from anyview.vae import load_vae

    config = AnyViewConfig(
        checkpoint_path=str(settings.anyview_checkpoint),
        tokenizer_path=str(settings.anyview_tokenizer),
        text_emb_path=str(settings.anyview_text_embedding),
    )
    frames, cameras, target_size = module.load_episode(args.episode)
    cameras["world2cam"] = module.reanchor_world2cam(cameras["world2cam"])
    scale = resolve_scale_factor(cameras)
    device = torch.device("cuda")
    print("Loading AnyView video tokenizer", flush=True)
    vae = load_vae(config.tokenizer_path, device)
    print("Loading AnyView diffusion transformer", flush=True)
    pipeline = load_pipeline(config, device)
    print("Generating target-view frames", flush=True)
    with torch.inference_mode():
        prediction, info = module.generate_target_view(
            pipeline, vae, config, frames, cameras, scale,
            seed=0, num_steps=settings.anyview_steps, guidance=0.0,
            device=device, hw_target=target_size,
        )
    args.out.mkdir(parents=True, exist_ok=True)
    module.save_video(prediction, str(args.out / "pred.mp4"), settings.fps)
    module.save_frames(prediction, str(args.out / "frames"))
    (args.out / "info.json").write_text(json.dumps(info, indent=2))
    print("Target-view video saved", flush=True)


if __name__ == "__main__":
    main()
