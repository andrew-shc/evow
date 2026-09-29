"""Run AnyView in isolation and publish its atomic inference stages."""

import argparse
import importlib.util
import json
from pathlib import Path
import sys

import torch

from .overrides import apply_settings_json
from .worker_events import WorkerEventWriter


def build_parser() -> argparse.ArgumentParser:
    """Expose the worker CLI for coordinator and direct invocations."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--episode", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--events", type=Path, default=None)
    parser.add_argument("--settings-json", default=None)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    events = WorkerEventWriter(args.events)
    settings = apply_settings_json(args.settings_json)
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
    from anyview.logistics import build_dvs_entries, unpack_entries_from_streams
    from anyview.pipe import load_pipeline
    from anyview.vae import load_vae

    config = AnyViewConfig(
        checkpoint_path=str(settings.anyview_checkpoint),
        tokenizer_path=str(settings.anyview_tokenizer),
        text_emb_path=str(settings.anyview_text_embedding),
    )
    device = torch.device("cuda")

    events.emit("load_anyview_vae", "running", "Loading AnyView video tokenizer.")
    vae = load_vae(config.tokenizer_path, device)
    events.emit("load_anyview_vae", "complete", "Video tokenizer is ready.")

    events.emit("load_anyview_diffusion", "running", "Loading AnyView diffusion transformer.")
    pipeline = load_pipeline(config, device)
    events.emit("load_anyview_diffusion", "complete", "Diffusion transformer is ready.")

    events.emit("load_anyview_episode", "running", "Reading source video and camera records.")
    frames, cameras, target_size = module.load_episode(args.episode)
    cameras["world2cam"] = module.reanchor_world2cam(cameras["world2cam"])
    scale = resolve_scale_factor(cameras)
    events.emit("load_anyview_episode", "complete", "Loaded source episode.", metrics={"frames": len(frames)})

    frame_count, source_height, source_width, _ = frames.shape
    if (frame_count - 1) % config.vae_temporal_factor:
        raise ValueError(f"AnyView requires 1 + {config.vae_temporal_factor}k frames, got {frame_count}.")
    input_shape = module.snap_shape(source_height, source_width)
    target_native = tuple(target_size) if target_size else (source_height, source_width)
    target_shape = module.snap_shape(*target_native)
    source_video = module.resize_video(frames, input_shape)

    with torch.inference_mode():
        events.emit("encode_source_rgb", "running", "Encoding source RGB video.")
        source_latent = vae.encode_rgb(source_video.unsqueeze(0).to(device))
        events.emit("encode_source_rgb", "complete", "Encoded source RGB video.")

        events.emit("encode_target_camera", "running", "Encoding target-camera Plücker rays.")
        target_cameras = module.prepare_cams_latent(
            vae, cameras["intrinsics"][0], cameras["world2cam"][0],
            target_native, target_shape, scale, device,
        )
        events.emit("encode_target_camera", "complete", "Encoded target-camera rays.")

        events.emit("encode_source_camera", "running", "Encoding source-camera Plücker rays.")
        source_cameras = module.prepare_cams_latent(
            vae, cameras["intrinsics"][1], cameras["world2cam"][1],
            (source_height, source_width), input_shape, scale, device,
        )
        events.emit("encode_source_camera", "complete", "Encoded source-camera rays.")

        events.emit("build_anyview_conditioning", "running", "Building video-and-camera conditioning.")
        entries = build_dvs_entries(None, source_latent, target_cameras, source_cameras)
        events.emit("build_anyview_conditioning", "complete", "Built AnyView conditioning.")

        if settings.anyview_guidance_scale != 0.0:
            raise ValueError("AnyView currently supports only zero guidance.")
        events.emit("sample_target_video", "running", "Denoising target-view video.", metrics={"steps": settings.anyview_steps})
        samples = pipeline.generate(entries, seed=0, num_sampling_steps=settings.anyview_steps)
        events.emit("sample_target_video", "complete", "Denoised target-view video.", metrics={"steps": settings.anyview_steps})

        events.emit("decode_target_video", "running", "Decoding target RGB video.")
        predicted_entries = unpack_entries_from_streams(samples["y0_pred_streams"])
        predicted_video = vae.decode_rgb(predicted_entries["rgb0"])
        prediction = predicted_video[0].to(torch.float32).clamp(0.0, 1.0).permute(1, 0, 2, 3).cpu()
        events.emit("decode_target_video", "complete", "Decoded target RGB video.", metrics={"frames": len(prediction)})

    args.out.mkdir(parents=True, exist_ok=True)
    events.emit("write_target_video", "running", "Writing target-view MP4.")
    module.save_video(prediction, str(args.out / "pred.mp4"), settings.fps)
    events.emit("write_target_video", "complete", "Saved target-view MP4.", metrics={"frames": len(prediction), "fps": settings.fps})

    events.emit("write_frames_metadata", "running", "Writing frame previews and metadata.")
    module.save_frames(prediction, str(args.out / "frames"))
    (args.out / "info.json").write_text(json.dumps({"hw_input": list(input_shape), "hw_target": list(target_shape)}, indent=2))
    events.emit(
        "write_frames_metadata", "complete", "Saved frame previews and metadata.",
        preview=args.out / "frames" / "000000.png", metrics={"frames": len(prediction)},
    )
    print("Target-view video saved", flush=True)


if __name__ == "__main__":
    main()
