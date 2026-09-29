"""Local Stable Video Diffusion LoRA optimizer for Future View temporal pairs."""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import numpy as np


def _progress(path: Path, **record) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(record))
    temporary.replace(path)



def _training_image(frame: np.ndarray, max_side: int):
    """Convert a sampled RGB frame to a VAE-safe, memory-bounded PIL image."""
    from PIL import Image

    image = Image.fromarray(frame.astype(np.uint8))
    width, height = image.size
    scale = min(1.0, max_side / max(width, height))
    size = (max(8, round(width * scale / 8) * 8), max(8, round(height * scale / 8) * 8))
    return image if size == image.size else image.resize(size, Image.Resampling.LANCZOS)


def _conditioning_inputs(pipe, frame: np.ndarray, device: str, dtype, max_side: int | None = None):
    """Match SVD inference: CLIP receives a PIL image, VAE receives processed pixels."""
    image = _training_image(frame, max_side or max(frame.shape[:2]))
    # Passing a Tensor here skips SVD's internal 224px CLIP resize. PIL preserves
    # the exact inference path: resize/normalize for CLIP, then original-aspect
    # processed pixels for the VAE condition latent.
    image_embeddings = pipe._encode_image(image, device, 1, False)
    vae_pixels = pipe.video_processor.preprocess(image).to(device=device, dtype=dtype)
    image_latents = pipe._encode_vae_image(vae_pixels, device, 1, False)
    return image_embeddings, image_latents


def _target_pixels(pipe, frames: np.ndarray, device: str, dtype, max_side: int | None = None):
    """Normalize target frames through the same SVD video processor as inference."""
    limit = max_side or max(frames.shape[1:3])
    images = [_training_image(frame, limit) for frame in frames]
    return pipe.video_processor.preprocess(images).to(device=device, dtype=dtype).unsqueeze(0)


def _initialize_training_scheduler(scheduler, device: str) -> None:
    """Populate SVD's continuous Euler schedule before sampling training noise."""
    scheduler.set_timesteps(scheduler.config.num_train_timesteps, device=device)
    if not len(scheduler.timesteps):
        raise RuntimeError("Stable Video Diffusion produced an empty training timestep schedule.")


def _sample_training_timestep(scheduler, device: str):
    """Choose a real continuous scheduler value, never an integer index."""
    import torch

    if not len(scheduler.timesteps):
        raise RuntimeError("Stable Video Diffusion's training timestep schedule was not initialized.")
    index = torch.randint(len(scheduler.timesteps), (1,), device=device)
    return scheduler.timesteps.index_select(0, index).to(device=device)


def _scale_training_input(scheduler, noisy, timestep):
    """Apply Euler's stateless sigma preconditioning for randomly ordered batches."""
    # ``scale_model_input`` caches an inference-loop step index. Training samples
    # timesteps independently, so derive the documented scaling directly instead.
    index = scheduler.index_for_timestep(timestep, scheduler.timesteps)
    sigma = scheduler.sigmas[index].to(device=noisy.device, dtype=noisy.dtype)
    return noisy / (sigma.square() + 1).sqrt()


def _training_target(scheduler, latents, noise, timestep):
    """Match the local checkpoint's scheduler prediction parameterization."""
    prediction_type = scheduler.config.prediction_type
    if prediction_type == "epsilon":
        return noise
    if prediction_type == "v_prediction":
        return scheduler.get_velocity(latents, noise, timestep)
    if prediction_type in {"sample", "original_sample"}:
        return latents
    raise ValueError(f"Unsupported Stable Video Diffusion prediction type: {prediction_type}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--progress", type=Path, required=True)
    parser.add_argument("--steps", type=int, required=True)
    parser.add_argument("--rank", type=int, required=True)
    parser.add_argument("--fps", type=int, required=True)
    parser.add_argument("--max-side", type=int, required=True)
    args = parser.parse_args()
    if not args.checkpoint.is_dir():
        raise FileNotFoundError(f"Missing Stable Video Diffusion checkpoint: {args.checkpoint}")
    import torch
    from diffusers import StableVideoDiffusionPipeline
    from peft import LoraConfig, get_peft_model

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device != "cuda":
        raise RuntimeError("Future View implicit LoRA training requires CUDA.")
    data = np.load(args.pairs)
    conditions = data["conditions"]
    targets = data["targets"]
    if not len(conditions):
        raise ValueError("No temporal pairs were supplied for LoRA training.")
    # Training owns its process and CUDA context; unlike browser inference it does not CPU-offload.
    pipe = StableVideoDiffusionPipeline.from_pretrained(args.checkpoint, local_files_only=True, torch_dtype=torch.float16)
    pipe.to(device)
    pipe.vae.requires_grad_(False); pipe.image_encoder.requires_grad_(False); pipe.unet.requires_grad_(False)
    pipe.vae.eval(); pipe.image_encoder.eval()
    # Frozen UNet layers still need their activations for LoRA gradients. Checkpointing
    # keeps a 14-frame source window within the dashboard GPU's practical memory bound.
    pipe.unet.enable_gradient_checkpointing()
    _initialize_training_scheduler(pipe.scheduler, device)
    # SVD's temporal UNet uses the generic PEFT wrapper rather than Diffusers' pipeline
    # LoRA mixin, which this checkpoint family does not expose.
    pipe.unet = get_peft_model(pipe.unet, LoraConfig(r=args.rank, lora_alpha=args.rank, target_modules=["to_q", "to_k", "to_v", "to_out.0"]))
    parameters = [parameter for parameter in pipe.unet.parameters() if parameter.requires_grad]
    if not parameters:
        raise RuntimeError("Stable Video Diffusion exposed no trainable LoRA parameters.")
    optimizer = torch.optim.AdamW(parameters, lr=1e-4)
    generator = np.random.default_rng(0)
    scaling = pipe.vae.config.scaling_factor
    pipe.unet.train()
    for step in range(args.steps):
        index = int(generator.integers(len(conditions)))
        target_pixels = _target_pixels(pipe, targets[index], device, torch.float16, args.max_side)
        with torch.no_grad():
            flat = target_pixels.flatten(0, 1)
            latents = pipe.vae.encode(flat).latent_dist.sample().mul(scaling).unflatten(0, target_pixels.shape[:2])
            image_embeddings, image_latents = _conditioning_inputs(pipe, conditions[index], device, torch.float16, args.max_side)
            image_latents = image_latents.unsqueeze(1).repeat(1, target_pixels.shape[1], 1, 1, 1)
        noise = torch.randn_like(latents)
        timesteps = _sample_training_timestep(pipe.scheduler, device)
        noisy = pipe.scheduler.add_noise(latents, noise, timesteps)
        model_input = _scale_training_input(pipe.scheduler, noisy, timesteps)
        prediction_target = _training_target(pipe.scheduler, latents, noise, timesteps)
        # The SVD pipeline subtracts one before embedding FPS; mirror inference so
        # the adapter sees the same micro-conditioning at forecast time.
        added_time_ids = pipe._get_add_time_ids(args.fps - 1, 127, 0.02, image_embeddings.dtype, 1, 1, False).to(device)
        prediction = pipe.unet(torch.cat([model_input, image_latents], dim=2), timesteps, encoder_hidden_states=image_embeddings, added_time_ids=added_time_ids).sample
        loss = torch.nn.functional.mse_loss(prediction.float(), prediction_target.float())
        optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step()
        _progress(args.progress, phase="lora", step=step + 1, total=args.steps, loss=round(float(loss.detach().cpu()), 6))
    args.out.mkdir(parents=True, exist_ok=True)
    pipe.unet.save_pretrained(args.out)
    _progress(args.progress, phase="saved_lora", step=args.steps, total=args.steps)

if __name__ == "__main__":
    main()
