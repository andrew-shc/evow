"""Fit a persistent, time-varying 3D Gaussian scene to one camera clip."""

import argparse
import json
from pathlib import Path

import cv2
import imageio.v2 as imageio
import numpy as np
import torch
from gsplat.rendering import rasterization

from .clip import write_video
from .depth_prior import scene_depths
from .pose import ViewRequest, intrinsics, target_cam_to_world
from .settings import load_settings
from .splat import save_splat, timeline_keyframe_indices


def _initial_gaussians(frames: np.ndarray, depths: np.ndarray, fov: float, stride: int):
    """Seed a volume from monocular depth and optical flow; training changes the 3D primitives."""
    count, height, width = frames.shape[:3]
    camera = intrinsics(height, width, fov)
    rows = np.arange(stride // 2, height, stride)
    cols = np.arange(stride // 2, width, stride)
    x, y = np.meshgrid(cols, rows)
    pixels = np.stack((x.ravel(), y.ravel()), axis=1)
    z = depths[0, y, x].ravel()
    centers = np.stack(((pixels[:, 0] - camera[0, 2]) * z / camera[0, 0],
                        (pixels[:, 1] - camera[1, 2]) * z / camera[1, 1], z), axis=1)
    base_color = frames[0, y, x].reshape(-1, 3).astype(np.float32) / 255.0
    footprint = z * stride / camera[0, 0]
    scales = np.stack((footprint * 0.9, footprint * 0.9,
                       np.maximum(footprint * 0.45, 0.015)), axis=1)
    flow_offsets = np.zeros((count, len(z), 3), dtype=np.float32)
    color_offsets = np.zeros((count, len(z), 3), dtype=np.float32)
    reference = cv2.cvtColor(frames[0], cv2.COLOR_RGB2GRAY)
    for index in range(1, count):
        target = cv2.cvtColor(frames[index], cv2.COLOR_RGB2GRAY)
        flow = cv2.calcOpticalFlowFarneback(reference, target, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        sampled_flow = flow[y, x].reshape(-1, 2)
        flow_offsets[index, :, 0] = sampled_flow[:, 0] * z / camera[0, 0]
        flow_offsets[index, :, 1] = sampled_flow[:, 1] * z / camera[1, 1]
        sample_x = np.clip(np.rint(pixels[:, 0] + sampled_flow[:, 0]).astype(int), 0, width - 1)
        sample_y = np.clip(np.rint(pixels[:, 1] + sampled_flow[:, 1]).astype(int), 0, height - 1)
        color_offsets[index] = frames[index, sample_y, sample_x] / 255.0 - base_color
    return centers.astype(np.float32), scales.astype(np.float32), base_color, flow_offsets, color_offsets


def _render(means, quats, scales, opacities, colors, viewmat, camera, width, height):
    rgb, alpha, _ = rasterization(
        means, quats, scales, opacities, colors,
        viewmat[None], camera[None], width, height,
        packed=False, backgrounds=torch.tensor([[0.5, 0.68, 0.86]], device=means.device),
    )
    return rgb[0], alpha[0]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames", type=Path, required=True)
    parser.add_argument("--depths", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--render-yaw", type=float, required=True)
    parser.add_argument("--render-shift", type=float, required=True)
    parser.add_argument("--render-fov", type=float, required=True)
    parser.add_argument("--source-fov", type=float, default=70.0)
    parser.add_argument("--forecast-frames", type=int, default=0)
    parser.add_argument("--forecast-fps", type=int, default=6)
    parser.add_argument("--motion-fit-frames", type=int, default=12)
    parser.add_argument("--history-fps", type=float, default=0)
    parser.add_argument("--splat-keyframe-fps", type=float, default=0)
    args = parser.parse_args()
    settings = load_settings()
    request = ViewRequest(args.render_yaw, args.render_shift, args.render_fov)
    frames = np.load(args.frames)
    with np.load(args.depths) as data:
        depths = scene_depths(data["depths"])
    count, height, width = frames.shape[:3]
    args.out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda")
    torch.manual_seed(0)
    centers, scales, colors, offsets, color_offsets = _initial_gaussians(
        frames, depths, args.source_fov, settings.gaussian_stride,
    )
    print(f"Optimizing {len(centers)} persistent 3D Gaussians over {count} frames", flush=True)
    xyz = torch.nn.Parameter(torch.tensor(centers, device=device, dtype=torch.float32))
    log_scales = torch.nn.Parameter(torch.tensor(np.log(scales), device=device, dtype=torch.float32))
    color_logits = torch.nn.Parameter(torch.logit(torch.tensor(colors, device=device).clamp(0.01, 0.99)))
    opacity_logits = torch.nn.Parameter(torch.full((len(centers),), 2.2, device=device))
    # The same primitive identifiers persist across time; these are per-frame deformations.
    motion = torch.nn.Parameter(torch.tensor(offsets, device=device, dtype=torch.float32))
    temporal_color = torch.nn.Parameter(torch.tensor(color_offsets, device=device))
    quats = torch.nn.Parameter(torch.zeros((len(centers), 4), device=device))
    with torch.no_grad():
        quats[:, 0] = 1
    camera = torch.tensor(intrinsics(height, width, args.source_fov), device=device)
    source_view = torch.eye(4, device=device)
    targets = torch.tensor(frames, device=device, dtype=torch.float32) / 255.0
    initial_depth = xyz[:, 2].detach().clone()
    optimizer = torch.optim.Adam([
        {"params": [xyz, quats], "lr": 0.003},
        {"params": [log_scales], "lr": 0.003},
        {"params": [color_logits, temporal_color], "lr": 0.015},
        {"params": [opacity_logits], "lr": 0.01},
        {"params": [motion], "lr": 0.002},
    ])
    progress_path = args.out / "progress.json"

    def write_progress(record: dict) -> None:
        """Atomically publish one complete optimizer update for the UI reader."""
        temporary = progress_path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(record))
        temporary.replace(progress_path)

    for step in range(settings.gaussian_steps):
        frame_index = step % count
        optimizer.zero_grad(set_to_none=True)
        means = xyz + motion[frame_index]
        rgb, _ = _render(
            means, quats, log_scales.exp(), opacity_logits.sigmoid(),
            (color_logits + temporal_color[frame_index]).sigmoid(),
            source_view, camera, width, height,
        )
        reconstruction = (rgb - targets[frame_index]).abs().mean()
        # A static camera leaves depth and hidden geometry ambiguous. Keep the learned
        # scene near its prior and make deformation smooth across consecutive frames.
        depth_penalty = (xyz[:, 2] - initial_depth).abs().mean()
        smoothness = (motion[1:] - motion[:-1]).square().mean()
        loss = reconstruction + 0.002 * depth_penalty + 0.01 * smoothness
        loss.backward()
        optimizer.step()
        record = {"phase": "fit_gaussians", "step": step + 1, "total": settings.gaussian_steps,
                  "loss": round(float(reconstruction.detach().cpu()), 5), "gaussians": len(centers)}
        # Preview encoding stays sparse; numerical progress is atomically written every step.
        if step == 0 or (step + 1) % 25 == 0 or step + 1 == settings.gaussian_steps:
            preview = args.out / f"training_{step + 1:04d}.png"
            imageio.imwrite(preview, (rgb.detach().cpu().numpy().clip(0, 1) * 255).astype(np.uint8))
            record["preview"] = str(preview)
        write_progress(record)
        print(f"step {step + 1}/{settings.gaussian_steps} loss={loss.item():.5f}", flush=True)

    target_pose = torch.tensor(target_cam_to_world(request), device=device)
    target_view = torch.linalg.inv(target_pose)
    all_renders = []
    render_camera = torch.tensor(intrinsics(height, width, args.render_fov), device=device)
    observed_indices = (
        timeline_keyframe_indices(count, args.history_fps, args.splat_keyframe_fps)
        if args.splat_keyframe_fps else tuple(range(count))
    )
    observed_index_set = set(observed_indices)
    with torch.inference_mode():
        for index in range(count):
            means = xyz + motion[index]
            scale = log_scales.exp()
            opacity = opacity_logits.sigmoid()
            color = (color_logits + temporal_color[index]).sigmoid()
            rgb, _ = _render(means, quats, scale, opacity, color,
                             target_view, render_camera, width, height)
            all_renders.append((rgb.cpu().numpy().clip(0, 1) * 255).astype(np.uint8))
            if index in observed_index_set:
                exported = observed_indices.index(index) + 1
                write_progress({"phase": "export_observed_keyframes", "completed": exported,
                                "total": len(observed_indices), "gaussians": len(centers)})
                save_splat(args.out / f"splat_{index:03d}.splat", means.cpu().numpy(),
                           scale.cpu().numpy(), color.cpu().numpy(), opacity.cpu().numpy(), quats.cpu().numpy())
    write_video(np.stack(all_renders), args.out / "rendered.mp4", settings.fps)
    if args.forecast_frames:
        fit = min(args.motion_fit_frames, count)
        times = torch.arange(fit, device=device, dtype=torch.float32)
        centered = times - times.mean()
        velocity = (centered[:, None, None] * (motion[-fit:] - motion[-fit:].mean(dim=0))).sum(dim=0) / centered.square().sum().clamp_min(1e-6)
        forecast = []
        forecast_indices = (
            timeline_keyframe_indices(args.forecast_frames, args.forecast_fps, args.splat_keyframe_fps)
            if args.splat_keyframe_fps else tuple(range(args.forecast_frames))
        )
        forecast_index_set = set(forecast_indices)
        with torch.inference_mode():
            for index in range(args.forecast_frames):
                delta = (index + 1) * settings.fps / args.forecast_fps
                means = xyz + motion[-1] + velocity * delta
                scale = log_scales.exp()
                opacity = opacity_logits.sigmoid()
                color = (color_logits + temporal_color[-1]).sigmoid()
                rgb, _ = _render(means, quats, scale, opacity, color, source_view, camera, width, height)
                forecast.append((rgb.cpu().numpy().clip(0, 1) * 255).astype(np.uint8))
                if index in forecast_index_set:
                    exported = forecast_indices.index(index) + 1
                    write_progress({"phase": "render_forecast_and_export_keyframes", "completed": exported,
                                    "total": len(forecast_indices), "rendered_frames": index + 1,
                                    "forecast_frames": args.forecast_frames, "gaussians": len(centers)})
                    save_splat(args.out / f"forecast_splat_{index:03d}.splat", means.cpu().numpy(),
                               scale.cpu().numpy(), color.cpu().numpy(), opacity.cpu().numpy(), quats.cpu().numpy())
        write_progress({"phase": "encode_forecast_video", "completed": args.forecast_frames,
                        "total": args.forecast_frames, "gaussians": len(centers)})
        write_video(np.stack(forecast), args.out / "forecast.mp4", args.forecast_fps)
        torch.save({"velocity": velocity.cpu(), "frames": args.forecast_frames, "fps": args.forecast_fps}, args.out / "forecast_scene.pt")
    imageio.imwrite(args.out / "render_000.png", all_renders[0])
    torch.save({
        "means": xyz.detach().cpu(), "log_scales": log_scales.detach().cpu(),
        "color_logits": color_logits.detach().cpu(),
        "opacity_logits": opacity_logits.detach().cpu(),
        "motion": motion.detach().cpu(), "temporal_color": temporal_color.detach().cpu(),
        "quaternions": quats.detach().cpu(), "source_fov_degrees": args.source_fov, "render_fov_degrees": args.render_fov, "fps": settings.fps,
    }, args.out / "scene.pt")
    (args.out / "scene.json").write_text(json.dumps({
        "representation": "time_varying_3d_gaussians", "gaussians": len(centers),
        "frames": count, "source_fov_degrees": args.source_fov, "render_fov_degrees": args.render_fov,
        "limitations": "Single fixed camera; unobserved geometry and absolute scale are uncertain.",
    }, indent=2))
    print("Native 4D Gaussian scene and video saved", flush=True)


if __name__ == "__main__":
    main()
