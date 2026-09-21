# 4D replay on the local network

This is the first of the four camera features. Open `http://192.168.4.254:7860` from a device on this LAN. Upload a recording or browser camera capture from a stationary view, use the included trees sample, or select a saved run.

- **Implicit:** [AnyView-DVS](https://github.com/TRI-ML/AnyView-DVS) generates a synchronized target-view video from the source clip and requested camera path.
- **Explicit:** [Video Depth Anything](https://github.com/DepthAnything/Video-Depth-Anything) supplies only a depth initialization prior. [gsplat](https://github.com/nerfstudio-project/gsplat) then fits persistent 3D Gaussian positions, size, rotation, opacity, color, and per-frame deformation to the video. The native scene is saved as a checkpoint and 13, 29, or 41 time-varying `.splat` frames. An orbitable [Three.js Gaussian splat viewer](https://github.com/mkkellogg/GaussianSplats3D) preloads the time-indexed frames once, then its local timeline and Play control switch frames without rebuilding the viewer. The requested nearby camera path is also rendered to MP4.

Both modes use the same sampled frames at 12 fps. The application assumes that your source has a stationary physical viewpoint; objects and weather may move within that view. It does not run camera-motion validation. **Show internal execution flow** reveals the route from source through the selected method to the result. Expand a row to see its implementation function, inputs, outputs, left-side editable control, and recorded live detail. Each collapsible stage contains a two-column program contract and live run record: source file, function chain, inputs, outputs, linked control, preview, timings, metrics, request settings, and raw event JSON. Every successful run stores `trace.json`, `result.json`, the sampled source clip, and generated assets under `ASSETS/replay/runs/<run-id>/`. Select a saved run and click **Load selected run** to open its result and load its sampled clip into the input field. The earlier rollercoaster smoke-test files are retained under `ASSETS/` for debugging but deliberately excluded because their moving physical camera violates this application’s input contract. Older saved explicit demos use depth meshes and remain viewable; new explicit runs use Gaussian scenes.

## Run

From the repo root:

```bash
conda activate evow
python -m GREENFIELD.replay.app
```

To reproduce the environment on this CUDA workstation:

```bash
conda create -n evow python=3.11
conda activate evow
python -m pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu126
python -m pip install -r CONFIGS/replay_requirements.txt
```

Place official source checkouts in `AnyView-DVS/` and `Video-Depth-Anything/`. Download the AnyView 2B, tokenizer, and default text embedding checkpoints from the [official AnyView release](https://github.com/TRI-ML/AnyView-DVS#checkpoints) into `ASSETS/checkpoints/anyview/`, and the VITS checkpoint from the [Video Depth Anything release](https://github.com/DepthAnything/Video-Depth-Anything#models) into `ASSETS/checkpoints/video_depth_anything/`. The exact filenames are in `settings.py`. The three AnyView files on this machine passed the release SHA-256 checksums.

An included six-second tree-and-wind sample is derived from [Pexels video 12644693](https://www.pexels.com/video/trees-swaying-in-the-wind-in-autumn-12644693/), whose page marks it free to use. It is stored under `ASSETS/replay/samples/` and is ready to use in the dashboard.

The app binds to all local IPv4 interfaces (`0.0.0.0`) by default, including the LAN and Tailscale interfaces, and does not create a public share link. Set `EVOW_BIND_HOST` to restrict or override the listening address. Devices permitted by the host firewall and Tailscale ACLs can start processing jobs and inspect saved runs. The Gaussian viewer JavaScript bundle is served from this app, so LAN clients do not need CDN access. `gaussian_viewer.bundle.js` was built from `@mkkellogg/gaussian-splats-3d@0.4.4` and Three.js with esbuild; its upstream MIT license is copied beside it.

A single fixed camera cannot reveal geometry behind visible surfaces, absolute scene scale, or true camera calibration. The Gaussian scene is a native 3D representation and can be orbited, but its unseen regions remain underconstrained. The field-of-view slider is an assumed calibration. Keep camera movement modest for credible results. AnyView's code is under CC BY-NC 4.0 and its model includes NVIDIA Open Model License components; review those terms before commercial or public deployment.
