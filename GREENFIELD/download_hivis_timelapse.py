import os
import requests
import subprocess
from pathlib import Path

BASE = "https://api.waterdata.usgs.gov/nims/v0"
CAM_ID = "VA_DIFFICULT_RUN_ABOVE_FOX_LAKE_NEAR_FAIRFAX"

# Converted from EDT to UTC:
AFTER = "2025-06-15T21:15:02Z"
BEFORE = "2025-06-17T13:15:02Z"

OUTDIR = Path("hivis_frames")
OUTDIR.mkdir(exist_ok=True)

# Optional: put your USGS API key here if needed
API_KEY = os.environ.get("USGS_API_KEY")  # or set to a string directly

session = requests.Session()
headers = {}
if API_KEY:
    headers["X-Api-Key"] = API_KEY

# 1) Get camera metadata so we know the image base path
cam_resp = session.get(
    f"{BASE}/cameras",
    params={"camId": CAM_ID},
    headers=headers,
    timeout=60,
)
cam_resp.raise_for_status()
cam_data = cam_resp.json()

if not cam_data:
    raise RuntimeError(f"No camera found for camId={CAM_ID}")

camera = cam_data[0]

# Choose smallDir (720px) for easier download; use overlayDir for full size
image_base = camera["smallDir"]

print("Using image base:", image_base)

# 2) Get filenames for the exact time range
files_resp = session.get(
    f"{BASE}/listFiles",
    params={
        "camId": CAM_ID,
        "recent": "false",
        "after": AFTER,
        "before": BEFORE,
        "rawItem": "true",
        "limit": 50000,
    },
    headers=headers,
    timeout=120,
)
files_resp.raise_for_status()
items = files_resp.json()

if not items:
    raise RuntimeError("No images returned for that time window.")

print(f"Found {len(items)} images")

# 3) Download each image
downloaded = []
for i, item in enumerate(items, start=1):
    filename = item["filename"]
    url = image_base + filename
    outpath = OUTDIR / f"{i:06d}.jpg"

    if outpath.exists():
        downloaded.append(outpath)
        continue

    r = session.get(url, headers=headers, timeout=120)
    r.raise_for_status()
    outpath.write_bytes(r.content)
    downloaded.append(outpath)

    if i % 100 == 0 or i == len(items):
        print(f"Downloaded {i}/{len(items)}")

print("All frames downloaded to", OUTDIR)

# 4) Compile to MP4 with ffmpeg
# Change -framerate as desired:
#   10 fps = slower timelapse
#   24 or 30 fps = faster timelapse
video_name = "VA_DIFFICULT_RUN_ABOVE_FOX_LAKE_NEAR_FAIRFAX_2025-06-15_to_2025-06-17.mp4"

cmd = [
    "ffmpeg",
    "-y",
    "-framerate", "24",
    "-i", str(OUTDIR / "%06d.jpg"),
    "-c:v", "libx264",
    "-pix_fmt", "yuv420p",
    video_name,
]

print("Running:", " ".join(cmd))
subprocess.run(cmd, check=True)
print("Created:", video_name)
