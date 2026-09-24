"""Owner-run local model setup for Text Query and Text Manipulation."""

import argparse
from datetime import datetime, timezone
import json
import os

from dotenv import load_dotenv

from GREENFIELD.app_core.models import require_package
from .text_settings import load_text_settings


def _models(settings) -> dict[str, tuple[str, object]]:
    """Return named snapshots without embedding checkpoint paths in UI code."""
    return {
        "siglip": (settings.semantic_model_id, settings.semantic_checkpoint),
        "grounding-dino": (settings.grounding_model_id, settings.grounding_checkpoint),
        "sam2": (settings.segmentation_model_id, settings.segmentation_checkpoint),
        "wan-vace": (settings.edit_model_id, settings.edit_checkpoint),
    }


def install(name: str, model_id: str, checkpoint, token: str | None) -> None:
    """Download one explicit owner-selected model and record its immutable revision."""
    from huggingface_hub import HfApi, snapshot_download

    info = HfApi(token=token).model_info(model_id)
    checkpoint.mkdir(parents=True, exist_ok=True)
    snapshot_download(model_id, revision=info.sha, token=token, local_dir=checkpoint)
    (checkpoint / "model_manifest.json").write_text(json.dumps({
        "name": name,
        "model_id": model_id,
        "revision": info.sha,
        "downloaded_at": datetime.now(timezone.utc).isoformat(),
    }, indent=2))
    print(f"Installed {name}: {model_id} at {checkpoint}")


def main() -> None:
    """Install all text models, or a named subset, outside dashboard requests."""
    require_package("huggingface_hub", "models")
    settings = load_text_settings()
    choices = _models(settings)
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", choices=tuple(choices), action="append", help="Install only this model; repeatable.")
    args = parser.parse_args()
    # Load the token into process state without inspecting or emitting .env.
    load_dotenv(settings.root / ".env", override=False)
    token = os.environ.get("HF_TOKEN")
    selected = args.only or list(choices)
    for name in selected:
        model_id, checkpoint = choices[name]
        install(name, model_id, checkpoint, token)


if __name__ == "__main__":
    main()
