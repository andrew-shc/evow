"""Owner-run Hugging Face setup; dashboard inference never downloads weights."""
import json
import os
from datetime import datetime, timezone
from dotenv import load_dotenv
from GREENFIELD.app_core.models import require_package
from .future_settings import load_future_settings

def main() -> None:
    require_package("huggingface_hub", "models")
    from huggingface_hub import HfApi, snapshot_download
    settings = load_future_settings()
    load_dotenv(settings.root / ".env", override=False)
    token = os.environ.get("HF_TOKEN")
    try:
        info = HfApi(token=token).model_info(settings.model_id)
    except Exception as error:
        raise RuntimeError("Cannot access the Future View model. Check network access, or provide HF_TOKEN if Hugging Face requires authentication.") from error
    settings.checkpoint.mkdir(parents=True, exist_ok=True)
    snapshot_download(settings.model_id, token=token, local_dir=settings.checkpoint)
    (settings.checkpoint / "model_manifest.json").write_text(json.dumps({"model_id": settings.model_id, "revision": info.sha, "downloaded_at": datetime.now(timezone.utc).isoformat()}, indent=2))
    print(f"Installed {settings.model_id} at {settings.checkpoint}")
if __name__ == "__main__": main()
