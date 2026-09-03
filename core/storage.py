"""Model weight caching, verification, and download utilities for Hugging Face Hub."""

from __future__ import annotations

from pathlib import Path
from typing import Callable, NamedTuple, Optional

from huggingface_hub import hf_hub_download, try_to_load_from_cache

from core.exceptions import ModelDownloadError
from core.registry import get_model_spec

MODEL_FILES = ("model_config.json", "model.safetensors", "t5gemma-b-b-ul2/model.safetensors")


class CacheStatus(NamedTuple):
    """Cache presence and disk footprint for a model."""

    downloaded: bool
    size_gb: float
    status_text: str

    def to_dict(self) -> dict:
        return {"downloaded": self.downloaded, "size_gb": self.size_gb, "status_text": self.status_text}


def check_model_cache(model_name: str) -> CacheStatus:
    """Check if model files exist in local cache and calculate size on disk."""
    spec = get_model_spec(model_name)
    total_bytes = 0
    all_present = True

    for filename in MODEL_FILES:
        cached = try_to_load_from_cache(spec.repo_id, filename)
        if cached:
            p = Path(cached)
            if p.is_file() and p.stat().st_size > 0:
                total_bytes += p.stat().st_size
                continue
        all_present = False
        break

    size_gb = round(total_bytes / (1024**3), 2)
    status_text = f"Ready ({size_gb:.2f} GB)" if all_present else "Not Downloaded"
    return CacheStatus(downloaded=all_present, size_gb=size_gb, status_text=status_text)


def is_model_cached(model_name: str) -> bool:
    """Return True if model is completely cached locally."""
    return check_model_cache(model_name).downloaded


def download_model(
    model_name: str,
    force: bool = False,
    progress_callback: Optional[Callable[[float, str], None]] = None,
) -> bool:
    """Download model files from Hugging Face Hub with progress reporting."""
    spec = get_model_spec(model_name)
    if not force and is_model_cached(model_name):
        return True

    total = len(MODEL_FILES)
    for idx, filename in enumerate(MODEL_FILES):
        desc = f"Downloading {filename} ({idx + 1}/{total})..."
        if progress_callback:
            progress_callback(idx / total, desc)
        print(f"[{model_name}] {desc}")

        try:
            hf_hub_download(repo_id=spec.repo_id, filename=filename, force_download=force)
        except Exception as e:
            err_msg = f"Failed to download {filename} from {spec.repo_id}: {e}"
            print(f"Error: {err_msg}")
            if progress_callback:
                progress_callback(0.0, err_msg)
            return False

    if progress_callback:
        progress_callback(1.0, "Download complete")
    return True


def ensure_model_ready(
    model_name: str,
    progress_callback: Optional[Callable[[float, str], None]] = None,
) -> bool:
    """Ensure all required model files are cached before loading."""
    if not is_model_cached(model_name):
        print(f"[{model_name}] Weights not cached locally. Starting download...")
        if progress_callback:
            progress_callback(0.0, f"Downloading '{model_name}' weights from Hugging Face...")
        if not download_model(model_name, progress_callback=progress_callback):
            spec = get_model_spec(model_name)
            raise ModelDownloadError(
                f"Failed to download weights for '{model_name}'. "
                f"Verify network connection and access at https://huggingface.co/{spec.repo_id}"
            )
    return True
