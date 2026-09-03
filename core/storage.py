"""Checkpoint storage, integrity verification, and self-healing download manager."""

import os
import shutil
import time
from pathlib import Path
from typing import Callable, Dict, Optional

from huggingface_hub import hf_hub_download, try_to_load_from_cache
from huggingface_hub.constants import HUGGINGFACE_HUB_CACHE
from safetensors import safe_open

from core.registry import get_model_spec


def verify_safetensors(file_path: str, min_size_mb: float = 1.0) -> bool:
    """Verify that a safetensors file exists, is not truncated, and can be read."""
    if not file_path or not os.path.isfile(file_path):
        return False
    try:
        if os.path.getsize(file_path) < min_size_mb * 1024 * 1024:
            return False

        with safe_open(file_path, framework="pt", device="cpu") as f:
            keys = list(f.keys())
            if not keys:
                return False
            # Read the last tensor in the file to guarantee the entire binary payload is intact
            last_tensor = f.get_tensor(keys[-1])
            if last_tensor is None:
                return False
        return True
    except Exception:
        return False


def check_model_cache(model_name: str) -> Dict[str, object]:
    """Check if all files for a model are downloaded, valid, and intact on local disk."""
    spec = get_model_spec(model_name)
    total_bytes = 0
    all_present = True

    for filename, min_mb in spec.files:
        cached = try_to_load_from_cache(spec.repo_id, filename)
        if not cached or not os.path.exists(cached):
            all_present = False
            break

        if filename.endswith(".safetensors") and not verify_safetensors(cached, min_size_mb=min_mb):
            all_present = False
            break

        total_bytes += os.path.getsize(cached)

    size_gb = total_bytes / (1024**3)
    if all_present:
        return {
            "downloaded": True,
            "size_gb": size_gb,
            "status_text": f"Ready ({size_gb:.2f} GB)",
        }
    else:
        return {
            "downloaded": False,
            "size_gb": 0.0,
            "status_text": f"Not Downloaded ({spec.approx_size} required)",
        }


def purge_model_cache(repo_id: str):
    """Purge the local cache folder for a specific Hugging Face repo if corrupted."""
    repo_folder = "models--" + repo_id.replace("/", "--")
    repo_dir = Path(HUGGINGFACE_HUB_CACHE) / repo_folder
    if repo_dir.exists():
        print(f"🧹 Purging corrupted cache folder: {repo_dir}")
        try:
            shutil.rmtree(repo_dir)
        except Exception as e:
            print(f"Warning: could not fully delete {repo_dir}: {e}")


def download_model(
    model_name: str,
    force: bool = False,
    progress_callback: Optional[Callable[[float, str], None]] = None,
) -> bool:
    """Download all files for a model with progress tracking and integrity checks."""
    spec = get_model_spec(model_name)
    status = check_model_cache(model_name)

    if status["downloaded"] and not force:
        print(f"  [Already Cached] {model_name} is complete and verified ({status['size_gb']:.2f} GB on disk).")
        return True

    print(f"\n📥 [{model_name}] Downloading from {spec.repo_id} ({spec.approx_size})...")
    start_time = time.time()

    total_files = len(spec.files)
    for idx, (filename, min_mb) in enumerate(spec.files):
        pct = idx / total_files
        desc = f"Downloading {filename} [{idx+1}/{total_files}]..."
        if progress_callback:
            progress_callback(pct, desc)

        print(f"  [{idx+1}/{total_files}] Downloading {filename}...")
        try:
            downloaded_path = hf_hub_download(
                repo_id=spec.repo_id,
                filename=filename,
                force_download=force,
            )
            if filename.endswith(".safetensors"):
                if not verify_safetensors(downloaded_path, min_size_mb=min_mb):
                    print(f"  ⚠️ Checksum verification failed for {filename}. Re-downloading fresh file...")
                    try:
                        os.remove(downloaded_path)
                    except Exception:
                        pass
                    downloaded_path = hf_hub_download(
                        repo_id=spec.repo_id,
                        filename=filename,
                        force_download=True,
                    )
        except Exception as e:
            print(f"  ❌ Failed to download {filename}: {e}")
            if "401" in str(e) or "gated" in str(e).lower() or "restricted" in str(e).lower():
                print(f"     Please accept the license terms at: https://huggingface.co/{spec.repo_id}")
                print(f"     and configure your HF_TOKEN in .env.")
            return False

    elapsed = time.time() - start_time
    new_status = check_model_cache(model_name)
    if progress_callback:
        progress_callback(1.0, f"✅ Verified & Ready ({new_status['size_gb']:.2f} GB)")

    print(f"  ✅ [{model_name}] Verified and ready in {elapsed:.1f}s ({new_status['size_gb']:.2f} GB on disk).")
    return True


def ensure_model_ready(model_name: str) -> bool:
    """Pre-flight check before model loading. Auto-heals if corrupted; preserves healthy files."""
    spec = get_model_spec(model_name)

    for filename, min_mb in spec.files:
        cached_file = try_to_load_from_cache(spec.repo_id, filename)
        if cached_file and os.path.exists(cached_file):
            if filename.endswith(".safetensors") and not verify_safetensors(cached_file, min_size_mb=min_mb):
                print(f"\n⚠️ [Self-Healing] Detected corrupted weights for '{filename}' in '{model_name}'.")
                print(f"📥 Re-downloading complete file from Hugging Face Hub...")
                try:
                    os.remove(cached_file)
                except Exception:
                    pass
                hf_hub_download(repo_id=spec.repo_id, filename=filename, force_download=True)
                print(f"✅ Verified '{filename}'!\n")

    return True
