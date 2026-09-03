#!/usr/bin/env python3
"""Environment and GPU diagnostics for Stable Audio Lab."""

import os
import sys
from dotenv import load_dotenv

load_dotenv()

from core.compat import get_device_info
from core import MODELS, check_model_cache


def main():
    print("=" * 60)
    print("       Stable Audio Lab - Environment Diagnostics")
    print("=" * 60)

    print(f"Python       : {sys.version.split()[0]} ({sys.executable})")
    print(f"Platform     : {sys.platform}")

    try:
        import torch
        print(f"PyTorch      : {torch.__version__}")
        print(f"Diagnostics  : {get_device_info()}")
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            print(f"GPU Memory   : {props.total_memory / 1024**3:.1f} GB dedicated VRAM")
    except ImportError as e:
        print(f"PyTorch      : NOT INSTALLED ({e})")
        return 1

    try:
        import stable_audio_3
        print("Stable Audio : Installed")
    except ImportError as e:
        print(f"Stable Audio : NOT INSTALLED ({e})")
        return 1

    # Authentication
    token = os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN")
    try:
        from huggingface_hub import get_token, whoami
        active_token = token or get_token()
        if active_token:
            user = whoami(token=active_token).get("name", "Authenticated User")
            print(f"Hugging Face : Logged in as '{user}'")
        else:
            print("Hugging Face : No token (set HF_TOKEN in .env for gated models)")
    except Exception as e:
        print(f"Hugging Face : Auth check failed ({e})")

    # Model cache status
    print(f"\n{'Model':<15} | {'Status':<20} | {'Description'}")
    print("-" * 70)
    for name, spec in MODELS.items():
        st = check_model_cache(name)
        print(f"{name:<15} | {st.status_text:<20} | {spec.description}")

    print("=" * 60)
    print("Diagnostics complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
