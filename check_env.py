#!/usr/bin/env python3
"""Environment and GPU Diagnostics for Stable Audio Lab."""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env if present
load_dotenv()


def main():
    print("=" * 60)
    print("       Stable Audio Lab - Environment Diagnostics")
    print("=" * 60)

    # 1. Python Information
    print(f"Python Version    : {sys.version.split()[0]}")
    print(f"Executable        : {sys.executable}")
    print(f"Platform          : {sys.platform}")

    # 2. PyTorch & CUDA Diagnostics
    try:
        import torch
        import torchaudio

        print(f"PyTorch Version   : {torch.__version__}")
        print(f"TorchAudio Version: {torchaudio.__version__}")

        cuda_available = torch.cuda.is_available()
        print(f"CUDA Available    : {cuda_available}")

        if cuda_available:
            device_count = torch.cuda.device_count()
            current_device = torch.cuda.current_device()
            device_name = torch.cuda.get_device_name(current_device)
            total_vram_gb = torch.cuda.get_device_properties(current_device).total_memory / (1024**3)
            print(f"Device Count      : {device_count}")
            print(f"Active GPU        : {device_name} (ID: {current_device})")
            print(f"Total VRAM        : {total_vram_gb:.2f} GB")
        else:
            print("WARNING: CUDA is not available. Generation will fall back to CPU (slower).")
    except ImportError as e:
        print(f"ERROR: Failed to import PyTorch: {e}")
        return 1

    # 3. Stable Audio 3 Diagnostics
    try:
        import stable_audio_3
        from stable_audio_3 import StableAudioModel

        print("Stable Audio 3    : Installed and importable")
    except ImportError as e:
        print(f"ERROR: Failed to import stable_audio_3: {e}")
        return 1

    # 4. Hugging Face Authentication Check
    print("-" * 60)
    token = os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN")
    try:
        from huggingface_hub import get_token, whoami

        cached_token = get_token()
        active_token = token or cached_token

        if active_token:
            user_info = whoami(token=active_token)
            print(f"Hugging Face Auth : Logged in as '{user_info.get('name', 'Unknown')}'")
        else:
            print("Hugging Face Auth : No token detected.")
            print("                    Note: 'Stable Audio 3 Small' requires accepting the license")
            print("                    at https://huggingface.co/stabilityai/stable-audio-3-small-music")
            print("                    and setting HF_TOKEN in your .env file or running `huggingface-cli login`.")
    except Exception as e:
        print(f"Hugging Face Auth : Unable to verify token: {e}")

    print("=" * 60)
    print("Diagnostics completed successfully!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
