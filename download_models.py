#!/usr/bin/env python3
"""Pre-download and verify Stable Audio 3 model weights from the terminal."""

import argparse
import os
import sys
from dotenv import load_dotenv

# Load environment / tokens
load_dotenv()

from core.compat import init_platform_compat
from core.registry import MODELS
from core.storage import check_model_cache, download_model

# Apply runtime compatibility
init_platform_compat()


def main():
    parser = argparse.ArgumentParser(
        description="Pre-download Stable Audio 3 models with progress bars (Terminal CLI)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model",
        "-m",
        type=str,
        default="all",
        choices=["all"] + list(MODELS.keys()),
        help="Which model to download, or 'all' to pre-download everything",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-download even if already cached on disk",
    )
    parser.add_argument(
        "--status",
        action="store_true",
        help="Show cache status of all models without downloading",
    )

    args = parser.parse_args()

    print("=" * 65)
    print("       Stable Audio Lab - Model Pre-Download Manager")
    print("=" * 65)

    if args.status:
        print(f"{'Model':<15} | {'Status':<25} | {'Details'}")
        print("-" * 65)
        for name, spec in MODELS.items():
            st = check_model_cache(name)
            print(f"{name:<15} | {st['status_text']:<25} | {spec.description}")
        print("=" * 65)
        return 0

    token = os.environ.get("HF_TOKEN")
    if not token:
        print("⚠️ No HF_TOKEN found in environment or .env file.")
        print("Gated model downloads may require authentication.")
        print("=" * 65)

    models_to_download = list(MODELS.keys()) if args.model == "all" else [args.model]
    print(f"Target models: {', '.join(models_to_download)}")
    if args.force:
        print("Force re-download mode: ON")

    success_count = 0
    for m in models_to_download:
        if download_model(m, force=args.force):
            success_count += 1

    print("\n" + "=" * 65)
    print(f"Summary: {success_count}/{len(models_to_download)} models ready on disk.")
    print("You can now launch the studio with: pixi run ui")
    print("=" * 65)
    return 0 if success_count == len(models_to_download) else 1


if __name__ == "__main__":
    sys.exit(main())
