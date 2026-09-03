#!/usr/bin/env python3
"""Pre-download and cache manager for Stable Audio 3 models."""

from __future__ import annotations

import argparse
import sys
from dotenv import load_dotenv

load_dotenv()

from core import MODELS, check_model_cache, download_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Pre-download Stable Audio 3 models",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model", "-m", type=str, default="all", choices=["all"] + list(MODELS.keys()), help="Target model"
    )
    parser.add_argument("--force", action="store_true", help="Force re-download cached files")
    parser.add_argument("--status", action="store_true", help="Show cache status without downloading")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if args.status:
        print(f"\n{'Model':<15} | {'Status':<20} | {'Description'}")
        print("-" * 65)
        for name, spec in MODELS.items():
            st = check_model_cache(name)
            print(f"{name:<15} | {st.status_text:<20} | {spec.description}")
        print()
        return 0

    targets = list(MODELS.keys()) if args.model == "all" else [args.model]
    print(f"Checking {len(targets)} model(s): {', '.join(targets)}")

    for name in targets:
        st = check_model_cache(name)
        if st.downloaded and not args.force:
            print(f"[{name}] Already cached ({st.size_gb:.2f} GB).")
            continue

        print(f"[{name}] Downloading...")
        success = download_model(name, force=args.force)
        if not success:
            print(f"Error: [{name}] Download failed. Verify network connection and HF_TOKEN.")
            return 1

    print("All requested models are ready on disk.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
