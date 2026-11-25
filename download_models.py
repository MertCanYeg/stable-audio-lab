#!/usr/bin/env python3
"""Pre-download Stable Audio 3 model weights into the local Hugging Face cache."""

from __future__ import annotations

import argparse
import sys
from dotenv import load_dotenv

load_dotenv()

from core import MODELS, check_model_cache, download_model, get_device_info


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command line arguments for model weight pre-downloading."""
    parser = argparse.ArgumentParser(
        description="Download Stable Audio 3 model weights to the local cache",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--models",
        "-m",
        nargs="+",
        choices=list(MODELS.keys()),
        default=list(MODELS.keys()),
        help="Model(s) to download",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-download even if already cached",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    print(f"Stable Audio Lab | {get_device_info()}")

    all_ok = True
    for model_name in args.models:
        status = check_model_cache(model_name)
        if status.downloaded and not args.force:
            print(f"[{model_name}] Already cached: {status.status_text}")
            continue
        ok = download_model(model_name, force=args.force)
        if ok:
            updated = check_model_cache(model_name)
            print(f"[{model_name}] Ready: {updated.status_text}")
        else:
            print(f"[{model_name}] Download failed.")
            all_ok = False

    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
