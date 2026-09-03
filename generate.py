#!/usr/bin/env python3
"""CLI Audio Generator for Stable Audio 3."""

import argparse
import os
import sys
from dotenv import load_dotenv

# Load environment / tokens
load_dotenv()

from core.compat import init_platform_compat
from core.engine import generate_audio
from core.registry import MODELS, get_model_spec

# Apply runtime compatibility
init_platform_compat()


def main():
    parser = argparse.ArgumentParser(
        description="Generate audio with Stable Audio 3 (CLI)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--prompt",
        "-p",
        type=str,
        required=True,
        help="Text prompt describing the audio to generate",
    )
    parser.add_argument(
        "--negative-prompt",
        "-n",
        type=str,
        default=None,
        help="Optional negative prompt",
    )
    parser.add_argument(
        "--model",
        "-m",
        type=str,
        default="small-music",
        choices=list(MODELS.keys()),
        help="Model variant to use",
    )
    parser.add_argument(
        "--duration",
        "-d",
        type=float,
        default=30.0,
        help="Duration of audio in seconds (Small up to 120s, Medium up to 380s)",
    )
    parser.add_argument(
        "--steps",
        "-s",
        type=int,
        default=8,
        help="Diffusion sampling steps (8 is optimal for post-trained models)",
    )
    parser.add_argument(
        "--cfg-scale",
        type=float,
        default=1.0,
        help="Classifier-free guidance scale",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=-1,
        help="Random seed (-1 for randomized seed)",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=None,
        help="Destination path for .wav file (defaults to outputs/<timestamp>_<model>_<prompt>.wav)",
    )

    args = parser.parse_args()

    spec = get_model_spec(args.model)
    if args.duration > spec.max_duration:
        print(f"⚠️ Warning: Requested duration {args.duration}s exceeds {args.model} maximum ({spec.max_duration}s).")

    print(f"Generating with '{args.model}'...")
    print(f"  Prompt   : \"{args.prompt}\"")
    print(f"  Duration : {args.duration}s")
    print(f"  Steps    : {args.steps}")
    print(f"  Seed     : {args.seed}")

    try:
        out_file, status = generate_audio(
            model_name=args.model,
            prompt=args.prompt,
            negative_prompt=args.negative_prompt,
            duration=args.duration,
            steps=args.steps,
            cfg_scale=args.cfg_scale,
            seed=args.seed,
            output_path=args.output,
        )
        print(f"\n{status}")
        return 0
    except Exception as e:
        print(f"\n❌ Error during generation: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
