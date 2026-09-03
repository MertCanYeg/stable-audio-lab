#!/usr/bin/env python3
"""CLI Audio Generator for Stable Audio 3."""

from __future__ import annotations

import argparse
import sys
import traceback
from dotenv import load_dotenv

load_dotenv()

from core import (
    MODELS,
    GenerationConfig,
    StableAudioError,
    generate_audio,
    get_device_info,
    get_model_spec,
)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate audio with Stable Audio 3",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--prompt", "-p", type=str, required=True, help="Text prompt describing the audio")
    parser.add_argument("--negative-prompt", "-n", type=str, default=None, help="Optional negative prompt")
    parser.add_argument("--model", "-m", type=str, default="small-music", choices=list(MODELS.keys()), help="Model")
    parser.add_argument("--duration", "-d", type=float, default=15.0, help="Duration in seconds")
    parser.add_argument("--steps", "-s", type=int, default=8, help="Sampling steps (8 is optimal)")
    parser.add_argument("--cfg-scale", type=float, default=1.0, help="Classifier-free guidance scale")
    parser.add_argument("--seed", type=int, default=-1, help="Random seed (-1 for random)")
    parser.add_argument("--output", "-o", type=str, default=None, help="Output .wav path")
    parser.add_argument("--verbose", "-v", action="store_true", help="Print full traceback on errors")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    spec = get_model_spec(args.model)
    if args.duration > spec.max_duration:
        print(f"Warning: {args.duration}s exceeds '{args.model}' maximum of {spec.max_duration}s. "
              f"Output will be clamped by the model.")

    config = GenerationConfig(
        model_name=args.model,
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        duration=args.duration,
        steps=args.steps,
        cfg_scale=args.cfg_scale,
        seed=args.seed,
        output_path=args.output,
    )

    print(f"\nStable Audio 3 CLI")
    print(f"{get_device_info()}")
    seed_desc = str(config.seed) if config.seed != -1 else "Random"
    print(f"Model: {config.model_name} ({spec.description})")
    print(f"Duration: {config.duration:.1f}s | Steps: {config.steps} | CFG: {config.cfg_scale} | Seed: {seed_desc}")
    print(f"Prompt: \"{config.prompt}\"")
    if config.negative_prompt:
        print(f"Negative: \"{config.negative_prompt}\"")
    print()

    try:
        result = generate_audio(config=config)
        print(f"\n{result.status_message}\n")
        return 0
    except KeyboardInterrupt:
        print("\nCancelled by user.")
        return 130
    except StableAudioError as e:
        print(f"\nError: {e}")
        if args.verbose:
            traceback.print_exc()
        return 1
    except Exception as e:
        print(f"\nUnexpected error: {e}")
        if args.verbose:
            traceback.print_exc()
        else:
            print("Run with --verbose for the full traceback.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
