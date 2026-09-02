#!/usr/bin/env python3
"""CLI Audio Generator for Stable Audio 3 Small."""

import argparse
import os
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv

# Load .env file if present
load_dotenv()


def slugify(text: str, max_len: int = 40) -> str:
    """Convert text into a safe filename slug."""
    text = re.sub(r"[^\w\s-]", "", text).strip().lower()
    slug = re.sub(r"[-\s]+", "_", text)
    return slug[:max_len] if slug else "generated_audio"


def main():
    parser = argparse.ArgumentParser(
        description="Generate audio with Stable Audio 3 Small (CLI)",
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
        choices=["small-music", "small-sfx", "medium"],
        help="Model variant to use",
    )
    parser.add_argument(
        "--duration",
        "-d",
        type=float,
        default=30.0,
        help="Duration of audio in seconds (Small supports up to 120s)",
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
        help="Destination path for .wav file (defaults to outputs/<timestamp>_<prompt>.wav)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use ('cuda', 'cpu', or auto-detect)",
    )
    parser.add_argument(
        "--no-half",
        action="store_true",
        help="Force full float32 precision instead of fp16",
    )

    args = parser.parse_args()

    # Verify Hugging Face authentication
    token = os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN")
    try:
        from huggingface_hub import get_token

        if not token and not get_token():
            print("=" * 65)
            print("WARNING: No Hugging Face access token found!")
            print("Stable Audio 3 models are gated. Please ensure you:")
            print("1. Accepted terms at: https://huggingface.co/stabilityai/stable-audio-3-" + args.model)
            print("2. Set HF_TOKEN in your .env file or run `huggingface-cli login`")
            print("=" * 65)
    except Exception:
        pass

    import torch
    import torchaudio
    from stable_audio_3 import StableAudioModel

    device = args.device
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    use_half = (device == "cuda") and (not args.no_half)

    print(f"Loading model '{args.model}' on {device.upper()} (fp16={use_half})...")
    load_start = time.time()
    try:
        model = StableAudioModel.from_pretrained(args.model, device=device, model_half=use_half)
    except Exception as e:
        print(f"\nFailed to load model '{args.model}': {e}")
        print("\nTroubleshooting tips:")
        print("1. Did you accept the license agreement at https://huggingface.co/stabilityai/stable-audio-3-" + args.model + " ?")
        print("2. Is your HF_TOKEN set in .env or via `huggingface-cli login`?")
        return 1

    load_time = time.time() - load_start
    print(f"Model loaded in {load_time:.2f}s.")

    print(f"\nGenerating audio...")
    print(f"  Prompt   : \"{args.prompt}\"")
    print(f"  Duration : {args.duration}s")
    print(f"  Steps    : {args.steps}")
    print(f"  Seed     : {args.seed}")

    gen_start = time.time()
    audio = model.generate(
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        duration=args.duration,
        steps=args.steps,
        cfg_scale=args.cfg_scale,
        seed=args.seed,
    )
    gen_time = time.time() - gen_start

    # Prepare output path
    output_dir = Path("outputs")
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.output:
        out_path = Path(args.output)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        slug = slugify(args.prompt)
        out_path = output_dir / f"{timestamp}_{slug}.wav"

    # Save audio: shape is [1, channels, samples]
    sample_rate = model.model.sample_rate
    audio_tensor = audio[0].detach().cpu()
    torchaudio.save(str(out_path), audio_tensor, sample_rate)

    print(f"\nGeneration complete in {gen_time:.2f}s!")
    print(f"  Saved to : {out_path.resolve()}")
    print(f"  Format   : {sample_rate} Hz, {audio_tensor.shape[0]} channels, {args.duration}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
