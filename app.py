#!/usr/bin/env python3
"""Interactive Gradio Web UI for Stable Audio 3 Small."""

import argparse
import os
import sys
import numpy as np
from dotenv import load_dotenv

# Load .env file
load_dotenv()


def patch_torchaudio_save():
    """Ensure torchaudio.save uses soundfile reliably without torchcodec errors."""
    import torch
    import torchaudio
    import soundfile as sf

    def _safe_torchaudio_save(uri, src, sample_rate, *args, **kwargs):
        if isinstance(src, torch.Tensor):
            data = src.detach().cpu().numpy()
        else:
            data = np.asarray(src)
        # If shape is [channels, samples], transpose to [samples, channels]
        if data.ndim == 2 and data.shape[0] in (1, 2) and data.shape[1] > data.shape[0]:
            data = data.T
        sf.write(str(uri), data, sample_rate)

    torchaudio.save = _safe_torchaudio_save


def main():
    parser = argparse.ArgumentParser(
        description="Launch Stable Audio 3 Small Gradio Web Interface",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model",
        "-m",
        type=str,
        default="small-music",
        choices=["small-music", "small-sfx", "medium"],
        help="Model checkpoint to load",
    )
    parser.add_argument(
        "--port",
        "-p",
        type=int,
        default=7860,
        help="Local port for the Gradio web server",
    )
    parser.add_argument(
        "--host",
        type=str,
        default="127.0.0.1",
        help="Server host/IP to bind",
    )
    parser.add_argument(
        "--share",
        action="store_true",
        help="Create a public Gradio share link",
    )
    parser.add_argument(
        "--no-half",
        action="store_true",
        help="Disable fp16 half precision",
    )

    args = parser.parse_args()

    # Check Hugging Face token
    token = os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN")
    try:
        from huggingface_hub import get_token

        if not token and not get_token():
            print("=" * 65)
            print("WARNING: No Hugging Face access token detected.")
            print("If downloading weights for the first time, please ensure:")
            print(f"1. You accepted the license at: https://huggingface.co/stabilityai/stable-audio-3-{args.model}")
            print("2. You set HF_TOKEN in your .env file or run `huggingface-cli login`")
            print("=" * 65)
    except Exception:
        pass

    # Apply safe audio saving patch
    patch_torchaudio_save()

    import torch
    from stable_audio_3 import StableAudioModel
    from stable_audio_3.interface.diffusion_cond import create_diffusion_cond_ui

    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_half = (device == "cuda") and (not args.no_half)

    print(f"Loading '{args.model}' on {device.upper()} (fp16={use_half})...")
    try:
        model = StableAudioModel.from_pretrained(args.model, device=device, model_half=use_half)
    except Exception as e:
        print(f"\nFailed to load model '{args.model}': {e}")
        print("\nTroubleshooting tips:")
        print(f"1. Check model license acceptance at https://huggingface.co/stabilityai/stable-audio-3-{args.model}")
        print("2. Make sure HF_TOKEN is configured in .env or via huggingface-cli login.")
        return 1

    print(f"Starting Gradio web interface on http://{args.host}:{args.port} ...")
    interface = create_diffusion_cond_ui(
        model,
        gradio_title=f"Stable Audio 3 ({args.model})",
        default_prompt="A melodic lo-fi beat with relaxing electric piano chords, soft warm bass, and subtle vinyl crackle",
    )
    interface.queue()
    interface.launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
        inbrowser=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
