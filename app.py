#!/usr/bin/env python3
"""Interactive Multi-Model Tabbed Studio for Stable Audio 3."""

import argparse
import os
import sys
import time
from datetime import datetime
from pathlib import Path
import numpy as np
import soundfile as sf
import torch
from dotenv import load_dotenv

# Load .env file for HF token
load_dotenv()

# Global model cache to avoid reloading weights into VRAM repeatedly
_MODEL_CACHE = {}


def get_device_info():
    """Return device and VRAM status string."""
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        vram = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        cuda_ver = torch.version.cuda
        return f"🚀 **GPU Accelerated:** {gpu_name} ({vram:.1f} GB VRAM) | **PyTorch:** {torch.__version__} | **CUDA:** {cuda_ver}"
    return f"⚠️ **Running on CPU** | **PyTorch:** {torch.__version__}"


def load_model(model_name: str, use_half: bool = True):
    """Retrieve model from cache or load it into memory."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    half = (device == "cuda") and use_half

    if model_name in _MODEL_CACHE:
        return _MODEL_CACHE[model_name]

    # When switching to Medium on a 6GB GPU, evict smaller models to maximize VRAM headroom
    if model_name == "medium" and torch.cuda.is_available():
        _MODEL_CACHE.clear()
        torch.cuda.empty_cache()

    from stable_audio_3 import StableAudioModel

    print(f"Loading '{model_name}' on {device.upper()} (fp16={half})...")
    try:
        model = StableAudioModel.from_pretrained(model_name, device=device, model_half=half)
    except Exception as e:
        if "401" in str(e) or "gated" in str(e).lower() or "restricted" in str(e).lower():
            raise RuntimeError(
                f"Cannot access model '{model_name}'. Please ensure you have accepted the license terms at "
                f"https://huggingface.co/stabilityai/stable-audio-3-{model_name} and configured your HF_TOKEN."
            ) from e
        raise e

    _MODEL_CACHE[model_name] = model
    return model


def generate_audio(
    model_name: str,
    prompt: str,
    negative_prompt: str,
    duration: float,
    steps: int,
    cfg_scale: float,
    seed: int,
    progress=None,
):
    """Generate audio and save it to outputs/."""
    if not prompt or not prompt.strip():
        raise ValueError("Please enter a text prompt.")

    start_time = time.time()

    # Empty cache before generation to maximize available VRAM
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    model = load_model(model_name)

    seed_val = int(seed) if seed is not None and seed != -1 else -1

    # Extract model's configured max sample size (e.g. 5292032 for small models, 16777216 for medium)
    max_sample_size = model.model_config.get("sample_size", 5292032)

    try:
        audio = model.generate(
            prompt=prompt.strip(),
            negative_prompt=negative_prompt.strip() if negative_prompt else None,
            duration=float(duration),
            steps=int(steps),
            cfg_scale=float(cfg_scale),
            seed=seed_val,
            sample_size=max_sample_size,
        )
    except torch.cuda.OutOfMemoryError as e:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        import gradio as gr
        raise gr.Error(
            f"CUDA Out of Memory! Requested {duration:.0f}s on '{model_name}' exceeded your GPU's VRAM. "
            f"Tip: Try reducing duration (e.g. 15s-30s) or switch to 'small-music' (~2GB VRAM)."
        ) from e
    except Exception as e:
        if "401" in str(e) or "gated" in str(e).lower():
            import gradio as gr
            raise gr.Error(
                f"License not accepted for '{model_name}'. Please visit "
                f"https://huggingface.co/stabilityai/stable-audio-3-{model_name} and click 'Agree and access repository'."
            ) from e
        raise e

    gen_time = time.time() - start_time
    speed = float(steps) / gen_time if gen_time > 0 else 0

    # Clean up VRAM after generation
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Save audio file
    output_dir = Path("outputs")
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    sanitized = "".join(c if c.isalnum() else "_" for c in prompt[:30]).strip("_")
    out_path = output_dir / f"{timestamp}_{model_name}_{sanitized}.wav"

    sample_rate = model.model.sample_rate
    audio_tensor = audio[0].detach().cpu()
    audio_np = audio_tensor.numpy().T
    sf.write(str(out_path), audio_np, sample_rate)

    status_msg = (
        f"✅ Generated **{duration:.1f}s** in **{gen_time:.2f}s** "
        f"({speed:.1f} steps/s) | Model: `{model_name}` | Saved to `{out_path.name}`"
    )
    return str(out_path), status_msg


def build_generation_tab(
    model_name: str,
    default_prompt: str,
    default_duration: float,
    examples: list,
    max_duration: float = 120.0,
    note: str = None,
):
    """Construct a clean, responsive generation interface for a specific model."""
    import gradio as gr

    if note:
        gr.Markdown(note)

    with gr.Row():
        with gr.Column(scale=5):
            prompt_input = gr.Textbox(
                label="Prompt",
                value=default_prompt,
                lines=3,
                placeholder="Describe the audio you want to generate...",
            )
            with gr.Row():
                duration_slider = gr.Slider(
                    minimum=1.0,
                    maximum=max_duration,
                    value=default_duration,
                    step=1.0,
                    label="Duration (seconds)",
                )
                steps_slider = gr.Slider(
                    minimum=4,
                    maximum=50,
                    value=8,
                    step=1,
                    label="Sampling Steps (8 is optimal)",
                )

            with gr.Accordion("Advanced Settings", open=False):
                negative_prompt_input = gr.Textbox(
                    label="Negative Prompt",
                    placeholder="Qualities or instruments to avoid...",
                    lines=1,
                )
                with gr.Row():
                    cfg_scale_slider = gr.Slider(
                        minimum=1.0,
                        maximum=15.0,
                        value=1.0,
                        step=0.5,
                        label="CFG Scale",
                    )
                    seed_input = gr.Number(
                        label="Seed (-1 for random)",
                        value=-1,
                        precision=0,
                    )

            generate_btn = gr.Button("⚡ Generate Audio", variant="primary", size="lg")

        with gr.Column(scale=4):
            audio_output = gr.Audio(
                label="Generated Audio",
                type="filepath",
                interactive=False,
                waveform_options=gr.WaveformOptions(
                    show_recording_waveform=False,
                    waveform_color="#3b82f6",
                    waveform_progress_color="#1d4ed8",
                ),
            )
            status_output = gr.Markdown("Ready to generate.")

    if examples:
        gr.Examples(
            examples=examples,
            inputs=[prompt_input, duration_slider],
            label="Inspiration Presets",
        )

    def on_generate(p, np_prompt, dur, st, cfg, sd):
        return generate_audio(model_name, p, np_prompt, dur, st, cfg, sd)

    generate_btn.click(
        fn=on_generate,
        inputs=[
            prompt_input,
            negative_prompt_input,
            duration_slider,
            steps_slider,
            cfg_scale_slider,
            seed_input,
        ],
        outputs=[audio_output, status_output],
    )


def create_app(model_mode: str = "all"):
    """Create the Gradio interface with tabs or single model mode."""
    import gradio as gr

    music_examples = [
        ["Upbeat funky bassline with warm rhodes piano and crisp drums", 15.0],
        ["Dreamy cinematic ambient synth pad with shimmering reverb and tape warmth", 20.0],
        ["Smooth jazz trumpet solo over mellow acoustic drums and upright bass", 15.0],
        ["TrackType: Music, traditional Turkish classical art music with Oud, Kanun, and Bendir", 20.0],
        ["Lofi hip hop drum groove with relaxing electric piano chords and vinyl crackle", 30.0],
    ]

    sfx_examples = [
        ["TrackType: SFX, a funny high-pitched rubber clown nose squeak honk sound with a quick double squeeze", 3.0],
        ["TrackType: SFX, deep campfire crackling and popping in a dense pine forest with gentle whistling night wind", 15.0],
        ["TrackType: SFX, powerful sci-fi plasma rifle blaster shot with metallic dissipation", 3.0],
        ["TrackType: SFX, classic cartoon boing spring bounce sound effect, comedic and bouncy", 3.0],
        ["TrackType: SFX, heavy pneumatic spaceship airlock door depressurizing with a loud hiss", 6.0],
    ]

    medium_examples = [
        ["An epic cinematic orchestral trailer theme with thundering percussion, brass swells, and soaring strings", 15.0],
        ["A soulful 70s funk groove with live brass section, slap bass, and vintage electric piano", 20.0],
        ["Dreamy synthwave anthem with pulsating analog arpeggios, nostalgic pads, and gated drums", 20.0],
        ["A dark industrial cyberpunk bassline with distorted synthesizer stabs and heavy driving kick", 15.0],
    ]

    theme = gr.themes.Soft(
        primary_hue="indigo",
        secondary_hue="blue",
        neutral_hue="slate",
    )

    with gr.Blocks(title="Stable Audio Lab Studio") as demo:
        gr.Markdown("# 🎧 Stable Audio Lab Studio")
        gr.Markdown(get_device_info())

        if model_mode == "all":
            with gr.Tabs():
                with gr.Tab("🎵 Music (Small-Music)"):
                    build_generation_tab(
                        model_name="small-music",
                        default_prompt="Upbeat funky bassline with warm rhodes piano and crisp drums",
                        default_duration=15.0,
                        max_duration=120.0,
                        examples=music_examples,
                    )
                with gr.Tab("🔊 Sound Effects (Small-SFX)"):
                    build_generation_tab(
                        model_name="small-sfx",
                        default_prompt="TrackType: SFX, a funny high-pitched rubber clown nose squeak honk sound with a quick double squeeze",
                        default_duration=3.0,
                        max_duration=30.0,
                        examples=sfx_examples,
                    )
                with gr.Tab("🎛️ Medium (1.4B Quality)"):
                    build_generation_tab(
                        model_name="medium",
                        default_prompt="An epic cinematic orchestral trailer theme with thundering percussion, brass swells, and soaring strings",
                        default_duration=15.0,
                        max_duration=180.0,
                        examples=medium_examples,
                        note="💡 **Hardware Guide for Medium (1.4B):** On 6GB GPUs (e.g. RTX 4050), stick to **10–30s**. On 8GB GPUs (e.g. RTX 5060), you can reach **45–60s**. For longer clips, 12GB+ VRAM is recommended.",
                    )
                with gr.Tab("ℹ️ System Diagnostics"):
                    gr.Markdown("### Workspace & Hardware Information")
                    gr.Markdown(f"""
- **Active GPU**: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None (CPU)'}
- **PyTorch Version**: `{torch.__version__}`
- **CUDA Runtime**: `{torch.version.cuda if torch.cuda.is_available() else 'N/A'}`
- **Supported Models**:
  - `small-music` (433M parameters, 44.1 kHz stereo composition, max 120s)
  - `small-sfx` (433M parameters, 44.1 kHz stereo sound effects, max 120s)
  - `medium` (1.4B parameters, production audio, max 380s)
- **Audio Output Directory**: `outputs/`
- **Hugging Face License Links**:
  - [small-music](https://huggingface.co/stabilityai/stable-audio-3-small-music)
  - [small-sfx](https://huggingface.co/stabilityai/stable-audio-3-small-sfx)
  - [medium](https://huggingface.co/stabilityai/stable-audio-3-medium)
                    """)
        elif model_mode == "small-sfx":
            gr.Markdown("### 🔊 Stable Audio 3 Small (Sound Effects)")
            build_generation_tab(
                model_name="small-sfx",
                default_prompt="TrackType: SFX, a funny high-pitched rubber clown nose squeak honk sound with a quick double squeeze",
                default_duration=3.0,
                max_duration=30.0,
                examples=sfx_examples,
            )
        elif model_mode == "small-music":
            gr.Markdown("### 🎵 Stable Audio 3 Small (Music)")
            build_generation_tab(
                model_name="small-music",
                default_prompt="Upbeat funky bassline with warm rhodes piano and crisp drums",
                default_duration=15.0,
                max_duration=120.0,
                examples=music_examples,
            )
        elif model_mode == "medium":
            gr.Markdown("### 🎛️ Stable Audio 3 Medium (1.4B)")
            build_generation_tab(
                model_name="medium",
                default_prompt="An epic cinematic orchestral trailer theme with thundering percussion, brass swells, and soaring strings",
                default_duration=15.0,
                max_duration=180.0,
                examples=medium_examples,
                note="💡 **Hardware Guide for Medium (1.4B):** On 6GB GPUs (e.g. RTX 4050), stick to **10–30s**. On 8GB GPUs (e.g. RTX 5060), you can reach **45–60s**. For longer clips, 12GB+ VRAM is recommended.",
            )
        else:
            gr.Markdown(f"### 🎛️ Stable Audio 3 ({model_mode})")
            build_generation_tab(
                model_name=model_mode,
                default_prompt="Upbeat lo-fi beat",
                default_duration=15.0,
                max_duration=60.0,
                examples=music_examples,
            )

    return demo


def main():
    parser = argparse.ArgumentParser(
        description="Launch Stable Audio Lab Gradio Web Studio",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model",
        "-m",
        type=str,
        default="all",
        choices=["all", "small-music", "small-sfx", "medium"],
        help="Launch with all switchable model tabs ('all'), or specify a single model ('small-music', 'small-sfx', 'medium')",
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

    args = parser.parse_args()

    app = create_app(model_mode=args.model)
    app.queue()
    app.launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
        inbrowser=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
