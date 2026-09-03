#!/usr/bin/env python3
"""Interactive Multi-Model Tabbed Studio for Stable Audio 3."""

import argparse
import sys
from dotenv import load_dotenv

# Load environment & tokens
load_dotenv()

import gradio as gr
import torch

from core.compat import init_platform_compat, get_device_banner
from core.engine import generate_audio
from core.registry import MODELS, ModelSpec, get_model_spec
from core.storage import check_model_cache, download_model

# Apply runtime compatibility patches (Triton fallback, asyncio reset, warning filters)
init_platform_compat()


def build_generation_tab(spec: ModelSpec):
    """Construct a clean, responsive generation interface for a specific model."""
    init_status = check_model_cache(spec.name)
    is_ready = init_status["downloaded"]

    # Model status & quick-download action bar
    with gr.Row(variant="panel"):
        status_badge = gr.Markdown(
            f"🟢 **Installed** ({init_status['size_gb']:.2f} GB) • {spec.note}"
            if is_ready
            else f"🟠 **Weights not downloaded** ({spec.approx_size} required) • Click **Download Weights** or **Generate Audio** to fetch."
        )
        dl_btn = gr.Button(
            "📥 Download Weights",
            variant="secondary",
            size="sm",
            scale=0,
            visible=not is_ready,
        )

    def on_tab_download(progress=gr.Progress(track_tqdm=True)):
        def cb(pct, desc):
            progress(pct, desc=desc)

        print(f"\n📥 [UI Action] Starting download for '{spec.name}'...")
        success = download_model(spec.name, progress_callback=cb)
        new_st = check_model_cache(spec.name)
        if success and new_st["downloaded"]:
            gr.Info(f"Model '{spec.name}' downloaded and verified successfully!")
            return (
                f"🟢 **Installed** ({new_st['size_gb']:.2f} GB) • {spec.note}",
                gr.update(visible=False),
            )
        else:
            raise gr.Error(f"Could not complete {spec.name} download. Check your connection or HF_TOKEN.")

    dl_btn.click(
        fn=on_tab_download,
        outputs=[status_badge, dl_btn],
        concurrency_id="download_worker",
        concurrency_limit=1,
    )

    with gr.Row():
        with gr.Column(scale=5):
            prompt_input = gr.Textbox(
                label="Prompt",
                value=spec.default_prompt,
                lines=3,
                placeholder="Describe the audio you want to generate...",
            )
            with gr.Row():
                duration_slider = gr.Slider(
                    minimum=1.0,
                    maximum=spec.max_duration,
                    value=spec.default_duration,
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
                    placeholder="Describe sounds or artifacts to avoid...",
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

    # Inspiration Presets
    if spec.examples:
        if isinstance(spec.examples, dict):
            with gr.Tabs():
                for cat_label, cat_items in spec.examples.items():
                    with gr.Tab(cat_label):
                        gr.Examples(
                            examples=cat_items,
                            inputs=[prompt_input, duration_slider],
                            examples_per_page=25,
                            label="Presets",
                        )
        else:
            gr.Examples(
                examples=spec.examples,
                inputs=[prompt_input, duration_slider],
                examples_per_page=25,
                label="Presets",
            )

    def on_generate(p, np_prompt, dur, st, cfg, sd, progress=gr.Progress(track_tqdm=True)):
        out_file, status = generate_audio(
            model_name=spec.name,
            prompt=p,
            negative_prompt=np_prompt,
            duration=dur,
            steps=st,
            cfg_scale=cfg,
            seed=sd,
            progress=progress,
        )
        return out_file, status

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
        concurrency_id="gpu_worker",
        concurrency_limit=1,
    )


def build_diagnostics_tab():
    """Construct hardware and model cache diagnostics tab."""
    gr.Markdown("### 🖥️ Hardware & Environment")
    gr.Markdown(f"""
- **Active GPU**: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None (CPU)'}
- **PyTorch Version**: `{torch.__version__}`
- **CUDA Runtime**: `{torch.version.cuda if torch.cuda.is_available() else 'N/A'}`
- **Audio Output Directory**: `outputs/`
    """)

    gr.Markdown("### 📦 Installed Models & Local Cache")
    table_lines = [
        "| Model | Status | Size on Disk | Parameters | Max Duration | Repository |",
        "| :--- | :---: | :---: | :---: | :---: | :--- |",
    ]
    for name, spec in MODELS.items():
        st = check_model_cache(name)
        status_icon = "🟢 Installed" if st["downloaded"] else "⚪ Not Downloaded"
        size_str = f"{st['size_gb']:.2f} GB" if st["downloaded"] else "—"
        table_lines.append(
            f"| **{name}** | {status_icon} | {size_str} | {spec.parameters} | {spec.max_duration:.0f}s | [`{spec.repo_id}`](https://huggingface.co/{spec.repo_id}) |"
        )

    gr.Markdown("\n".join(table_lines))


def create_app(model_mode: str = "all"):
    """Create the Gradio Blocks application."""
    theme = gr.themes.Soft(
        primary_hue="indigo",
        secondary_hue="blue",
        neutral_hue="slate",
    )

    with gr.Blocks(title="Stable Audio Lab Studio") as demo:
        gr.Markdown("# 🎧 Stable Audio Lab Studio")
        gr.Markdown(get_device_banner())

        if model_mode == "all":
            with gr.Tabs():
                for name, spec in MODELS.items():
                    with gr.Tab(spec.display_name):
                        build_generation_tab(spec)
                with gr.Tab("ℹ️ System Diagnostics"):
                    build_diagnostics_tab()
        else:
            spec = get_model_spec(model_mode)
            gr.Markdown(f"### {spec.display_name}")
            build_generation_tab(spec)

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
        choices=["all"] + list(MODELS.keys()),
        help="Launch with all switchable model tabs ('all'), or specify a single model",
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
    app.queue(default_concurrency_limit=1)
    app.launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
        inbrowser=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
