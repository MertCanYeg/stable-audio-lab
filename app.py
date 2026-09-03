#!/usr/bin/env python3
"""Interactive Web Studio for Stable Audio 3."""

import argparse
import functools
import gc
import queue
import threading
from dotenv import load_dotenv
import gradio as gr
import torch

load_dotenv()

from core import MODELS, GenerationConfig, StableAudioError, generate_audio, get_device_info

AUTO_SCROLL_JS = """
() => {
    const observer = new MutationObserver(() => {
        document.querySelectorAll('.status-console textarea').forEach(el => {
            el.scrollTop = el.scrollHeight;
        });
    });
    observer.observe(document.body, { childList: true, subtree: true, characterData: true });
}
"""

CUSTOM_CSS = """
.gradio-container {
    max-width: 95% !important;
    width: 95% !important;
    margin: 0 auto !important;
    padding: 12px 24px !important;
}
.table-wrap {
    max-height: 280px !important;
    overflow-y: auto !important;
    border-radius: 6px !important;
    border: 1px solid var(--border-color-primary) !important;
}
.table-wrap tr:hover {
    background-color: var(--background-fill-secondary) !important;
    cursor: pointer !important;
}
.model-info-banner {
    padding: 8px 14px;
    border-radius: 6px;
    background-color: var(--background-fill-secondary);
    margin-bottom: 12px;
    font-size: 0.88em;
    min-height: 38px !important;
    display: flex !important;
    align-items: center !important;
    box-sizing: border-box !important;
}
.prompt-box textarea {
    height: 76px !important;
    min-height: 76px !important;
    max-height: 76px !important;
    resize: none !important;
}
.neg-prompt-box textarea {
    height: 42px !important;
    min-height: 42px !important;
    max-height: 42px !important;
    resize: none !important;
}
.gradio-slider input[type="number"] {
    width: 72px !important;
    min-width: 72px !important;
    text-align: center !important;
}
.gradio-slider span, .gradio-slider label {
    white-space: nowrap !important;
}
.status-console textarea {
    font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", monospace !important;
    font-size: 0.82rem !important;
    line-height: 1.45 !important;
    background-color: #0d1117 !important;
    color: #e6edf3 !important;
    border: 1px solid #30363d !important;
    border-radius: 6px !important;
    height: 310px !important;
    white-space: pre-wrap !important;
    word-break: break-word !important;
    overflow-y: auto !important;
    overflow-x: hidden !important;
}
"""

TAB_CONFIG = [
    (
        "small-music",
        "Describe the musical track: genre, instruments, mood, tempo...",
        "e.g. vocals, speech, harsh treble, distortion, low quality, muddy bass, background noise",
        "Optional: Suppress unwanted instruments, vocals, or acoustic flaws.",
    ),
    (
        "small-sfx",
        "TrackType: SFX, describe the sound effect or foley soundscape...",
        "e.g. music, melody, singing, synthesizer, speech, hum",
        "Optional: Suppress unwanted music, melody, vocals, or background noise.",
    ),
    (
        "medium",
        "Describe the cinematic music or high-fidelity sound design...",
        "e.g. clipping, distortion, low quality, noise, out of tune, artifacts",
        "Optional: Suppress acoustic flaws, distortion, or unwanted elements.",
    ),
]


def generate(
    model_name: str,
    prompt: str,
    negative_prompt: str,
    duration: float,
    steps: int,
    cfg: float,
    seed: int,
    progress=gr.Progress(track_tqdm=False),
):
    """Generate audio with live streaming status, interactive UI sampling bar, and error recovery."""
    config = GenerationConfig(
        model_name=model_name,
        prompt=prompt,
        negative_prompt=negative_prompt,
        duration=duration,
        steps=steps,
        cfg_scale=cfg,
        seed=seed,
    )
    config.validate()

    q: queue.Queue = queue.Queue()
    logs: list[str] = []
    init_msg = f"Initializing {model_name}..."
    logs.append(init_msg)
    yield None, init_msg

    def worker():
        try:
            result = generate_audio(
                config=config,
                progress=progress,
                status_callback=lambda msg: q.put(("status", msg)),
            )
            q.put(("done", result))
        except Exception as e:
            q.put(("error", e))

    thread = threading.Thread(target=worker, daemon=True)
    thread.start()

    active_sampling_bar = None

    while True:
        try:
            event_type, payload = q.get(timeout=0.05)
        except queue.Empty:
            if not thread.is_alive() and q.empty():
                break
            continue

        if event_type == "status":
            # Once real stages start, replace the initial placeholder
            if logs == [init_msg]:
                logs.clear()

            if payload.startswith("[sampling]"):
                active_sampling_bar = payload
            else:
                logs.append(payload)
                active_sampling_bar = None

            display_text = "\n".join(logs)
            if active_sampling_bar:
                display_text = f"{display_text}\n{active_sampling_bar}" if display_text else active_sampling_bar
            yield None, display_text

        elif event_type == "done":
            summary_display = f"{chr(10).join(logs)}\n\n✅ {payload.status_message}"
            yield payload.output_path, summary_display
            break

        elif event_type == "error":
            err = payload
            if isinstance(err, StableAudioError):
                err_msg = f"❌ Error: {err}"
                yield None, f"{chr(10).join(logs)}\n\n{err_msg}" if logs else err_msg
                raise gr.Error(str(err)) from err
            elif isinstance(err, torch.cuda.OutOfMemoryError):
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                oom_msg = (
                    f"❌ GPU out of memory generating {duration:.0f}s with '{model_name}'. "
                    f"Try reducing duration or switching to a smaller model."
                )
                yield None, f"{chr(10).join(logs)}\n\n{oom_msg}" if logs else oom_msg
                raise gr.Error(oom_msg) from err
            else:
                err_msg = f"❌ Generation failed: {err}"
                yield None, f"{chr(10).join(logs)}\n\n{err_msg}" if logs else err_msg
                raise gr.Error(err_msg) from err


def reset_status():
    """Reset the status textbox to ready state."""
    return None, "Ready to generate."


def build_model_tab(
    model_name: str,
    placeholder: str,
    neg_placeholder: str,
    neg_info: str,
):
    """Build UI layout and bindings for a specific model tab."""
    spec = MODELS[model_name]

    banner = f"**{spec.name}** | Repo: `{spec.repo_id}` | Max Duration: {spec.max_duration:.0f}s | Audio: 44.1 kHz Stereo"
    gr.Markdown(banner, elem_classes=["model-info-banner"])

    with gr.Row():
        with gr.Column(scale=1):
            prompt = gr.Textbox(
                label="Prompt",
                value=spec.default_prompt,
                lines=3,
                max_lines=3,
                placeholder=placeholder,
                elem_classes=["prompt-box"],
            )
            negative_prompt = gr.Textbox(
                label="Negative Prompt",
                lines=1,
                max_lines=1,
                placeholder=neg_placeholder,
                info=neg_info,
                elem_classes=["neg-prompt-box"],
            )
            duration = gr.Slider(
                minimum=1,
                maximum=int(spec.max_duration),
                value=int(spec.default_duration),
                step=1,
                label="Duration (seconds)",
            )
            steps = gr.Slider(
                minimum=4,
                maximum=50,
                value=8,
                step=1,
                label="Sampling Steps",
            )
            with gr.Row():
                cfg = gr.Slider(
                    minimum=1.0,
                    maximum=15.0,
                    value=1.0,
                    step=0.5,
                    label="CFG",
                    scale=2,
                )
                seed = gr.Number(
                    value=-1,
                    precision=0,
                    label="Seed (-1 = random)",
                    scale=1,
                )
            with gr.Row():
                generate_btn = gr.Button(
                    f"✨ Generate with {model_name}",
                    variant="primary",
                    size="lg",
                    scale=4,
                )
                clear_btn = gr.Button(
                    "🗑️ Clear Output",
                    variant="secondary",
                    size="lg",
                    scale=1,
                )

        with gr.Column(scale=1):
            output_audio = gr.Audio(label="Generated Audio", type="filepath")
            status = gr.Textbox(
                label="Generation Status & Telemetry",
                value="Ready to generate.",
                interactive=False,
                lines=8,
                max_lines=16,
                elem_classes=["status-console"],
            )

    generate_btn.click(
        fn=functools.partial(generate, model_name),
        inputs=[prompt, negative_prompt, duration, steps, cfg, seed],
        outputs=[output_audio, status],
    )

    clear_btn.click(
        fn=reset_status,
        inputs=[],
        outputs=[output_audio, status],
    )

    if spec.examples:
        gr.Markdown("#### Prompt Templates *(click any row to load)*")
        gr.Examples(
            examples=spec.examples,
            inputs=[prompt, duration],
            examples_per_page=100,
        )


def create_app():
    """Create the root Gradio Blocks application."""
    with gr.Blocks(title="Stable Audio Lab") as demo:
        demo.load(None, js=AUTO_SCROLL_JS)
        gr.Markdown("# Stable Audio Lab")
        gr.Markdown(f"*{get_device_info()}*")

        with gr.Tabs():
            for model_name, placeholder, neg_placeholder, neg_info in TAB_CONFIG:
                with gr.Tab(model_name):
                    build_model_tab(
                        model_name,
                        placeholder,
                        neg_placeholder,
                        neg_info,
                    )

    return demo


def main():
    parser = argparse.ArgumentParser(description="Launch Stable Audio Lab Web UI")
    parser.add_argument("--port", "-p", type=int, default=7860)
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--share", action="store_true")
    args = parser.parse_args()

    demo = create_app()
    demo.launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
        inbrowser=True,
        css=CUSTOM_CSS,
    )


if __name__ == "__main__":
    main()
