#!/usr/bin/env python3
"""Interactive Web Studio for Stable Audio 3."""

import argparse
import functools
import gc
import queue
import random
import threading
from dotenv import load_dotenv
import gradio as gr
import torch

load_dotenv()

from core import (
    MODELS,
    GenerationConfig,
    StableAudioError,
    generate_audio,
    get_device_info,
    parse_cfg_sweep,
)

AUTO_SCROLL_JS = """
() => {
    const observer = new MutationObserver((mutations) => {
        for (const mutation of mutations) {
            const target = mutation.target;
            const consoleEl = target.nodeType === 1 
                ? target.closest('.status-console') 
                : target.parentElement?.closest('.status-console');
            
            if (consoleEl) {
                const textarea = consoleEl.querySelector('textarea');
                if (textarea) {
                    const isNearBottom = (textarea.scrollHeight - textarea.clientHeight - textarea.scrollTop) <= 60;
                    if (isNearBottom) {
                        textarea.scrollTop = textarea.scrollHeight;
                    }
                }
            }
        }
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
    border-radius: 6px !important;
    background-color: var(--background-fill-secondary) !important;
    margin-top: 0px !important;
    margin-bottom: 8px !important;
    font-size: 0.88em !important;
    min-height: 38px !important;
    height: 38px !important;
    display: flex !important;
    align-items: center !important;
    box-sizing: border-box !important;
    padding: 0 14px !important;
    overflow: hidden !important;
}
.model-info-banner.block {
    padding: 0 14px !important;
    margin-top: 0px !important;
    margin-bottom: 8px !important;
}
.model-info-banner div,
.model-info-banner span,
.model-info-banner p {
    margin: 0 !important;
    padding: 0 !important;
    line-height: normal !important;
    display: block !important;
}
.model-info-banner .prose {
    display: block !important;
}
.model-info-banner code {
    all: unset !important;
    font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace !important;
    font-size: 0.85em !important;
    background-color: var(--background-fill-primary) !important;
    border: 1px solid var(--border-color-primary) !important;
    border-radius: 4px !important;
    padding: 1px 3px !important;
    margin: 0 !important;
    display: inline-block !important;
    vertical-align: 0px !important;
    line-height: 1.3 !important;
}
.prompt-box textarea {
    height: 82px !important;
    min-height: 82px !important;
    max-height: 82px !important;
    line-height: 1.4 !important;
    overflow-y: hidden !important;
    resize: none !important;
}
.neg-prompt-box textarea {
    height: 42px !important;
    min-height: 42px !important;
    max-height: 42px !important;
    resize: none !important;
}
.gradio-slider input[type="number"] {
    width: 62px !important;
    min-width: 62px !important;
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
    height: 296px !important;
    white-space: pre-wrap !important;
    word-break: break-word !important;
    overflow-y: auto !important;
    overflow-x: hidden !important;
    padding-bottom: 8px !important;
}
.single-audio-box,
.single-audio-box .gr-group,
.sweep-box,
.sweep-box .gr-group {
    border: none !important;
    background: transparent !important;
    padding: 0 !important;
    box-shadow: none !important;
}
.single-audio-box .styler,
.sweep-box .styler {
    border: none !important;
    background: transparent !important;
    padding: 0 !important;
}
.single-audio-box .block,
.sweep-box .block {
    border: 1px solid var(--block-border-color) !important;
    border-radius: 4px !important;
    background: var(--block-background-fill) !important;
    box-shadow: none !important;
    height: auto !important;
    min-height: unset !important;
    max-height: unset !important;
}

/* CFG Group - Unified card with seamless styling, balanced padding, and invariant height */
.cfg-group {
    border: 1px solid var(--block-border-color) !important;
    border-radius: var(--block-radius) !important;
    background: var(--block-background-fill) !important;
    padding: 0 !important;
    box-sizing: border-box !important;
    height: 114px !important;
    min-height: 114px !important;
    max-height: 114px !important;
    display: flex !important;
    flex-direction: column !important;
    justify-content: space-between !important;
    overflow: hidden !important;
}
.cfg-group .cfg-group,
.cfg-group .styler,
.cfg-group .form {
    border: none !important;
    border-radius: 0 !important;
    background: transparent !important;
    box-sizing: border-box !important;
    margin: 0 !important;
    padding: 0 !important;
    height: 100% !important;
    display: flex !important;
    flex-direction: column !important;
    justify-content: space-between !important;
}
.cfg-group .block,
.cfg-group .gradio-slider,
.cfg-group .gradio-textbox {
    border: none !important;
    background: transparent !important;
    box-shadow: none !important;
    padding: 8px 10px 6px 10px !important;
    height: 76px !important;
    box-sizing: border-box !important;
    display: flex !important;
    flex-direction: column !important;
    justify-content: center !important;
}
.cfg-group textarea {
    height: 36px !important;
    min-height: 36px !important;
    max-height: 36px !important;
    resize: none !important;
}
.cfg-sweep-toggle,
.cfg-group .cfg-sweep-toggle.block {
    border: none !important;
    border-top: 1px solid var(--block-border-color) !important;
    border-radius: 0 !important;
    background: transparent !important;
    box-shadow: none !important;
    padding: 7px 10px !important;
    margin: 0 !important;
    height: 36px !important;
    box-sizing: border-box !important;
    font-size: 0.85rem !important;
    display: flex !important;
    align-items: center !important;
    justify-content: flex-start !important;
    width: 100% !important;
}
.cfg-sweep-toggle > *,
.cfg-sweep-toggle label,
.cfg-sweep-toggle .checkbox-container {
    justify-content: flex-start !important;
    align-items: center !important;
    margin: 0 !important;
    margin-right: auto !important;
    width: auto !important;
    text-align: left !important;
}

/* Seed Column and Box - Invariant matching height */
.seed-col {
    display: flex !important;
    flex-direction: column !important;
}
.seed-col > .form {
    border: 1px solid var(--block-border-color) !important;
    border-radius: var(--block-radius) !important;
    background: var(--block-background-fill) !important;
    height: 114px !important;
    min-height: 114px !important;
    max-height: 114px !important;
    box-sizing: border-box !important;
    display: flex !important;
    flex-direction: column !important;
    justify-content: center !important;
    padding: 0 !important;
}
.seed-box {
    border: none !important;
    background: transparent !important;
    height: 100% !important;
    box-sizing: border-box !important;
    display: flex !important;
    flex-direction: column !important;
    justify-content: center !important;
    padding: 8px 12px !important;
}
.seed-box > label {
    display: flex !important;
    flex-direction: column !important;
    gap: 4px !important;
}
.form:has(> .metadata-toggle),
.form:has(.metadata-toggle) {
    border: none !important;
    background: transparent !important;
    box-shadow: none !important;
    padding: 0 !important;
    margin: 4px 0 !important;
}
.metadata-toggle,
.metadata-toggle.block {
    border: none !important;
    background: transparent !important;
    box-shadow: none !important;
    padding: 2px 4px !important;
    margin: 0 !important;
}
.action-btn-row {
    align-items: stretch !important;
}
.action-btn-row button {
    height: 100% !important;
    min-height: 48px !important;
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
    text-align: center !important;
    line-height: 1.25 !important;
    font-size: 1rem !important;
    padding: 8px 12px !important;
    box-sizing: border-box !important;
}
.sweep-box {
    display: flex !important;
    flex-direction: column !important;
    gap: 8px !important;
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
    cfg_sweep_text: str,
    is_sweep: bool,
    embed_metadata: bool,
    seed: int,
    progress=gr.Progress(track_tqdm=False),
):
    """Generate audio (single or CFG sweep) with live streaming status, interactive UI sampling bar, and error recovery."""
    q: queue.Queue = queue.Queue()
    logs: list[str] = []
    init_msg = f"Initializing {model_name}..."
    logs.append(init_msg)
    yield None, None, None, None, None, init_msg

    sweep_files: list[str | None] = [None, None, None, None]

    def worker():
        try:
            if not is_sweep:
                config = GenerationConfig(
                    model_name=model_name,
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    duration=duration,
                    steps=steps,
                    cfg_scale=cfg,
                    seed=seed,
                    embed_metadata=embed_metadata,
                )
                config.validate()
                result = generate_audio(
                    config=config,
                    progress=progress,
                    status_callback=lambda msg: q.put(("status", msg)),
                )
                q.put(("done_single", result))
            else:
                cfgs = parse_cfg_sweep(cfg_sweep_text)
                shared_seed = random.randint(0, 2**31 - 1) if (seed is None or int(seed) == -1) else int(seed)
                q.put(("status", f"[sweep] Starting CFG Sweep ({len(cfgs)} variations) | Shared Seed: {shared_seed}"))
                for idx, c_val in enumerate(cfgs):
                    q.put(("status", f"\n[variation {idx + 1}/{len(cfgs)}] CFG: {c_val} (Seed: {shared_seed})"))
                    var_cfg = GenerationConfig(
                        model_name=model_name,
                        prompt=prompt,
                        negative_prompt=negative_prompt,
                        duration=duration,
                        steps=steps,
                        cfg_scale=c_val,
                        seed=shared_seed,
                        embed_metadata=embed_metadata,
                    )
                    var_cfg.validate()
                    res = generate_audio(
                        config=var_cfg,
                        progress=progress,
                        status_callback=lambda msg: q.put(("status", msg)),
                    )
                    q.put(("done_variation", (idx, res)))
                q.put(("done_sweep", (len(cfgs), shared_seed)))
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
            yield None, sweep_files[0], sweep_files[1], sweep_files[2], sweep_files[3], display_text

        elif event_type == "done_single":
            summary_display = f"{chr(10).join(logs)}\n\n✅ {payload.status_message}"
            yield payload.output_path, None, None, None, None, summary_display
            break

        elif event_type == "done_variation":
            idx, res = payload
            if idx < len(sweep_files):
                sweep_files[idx] = res.output_path
            display_text = "\n".join(logs)
            yield None, sweep_files[0], sweep_files[1], sweep_files[2], sweep_files[3], display_text

        elif event_type == "done_sweep":
            total_vars, s_seed = payload
            summary_display = f"{chr(10).join(logs)}\n\n✅ Completed CFG Sweep: {total_vars} variations generated with Seed {s_seed}."
            yield None, sweep_files[0], sweep_files[1], sweep_files[2], sweep_files[3], summary_display
            break

        elif event_type == "error":
            err = payload
            if isinstance(err, StableAudioError):
                err_msg = f"❌ Error: {err}"
                yield None, sweep_files[0], sweep_files[1], sweep_files[2], sweep_files[3], f"{chr(10).join(logs)}\n\n{err_msg}" if logs else err_msg
                raise gr.Error(str(err)) from err
            elif isinstance(err, torch.cuda.OutOfMemoryError):
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                oom_msg = (
                    f"❌ GPU out of memory generating {duration:.0f}s with '{model_name}'. "
                    f"Try reducing duration or switching to a smaller model."
                )
                yield None, sweep_files[0], sweep_files[1], sweep_files[2], sweep_files[3], f"{chr(10).join(logs)}\n\n{oom_msg}" if logs else oom_msg
                raise gr.Error(oom_msg) from err
            else:
                err_msg = f"❌ Generation failed: {err}"
                yield None, sweep_files[0], sweep_files[1], sweep_files[2], sweep_files[3], f"{chr(10).join(logs)}\n\n{err_msg}" if logs else err_msg
                raise gr.Error(err_msg) from err


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
            with gr.Row():
                duration = gr.Slider(
                    minimum=1,
                    maximum=int(spec.max_duration),
                    value=int(spec.default_duration),
                    step=1,
                    label="Duration (seconds)",
                    scale=1,
                )
                steps = gr.Slider(
                    minimum=4,
                    maximum=50,
                    value=8,
                    step=1,
                    label="Sampling Steps",
                    scale=1,
                )
            with gr.Row(equal_height=True):
                with gr.Column(scale=2, min_width=200):
                    with gr.Group(elem_classes=["cfg-group"]):
                        cfg = gr.Slider(
                            minimum=1.0,
                            maximum=15.0,
                            value=1.0,
                            step=0.5,
                            label="CFG",
                            visible=True,
                        )
                        cfg_sweep_input = gr.Textbox(
                            label="CFG Values (comma-separated)",
                            value="1.0, 1.5, 2.0, 3.0",
                            placeholder="1.0, 1.5, 2.0, 3.0",
                            visible=False,
                        )
                        cfg_sweep_toggle = gr.Checkbox(
                            label="🔬 CFG Sweep (4 Variations)",
                            value=False,
                            elem_classes=["cfg-sweep-toggle"],
                        )
                with gr.Column(scale=1, min_width=110, elem_classes=["seed-col"]):
                    seed = gr.Number(
                        value=-1,
                        precision=0,
                        label="Seed",
                        info="Set to -1 for random seed",
                        elem_classes=["seed-box"],
                    )
            embed_metadata_toggle = gr.Checkbox(
                label="🏷️ Embed generation metadata in WAV",
                value=True,
                elem_classes=["metadata-toggle"],
            )
            with gr.Row(equal_height=True, elem_classes=["action-btn-row"]):
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
            with gr.Group(visible=True, elem_classes=["single-audio-box"]) as single_box:
                output_audio = gr.Audio(label="Generated Audio", type="filepath", interactive=False)
            with gr.Group(visible=False, elem_classes=["sweep-box"]) as sweep_box:
                sweep_audio_1 = gr.Audio(label="CFG 1.0", type="filepath", interactive=False)
                sweep_audio_2 = gr.Audio(label="CFG 1.5", type="filepath", interactive=False)
                sweep_audio_3 = gr.Audio(label="CFG 2.0", type="filepath", interactive=False)
                sweep_audio_4 = gr.Audio(label="CFG 3.0", type="filepath", interactive=False)
            status = gr.Textbox(
                label="Generation Status & Telemetry",
                value="Ready to generate.",
                interactive=False,
                lines=8,
                max_lines=16,
                elem_classes=["status-console"],
            )

    def on_sweep_toggle(is_sweep: bool, cfg_text: str):
        cfgs = parse_cfg_sweep(cfg_text)
        c1 = f"CFG {cfgs[0]}" if len(cfgs) > 0 else "CFG 1"
        c2 = f"CFG {cfgs[1]}" if len(cfgs) > 1 else "CFG 2"
        c3 = f"CFG {cfgs[2]}" if len(cfgs) > 2 else "CFG 3"
        c4 = f"CFG {cfgs[3]}" if len(cfgs) > 3 else "CFG 4"
        return (
            gr.update(visible=not is_sweep),
            gr.update(visible=is_sweep),
            gr.update(visible=not is_sweep),
            gr.update(visible=is_sweep),
            gr.update(label=c1),
            gr.update(label=c2),
            gr.update(label=c3),
            gr.update(label=c4),
        )

    cfg_sweep_toggle.change(
        fn=on_sweep_toggle,
        inputs=[cfg_sweep_toggle, cfg_sweep_input],
        outputs=[
            cfg,
            cfg_sweep_input,
            single_box,
            sweep_box,
            sweep_audio_1,
            sweep_audio_2,
            sweep_audio_3,
            sweep_audio_4,
        ],
    )

    def on_cfg_input_change(cfg_text: str):
        cfgs = parse_cfg_sweep(cfg_text)
        c1 = f"CFG {cfgs[0]}" if len(cfgs) > 0 else "CFG 1"
        c2 = f"CFG {cfgs[1]}" if len(cfgs) > 1 else "CFG 2"
        c3 = f"CFG {cfgs[2]}" if len(cfgs) > 2 else "CFG 3"
        c4 = f"CFG {cfgs[3]}" if len(cfgs) > 3 else "CFG 4"
        return (
            gr.update(label=c1),
            gr.update(label=c2),
            gr.update(label=c3),
            gr.update(label=c4),
        )

    cfg_sweep_input.change(
        fn=on_cfg_input_change,
        inputs=[cfg_sweep_input],
        outputs=[
            sweep_audio_1,
            sweep_audio_2,
            sweep_audio_3,
            sweep_audio_4,
        ],
    )

    generate_btn.click(
        fn=functools.partial(generate, model_name),
        inputs=[
            prompt,
            negative_prompt,
            duration,
            steps,
            cfg,
            cfg_sweep_input,
            cfg_sweep_toggle,
            embed_metadata_toggle,
            seed,
        ],
        outputs=[
            output_audio,
            sweep_audio_1,
            sweep_audio_2,
            sweep_audio_3,
            sweep_audio_4,
            status,
        ],
    )

    def reset_all():
        return None, None, None, None, None, "Ready to generate."

    clear_btn.click(
        fn=reset_all,
        inputs=[],
        outputs=[
            output_audio,
            sweep_audio_1,
            sweep_audio_2,
            sweep_audio_3,
            sweep_audio_4,
            status,
        ],
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
