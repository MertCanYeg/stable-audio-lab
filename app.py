#!/usr/bin/env python3
"""Interactive Web Studio for Stable Audio 3."""

import argparse
import functools
import gc
import queue
import random
import threading
import time
from dotenv import load_dotenv
import gradio as gr
import torch

load_dotenv()

from core import (
    MODELS,
    GenerationCancelledError,
    GenerationConfig,
    GenerationError,
    StableAudioError,
    check_model_cache,
    delete_output_file,
    generate_audio,
    get_device_info,
    list_recent_outputs,
    parse_cfg_sweep,
    read_wav_metadata,
    unload_model,
    validate_cfg_sweep,
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
.model-info-banner.block {
    border-radius: 6px !important;
    background-color: var(--background-fill-secondary) !important;
    margin-top: 0px !important;
    margin-bottom: 8px !important;
    font-size: 0.88em !important;
    min-height: 38px !important;
    height: 38px !important;
    max-height: 38px !important;
    display: flex !important;
    align-items: center !important;
    box-sizing: border-box !important;
    padding: 0 14px !important;
    overflow: hidden !important;
}
.model-info-banner .prose,
.model-info-banner [data-testid="markdown-wrapper"],
.model-info-banner span.md,
.model-info-banner div:not(.block) {
    height: auto !important;
    min-height: 0 !important;
    max-height: none !important;
    margin: 0 !important;
    padding: 0 !important;
    display: block !important;
    width: 100% !important;
}
.model-info-banner p {
    margin: 0 !important;
    padding: 0 !important;
    line-height: normal !important;
    display: block !important;
    width: 100% !important;
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
    gap: 0 !important;
    row-gap: 0 !important;
    column-gap: 0 !important;
    height: 112px !important;
    min-height: 112px !important;
    max-height: 112px !important;
    display: flex !important;
    flex-direction: column !important;
    flex-wrap: nowrap !important;
    justify-content: space-between !important;
    overflow: hidden !important;
    overflow-x: hidden !important;
    overflow-y: hidden !important;
}
.cfg-group .block:not(.variation-mode-radio),
.cfg-group .gradio-slider,
.cfg-group .gradio-textbox {
    border: none !important;
    background: transparent !important;
    box-shadow: none !important;
    padding: 8px 10px 6px 10px !important;
    height: 76px !important;
    min-height: 76px !important;
    max-height: 76px !important;
    box-sizing: border-box !important;
    display: flex !important;
    flex-direction: column !important;
    justify-content: center !important;
    overflow: hidden !important;
    width: 100% !important;
}
.cfg-group textarea {
    height: 36px !important;
    min-height: 36px !important;
    max-height: 36px !important;
    resize: none !important;
    font-size: 0.85rem !important;
    padding: 6px 10px !important;
    box-sizing: border-box !important;
}
.input-error textarea {
    border: 1px solid #f85149 !important;
    box-shadow: 0 0 0 1px #f85149 !important;
}
.input-error label > span,
.input-error span[data-testid="block-info"] {
    color: #f85149 !important;
}
.variation-mode-radio,
.cfg-group .variation-mode-radio,
.cfg-group .variation-mode-radio.block {
    border: none !important;
    border-top: 1px solid var(--block-border-color) !important;
    border-radius: 0 !important;
    background: transparent !important;
    box-shadow: none !important;
    padding: 4px 10px !important;
    margin: 0 !important;
    height: 36px !important;
    min-height: 36px !important;
    max-height: 36px !important;
    box-sizing: border-box !important;
    display: flex !important;
    align-items: center !important;
    justify-content: flex-start !important;
    width: 100% !important;
    overflow: hidden !important;
}
.variation-mode-radio .wrap {
    display: flex !important;
    flex-direction: row !important;
    flex-wrap: nowrap !important;
    align-items: center !important;
    gap: 8px !important;
    width: 100% !important;
    margin: 0 !important;
    padding: 0 !important;
}
.variation-mode-radio label {
    display: inline-flex !important;
    align-items: center !important;
    gap: 4px !important;
    white-space: nowrap !important;
    font-size: 0.78rem !important;
    padding: 2px 4px !important;
    margin: 0 !important;
    border: none !important;
    background: transparent !important;
    box-shadow: none !important;
}
.option-accordion {
    border-radius: var(--block-radius) !important;
    margin-top: 2px !important;
    margin-bottom: 2px !important;
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
    padding: 10px 10px !important;
    overflow: hidden !important;
}
.seed-box > label {
    display: flex !important;
    flex-direction: column !important;
    justify-content: center !important;
    gap: 0 !important;
    height: 100% !important;
    margin: 0 !important;
    padding: 0 !important;
}
.seed-box span.has-info {
    margin: 0 0 2px 0 !important;
    line-height: 1.2 !important;
    font-size: 0.85rem !important;
}
.seed-box .info-text {
    font-size: 0.75rem !important;
    line-height: 1.2 !important;
    margin: 0 0 4px 0 !important;
}
.seed-box input {
    height: 36px !important;
    min-height: 36px !important;
    max-height: 36px !important;
    padding: 6px 10px !important;
    box-sizing: border-box !important;
    font-size: 0.85rem !important;
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
    display: flex !important;
    align-items: stretch !important;
}
.action-btn-row button {
    height: auto !important;
    min-height: 48px !important;
    align-self: stretch !important;
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
.header-row {
    display: flex !important;
    align-items: center !important;
    justify-content: space-between !important;
    gap: 12px !important;
    margin-bottom: 6px !important;
}
.header-row button {
    white-space: nowrap !important;
}
.history-table {
    max-height: 360px !important;
    overflow-y: auto !important;
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
    is_sweep: bool | str,
    embed_metadata: bool,
    seed: int,
    seed_batch_count: int = 2,
    audio_mode: str = "Off",
    source_audio: str | None = None,
    init_noise_level: float = 0.7,
    inpaint_start: float = 0.0,
    inpaint_end: float = 5.0,
    normalize_audio: bool = False,
    fade_out_seconds: float = 0.0,
    progress=gr.Progress(track_tqdm=False),
):
    """Generate audio (Single, CFG Sweep, or Seed Batch) with optional audio conditioning and post-processing."""
    # 1. Strict generate-time parameter validation BEFORE any yield so existing outputs are preserved on failure
    if not prompt or not prompt.strip():
        raise gr.Error("❌ Prompt cannot be empty. Please enter a prompt.")

    if isinstance(is_sweep, bool):
        mode = "CFG Sweep" if is_sweep else "Single"
    else:
        mode = is_sweep or "Single"

    cfgs: list[float] = [cfg]
    seeds_list: list[int] = []
    variation_labels: list[str] = []

    if mode == "CFG Sweep":
        is_valid, cfgs, err_msg = validate_cfg_sweep(cfg_sweep_text)
        if not is_valid:
            raise gr.Error(f"❌ Invalid CFG sweep input: {err_msg}")
        variation_labels = [f"CFG {c}" for c in cfgs]
    elif mode == "Seed Batch":
        batch_n = max(2, min(5, int(seed_batch_count)))
        if seed is None or int(seed) == -1:
            seeds_list = [random.randint(0, 2**31 - 1) for _ in range(batch_n)]
        else:
            base_s = int(seed)
            seeds_list = [base_s + i for i in range(batch_n)]
        variation_labels = [f"Seed {s}" for s in seeds_list]

    init_path: str | None = None
    inpaint_path: str | None = None
    if audio_mode == "Audio-to-Audio (Remix)":
        if not source_audio:
            raise gr.Error("❌ Please upload or select a Source Audio file for Remix mode.")
        init_path = source_audio
    elif audio_mode == "Inpainting":
        if not source_audio:
            raise gr.Error("❌ Please upload or select a Source Audio file for Inpainting mode.")
        if inpaint_start is None or inpaint_end is None or float(inpaint_start) < 0 or float(inpaint_end) <= float(inpaint_start):
            raise gr.Error("❌ Inpainting requires valid start/end times with End > Start >= 0.")
        inpaint_path = source_audio

    is_multi = mode in ("CFG Sweep", "Seed Batch")

    q: queue.Queue = queue.Queue()
    logs: list[str] = []
    init_msg = f"Initializing {model_name}..."
    logs.append(init_msg)

    # 2. Validation passed -> immediately clear outputs to signify start of a fresh generation
    yield None, gr.update(value=None), gr.update(value=None), gr.update(value=None), gr.update(value=None), gr.update(value=None), init_msg

    sweep_files: list[str | None] = [None, None, None, None, None]

    def _sweep_updates(files: list[str | None], labels_list: list[str], multi_active: bool):
        if not multi_active:
            return [gr.update(value=None)] * 5
        updates = []
        for i in range(5):
            if i < len(labels_list):
                f_path = files[i] if i < len(files) else None
                updates.append(gr.update(value=f_path, visible=True, label=labels_list[i]))
            else:
                updates.append(gr.update(value=None, visible=False))
        return updates

    def queue_progress(value, desc=None, unit=None):
        q.put(("progress", (value, desc, unit)))

    cancel_event = threading.Event()

    def _make_config(cfg_val: float, seed_val: int) -> GenerationConfig:
        return GenerationConfig(
            model_name=model_name,
            prompt=prompt,
            negative_prompt=negative_prompt,
            duration=duration,
            steps=steps,
            cfg_scale=cfg_val,
            seed=seed_val,
            embed_metadata=embed_metadata,
            init_audio_path=init_path,
            init_noise_level=float(init_noise_level),
            inpaint_audio_path=inpaint_path,
            inpaint_start=float(inpaint_start) if inpaint_path else None,
            inpaint_end=float(inpaint_end) if inpaint_path else None,
            normalize_audio=bool(normalize_audio),
            fade_out_seconds=float(fade_out_seconds),
        )

    def worker():
        try:
            if mode == "Single":
                config = _make_config(cfg, seed)
                config.validate()
                result = generate_audio(
                    config=config,
                    progress=queue_progress,
                    status_callback=lambda msg: q.put(("status", msg)),
                    cancel_event=cancel_event,
                )
                q.put(("done_single", result))
            elif mode == "CFG Sweep":
                shared_seed = random.randint(0, 2**31 - 1) if (seed is None or int(seed) == -1) else int(seed)
                q.put(("status", f"[sweep] Starting CFG Sweep ({len(cfgs)} variations) | Shared Seed: {shared_seed}"))
                for idx, c_val in enumerate(cfgs):
                    if cancel_event.is_set():
                        raise GenerationCancelledError("Generation cancelled.")
                    q.put(("status", f"\n[variation {idx + 1}/{len(cfgs)}] CFG: {c_val} (Seed: {shared_seed})"))
                    var_cfg = _make_config(c_val, shared_seed)
                    var_cfg.validate()
                    res = generate_audio(
                        config=var_cfg,
                        progress=queue_progress,
                        status_callback=lambda msg: q.put(("status", msg)),
                        cancel_event=cancel_event,
                    )
                    q.put(("done_variation", (idx, res)))
                q.put(("done_sweep", f"Completed CFG Sweep: {len(cfgs)} variations generated with Seed {shared_seed}."))
            else:
                q.put(("status", f"[batch] Starting Seed Batch ({len(seeds_list)} variations) | CFG: {cfg}"))
                for idx, s_val in enumerate(seeds_list):
                    if cancel_event.is_set():
                        raise GenerationCancelledError("Generation cancelled.")
                    q.put(("status", f"\n[variation {idx + 1}/{len(seeds_list)}] Seed: {s_val} (CFG: {cfg})"))
                    var_cfg = _make_config(cfg, s_val)
                    var_cfg.validate()
                    res = generate_audio(
                        config=var_cfg,
                        progress=queue_progress,
                        status_callback=lambda msg: q.put(("status", msg)),
                        cancel_event=cancel_event,
                    )
                    q.put(("done_variation", (idx, res)))
                q.put(("done_sweep", f"Completed Seed Batch: {len(seeds_list)} variations generated."))
        except GenerationCancelledError as e:
            q.put(("cancelled", str(e)))
        except Exception as e:
            q.put(("error", e))

    thread = threading.Thread(target=worker, daemon=True)
    thread.start()

    active_sampling_bar = None
    last_activity_time = time.time()
    stage_start_time = time.time()

    try:
        while True:
            try:
                event_type, payload = q.get(timeout=0.1)
            except queue.Empty:
                if not thread.is_alive() and q.empty():
                    break
                now = time.time()
                if now - last_activity_time >= 2.0 and active_sampling_bar is None and logs:
                    last_activity_time = now
                    elapsed_s = int(now - stage_start_time)
                    heartbeat_logs = list(logs)
                    heartbeat_logs[-1] = f"{heartbeat_logs[-1]} ({elapsed_s}s)"
                    yield None, *(_sweep_updates(sweep_files, variation_labels, is_multi)), "\n".join(heartbeat_logs)
                continue

            last_activity_time = time.time()

            if event_type == "progress":
                val, desc, unit = payload
                if progress is not None:
                    if unit is not None:
                        progress(val, desc=desc, unit=unit)
                    else:
                        progress(val, desc=desc)

            elif event_type == "status":
                if logs == [init_msg]:
                    logs.clear()

                if payload.startswith("[sampling]"):
                    active_sampling_bar = payload
                else:
                    logs.append(payload)
                    active_sampling_bar = None
                    stage_start_time = time.time()

                display_text = "\n".join(logs)
                if active_sampling_bar:
                    display_text = f"{display_text}\n{active_sampling_bar}" if display_text else active_sampling_bar
                yield None, *(_sweep_updates(sweep_files, variation_labels, is_multi)), display_text

            elif event_type == "done_single":
                summary_display = f"{chr(10).join(logs)}\n\n✅ {payload.status_message}"
                yield payload.output_path, *([gr.update(value=None)] * 5), summary_display
                break

            elif event_type == "done_variation":
                idx, res = payload
                if idx < len(sweep_files):
                    sweep_files[idx] = res.output_path
                display_text = "\n".join(logs)
                yield None, *(_sweep_updates(sweep_files, variation_labels, is_multi)), display_text

            elif event_type == "done_sweep":
                summary_display = f"{chr(10).join(logs)}\n\n✅ {payload}"
                yield None, *(_sweep_updates(sweep_files, variation_labels, is_multi)), summary_display
                break

            elif event_type == "cancelled":
                cancel_msg = f"⏹️ {payload or 'Generation cancelled.'}"
                display_text = f"{chr(10).join(logs)}\n\n{cancel_msg}" if logs else cancel_msg
                yield None, *(_sweep_updates(sweep_files, variation_labels, is_multi)), display_text
                break

            elif event_type == "error":
                err = payload
                if isinstance(err, StableAudioError):
                    err_msg = f"❌ Error: {err}"
                    yield None, *(_sweep_updates(sweep_files, variation_labels, is_multi)), f"{chr(10).join(logs)}\n\n{err_msg}" if logs else err_msg
                    raise gr.Error(str(err)) from err
                elif isinstance(err, torch.cuda.OutOfMemoryError):
                    unload_model()
                    oom_msg = (
                        f"❌ GPU out of memory generating {duration:.0f}s with '{model_name}'. "
                        f"Model cache cleared; try reducing duration or switching to a smaller model."
                    )
                    yield None, *(_sweep_updates(sweep_files, variation_labels, is_multi)), f"{chr(10).join(logs)}\n\n{oom_msg}" if logs else oom_msg
                    raise gr.Error(oom_msg) from err
                elif isinstance(err, MemoryError):
                    unload_model()
                    ram_msg = (
                        f"❌ System RAM exhausted loading or running '{model_name}'. "
                        f"Close other memory-heavy apps or switch to a smaller model."
                    )
                    yield None, *(_sweep_updates(sweep_files, variation_labels, is_multi)), f"{chr(10).join(logs)}\n\n{ram_msg}" if logs else ram_msg
                    raise gr.Error(ram_msg) from err
                else:
                    err_msg = f"❌ Generation failed: {err}"
                    yield None, *(_sweep_updates(sweep_files, variation_labels, is_multi)), f"{chr(10).join(logs)}\n\n{err_msg}" if logs else err_msg
                    raise gr.Error(err_msg) from err
    finally:
        cancel_event.set()


def build_model_tab(
    model_name: str,
    placeholder: str,
    neg_placeholder: str,
    neg_info: str,
):
    """Build UI layout and bindings for a specific model tab."""
    spec = MODELS[model_name]
    cache_st = check_model_cache(model_name)

    banner = (
        f"**{spec.name}** | Repo: `{spec.repo_id}` | Max Duration: {spec.max_duration:.0f}s | "
        f"Audio: 44.1 kHz Stereo | Cache: {cache_st.status_text}"
    )
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
                            elem_classes=["cfg-sweep-input"],
                            visible=False,
                        )
                        seed_batch_count = gr.Slider(
                            minimum=2,
                            maximum=5,
                            value=3,
                            step=1,
                            label="Seed Variations",
                            visible=False,
                        )
                        variation_mode = gr.Radio(
                            choices=["Single", "CFG Sweep", "Seed Batch"],
                            value="Single",
                            show_label=False,
                            container=False,
                            elem_classes=["variation-mode-radio"],
                        )
                with gr.Column(scale=1, min_width=110, elem_classes=["seed-col"]):
                    seed = gr.Number(
                        value=-1,
                        precision=0,
                        label="Seed",
                        info="Set to -1 for random seed",
                        elem_classes=["seed-box"],
                    )
            with gr.Accordion("🎛️ Audio Input (Remix / Inpainting)", open=False, elem_classes=["option-accordion"]) as audio_accordion:
                audio_mode = gr.Radio(
                    choices=["Off", "Audio-to-Audio (Remix)", "Inpainting"],
                    value="Off",
                    label="Conditioning Mode",
                )
                source_audio = gr.Audio(
                    label="Source Audio",
                    type="filepath",
                    visible=False,
                )
                init_noise_level = gr.Slider(
                    minimum=0.05,
                    maximum=1.0,
                    value=0.7,
                    step=0.05,
                    label="Remix Strength (Noise Level)",
                    info="Lower values preserve more of the source audio; 1.0 regenerates freely.",
                    visible=False,
                )
                with gr.Row(visible=False) as inpaint_row:
                    inpaint_start = gr.Number(value=0.0, minimum=0.0, label="Inpaint Start (s)")
                    inpaint_end = gr.Number(value=5.0, minimum=0.1, label="Inpaint End (s)")
            with gr.Accordion("🎚️ Output & Post-Processing", open=False, elem_classes=["option-accordion"]):
                embed_metadata_toggle = gr.Checkbox(
                    label="🏷️ Embed generation metadata in WAV",
                    value=True,
                    elem_classes=["metadata-toggle"],
                )
                with gr.Row():
                    normalize_audio_toggle = gr.Checkbox(
                        label="🔊 Peak Normalize (-1 dBFS)",
                        value=False,
                        elem_classes=["metadata-toggle"],
                    )
                    fade_out_slider = gr.Slider(
                        minimum=0.0,
                        maximum=5.0,
                        value=0.0,
                        step=0.1,
                        label="📉 Fade-Out (seconds)",
                    )
            with gr.Row(equal_height=True, elem_classes=["action-btn-row"]):
                generate_btn = gr.Button(
                    f"✨ Generate with {model_name}",
                    variant="primary",
                    size="lg",
                    scale=4,
                )
                stop_btn = gr.Button(
                    "⏹️ Stop",
                    variant="stop",
                    size="lg",
                    scale=1,
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
                sweep_audio_5 = gr.Audio(label="CFG 4.0", type="filepath", interactive=False, visible=False)
            status = gr.Textbox(
                label="Generation Status & Telemetry",
                value="Ready to generate.",
                interactive=False,
                lines=8,
                max_lines=16,
                elem_classes=["status-console"],
            )

    def on_variation_mode_change(mode: str, cfg_text: str, batch_count: int):
        if mode == "CFG Sweep":
            valid, cfgs, _ = validate_cfg_sweep(cfg_text)
            if not valid or not cfgs:
                cfgs = parse_cfg_sweep(cfg_text)
            audio_updates = [
                gr.update(visible=(i < len(cfgs)), label=f"CFG {cfgs[i]}")
                if i < len(cfgs)
                else gr.update(visible=False)
                for i in range(5)
            ]
            return (
                gr.update(visible=False),
                gr.update(visible=True, label="CFG Values (comma-separated)", elem_classes=["cfg-sweep-input"]),
                gr.update(visible=False),
                gr.update(visible=False),
                gr.update(visible=True),
                *audio_updates,
            )
        if mode == "Seed Batch":
            count = max(2, min(5, int(batch_count)))
            audio_updates = [
                gr.update(visible=(i < count), label=f"Variation {i + 1}")
                for i in range(5)
            ]
            return (
                gr.update(visible=False),
                gr.update(visible=False),
                gr.update(visible=True),
                gr.update(visible=False),
                gr.update(visible=True),
                *audio_updates,
            )
        return (
            gr.update(visible=True),
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(visible=True),
            gr.update(visible=False),
            *([gr.update()] * 5),
        )

    variation_mode.change(
        fn=on_variation_mode_change,
        inputs=[variation_mode, cfg_sweep_input, seed_batch_count],
        outputs=[
            cfg,
            cfg_sweep_input,
            seed_batch_count,
            single_box,
            sweep_box,
            sweep_audio_1,
            sweep_audio_2,
            sweep_audio_3,
            sweep_audio_4,
            sweep_audio_5,
        ],
    )

    def on_seed_batch_count_change(mode: str, batch_count: int):
        if mode != "Seed Batch":
            return [gr.update()] * 5
        count = max(2, min(5, int(batch_count)))
        return [
            gr.update(visible=(i < count), label=f"Variation {i + 1}")
            for i in range(5)
        ]

    seed_batch_count.change(
        fn=on_seed_batch_count_change,
        inputs=[variation_mode, seed_batch_count],
        outputs=[
            sweep_audio_1,
            sweep_audio_2,
            sweep_audio_3,
            sweep_audio_4,
            sweep_audio_5,
        ],
    )

    def on_audio_mode_change(mode: str):
        if mode == "Audio-to-Audio (Remix)":
            return gr.update(visible=True), gr.update(visible=True), gr.update(visible=False)
        if mode == "Inpainting":
            return gr.update(visible=True), gr.update(visible=False), gr.update(visible=True)
        return gr.update(visible=False), gr.update(visible=False), gr.update(visible=False)

    audio_mode.change(
        fn=on_audio_mode_change,
        inputs=[audio_mode],
        outputs=[source_audio, init_noise_level, inpaint_row],
    )

    def on_cfg_change(cfg_text: str):
        is_valid, vals, _ = validate_cfg_sweep(cfg_text)
        if is_valid:
            audio_updates = []
            for i in range(5):
                if i < len(vals):
                    audio_updates.append(gr.update(visible=True, label=f"CFG {vals[i]}"))
                else:
                    audio_updates.append(gr.update(visible=False))
            return (
                *audio_updates,
                gr.update(
                    label="CFG Values (comma-separated)",
                    elem_classes=["cfg-sweep-input"],
                ),
            )
        return (
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
        )

    cfg_sweep_input.change(
        fn=on_cfg_change,
        inputs=[cfg_sweep_input],
        outputs=[
            sweep_audio_1,
            sweep_audio_2,
            sweep_audio_3,
            sweep_audio_4,
            sweep_audio_5,
            cfg_sweep_input,
        ],
    )

    def on_cfg_blur(cfg_text: str):
        is_valid, vals, msg = validate_cfg_sweep(cfg_text)
        if not is_valid:
            gr.Warning(f"CFG Sweep: {msg}")
            return (
                gr.update(),
                gr.update(),
                gr.update(),
                gr.update(),
                gr.update(),
                gr.update(
                    label="CFG Values (comma-separated)",
                    elem_classes=["cfg-sweep-input", "input-error"],
                ),
            )
        audio_updates = []
        for i in range(5):
            if i < len(vals):
                audio_updates.append(gr.update(visible=True, label=f"CFG {vals[i]}"))
            else:
                audio_updates.append(gr.update(visible=False))
        return (
            *audio_updates,
            gr.update(
                label="CFG Values (comma-separated)",
                elem_classes=["cfg-sweep-input"],
            ),
        )

    cfg_sweep_input.blur(
        fn=on_cfg_blur,
        inputs=[cfg_sweep_input],
        outputs=[
            sweep_audio_1,
            sweep_audio_2,
            sweep_audio_3,
            sweep_audio_4,
            sweep_audio_5,
            cfg_sweep_input,
        ],
    )

    gen_event = generate_btn.click(
        fn=functools.partial(generate, model_name),
        inputs=[
            prompt,
            negative_prompt,
            duration,
            steps,
            cfg,
            cfg_sweep_input,
            variation_mode,
            embed_metadata_toggle,
            seed,
            seed_batch_count,
            audio_mode,
            source_audio,
            init_noise_level,
            inpaint_start,
            inpaint_end,
            normalize_audio_toggle,
            fade_out_slider,
        ],
        outputs=[
            output_audio,
            sweep_audio_1,
            sweep_audio_2,
            sweep_audio_3,
            sweep_audio_4,
            sweep_audio_5,
            status,
        ],
    )

    stop_btn.click(fn=None, cancels=[gen_event])

    def reset_all():
        return (
            None,
            None,
            None,
            None,
            None,
            None,
            "Ready to generate.",
            gr.update(
                label="CFG Values (comma-separated)",
                elem_classes=["cfg-sweep-input"],
            ),
        )

    clear_btn.click(
        fn=reset_all,
        inputs=[],
        outputs=[
            output_audio,
            sweep_audio_1,
            sweep_audio_2,
            sweep_audio_3,
            sweep_audio_4,
            sweep_audio_5,
            status,
            cfg_sweep_input,
        ],
    )

    if spec.examples:
        gr.Markdown("#### Prompt Templates *(click any row to load)*")
        gr.Examples(
            examples=spec.examples,
            inputs=[prompt, duration],
            examples_per_page=100,
        )

    return {
        "prompt": prompt,
        "negative_prompt": negative_prompt,
        "duration": duration,
        "steps": steps,
        "cfg": cfg,
        "variation_mode": variation_mode,
        "seed": seed,
        "normalize_audio": normalize_audio_toggle,
        "fade_out": fade_out_slider,
        "audio_accordion": audio_accordion,
        "audio_mode": audio_mode,
        "source_audio": source_audio,
        "init_noise_level": init_noise_level,
        "inpaint_row": inpaint_row,
    }


def _format_history_rows(records: list[dict]) -> list[list[str]]:
    """Format output records into table rows for the History dataframe."""
    rows = []
    for rec in records:
        prompt_text = rec.get("prompt") or "(no embedded metadata)"
        if len(prompt_text) > 70:
            prompt_text = prompt_text[:67] + "..."
        dur = f"{rec['duration']:.0f}s" if rec.get("duration") is not None else "-"
        cfg_val = f"{rec['cfg_scale']}" if rec.get("cfg_scale") is not None else "-"
        seed_val = f"{rec['seed']}" if rec.get("seed") is not None else "-"
        rows.append(
            [
                rec["created"],
                rec.get("model") or "-",
                dur,
                cfg_val,
                seed_val,
                prompt_text,
                f"{rec['filename']} ({rec['size_str']})",
            ]
        )
    return rows


def _format_metadata_summary(rec: dict) -> str:
    """Format full record details for the History metadata inspector."""
    if not rec.get("has_metadata"):
        return (
            f"File: {rec['filename']} ({rec['size_str']})\n"
            f"Created: {rec['created']}\n\n"
            "No embedded Stable Audio Lab metadata found in this WAV file."
        )
    lines = [
        f"File: {rec['filename']} ({rec['size_str']})",
        f"Created: {rec['created']}",
        f"Model: {rec.get('model') or '-'} | Duration: {rec.get('duration')}s | "
        f"Steps: {rec.get('steps')} | CFG: {rec.get('cfg_scale')} | Seed: {rec.get('seed')}",
        f"Prompt: {rec.get('prompt') or ''}",
    ]
    if rec.get("negative_prompt"):
        lines.append(f"Negative Prompt: {rec['negative_prompt']}")
    return "\n".join(lines)


def create_app():
    """Create the root Gradio Blocks application."""
    with gr.Blocks(title="Stable Audio Lab") as demo:
        demo.load(None, js=AUTO_SCROLL_JS)
        gr.Markdown("# Stable Audio Lab")
        with gr.Row(elem_classes=["header-row"]):
            gr.Markdown(f"*{get_device_info()}*")
            unload_btn = gr.Button(
                "🧹 Unload Model / Free VRAM",
                variant="secondary",
                size="sm",
                scale=0,
            )

        def on_unload_model():
            try:
                evicted = unload_model()
            except GenerationError as e:
                gr.Warning(str(e))
                return
            if evicted:
                gr.Info(f"Unloaded '{evicted}' and cleared VRAM cache.")
            else:
                gr.Info("No model was loaded; memory cache cleared.")

        unload_btn.click(fn=on_unload_model, inputs=[], outputs=[])

        tab_controls: dict[str, dict] = {}
        with gr.Tabs() as main_tabs:
            for model_name, placeholder, neg_placeholder, neg_info in TAB_CONFIG:
                with gr.Tab(model_name, id=model_name):
                    tab_controls[model_name] = build_model_tab(
                        model_name,
                        placeholder,
                        neg_placeholder,
                        neg_info,
                    )

            with gr.Tab("History", id="history") as history_tab:
                history_state = gr.State([])
                selected_wav_state = gr.State(None)

                with gr.Row(elem_classes=["action-btn-row"]):
                    refresh_history_btn = gr.Button("🔄 Refresh History", variant="secondary", scale=1)
                    load_settings_btn = gr.Button(
                        "📥 Load Settings into Tab",
                        variant="primary",
                        interactive=False,
                        scale=2,
                    )
                    send_remix_btn = gr.Button(
                        "🎛️ Send to Remix / Inpaint",
                        variant="secondary",
                        interactive=False,
                        scale=2,
                    )
                    delete_history_btn = gr.Button(
                        "🗑️ Delete Selected",
                        variant="stop",
                        interactive=False,
                        scale=1,
                    )

                with gr.Row():
                    history_audio = gr.Audio(label="Selected Output Preview", type="filepath", interactive=False)
                    history_meta = gr.Textbox(
                        label="Embedded Generation Metadata",
                        value="Select a WAV row below to inspect its settings.",
                        interactive=False,
                        lines=6,
                        max_lines=8,
                        elem_classes=["status-console"],
                    )

                history_table = gr.Dataframe(
                    headers=["Created", "Model", "Duration", "CFG", "Seed", "Prompt", "File"],
                    datatype=["str", "str", "str", "str", "str", "str", "str"],
                    interactive=False,
                    wrap=False,
                    elem_classes=["history-table"],
                )

                def refresh_history():
                    records = list_recent_outputs()
                    return (
                        _format_history_rows(records),
                        records,
                        None,
                        None,
                        "Select a WAV row below to inspect its settings.",
                        gr.update(interactive=False),
                        gr.update(interactive=False),
                        gr.update(interactive=False),
                    )

                def on_select_row(records: list[dict], evt: gr.SelectData):
                    row_idx = evt.index[0] if isinstance(evt.index, (list, tuple)) else evt.index
                    if not records or row_idx is None or row_idx < 0 or row_idx >= len(records):
                        return (
                            None,
                            None,
                            "Select a WAV row below to inspect its settings.",
                            gr.update(interactive=False),
                            gr.update(interactive=False),
                            gr.update(interactive=False),
                        )
                    rec = records[row_idx]
                    has_valid_meta = bool(rec.get("has_metadata") and rec.get("model") in MODELS)
                    return (
                        rec["path"],
                        rec["path"],
                        _format_metadata_summary(rec),
                        gr.update(interactive=has_valid_meta),
                        gr.update(interactive=True),
                        gr.update(interactive=True),
                    )

                tab_order = [m for m, _, _, _ in TAB_CONFIG]
                tab_output_components = []
                for m in tab_order:
                    ctrl = tab_controls[m]
                    tab_output_components.extend(
                        [
                            ctrl["prompt"],
                            ctrl["negative_prompt"],
                            ctrl["duration"],
                            ctrl["steps"],
                            ctrl["cfg"],
                            ctrl["variation_mode"],
                            ctrl["seed"],
                            ctrl["normalize_audio"],
                            ctrl["fade_out"],
                        ]
                    )

                remix_output_components = []
                for m in tab_order:
                    ctrl = tab_controls[m]
                    remix_output_components.extend(
                        [
                            ctrl["audio_accordion"],
                            ctrl["audio_mode"],
                            ctrl["source_audio"],
                            ctrl["init_noise_level"],
                            ctrl["inpaint_row"],
                        ]
                    )

                def on_load_settings(selected_path: str | None):
                    noop = [gr.update()] * (1 + len(tab_order) * 9)
                    if not selected_path:
                        gr.Warning("Please select an output file first.")
                        return tuple(noop)
                    meta = read_wav_metadata(selected_path)
                    if not meta:
                        gr.Warning("Selected WAV has no embedded generation metadata.")
                        return tuple(noop)
                    target_model = meta.get("model")
                    if target_model not in MODELS:
                        gr.Warning(f"Unknown model '{target_model}' in WAV metadata.")
                        return tuple(noop)

                    spec = MODELS[target_model]
                    dur_val = max(1, min(int(spec.max_duration), round(float(meta.get("duration", spec.default_duration)))))
                    steps_val = max(4, min(50, int(meta.get("steps", 8))))
                    cfg_val = max(1.0, min(15.0, float(meta.get("cfg_scale", 1.0))))
                    seed_val = int(meta.get("seed", -1))
                    norm_val = bool(meta.get("normalize_audio", False))
                    fade_val = max(0.0, min(5.0, float(meta.get("fade_out_seconds", 0.0))))

                    updates: list = [gr.Tabs(selected=target_model)]
                    for m in tab_order:
                        if m == target_model:
                            updates.extend(
                                [
                                    str(meta.get("prompt", "")),
                                    str(meta.get("negative_prompt", "") or ""),
                                    dur_val,
                                    steps_val,
                                    cfg_val,
                                    "Single",
                                    seed_val,
                                    norm_val,
                                    fade_val,
                                ]
                            )
                        else:
                            updates.extend([gr.update()] * 9)

                    gr.Info(f"Loaded settings into '{target_model}' tab.")
                    return tuple(updates)

                def on_send_to_remix(selected_path: str | None):
                    noop = [gr.update()] * (1 + len(tab_order) * 5)
                    if not selected_path:
                        gr.Warning("Please select an output file first.")
                        return tuple(noop)
                    meta = read_wav_metadata(selected_path)
                    target_model = meta.get("model") if (meta and meta.get("model") in MODELS) else "small-music"

                    updates: list = [gr.Tabs(selected=target_model)]
                    for m in tab_order:
                        if m == target_model:
                            updates.extend(
                                [
                                    gr.Accordion(open=True),
                                    "Audio-to-Audio (Remix)",
                                    gr.update(value=selected_path, visible=True),
                                    gr.update(visible=True),
                                    gr.update(visible=False),
                                ]
                            )
                        else:
                            updates.extend([gr.update()] * 5)

                    gr.Info(f"Sent audio to '{target_model}' Remix / Inpaint input.")
                    return tuple(updates)

                def on_delete_selected(selected_path: str | None):
                    if not selected_path:
                        gr.Warning("Please select an output file first.")
                        return refresh_history()
                    try:
                        deleted = delete_output_file(selected_path)
                    except GenerationError as e:
                        gr.Warning(str(e))
                        return refresh_history()
                    if deleted:
                        gr.Info("Deleted selected output WAV.")
                    else:
                        gr.Warning("Selected file was already removed.")
                    return refresh_history()

                history_outputs = [
                    history_table,
                    history_state,
                    selected_wav_state,
                    history_audio,
                    history_meta,
                    load_settings_btn,
                    send_remix_btn,
                    delete_history_btn,
                ]
                history_tab.select(fn=refresh_history, inputs=[], outputs=history_outputs)
                refresh_history_btn.click(fn=refresh_history, inputs=[], outputs=history_outputs)
                history_table.select(
                    fn=on_select_row,
                    inputs=[history_state],
                    outputs=[
                        selected_wav_state,
                        history_audio,
                        history_meta,
                        load_settings_btn,
                        send_remix_btn,
                        delete_history_btn,
                    ],
                )
                load_settings_btn.click(
                    fn=on_load_settings,
                    inputs=[selected_wav_state],
                    outputs=[main_tabs, *tab_output_components],
                )
                send_remix_btn.click(
                    fn=on_send_to_remix,
                    inputs=[selected_wav_state],
                    outputs=[main_tabs, *remix_output_components],
                )
                delete_history_btn.click(
                    fn=on_delete_selected,
                    inputs=[selected_wav_state],
                    outputs=history_outputs,
                )

    return demo


def main():
    parser = argparse.ArgumentParser(description="Launch Stable Audio Lab Web UI")
    parser.add_argument("--port", "-p", type=int, default=7860)
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--share", action="store_true")
    args = parser.parse_args()

    demo = create_app()
    demo.queue(default_concurrency_limit=1)
    demo.launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
        inbrowser=True,
        css=CUSTOM_CSS,
    )


if __name__ == "__main__":
    main()
