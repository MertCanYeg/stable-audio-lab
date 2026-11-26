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
import soundfile as sf
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
    height: auto !important;
    max-height: none !important;
    display: flex !important;
    align-items: center !important;
    box-sizing: border-box !important;
    padding: 6px 14px !important;
    overflow: visible !important;
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

/* Variation card - mode selector on top, guidance + seed controls below (native Gradio sizing, no clipping) */
.variation-card {
    border-radius: var(--block-radius) !important;
}
.seed-actions {
    gap: 6px !important;
    flex-wrap: nowrap !important;
}
.seed-actions button {
    min-height: 32px !important;
    font-size: 0.85rem !important;
    padding: 4px 8px !important;
    white-space: nowrap !important;
}
.input-error textarea {
    border: 1px solid #f85149 !important;
    box-shadow: 0 0 0 1px #f85149 !important;
}
.input-error label > span,
.input-error span[data-testid="block-info"] {
    color: #f85149 !important;
}
.option-accordion {
    border-radius: var(--block-radius) !important;
    margin-top: 2px !important;
    margin-bottom: 2px !important;
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
    display: flex !important;
    justify-content: flex-start !important;
    width: 100% !important;
    margin-right: auto !important;
}
.metadata-toggle > *,
.metadata-toggle label,
.metadata-toggle .checkbox-container {
    justify-content: flex-start !important;
    margin-right: auto !important;
}
.action-btn-row {
    display: flex !important;
    align-items: stretch !important;
    gap: 8px !important;
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
    overflow-x: hidden !important;
}
.history-table table {
    width: 100% !important;
}
.history-table td {
    word-break: break-word !important;
    white-space: normal !important;
}
.history-table th {
    white-space: nowrap !important;
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


VARIATION_MODES = ["Single", "CFG Sweep", "Seed Batch"]
VARIATION_MODE_INFO = {
    "Single": "One output using the CFG and Seed below.",
    "CFG Sweep": "2-5 outputs sharing one seed, each with a different CFG value.",
    "Seed Batch": "2-5 outputs sharing one CFG, each with a different seed.",
}
SEED_INFO = {
    "Single": "-1 = random seed",
    "CFG Sweep": "Shared by all CFG variations (-1 = random)",
    "Seed Batch": "Base seed, +1 per variation (-1 = random seeds)",
}
AUDIO_ACCORDION_LABEL = "🎛️ Audio Input (Remix / Inpainting)"
POSTPROC_ACCORDION_LABEL = "🎚️ Output & Post-Processing"
DELETE_BUTTON_LABEL = "🗑️ Delete Selected"
CONFIRM_DELETE_LABEL = "⚠️ Confirm Delete"
HISTORY_PLACEHOLDER = "Select a WAV row below to inspect its settings."
HISTORY_EMPTY_MSG = "No generated WAV files found in outputs/ yet. Generate audio in any model tab to see it here."


def audio_accordion_label(mode: str) -> str:
    """Accordion title that stays informative while collapsed (shows active conditioning mode)."""
    if mode == "Audio-to-Audio (Remix)":
        return f"{AUDIO_ACCORDION_LABEL} · Remix active"
    if mode == "Inpainting":
        return f"{AUDIO_ACCORDION_LABEL} · Inpainting active"
    return AUDIO_ACCORDION_LABEL


def postproc_accordion_label(embed_metadata: bool, normalize: bool, fade_out_seconds: float) -> str:
    """Accordion title summarizing non-default output options while collapsed."""
    parts = []
    if normalize:
        parts.append("Normalize")
    if fade_out_seconds and float(fade_out_seconds) > 0:
        parts.append(f"Fade {float(fade_out_seconds):g}s")
    if not embed_metadata:
        parts.append("Metadata off")
    return f"{POSTPROC_ACCORDION_LABEL} · {' · '.join(parts)}" if parts else POSTPROC_ACCORDION_LABEL


def _read_playhead_js(audio_elem_id: str) -> str:
    """Return a minimal JS snippet reading the current playback time (seconds) from gr.Audio."""
    return f"""(currentVal) => {{
        const root = document.getElementById('{audio_elem_id}');
        if (!root) return currentVal;
        const findAudio = (node) => {{
            if (!node) return null;
            if (node.tagName === 'AUDIO') return node;
            for (const el of node.querySelectorAll('*')) {{
                if (el.tagName === 'AUDIO') return el;
                if (el.shadowRoot) {{
                    const found = findAudio(el.shadowRoot);
                    if (found) return found;
                }}
            }}
            return null;
        }};
        const audio = findAudio(root);
        if (audio && Number.isFinite(audio.currentTime)) {{
            return Math.round(audio.currentTime * 10) / 10;
        }}
        const timeEl = root.querySelector('time');
        if (timeEl) {{
            const parts = timeEl.textContent.trim().split(':').map(Number);
            if (parts.length === 2 && parts.every(Number.isFinite)) {{
                return Math.round((parts[0] * 60 + parts[1]) * 10) / 10;
            }}
        }}
        return currentVal;
    }}"""


def _build_generation_plan(
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
) -> dict:
    """Validate inputs and construct a serializable execution plan for Single, CFG Sweep, or Seed Batch."""
    if not prompt or not prompt.strip():
        raise gr.Error("❌ Prompt cannot be empty. Please enter a prompt.")

    if isinstance(is_sweep, bool):
        mode = "CFG Sweep" if is_sweep else "Single"
    else:
        mode = is_sweep or "Single"

    cfgs: list[float] = [cfg]
    seeds_list: list[int] = []
    labels: list[str] = []

    if mode == "CFG Sweep":
        is_valid, cfgs, err_msg = validate_cfg_sweep(cfg_sweep_text)
        if not is_valid:
            raise gr.Error(f"❌ Invalid CFG sweep input: {err_msg}")
        labels = [f"CFG {c}" for c in cfgs]
    elif mode == "Seed Batch":
        batch_n = max(2, min(5, int(seed_batch_count)))
        if seed is None or int(seed) == -1:
            seeds_list = [random.randint(0, 2**31 - 1) for _ in range(batch_n)]
        else:
            base_s = int(seed)
            seeds_list = [base_s + i for i in range(batch_n)]
        labels = [f"Seed {s}" for s in seeds_list]

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
        if float(inpaint_start) >= float(duration):
            raise gr.Error(f"❌ Inpaint Start ({float(inpaint_start):g}s) must be earlier than Duration ({float(duration):g}s).")
        inpaint_path = source_audio

    def _cfg_dict(cfg_val: float, seed_val: int) -> dict:
        return {
            "model_name": model_name,
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "duration": float(duration),
            "steps": int(steps),
            "cfg_scale": float(cfg_val),
            "seed": int(seed_val),
            "embed_metadata": bool(embed_metadata),
            "init_audio_path": init_path,
            "init_noise_level": float(init_noise_level),
            "inpaint_audio_path": inpaint_path,
            "inpaint_start": float(inpaint_start) if inpaint_path else None,
            "inpaint_end": float(inpaint_end) if inpaint_path else None,
            "normalize_audio": bool(normalize_audio),
            "fade_out_seconds": float(fade_out_seconds),
        }

    if mode == "Single":
        return {
            "mode": "Single",
            "model_name": model_name,
            "configs": [_cfg_dict(cfg, seed if seed is not None else -1)],
            "labels": [],
            "header": "",
            "var_headers": [],
            "summary_msg": "",
            "last_seed": None,
        }

    if mode == "CFG Sweep":
        shared_seed = random.randint(0, 2**31 - 1) if (seed is None or int(seed) == -1) else int(seed)
        return {
            "mode": "CFG Sweep",
            "model_name": model_name,
            "configs": [_cfg_dict(c_val, shared_seed) for c_val in cfgs],
            "labels": labels,
            "header": f"[sweep] Starting CFG Sweep ({len(cfgs)} variations) | Shared Seed: {shared_seed}",
            "var_headers": [f"\n[variation {i + 1}/{len(cfgs)}] CFG: {c_val} (Seed: {shared_seed})" for i, c_val in enumerate(cfgs)],
            "summary_msg": f"Completed CFG Sweep: {len(cfgs)} variations generated with Seed {shared_seed}.",
            "last_seed": shared_seed,
        }

    seeds_str = ", ".join(str(s) for s in seeds_list)
    return {
        "mode": "Seed Batch",
        "model_name": model_name,
        "configs": [_cfg_dict(cfg, s_val) for s_val in seeds_list],
        "labels": labels,
        "header": f"[batch] Starting Seed Batch ({len(seeds_list)} variations) | CFG: {cfg}",
        "var_headers": [f"\n[variation {i + 1}/{len(seeds_list)}] Seed: {s_val} (CFG: {cfg})" for i, s_val in enumerate(seeds_list)],
        "summary_msg": f"Completed Seed Batch: {len(seeds_list)} variations generated with Seeds {seeds_str}.",
        "last_seed": seeds_list[0],
    }


def _format_slot_error(err: Exception, duration: float, model_name: str, logs: list[str]) -> tuple[str, str]:
    """Handle OOM/domain error cleanup and return (toast_error_msg, console_display_text)."""
    if isinstance(err, StableAudioError):
        toast_msg = str(err)
        console_err = f"❌ Error: {err}"
    elif isinstance(err, torch.cuda.OutOfMemoryError):
        unload_model()
        toast_msg = (
            f"❌ GPU out of memory generating {duration:.0f}s with '{model_name}'. "
            f"Model cache cleared; try reducing duration or switching to a smaller model."
        )
        console_err = toast_msg
    elif isinstance(err, MemoryError):
        unload_model()
        toast_msg = (
            f"❌ System RAM exhausted loading or running '{model_name}'. "
            f"Close other memory-heavy apps or switch to a smaller model."
        )
        console_err = toast_msg
    else:
        toast_msg = f"❌ Generation failed: {err}"
        console_err = toast_msg

    full_log = f"{chr(10).join(logs)}\n\n{console_err}" if logs else console_err
    return toast_msg, full_log


_UI_CANCEL_EVENT = threading.Event()
_LIVE_LOGS: dict[str, dict] = {}


def _set_live_logs(key: str | None, lines: list[str], sampling: str | None = None) -> None:
    """Record live status lines for a model tab so gr.Timer can stream them without yielding."""
    if key is None:
        return
    _LIVE_LOGS[key] = {"lines": list(lines), "sampling": sampling}


def _format_live_logs(key: str | None) -> str:
    """Format the latest live log buffer for display in Generation Status & Telemetry."""
    if key is None or key not in _LIVE_LOGS:
        return ""
    entry = _LIVE_LOGS[key]
    text = "\n".join(entry.get("lines", []))
    sampling = entry.get("sampling")
    if sampling:
        return f"{text}\n{sampling}" if text else sampling
    return text


def _run_job_blocking(
    config_dict: dict,
    progress,
    logs: list[str],
    cancel_event: threading.Event,
    header_lines: list[str] | None = None,
    init_msg: str | None = None,
    live_log_key: str | None = None,
) -> tuple[str, object, str]:
    """Run a single GenerationConfig without intermediate yields so gr.Progress never flickers."""
    q: queue.Queue = queue.Queue()

    if header_lines:
        if init_msg and logs == [init_msg]:
            logs.clear()
        logs.extend(header_lines)
        _set_live_logs(live_log_key, logs)

    last_desc = "Starting..."
    in_sampling = False
    active_sampling_bar: str | None = None

    def queue_progress(value, desc=None, unit=None):
        q.put(("progress", (value, desc, unit)))

    def worker():
        try:
            config = GenerationConfig(**config_dict)
            config.validate()
            result = generate_audio(
                config=config,
                progress=queue_progress,
                status_callback=lambda msg: q.put(("status", msg)),
                cancel_event=cancel_event,
            )
            q.put(("done", result))
        except GenerationCancelledError as e:
            q.put(("cancelled", str(e)))
        except Exception as e:
            q.put(("error", e))

    thread = threading.Thread(target=worker, daemon=True)
    thread.start()

    last_activity_time = time.time()
    stage_start_time = time.time()

    while True:
        try:
            event_type, payload = q.get(timeout=0.1)
        except queue.Empty:
            if not thread.is_alive() and q.empty():
                break
            now = time.time()
            if now - last_activity_time >= 2.0 and not in_sampling:
                last_activity_time = now
                elapsed_s = int(now - stage_start_time)
                if progress is not None:
                    progress((0, None), desc=f"{last_desc} ({elapsed_s}s)", unit="")
                if live_log_key is not None and logs:
                    hb_lines = list(logs)
                    hb_lines[-1] = f"{hb_lines[-1]} ({elapsed_s}s)"
                    _set_live_logs(live_log_key, hb_lines, active_sampling_bar)
            continue

        last_activity_time = time.time()

        if event_type == "progress":
            val, desc, unit = payload
            if desc:
                last_desc = desc
            in_sampling = unit == "steps"
            if progress is not None:
                if unit is not None:
                    progress(val, desc=desc, unit=unit)
                else:
                    progress(val, desc=desc)

        elif event_type == "status":
            if init_msg and logs == [init_msg]:
                logs.clear()
            if payload.startswith("[sampling]"):
                active_sampling_bar = payload
            else:
                logs.append(payload)
                active_sampling_bar = None
                stage_start_time = time.time()
                if not in_sampling:
                    last_desc = payload
            _set_live_logs(live_log_key, logs, active_sampling_bar)

        elif event_type == "done":
            return "done", payload, "\n".join(logs)

        elif event_type == "cancelled":
            cancel_msg = f"⏹️ {payload or 'Generation cancelled.'}"
            display_text = f"{chr(10).join(logs)}\n\n{cancel_msg}" if logs else cancel_msg
            return "cancelled", None, display_text

        elif event_type == "error":
            return "error", payload, "\n".join(logs)

    return "cancelled", None, "\n".join(logs)


def _run_job_stream(
    config_dict: dict,
    progress,
    logs: list[str],
    cancel_event: threading.Event,
    header_lines: list[str] | None = None,
    init_msg: str | None = None,
):
    """Run a single GenerationConfig in a worker thread, yielding thread-safe progress and SSE heartbeats."""
    q: queue.Queue = queue.Queue()

    if header_lines:
        if init_msg and logs == [init_msg]:
            logs.clear()
        logs.extend(header_lines)
        yield ("status", None, "\n".join(logs))

    def queue_progress(value, desc=None, unit=None):
        q.put(("progress", (value, desc, unit)))

    def worker():
        try:
            config = GenerationConfig(**config_dict)
            config.validate()
            result = generate_audio(
                config=config,
                progress=queue_progress,
                status_callback=lambda msg: q.put(("status", msg)),
                cancel_event=cancel_event,
            )
            q.put(("done", result))
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
                    yield ("status", None, "\n".join(heartbeat_logs))
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
                if init_msg and logs == [init_msg]:
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
                yield ("status", None, display_text)

            elif event_type == "done":
                yield ("done", payload, "\n".join(logs))
                break

            elif event_type == "cancelled":
                cancel_msg = f"⏹️ {payload or 'Generation cancelled.'}"
                display_text = f"{chr(10).join(logs)}\n\n{cancel_msg}" if logs else cancel_msg
                yield ("cancelled", None, display_text)
                break

            elif event_type == "error":
                yield ("error", payload, "\n".join(logs))
                break
    finally:
        cancel_event.set()


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
    """Generate audio (Single, CFG Sweep, or Seed Batch) in a single streaming generator."""
    plan = _build_generation_plan(
        model_name=model_name,
        prompt=prompt,
        negative_prompt=negative_prompt,
        duration=duration,
        steps=steps,
        cfg=cfg,
        cfg_sweep_text=cfg_sweep_text,
        is_sweep=is_sweep,
        embed_metadata=embed_metadata,
        seed=seed,
        seed_batch_count=seed_batch_count,
        audio_mode=audio_mode,
        source_audio=source_audio,
        init_noise_level=init_noise_level,
        inpaint_start=inpaint_start,
        inpaint_end=inpaint_end,
        normalize_audio=normalize_audio,
        fade_out_seconds=fade_out_seconds,
    )

    mode = plan["mode"]
    is_multi = mode != "Single"
    variation_labels = plan["labels"]
    init_msg = f"Initializing {model_name}..."
    logs: list[str] = [init_msg]

    yield None, gr.update(value=None), gr.update(value=None), gr.update(value=None), gr.update(value=None), gr.update(value=None), gr.update(), init_msg

    sweep_files: list[str | None] = [None, None, None, None, None]

    def _sweep_updates(files: list[str | None]):
        if not is_multi:
            return [gr.update(value=None)] * 5
        updates = []
        for i in range(5):
            if i < len(variation_labels):
                f_path = files[i] if i < len(files) else None
                updates.append(gr.update(value=f_path, visible=True, label=variation_labels[i]))
            else:
                updates.append(gr.update(value=None, visible=False))
        return updates

    cancel_event = threading.Event()
    configs = plan["configs"]
    total = len(configs)

    for idx, cfg_dict in enumerate(configs):
        header_lines: list[str] = []
        if is_multi:
            if idx == 0:
                header_lines.append(plan["header"])
            header_lines.append(plan["var_headers"][idx])

        def _prefixed_progress(val, desc=None, unit=None, _idx=idx):
            if progress is None:
                return
            if is_multi and desc:
                desc = f"[{_idx + 1}/{total} · {variation_labels[_idx]}] {desc}"
            if unit is not None:
                progress(val, desc=desc, unit=unit)
            else:
                progress(val, desc=desc)

        for kind, payload, display_text in _run_job_stream(
            cfg_dict,
            _prefixed_progress,
            logs,
            cancel_event,
            header_lines=header_lines,
            init_msg=init_msg,
        ):
            if kind == "status":
                yield None, *(_sweep_updates(sweep_files)), gr.update(), display_text
            elif kind == "done":
                if not is_multi:
                    summary_display = f"{display_text}\n\n✅ {payload.status_message}"
                    yield payload.output_path, *([gr.update(value=None)] * 5), payload.seed, summary_display
                    return
                sweep_files[idx] = payload.output_path
                if idx == total - 1:
                    summary_display = f"{display_text}\n\n✅ {plan['summary_msg']}"
                    yield None, *(_sweep_updates(sweep_files)), plan["last_seed"], summary_display
                    return
                yield None, *(_sweep_updates(sweep_files)), gr.update(), display_text
            elif kind == "cancelled":
                yield None, *(_sweep_updates(sweep_files)), gr.update(), display_text
                return
            elif kind == "error":
                toast_msg, full_log = _format_slot_error(payload, duration, model_name, logs)
                yield None, *(_sweep_updates(sweep_files)), gr.update(), full_log
                raise gr.Error(toast_msg) from payload


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
            with gr.Group(elem_classes=["variation-card"]):
                variation_mode = gr.Radio(
                    choices=VARIATION_MODES,
                    value="Single",
                    label="Variation Mode",
                    info=VARIATION_MODE_INFO["Single"],
                )
                with gr.Row(equal_height=True):
                    with gr.Column(scale=3, min_width=200):
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
                    with gr.Column(scale=2, min_width=150):
                        seed = gr.Number(
                            value=-1,
                            precision=0,
                            label="Seed",
                            info=SEED_INFO["Single"],
                        )
                        with gr.Row(elem_classes=["seed-actions"]):
                            random_seed_btn = gr.Button("🎲 Random", size="sm", min_width=60)
                            reuse_seed_btn = gr.Button("♻️ Reuse Last", size="sm", min_width=60, interactive=False)
            last_seed_box = gr.Number(value=None, precision=0, visible=False)
            with gr.Accordion(AUDIO_ACCORDION_LABEL, open=False, elem_classes=["option-accordion"]) as audio_accordion:
                audio_mode = gr.Radio(
                    choices=["Off", "Audio-to-Audio (Remix)", "Inpainting"],
                    value="Off",
                    label="Conditioning Mode",
                )
                source_audio = gr.Audio(
                    label="Source Audio",
                    type="filepath",
                    visible=False,
                    elem_id=f"source-audio-{model_name}",
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
                with gr.Group(visible=False) as inpaint_row:
                    with gr.Row():
                        inpaint_start = gr.Slider(
                            minimum=0.0,
                            maximum=float(spec.max_duration),
                            value=0.0,
                            step=0.1,
                            label="Inpaint Start (s)",
                        )
                        inpaint_end = gr.Slider(
                            minimum=0.1,
                            maximum=float(spec.max_duration),
                            value=5.0,
                            step=0.1,
                            label="Inpaint End (s)",
                        )
                    with gr.Row(elem_classes=["seed-actions"]):
                        set_start_btn = gr.Button("Set Start from Playhead", size="sm")
                        set_end_btn = gr.Button("Set End from Playhead", size="sm")
            with gr.Accordion(POSTPROC_ACCORDION_LABEL, open=False, elem_classes=["option-accordion"]) as postproc_accordion:
                embed_metadata_toggle = gr.Checkbox(
                    label="Embed generation metadata in WAV",
                    value=True,
                    elem_classes=["metadata-toggle"],
                )
                with gr.Row():
                    normalize_audio_toggle = gr.Checkbox(
                        label="Peak Normalize (-1 dBFS)",
                        value=False,
                        elem_classes=["metadata-toggle"],
                    )
                    fade_out_slider = gr.Slider(
                        minimum=0.0,
                        maximum=5.0,
                        value=0.0,
                        step=0.1,
                        label="Fade-Out (seconds)",
                    )
            with gr.Row(equal_height=True, elem_classes=["action-btn-row"]):
                generate_btn = gr.Button(
                    f"✨ Generate with {model_name}",
                    variant="primary",
                    size="lg",
                    scale=3,
                    min_width=130,
                )
                stop_btn = gr.Button(
                    "⏹️ Stop",
                    variant="stop",
                    size="lg",
                    scale=1,
                    min_width=76,
                    interactive=False,
                )
                clear_btn = gr.Button(
                    "🗑️ Clear",
                    variant="secondary",
                    size="lg",
                    scale=1,
                    min_width=82,
                )

        with gr.Column(scale=1):
            jobs_state = gr.State(None)
            logs_state = gr.State([])
            log_timer = gr.Timer(value=0.5, active=False)
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
        mode = mode if mode in VARIATION_MODE_INFO else "Single"
        is_sweep = mode == "CFG Sweep"
        is_batch = mode == "Seed Batch"
        if is_sweep:
            valid, sweep_vals, _ = validate_cfg_sweep(cfg_text)
            if not valid or not sweep_vals:
                sweep_vals = parse_cfg_sweep(cfg_text)
            labels = [f"CFG {c}" for c in sweep_vals]
        elif is_batch:
            labels = [f"Variation {i + 1}" for i in range(max(2, min(5, int(batch_count))))]
        else:
            labels = []
        audio_updates = [
            gr.update(visible=True, label=labels[i]) if i < len(labels) else gr.update(visible=False)
            for i in range(5)
        ]
        return (
            gr.update(info=VARIATION_MODE_INFO[mode]),
            gr.update(visible=not is_sweep),  # CFG slider stays visible in Single and Seed Batch
            gr.update(visible=is_sweep),
            gr.update(visible=is_batch),
            gr.update(info=SEED_INFO[mode]),
            gr.update(visible=not (is_sweep or is_batch)),
            gr.update(visible=is_sweep or is_batch),
            *audio_updates,
        )

    variation_mode.change(
        fn=on_variation_mode_change,
        inputs=[variation_mode, cfg_sweep_input, seed_batch_count],
        outputs=[
            variation_mode,
            cfg,
            cfg_sweep_input,
            seed_batch_count,
            seed,
            single_box,
            sweep_box,
            sweep_audio_1,
            sweep_audio_2,
            sweep_audio_3,
            sweep_audio_4,
            sweep_audio_5,
        ],
    )

    random_seed_btn.click(fn=lambda: -1, inputs=[], outputs=[seed])
    reuse_seed_btn.click(
        fn=lambda last: gr.update() if last is None else int(last),
        inputs=[last_seed_box],
        outputs=[seed],
    )
    last_seed_box.change(
        fn=lambda last: gr.update(interactive=last is not None),
        inputs=[last_seed_box],
        outputs=[reuse_seed_btn],
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
        is_remix = mode == "Audio-to-Audio (Remix)"
        is_inpaint = mode == "Inpainting"
        return (
            gr.update(visible=is_remix or is_inpaint),
            gr.update(visible=is_remix),
            gr.update(visible=is_inpaint),
            gr.update(label=audio_accordion_label(mode)),
        )

    audio_mode.change(
        fn=on_audio_mode_change,
        inputs=[audio_mode],
        outputs=[source_audio, init_noise_level, inpaint_row, audio_accordion],
    )

    def on_source_audio_change(path: str | None, current_start: float | None, current_end: float | None):
        """Match Duration and slider ranges from the uploaded source audio."""
        if not path:
            return gr.update(), gr.update(), gr.update()
        try:
            src_seconds = float(sf.info(path).duration)
        except Exception:
            gr.Warning("Could not read this audio file. Please use WAV, FLAC, OGG, or MP3.")
            return gr.update(), gr.update(), gr.update()
        dur_val = int(max(1, min(spec.max_duration, round(src_seconds))))
        max_s = max(0.2, round(src_seconds, 1))
        end_default = min(float(current_end), max_s) if (current_end and float(current_end) > 0) else min(5.0, max_s)
        end_val = round(max(0.1, end_default), 1)
        start_val = round(max(0.0, min(float(current_start or 0.0), max(0.0, end_val - 0.1))), 1)
        if src_seconds > spec.max_duration:
            gr.Info(f"Source is {src_seconds:.1f}s; Duration clamped to the model maximum of {spec.max_duration:.0f}s.")
        else:
            gr.Info(f"Duration set to {dur_val}s to match the source audio ({src_seconds:.1f}s).")
        return (
            dur_val,
            gr.update(maximum=max_s, value=start_val),
            gr.update(maximum=max_s, value=end_val),
        )

    source_audio.change(
        fn=on_source_audio_change,
        inputs=[source_audio, inpaint_start, inpaint_end],
        outputs=[duration, inpaint_start, inpaint_end],
    )

    inpaint_start.release(
        fn=lambda s: gr.update(playback_position=float(s)),
        inputs=[inpaint_start],
        outputs=[source_audio],
        show_progress="hidden",
    )
    inpaint_end.release(
        fn=lambda e: gr.update(playback_position=float(e)),
        inputs=[inpaint_end],
        outputs=[source_audio],
        show_progress="hidden",
    )
    set_start_btn.click(
        fn=None,
        inputs=[inpaint_start],
        outputs=[inpaint_start],
        js=_read_playhead_js(f"source-audio-{model_name}"),
    )
    set_end_btn.click(
        fn=None,
        inputs=[inpaint_end],
        outputs=[inpaint_end],
        js=_read_playhead_js(f"source-audio-{model_name}"),
    )

    def on_postproc_change(embed: bool, normalize: bool, fade: float):
        return gr.update(label=postproc_accordion_label(embed, normalize, fade))

    for postproc_input in (embed_metadata_toggle, normalize_audio_toggle, fade_out_slider):
        postproc_input.change(
            fn=on_postproc_change,
            inputs=[embed_metadata_toggle, normalize_audio_toggle, fade_out_slider],
            outputs=[postproc_accordion],
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

    def prepare_generation(
        prompt_val: str,
        neg_val: str,
        dur_val: float,
        steps_val: int,
        cfg_val: float,
        cfg_sweep_text: str,
        mode_val: str,
        embed_val: bool,
        seed_val: int,
        batch_count_val: int,
        audio_mode_val: str,
        source_audio_val: str | None,
        noise_val: float,
        inpaint_start_val: float,
        inpaint_end_val: float,
        norm_val: bool,
        fade_val: float,
    ):
        plan = _build_generation_plan(
            model_name=model_name,
            prompt=prompt_val,
            negative_prompt=neg_val,
            duration=dur_val,
            steps=steps_val,
            cfg=cfg_val,
            cfg_sweep_text=cfg_sweep_text,
            is_sweep=mode_val,
            embed_metadata=embed_val,
            seed=seed_val,
            seed_batch_count=batch_count_val,
            audio_mode=audio_mode_val,
            source_audio=source_audio_val,
            init_noise_level=noise_val,
            inpaint_start=inpaint_start_val,
            inpaint_end=inpaint_end_val,
            normalize_audio=norm_val,
            fade_out_seconds=fade_val,
        )
        _UI_CANCEL_EVENT.clear()
        init_msg = f"Initializing {model_name}..."
        _set_live_logs(model_name, [init_msg])
        if plan["mode"] == "Single":
            slot_clears = [gr.update(value=None)] * 5
        else:
            labels = plan["labels"]
            slot_clears = [
                gr.update(value=None, visible=True, label=labels[i])
                if i < len(labels)
                else gr.update(value=None, visible=False)
                for i in range(5)
            ]
        return (
            None,
            *slot_clears,
            init_msg,
            plan,
            [init_msg],
            gr.update(interactive=False),
            gr.update(interactive=True),
            gr.Timer(active=True),
        )

    def poll_live_logs(current_status: str):
        text = _format_live_logs(model_name)
        if not text or text == current_status:
            return gr.skip()
        return text

    log_timer.tick(
        fn=poll_live_logs,
        inputs=[status],
        outputs=[status],
        queue=False,
        show_progress="hidden",
    )

    def run_single_slot(
        plan: dict | None,
        current_logs: list[str] | None,
        progress=gr.Progress(track_tqdm=False),
    ):
        if not plan or plan.get("mode") != "Single":
            return (gr.skip(),) * 8

        logs = list(current_logs or [])
        cfg_dict = plan["configs"][0]
        init_msg = f"Initializing {plan['model_name']}..."

        kind, payload, display_text = _run_job_blocking(
            cfg_dict,
            progress,
            logs,
            _UI_CANCEL_EVENT,
            init_msg=init_msg,
            live_log_key=model_name,
        )
        if kind == "done":
            summary_display = f"{display_text}\n\n✅ {payload.status_message}"
            _set_live_logs(model_name, [summary_display])
            return payload.output_path, payload.seed, summary_display, logs, None, gr.update(interactive=True), gr.update(interactive=False), gr.Timer(active=False)
        if kind == "cancelled":
            _set_live_logs(model_name, [display_text])
            return None, gr.skip(), display_text, logs, None, gr.update(interactive=True), gr.update(interactive=False), gr.Timer(active=False)

        toast_msg, full_log = _format_slot_error(payload, cfg_dict["duration"], plan["model_name"], logs)
        _set_live_logs(model_name, [full_log])
        gr.Warning(toast_msg)
        return None, gr.skip(), full_log, logs, None, gr.update(interactive=True), gr.update(interactive=False), gr.Timer(active=False)

    def run_multi_slot(
        slot_idx: int,
        plan: dict | None,
        current_logs: list[str] | None,
        progress=gr.Progress(track_tqdm=False),
    ):
        if not plan or plan.get("mode") == "Single" or slot_idx >= len(plan.get("configs", [])):
            return (gr.skip(),) * 8

        configs = plan["configs"]
        total = len(configs)
        is_last = slot_idx == total - 1
        cfg_dict = configs[slot_idx]
        logs = list(current_logs or [])
        header_lines = []
        if slot_idx == 0:
            header_lines.append(plan["header"])
        header_lines.append(plan["var_headers"][slot_idx])
        init_msg = f"Initializing {plan['model_name']}..."

        kind, payload, display_text = _run_job_blocking(
            cfg_dict,
            progress,
            logs,
            _UI_CANCEL_EVENT,
            header_lines=header_lines,
            init_msg=init_msg,
            live_log_key=model_name,
        )
        if kind == "done":
            if is_last:
                summary_display = f"{display_text}\n\n✅ {plan['summary_msg']}"
                _set_live_logs(model_name, [summary_display])
                return payload.output_path, plan["last_seed"], summary_display, logs, None, gr.update(interactive=True), gr.update(interactive=False), gr.Timer(active=False)
            return payload.output_path, gr.skip(), display_text, logs, gr.skip(), gr.skip(), gr.skip(), gr.skip()
        if kind == "cancelled":
            _set_live_logs(model_name, [display_text])
            return None, gr.skip(), display_text, logs, None, gr.update(interactive=True), gr.update(interactive=False), gr.Timer(active=False)

        toast_msg, full_log = _format_slot_error(payload, cfg_dict["duration"], plan["model_name"], logs)
        _set_live_logs(model_name, [full_log])
        gr.Warning(toast_msg)
        return None, gr.skip(), full_log, logs, None, gr.update(interactive=True), gr.update(interactive=False), gr.Timer(active=False)

    ev_prep = generate_btn.click(
        fn=prepare_generation,
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
            jobs_state,
            logs_state,
            generate_btn,
            stop_btn,
            log_timer,
        ],
        show_progress="hidden",
    )

    ev_single = ev_prep.success(
        fn=run_single_slot,
        inputs=[jobs_state, logs_state],
        outputs=[output_audio, last_seed_box, status, logs_state, jobs_state, generate_btn, stop_btn, log_timer],
        show_progress="full",
        show_progress_on=[output_audio],
    )

    sweep_slots = [sweep_audio_1, sweep_audio_2, sweep_audio_3, sweep_audio_4, sweep_audio_5]
    chain_prev = ev_single
    sweep_events = []
    for idx, slot_comp in enumerate(sweep_slots):
        ev_slot = chain_prev.success(
            fn=functools.partial(run_multi_slot, idx),
            inputs=[jobs_state, logs_state],
            outputs=[slot_comp, last_seed_box, status, logs_state, jobs_state, generate_btn, stop_btn, log_timer],
            show_progress="full",
            show_progress_on=[slot_comp],
        )
        sweep_events.append(ev_slot)
        chain_prev = ev_slot

    def on_stop_click():
        _UI_CANCEL_EVENT.set()
        return gr.update(interactive=True), gr.update(interactive=False), gr.Timer(active=False)

    stop_btn.click(
        fn=on_stop_click,
        inputs=[],
        outputs=[generate_btn, stop_btn, log_timer],
        cancels=[ev_prep, ev_single, *sweep_events],
        show_progress="hidden",
    )

    def reset_all():
        _LIVE_LOGS.pop(model_name, None)
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
        gr.Examples(
            examples=spec.examples,
            inputs=[prompt, duration],
            examples_per_page=100,
            label="Prompt Templates (click any row to load)",
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
        "postproc_accordion": postproc_accordion,
        "audio_accordion": audio_accordion,
        "audio_mode": audio_mode,
        "source_audio": source_audio,
        "init_noise_level": init_noise_level,
        "inpaint_row": inpaint_row,
        "inpaint_start": inpaint_start,
        "inpaint_end": inpaint_end,
    }


def _format_history_rows(records: list[dict]) -> list[list[str]]:
    """Format output records into table rows for the History dataframe."""
    rows = []
    for rec in records:
        prompt_text = rec.get("prompt") or "(no embedded metadata)"
        if len(prompt_text) > 55:
            prompt_text = prompt_text[:52] + "..."
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
                "Unload VRAM",
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
                delete_armed_state = gr.State(False)

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
                        DELETE_BUTTON_LABEL,
                        variant="stop",
                        interactive=False,
                        scale=1,
                    )

                with gr.Row():
                    history_audio = gr.Audio(label="Selected Output Preview", type="filepath", interactive=False)
                    history_meta = gr.Textbox(
                        label="Embedded Generation Metadata",
                        value=HISTORY_PLACEHOLDER,
                        interactive=False,
                        lines=6,
                        max_lines=8,
                        elem_classes=["status-console"],
                    )

                history_table = gr.Dataframe(
                    headers=["Created", "Model", "Duration", "CFG", "Seed", "Prompt", "File"],
                    datatype=["str", "str", "str", "str", "str", "str", "str"],
                    interactive=False,
                    wrap=True,
                    column_widths=["14%", "9%", "7%", "5%", "11%", "30%", "24%"],
                    show_search="search",
                    elem_classes=["history-table"],
                )

                def refresh_history():
                    records = list_recent_outputs()
                    msg = HISTORY_PLACEHOLDER if records else HISTORY_EMPTY_MSG
                    return (
                        _format_history_rows(records),
                        records,
                        None,
                        None,
                        msg,
                        gr.update(interactive=False),
                        gr.update(interactive=False),
                        gr.update(interactive=False, value=DELETE_BUTTON_LABEL),
                        False,
                    )

                def on_select_row(records: list[dict], evt: gr.SelectData):
                    row_idx = evt.index[0] if isinstance(evt.index, (list, tuple)) else evt.index
                    if not records or row_idx is None or row_idx < 0 or row_idx >= len(records):
                        return (
                            None,
                            None,
                            HISTORY_PLACEHOLDER,
                            gr.update(interactive=False),
                            gr.update(interactive=False),
                            gr.update(interactive=False, value=DELETE_BUTTON_LABEL),
                            False,
                        )
                    rec = records[row_idx]
                    has_valid_meta = bool(rec.get("has_metadata") and rec.get("model") in MODELS)
                    return (
                        rec["path"],
                        rec["path"],
                        _format_metadata_summary(rec),
                        gr.update(interactive=has_valid_meta),
                        gr.update(interactive=True),
                        gr.update(interactive=True, value=DELETE_BUTTON_LABEL),
                        False,
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
                            ctrl["postproc_accordion"],
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
                            ctrl["inpaint_start"],
                            ctrl["inpaint_end"],
                        ]
                    )

                def on_load_settings(selected_path: str | None):
                    noop = [gr.update()] * (1 + len(tab_order) * 10)
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
                                    gr.update(label=postproc_accordion_label(True, norm_val, fade_val)),
                                ]
                            )
                        else:
                            updates.extend([gr.update()] * 10)

                    gr.Info(f"Loaded settings into '{target_model}' tab.")
                    return tuple(updates)

                def on_send_to_remix(selected_path: str | None):
                    noop = [gr.update()] * (1 + len(tab_order) * 7)
                    if not selected_path:
                        gr.Warning("Please select an output file first.")
                        return tuple(noop)
                    meta = read_wav_metadata(selected_path)
                    target_model = meta.get("model") if (meta and meta.get("model") in MODELS) else "small-music"

                    start_update = gr.update()
                    end_update = gr.update()
                    try:
                        src_seconds = float(sf.info(selected_path).duration)
                        max_s = max(0.2, round(src_seconds, 1))
                        end_val = round(min(5.0, max_s), 1)
                        start_update = gr.update(maximum=max_s, value=0.0)
                        end_update = gr.update(maximum=max_s, value=end_val)
                    except Exception:
                        pass

                    updates: list = [gr.Tabs(selected=target_model)]
                    for m in tab_order:
                        if m == target_model:
                            updates.extend(
                                [
                                    gr.update(open=True, label=audio_accordion_label("Audio-to-Audio (Remix)")),
                                    "Audio-to-Audio (Remix)",
                                    gr.update(value=selected_path, visible=True),
                                    gr.update(visible=True),
                                    gr.update(visible=False),
                                    start_update,
                                    end_update,
                                ]
                            )
                        else:
                            updates.extend([gr.update()] * 7)

                    gr.Info(f"Sent audio to '{target_model}' Remix / Inpaint input.")
                    return tuple(updates)

                def on_delete_selected(selected_path: str | None, armed: bool):
                    """Two-step delete: first click arms the button, second click deletes permanently."""
                    if not selected_path:
                        gr.Warning("Please select an output file first.")
                        return refresh_history()
                    if not armed:
                        return (
                            *([gr.skip()] * 7),
                            gr.update(value=CONFIRM_DELETE_LABEL),
                            True,
                        )
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
                    delete_armed_state,
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
                        delete_armed_state,
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
                    inputs=[selected_wav_state, delete_armed_state],
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
