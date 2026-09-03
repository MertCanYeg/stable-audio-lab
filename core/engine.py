"""Inference engine, memory lifecycle, and audio export for Stable Audio Lab."""

from __future__ import annotations

import gc
import json
import random
import re
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable, NamedTuple, Optional, Tuple, Union

from safetensors.torch import load_file
import soundfile as sf
import torch
from tqdm import tqdm

from core.exceptions import GenerationError
from core.storage import ensure_model_ready

_MODEL_CACHE: dict = {}
_INFERENCE_LOCK = threading.Lock()


def slugify(text: str, max_len: int = 30) -> str:
    """Sanitize prompt text into a clean filename slug."""
    text = re.sub(r"[^\w\s-]", "", text or "").strip().lower()
    slug = re.sub(r"[-\s]+", "_", text).strip("_")
    slug = slug[:max_len].rstrip("_")
    return slug if slug else "audio"


def _format_bytes(n: int) -> str:
    """Format byte count as human-readable string."""
    if n >= 1024**3:
        return f"{n / 1024**3:.1f} GB"
    if n >= 1024**2:
        return f"{n / 1024**2:.0f} MB"
    return f"{n / 1024:.0f} KB"


def _gpu_mem_summary() -> str:
    """Return current GPU memory usage as a formatted string, or empty if no GPU."""
    if not torch.cuda.is_available():
        return ""
    allocated = torch.cuda.memory_allocated()
    reserved = torch.cuda.memory_reserved()
    total = torch.cuda.get_device_properties(0).total_memory
    return (
        f"VRAM: {_format_bytes(allocated)} allocated, "
        f"{_format_bytes(reserved)} reserved / {_format_bytes(total)} total"
    )


@dataclass
class GenerationConfig:
    """Parameters for audio generation with validation."""

    prompt: str
    model_name: str = "small-music"
    negative_prompt: Optional[str] = None
    duration: float = 15.0
    steps: int = 8
    cfg_scale: float = 1.0
    seed: int = -1
    output_path: Optional[Union[str, Path]] = None

    def validate(self) -> None:
        """Validate generation parameters, raising GenerationError on failure."""
        if not self.prompt or not self.prompt.strip():
            raise GenerationError("Please enter a text prompt.")
        if self.duration <= 0:
            raise GenerationError(f"Duration must be greater than 0 seconds (got {self.duration}).")
        if self.steps < 1:
            raise GenerationError(f"Steps must be at least 1 (got {self.steps}).")
        if self.cfg_scale < 0:
            raise GenerationError(f"CFG scale must be non-negative (got {self.cfg_scale}).")

    def resolve_seed(self) -> int:
        """Resolve random seed if set to -1."""
        return random.randint(0, 2**31 - 1) if (self.seed is None or int(self.seed) == -1) else int(self.seed)


class GenerationResult(NamedTuple):
    """Result of an audio generation run, unpackable as (output_path_str, status_message)."""

    output_path: str
    status_message: str
    duration: float
    elapsed: float
    speed: float
    seed: int

    def __iter__(self):
        # Enables direct unpacking: `out_file, status = generate_audio(...)`
        return iter((self.output_path, self.status_message))


def _emit_log(msg: str, status_cb: Optional[Callable[[str], None]] = None) -> None:
    """Print to stdout and notify status callback with identical message."""
    print(msg)
    if status_cb:
        status_cb(msg)


def load_model(
    model_name: str,
    use_half: bool = True,
    progress=None,
    status_callback: Optional[Callable[[str], None]] = None,
):
    """Load model weights into GPU memory with granular stage-by-stage progress reporting."""
    if model_name in _MODEL_CACHE:
        cached_msg = f"[load] Using cached '{model_name}' ({'fp16' if use_half and torch.cuda.is_available() else 'fp32'})."
        _emit_log(cached_msg, status_callback)
        return _MODEL_CACHE[model_name]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    half = (device == "cuda") and use_half
    precision = "fp16" if half else "fp32"

    # Evict previous model if switching
    if _MODEL_CACHE:
        prev_name = next(iter(_MODEL_CACHE))
        evict_msg = f"[load] Evicting '{prev_name}' from memory to load '{model_name}'..."
        _emit_log(evict_msg, status_callback)
        _MODEL_CACHE.clear()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    ensure_model_ready(model_name)

    from stable_audio_3.factory import create_diffusion_cond_from_config
    from stable_audio_3.loading_utils import copy_state_dict
    from stable_audio_3.model import StableAudioModel, all_models

    model_cfg = all_models[model_name]
    cfg_path, ckpt_path = model_cfg.resolve()
    with open(cfg_path) as f:
        model_config = json.load(f)

    t0 = time.time()

    # Stage 1: Architecture & T5 conditioner
    stage1_msg = f"[load] [1/4] Building '{model_name}' architecture & text encoder..."
    if progress:
        progress(0.0, desc=stage1_msg)
    _emit_log(stage1_msg, status_callback)
    model = create_diffusion_cond_from_config(model_config)

    # Stage 2: Reading safetensors weights from disk
    if progress:
        progress(0.0, desc=f"[load] [2/4] Reading safetensors weights from disk...")
    t_read = time.time()
    sd = load_file(ckpt_path)
    stage2_done = f"[load] [2/4] Read {len(sd)} weights from disk in {time.time()-t_read:.2f}s."
    _emit_log(stage2_done, status_callback)

    # Stage 3: Mapping state dict
    if progress:
        progress(0.0, desc=f"[load] [3/4] Mapping weights into model...")
    t_copy = time.time()
    copy_state_dict(model, sd)
    stage3_done = f"[load] [3/4] Mapped {len(sd)} weights in {time.time()-t_copy:.2f}s."
    _emit_log(stage3_done, status_callback)
    del sd
    gc.collect()

    # Stage 4: Transfer to device
    stage4_start = f"[load] [4/4] Transferring weights to {device.upper()} ({precision})..."
    if progress:
        progress(0.0, desc=stage4_start)
    t_xfer = time.time()
    model.to(device).eval().requires_grad_(False)
    if half:
        model.to(torch.float16)
    stage4_done = f"[load] [4/4] Transferred to {device.upper()} in {time.time()-t_xfer:.2f}s."
    _emit_log(stage4_done, status_callback)

    wrapped = StableAudioModel(model, model_config, device, half)
    wrapped.use_lora = False
    wrapped.lora_names = []

    elapsed = time.time() - t0
    mem_summary = _gpu_mem_summary() if torch.cuda.is_available() else ""
    mem_str = f" ({mem_summary})" if mem_summary else ""
    load_done = f"[load] Loaded '{model_name}' in {elapsed:.2f}s{mem_str}."
    _emit_log(load_done, status_callback)

    _MODEL_CACHE[model_name] = wrapped
    return wrapped


def generate_audio(
    model_name: Optional[str] = None,
    prompt: Optional[str] = None,
    negative_prompt: Optional[str] = None,
    duration: float = 15.0,
    steps: int = 8,
    cfg_scale: float = 1.0,
    seed: int = -1,
    output_path: Optional[Union[str, Path]] = None,
    progress=None,
    step_callback: Optional[Callable[[int, int], None]] = None,
    status_callback: Optional[Callable[[str], None]] = None,
    config: Optional[GenerationConfig] = None,
) -> GenerationResult:
    """Generate audio from text prompt with reproducible seed and 16-bit PCM WAV export.

    Accepts either a GenerationConfig instance or individual keyword arguments.
    Returns a GenerationResult that can be directly unpacked as (out_path_str, status_str).
    """
    if config is not None:
        cfg = config
    else:
        cfg = GenerationConfig(
            model_name=model_name or "small-music",
            prompt=prompt or "",
            negative_prompt=negative_prompt,
            duration=duration,
            steps=steps,
            cfg_scale=cfg_scale,
            seed=seed,
            output_path=output_path,
        )

    cfg.validate()
    prompt_clean = cfg.prompt.strip()
    resolved_seed = cfg.resolve_seed()
    neg = cfg.negative_prompt.strip() if cfg.negative_prompt and cfg.negative_prompt.strip() else None
    total_steps = int(cfg.steps)

    # Terminal: log full generation parameters with single-line whitespace normalization
    prompt_display = " ".join(prompt_clean.split())
    print(f"\n{'=' * 60}")
    _emit_log(
        f"[generate] model={cfg.model_name} duration={cfg.duration:.1f}s steps={total_steps} "
        f"cfg={cfg.cfg_scale} seed={resolved_seed}",
        status_callback,
    )
    _emit_log(f"[generate] prompt=\"{prompt_display}\"", status_callback)
    if neg:
        neg_display = " ".join(neg.split())
        _emit_log(f"[generate] negative=\"{neg_display}\"", status_callback)
    print(f"{'=' * 60}")

    with _INFERENCE_LOCK:
        start_time = time.time()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Stage 1: Granular Model loading
        model = load_model(cfg.model_name, progress=progress, status_callback=status_callback)
        sample_size = model.model_config.get("sample_size", 5292032)
        load_elapsed = time.time() - start_time

        # Stage 2: Text encoding + diffusion sampling
        if progress:
            progress(0.0, desc="Encoding text prompt with T5 conditioning...")
        encode_msg = f"[generate] Encoding prompt and starting diffusion sampling ({total_steps} steps)..."
        _emit_log(encode_msg, status_callback)

        pbar = None
        first_step_time: float | None = None
        diffusion_start: float | None = None
        diffusion_elapsed: float = 0.0
        diffusion_end: float | None = None

        def _on_step(info: dict):
            nonlocal pbar, first_step_time, diffusion_start, diffusion_elapsed, diffusion_end
            now = time.time()
            current = info.get("i", 0) + 1

            # Live terminal progress bar (displayed for both UI and CLI runs)
            if pbar is None:
                pbar = tqdm(total=total_steps, desc=f"Sampling [{cfg.model_name}]", unit="step", ncols=70, leave=True)
            pbar.n = current
            pbar.refresh()
            if current >= total_steps:
                pbar.close()
                diffusion_end = now

            # External caller callback (if provided)
            if step_callback:
                step_callback(current, total_steps)

            # Measure step timing isolated from prior text encoding
            if first_step_time is None:
                first_step_time = now
                speed_str = ""
            else:
                if diffusion_start is None:
                    # Estimate step 1 duration from step 2 duration (homogeneous DiT steps)
                    step_duration = now - first_step_time
                    diffusion_start = first_step_time - step_duration
                elapsed = now - diffusion_start
                speed = current / elapsed if elapsed > 0 else 0.0
                eta = max(0.0, (total_steps - current) / speed) if speed > 0 else 0.0
                speed_str = f"{speed:.1f} st/s (ETA: {eta:.0f}s)"

            if current >= total_steps:
                if diffusion_start is not None:
                    diffusion_elapsed = now - diffusion_start
                elif first_step_time is not None:
                    diffusion_elapsed = max(0.01, now - first_step_time)

            # Live UI status & sampling progress bar callback (compact 12-char bar)
            if status_callback:
                pct = int((current / total_steps) * 100)
                bar_len = 12
                filled = int(bar_len * current / total_steps)
                bar = "█" * filled + "░" * (bar_len - filled)
                status_suffix = f" {speed_str}" if speed_str else ""
                status_callback(
                    f"[sampling] Step {current}/{total_steps} ({pct}%) [{bar}]{status_suffix}"
                )

            # Gradio progress tracking
            if progress:
                step_ratio = current / total_steps
                progress(step_ratio, desc=f"Sampling: step {current}/{total_steps} ({int(step_ratio * 100)}%)")

        audio = model.generate(
            prompt=prompt_clean,
            negative_prompt=neg,
            duration=float(cfg.duration),
            steps=total_steps,
            cfg_scale=float(cfg.cfg_scale),
            seed=resolved_seed,
            sample_size=sample_size,
            chunked_decode=False,
            callback=_on_step,
            disable_tqdm=True,
        )

        if diffusion_elapsed <= 0.0 and diffusion_start is not None and diffusion_end is not None:
            diffusion_elapsed = max(0.01, diffusion_end - diffusion_start)
        elif diffusion_elapsed <= 0.0:
            diffusion_elapsed = max(0.01, time.time() - start_time - load_elapsed)

        sample_speed = total_steps / diffusion_elapsed if diffusion_elapsed > 0 else 0.0
        sample_done_msg = (
            f"[generate] Sampling completed in {diffusion_elapsed:.2f}s "
            f"({sample_speed:.1f} steps/s)"
        )
        print()
        _emit_log(sample_done_msg, status_callback)

        # Stage 3: Decode latents and export WAV
        if progress:
            progress(None, desc="Decoding audio latents and saving WAV...")
        _emit_log("[generate] Decoding audio latents and saving WAV...", status_callback)
        decode_start = time.time()

        if cfg.output_path:
            out_file = Path(cfg.output_path)
            if out_file.suffix.lower() != ".wav":
                out_file = out_file.with_suffix(".wav")
            out_file.parent.mkdir(parents=True, exist_ok=True)
        else:
            out_dir = Path("outputs")
            out_dir.mkdir(parents=True, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            out_file = out_dir / f"{timestamp}_{cfg.model_name}_{slugify(prompt_clean)}.wav"

        sample_rate = model.model.sample_rate
        audio_tensor = audio[0].detach().cpu().clamp(-1.0, 1.0)
        audio_np = audio_tensor.numpy().T
        sf.write(str(out_file), audio_np, sample_rate, subtype="PCM_16")

        # Total decode time: VAE latent decoding inside model.generate + disk write
        if diffusion_end is not None:
            decode_elapsed = time.time() - diffusion_end
        else:
            decode_elapsed = time.time() - decode_start

        file_size = out_file.stat().st_size
        channels = audio_np.shape[1] if audio_np.ndim > 1 else 1

        del audio
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        if progress:
            progress(1.0, desc="Complete")

        total_elapsed = time.time() - start_time
        speed = round(total_steps / total_elapsed, 1) if total_elapsed > 0 else 0.0

        # Synchronized telemetry breakdown
        _emit_log(f"[generate] Decoded and saved in {decode_elapsed:.2f}s", status_callback)
        _emit_log(
            f"[generate] Output: {out_file.name} "
            f"({_format_bytes(file_size)}, {channels}ch, {sample_rate}Hz, PCM_16)",
            status_callback,
        )
        if torch.cuda.is_available():
            _emit_log(f"[generate] {_gpu_mem_summary()}", status_callback)
        _emit_log(
            f"[generate] Total: {total_elapsed:.2f}s "
            f"(load={load_elapsed:.1f}s, sample={diffusion_elapsed:.1f}s, decode={decode_elapsed:.1f}s)",
            status_callback,
        )
        print(f"{'=' * 60}\n")

        # UI status: clean user-facing summary
        status_msg = (
            f"Generated {cfg.duration:.1f}s audio in {total_elapsed:.2f}s ({speed} it/s) | "
            f"Model: {cfg.model_name} | Seed: {resolved_seed} | Saved: {out_file.name}"
        )

        return GenerationResult(
            output_path=str(out_file),
            status_message=status_msg,
            duration=cfg.duration,
            elapsed=round(total_elapsed, 2),
            speed=speed,
            seed=resolved_seed,
        )
