"""Inference engine, VRAM cache management, and audio serialization."""

import os
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterator, Optional, Tuple

import soundfile as sf
import torch

import threading

from core.registry import get_model_spec
from core.storage import download_model, ensure_model_ready

# Global model cache to avoid reloading weights into VRAM repeatedly
_MODEL_CACHE: Dict[str, object] = {}

# Strict hardware inference lock to prevent multiple models from colliding on the GPU
_GPU_INFERENCE_LOCK = threading.Lock()


def slugify(text: str, max_len: int = 30) -> str:
    """Convert prompt into a safe, readable filename slug."""
    text = re.sub(r"[^\w\s-]", "", text).strip().lower()
    slug = re.sub(r"[-\s]+", "_", text)
    return slug[:max_len] if slug else "audio"


def load_model(model_name: str, use_half: bool = True, progress=None):
    """Retrieve model from VRAM cache or load it into memory with self-healing checks."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    half = (device == "cuda") and use_half

    if model_name in _MODEL_CACHE:
        return _MODEL_CACHE[model_name]

    # When loading Medium, evict smaller models from cache to maximize VRAM headroom
    if model_name == "medium" and torch.cuda.is_available():
        print("🧹 Clearing smaller models from VRAM cache to allocate space for Medium...")
        _MODEL_CACHE.clear()
        torch.cuda.empty_cache()

    spec = get_model_spec(model_name)
    approx_size = spec.approx_size.split(" ")[0] if spec.approx_size else "weights"

    # Verify model files are 100% complete and healthy before loading
    if progress:
        progress(0.05, desc=f"Verifying '{model_name}' weights on disk...")
    print(f"📦 Verifying '{model_name}' weights on disk...")
    ensure_model_ready(model_name)

    from stable_audio_3 import StableAudioModel

    if progress:
        progress(0.12, desc=f"Loading '{model_name}' weights into {device.upper()} ({approx_size})...")
    print(f"⚡ Loading '{model_name}' on {device.upper()} (fp16={half}, size={approx_size})...")
    load_start = time.time()
    try:
        model = StableAudioModel.from_pretrained(model_name, device=device, model_half=half)
    except Exception as e:
        err_str = str(e).lower()
        if any(w in err_str for w in ["safetensor", "corrupted", "truncate", "header", "eof"]):
            print(f"⚠️ Checkpoint load failed: {e}. Auto-repairing model...")
            download_model(model_name, force=True)
            model = StableAudioModel.from_pretrained(model_name, device=device, model_half=half)
        elif "401" in str(e) or "gated" in str(e).lower() or "restricted" in str(e).lower():
            raise RuntimeError(
                f"Cannot access model '{model_name}'. Please ensure you have accepted the license terms at "
                f"https://huggingface.co/{spec.repo_id} and configured your HF_TOKEN in .env."
            ) from e
        else:
            raise e

    load_time = time.time() - load_start
    print(f"   ✓ '{model_name}' loaded in {load_time:.2f}s.")
    _MODEL_CACHE[model_name] = model
    return model


def generate_audio_stream(
    model_name: str,
    prompt: str,
    negative_prompt: Optional[str] = None,
    duration: float = 15.0,
    steps: int = 8,
    cfg_scale: float = 1.0,
    seed: int = -1,
    output_path: Optional[str] = None,
    progress=None,
) -> Iterator[Tuple[Optional[str], str]]:
    """Stream real-time stage status and final audio filepath from text prompt.

    Yields:
        (None, stage_markdown) during preparation and sampling,
        (output_filepath_str, completion_markdown) when complete.
    """
    if not prompt or not prompt.strip():
        raise ValueError("Please enter a text prompt.")

    spec = get_model_spec(model_name)
    approx_size = spec.approx_size.split(" ")[0] if spec.approx_size else "weights"

    # Serialize GPU access so concurrent tabs never collide or thrash VRAM
    was_locked = not _GPU_INFERENCE_LOCK.acquire(blocking=False)
    if was_locked:
        if progress:
            progress(0.01, desc="Waiting in queue for active generation to finish...")
        yield None, "⏳ **Waiting in queue for active generation to finish...**"
        print(f"\n⏳ [GPU Queue] '{model_name}' queued — waiting for active generation to finish...")
        _GPU_INFERENCE_LOCK.acquire(blocking=True)

    try:
        print("\n" + "=" * 60)
        print(f"🎵 Starting Audio Generation [{model_name}]")
        print(f"   Prompt   : \"{prompt.strip()}\"")
        if negative_prompt and negative_prompt.strip():
            print(f"   Negative : \"{negative_prompt.strip()}\"")
        print(f"   Duration : {duration:.1f}s | Steps: {steps} | CFG: {cfg_scale} | Seed: {seed}")
        print("=" * 60)

        start_time = time.time()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        if model_name not in _MODEL_CACHE:
            yield None, f"⏳ **Loading '{model_name}' weights into VRAM ({approx_size})...** *(first load takes ~10–15s)*"

        model = load_model(model_name, progress=progress)

        seed_val = int(seed) if seed is not None and seed != -1 else -1
        max_sample_size = model.model_config.get("sample_size", 5292032)

        # Stage 2: Prompt Conditioning
        if progress:
            progress(0.20, desc="Encoding prompt conditioning with T5...")
        yield None, "⏳ **Encoding text prompt conditioning with T5...**"
        print(f"🔤 Encoding prompt conditioning with T5...")

        # Stage 3: Diffusion Sampling
        print(f"🌊 Running diffusion sampling ({steps} steps)...")
        if progress:
            progress(0.28, desc=f"Diffusion sampling: step 1/{steps} (0%)...")
        yield None, f"⏳ **Diffusion sampling: step 1/{steps} (0%)...**"

        def sampling_callback(info: dict):
            step_idx = info.get("i", 0)
            curr_step = step_idx + 1
            pct = 0.28 + 0.64 * (curr_step / steps)
            step_desc = f"Diffusion sampling: step {curr_step}/{steps} ({int((curr_step/steps)*100)}%)"
            print(f"   -> Diffusion step {curr_step}/{steps} completed")
            if progress:
                progress(pct, desc=step_desc)

        try:
            audio = model.generate(
                prompt=prompt.strip(),
                negative_prompt=negative_prompt.strip() if negative_prompt else None,
                duration=float(duration),
                steps=int(steps),
                cfg_scale=float(cfg_scale),
                seed=seed_val,
                sample_size=max_sample_size,
                callback=sampling_callback,
            )
        except torch.cuda.OutOfMemoryError as e:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            raise RuntimeError(
                f"CUDA Out of Memory! Requested {duration:.0f}s on '{model_name}' exceeded dedicated VRAM. "
                f"Tip: Try reducing duration or switch to 'small-music' (~2GB VRAM)."
            ) from e

        # Stage 4: Audio Decoding and Export
        if progress:
            progress(0.95, desc="Decoding audio latents and saving .wav...")
        yield None, "⏳ **Decoding audio latents and saving .wav...**"
        print(f"💾 Decoding audio latents and saving .wav...")

        gen_time = time.time() - start_time
        speed = float(steps) / gen_time if gen_time > 0 else 0

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Determine destination file path
        if output_path:
            out_file = Path(output_path)
            out_file.parent.mkdir(parents=True, exist_ok=True)
        else:
            output_dir = Path("outputs")
            output_dir.mkdir(parents=True, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            slug = slugify(prompt)
            out_file = output_dir / f"{timestamp}_{model_name}_{slug}.wav"

        sample_rate = model.model.sample_rate
        audio_tensor = audio[0].detach().cpu()
        audio_np = audio_tensor.numpy().T
        sf.write(str(out_file), audio_np, sample_rate)

        if progress:
            progress(1.0, desc="✅ Audio ready!")

        status_msg = (
            f"✅ Generated **{duration:.1f}s** in **{gen_time:.2f}s** "
            f"({speed:.1f} steps/s) • Saved to `{out_file.name}`"
        )

        print(f"\n✨ Generation Complete in {gen_time:.2f}s ({speed:.1f} steps/s)!")
        print(f"   File saved to: {out_file.resolve()}")
        print("=" * 60 + "\n")

        yield str(out_file), status_msg
    finally:
        _GPU_INFERENCE_LOCK.release()


def generate_audio(*args, **kwargs) -> Tuple[str, str]:
    """Non-streaming wrapper around generate_audio_stream for CLI and scripts."""
    final_file, final_status = None, ""
    for out_file, status in generate_audio_stream(*args, **kwargs):
        if out_file:
            final_file = out_file
        final_status = status
    return final_file or "", final_status
