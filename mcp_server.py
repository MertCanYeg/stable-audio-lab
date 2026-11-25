#!/usr/bin/env python3
"""MCP server exposing Stable Audio generation to local agents."""

from __future__ import annotations

import contextlib
import sys
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from mcp.server import MCPServer

load_dotenv()

from core import (
    MODELS,
    GenerationConfig,
    StableAudioError,
    check_model_cache,
    generate_audio,
    get_loaded_model_name,
    list_recent_outputs,
    unload_model,
)


mcp = MCPServer("stable-audio-lab")


@mcp.tool()
def generate_audio_tool(
    prompt: str,
    model: str = "small-music",
    negative_prompt: str | None = None,
    duration: float = 15.0,
    steps: int = 8,
    cfg_scale: float = 1.0,
    seed: int = -1,
    output: str | None = None,
    embed_metadata: bool = True,
    init_audio: str | None = None,
    init_noise_level: float = 0.7,
    inpaint_audio: str | None = None,
    inpaint_start: float | None = None,
    inpaint_end: float | None = None,
    normalize_audio: bool = False,
    fade_out_seconds: float = 0.0,
) -> dict[str, Any]:
    """Generate music or sound effects (text, remix, or inpainting) and return the WAV path and metadata.

    Use `TrackType: SFX,` at the start of prompts for sound effects. The
    available models are `small-music`, `small-sfx`, and `medium`.
    """
    config = GenerationConfig(
        model_name=model,
        prompt=prompt,
        negative_prompt=negative_prompt,
        duration=duration,
        steps=steps,
        cfg_scale=cfg_scale,
        seed=seed,
        output_path=Path(output) if output else None,
        embed_metadata=embed_metadata,
        init_audio_path=Path(init_audio) if init_audio else None,
        init_noise_level=init_noise_level,
        inpaint_audio_path=Path(inpaint_audio) if inpaint_audio else None,
        inpaint_start=inpaint_start,
        inpaint_end=inpaint_end,
        normalize_audio=normalize_audio,
        fade_out_seconds=fade_out_seconds,
    )

    try:
        # MCP stdio uses stdout for protocol messages; preserve engine logs on stderr.
        with contextlib.redirect_stdout(sys.stderr):
            result = generate_audio(config=config)
    except StableAudioError:
        raise

    return {
        "output_path": result.output_path,
        "status_message": result.status_message,
        "duration": result.duration,
        "elapsed": result.elapsed,
        "speed": result.speed,
        "seed": result.seed,
    }


@mcp.tool()
def list_models_tool() -> dict[str, Any]:
    """List available Stable Audio 3 models, local cache status, and currently loaded model."""
    models = []
    for name, spec in MODELS.items():
        cache = check_model_cache(name)
        models.append(
            {
                "name": spec.name,
                "repo_id": spec.repo_id,
                "default_duration": spec.default_duration,
                "max_duration": spec.max_duration,
                "description": spec.description,
                "cache": cache.to_dict(),
            }
        )
    return {
        "loaded_model": get_loaded_model_name(),
        "models": models,
    }


@mcp.tool()
def list_recent_outputs_tool(limit: int = 20) -> list[dict[str, Any]]:
    """List recent generated WAV files and their embedded generation metadata."""
    return list_recent_outputs(limit=max(1, min(100, int(limit))))


@mcp.tool()
def unload_model_tool() -> dict[str, Any]:
    """Evict the currently cached model from memory and reclaim CUDA VRAM."""
    evicted = unload_model()
    return {
        "unloaded_model": evicted,
        "status": f"Unloaded '{evicted}'" if evicted else "No model was loaded; cache cleared.",
    }


if __name__ == "__main__":
    mcp.run(transport="stdio")
