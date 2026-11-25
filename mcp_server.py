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

from core import GenerationConfig, StableAudioError, generate_audio


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
) -> dict[str, Any]:
    """Generate music or sound effects and return the WAV path and metadata.

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


if __name__ == "__main__":
    mcp.run(transport="stdio")
