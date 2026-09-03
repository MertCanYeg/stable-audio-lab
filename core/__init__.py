"""Stable Audio Lab core package."""

from core.compat import get_device_info, setup_environment
from core.exceptions import (
    GenerationError,
    ModelDownloadError,
    ModelNotFoundError,
    StableAudioError,
)
from core.registry import MODELS, ModelSpec, get_model_spec
from core.storage import (
    CacheStatus,
    check_model_cache,
    download_model,
    ensure_model_ready,
    is_model_cached,
)

# Automatically initialize platform and console settings
setup_environment()

from core.engine import (
    GenerationConfig,
    GenerationResult,
    generate_audio,
    load_model,
    slugify,
)

__all__ = [
    # Compat & Diagnostics
    "setup_environment",
    "get_device_info",
    # Exceptions
    "StableAudioError",
    "ModelNotFoundError",
    "ModelDownloadError",
    "GenerationError",
    # Registry
    "MODELS",
    "ModelSpec",
    "get_model_spec",
    # Storage
    "CacheStatus",
    "check_model_cache",
    "is_model_cached",
    "download_model",
    "ensure_model_ready",
    # Engine
    "GenerationConfig",
    "GenerationResult",
    "load_model",
    "generate_audio",
    "slugify",
]
