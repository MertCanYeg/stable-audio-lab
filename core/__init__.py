"""Stable Audio Lab core package."""

from core.compat import get_device_info, setup_environment
from core.exceptions import (
    GenerationCancelledError,
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
    get_loaded_model_name,
    list_recent_outputs,
    load_model,
    parse_cfg_sweep,
    read_wav_metadata,
    slugify,
    unload_model,
    validate_cfg_sweep,
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
    "GenerationCancelledError",
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
    "get_loaded_model_name",
    "load_model",
    "unload_model",
    "generate_audio",
    "list_recent_outputs",
    "parse_cfg_sweep",
    "read_wav_metadata",
    "slugify",
    "validate_cfg_sweep",
]
