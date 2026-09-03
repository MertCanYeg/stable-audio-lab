"""Platform compatibility and hardware diagnostics for Stable Audio Lab."""

import logging
import os
import sys
import warnings
import torch

_INITIALIZED = False


def setup_environment():
    """Apply environment settings, warning filters, and platform fixes."""
    global _INITIALIZED
    if _INITIALIZED:
        return

    os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    warnings.filterwarnings("ignore", category=FutureWarning, message=".*weight_norm.*")
    warnings.filterwarnings("ignore", message=".*flop counting.*")
    logging.getLogger("torch.utils.flop_counter").setLevel(logging.ERROR)

    try:
        import transformers.utils.logging as _tl
        _tl.disable_progress_bar()
    except Exception:
        pass

    if sys.platform == "win32":
        for stream in (sys.stdout, sys.stderr):
            if stream and hasattr(stream, "reconfigure"):
                try:
                    stream.reconfigure(encoding="utf-8", errors="replace")
                except Exception:
                    pass

    from contextlib import redirect_stderr, redirect_stdout

    with open(os.devnull, "w") as devnull, redirect_stdout(devnull), redirect_stderr(devnull):
        try:
            import triton  # noqa: F401
        except ImportError:
            pass
        try:
            import stable_audio_3.models.transformer as _sat
            _sat.flex_attention_compiled = None
        except Exception:
            pass

    _INITIALIZED = True


def get_device_info() -> str:
    """Return a concise hardware diagnostic summary."""
    if torch.cuda.is_available():
        gpu = torch.cuda.get_device_name(0)
        vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        cuda = torch.version.cuda
        return f"Hardware: {gpu} ({vram_gb:.1f} GB VRAM) | PyTorch: {torch.__version__} (CUDA {cuda})"
    return f"Hardware: CPU Only | PyTorch: {torch.__version__}"
