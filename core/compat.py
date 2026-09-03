"""Platform and hardware compatibility utilities for Stable Audio Lab."""

import ctypes
import sys
import warnings
import torch


def init_platform_compat():
    """Apply platform-specific patches and warning suppressions."""
    # Suppress known upstream PyTorch / weight_norm deprecation notices
    warnings.filterwarnings("ignore", category=FutureWarning, message=".*weight_norm.*")
    warnings.filterwarnings("ignore", message=".*flop counting.*")

    # Windows without Triton: bypass failing Dynamo JIT flex_attention compilation
    # Forces PyTorch native C++ chunked SDPA, eliminating the 100-line Inductor crash
    try:
        import triton
    except ImportError:
        try:
            import stable_audio_3.models.transformer as _sat
            _sat.flex_attention_compiled = None
        except Exception:
            pass

    # Windows asyncio: silence harmless TCP connection reset (WinError 10054)
    # when web browsers finish downloading audio streams or refresh SSE sockets
    if sys.platform == "win32":
        try:
            from asyncio.proactor_events import _ProactorBasePipeTransport
            _orig_call_connection_lost = _ProactorBasePipeTransport._call_connection_lost

            def _patched_call_connection_lost(self, exc):
                try:
                    _orig_call_connection_lost(self, exc)
                except ConnectionResetError:
                    pass

            _ProactorBasePipeTransport._call_connection_lost = _patched_call_connection_lost
        except Exception:
            pass


def get_system_ram_gb() -> float | None:
    """Return total physical system RAM in GB via Windows API, or None if unavailable."""
    if sys.platform != "win32":
        return None
    try:
        class MEMORYSTATUSEX(ctypes.Structure):
            _fields_ = [
                ("dwLength", ctypes.c_ulong),
                ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong),
                ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong),
                ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong),
                ("ullAvailVirtual", ctypes.c_ulonglong),
                ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]
        stat = MEMORYSTATUSEX()
        stat.dwLength = ctypes.sizeof(stat)
        ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat))
        return stat.ullTotalPhys / (1024**3)
    except Exception:
        return None


def get_device_banner() -> str:
    """Format an informative hardware diagnostic string for UI banners."""
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        vram = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        cuda_ver = torch.version.cuda
        ram_info = ""
        total_ram = get_system_ram_gb()
        if total_ram:
            ram_info = f" | **System RAM:** {total_ram:.0f} GB (Shared GPU Memory)"
        return f"🚀 **GPU Accelerated:** {gpu_name} ({vram:.1f} GB VRAM){ram_info} | **PyTorch:** {torch.__version__} | **CUDA:** {cuda_ver}"
    return f"⚠️ **Running on CPU** | **PyTorch:** {torch.__version__}"
