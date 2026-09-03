"""Domain exceptions for Stable Audio Lab."""


class StableAudioError(Exception):
    """Base exception for all Stable Audio Lab errors."""


class ModelNotFoundError(StableAudioError, ValueError):
    """Raised when an unrecognized model identifier is requested."""


class ModelDownloadError(StableAudioError, RuntimeError):
    """Raised when model weights fail to download from Hugging Face Hub."""


class GenerationError(StableAudioError, ValueError):
    """Raised when audio generation parameters are invalid or inference fails."""
