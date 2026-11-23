"""Public interface shared between offline training and live FastAPI inference."""

from .config import (
    CHANNEL_TO_IDX,
    CHANNELS,
    MAX_DROPPED_PACKETS,
    VALID_SAMPLING_RATES,
    WINDOW_SIZE,
    TelemetryConfig,
    TelemetryReading,
)
from .features import (
    FEATURE_NAMES,
    extract_window_features,
    feature_dict_to_array,
    should_reset_buffer,
)

__all__ = [
    "CHANNELS",
    "CHANNEL_TO_IDX",
    "FEATURE_NAMES",
    "MAX_DROPPED_PACKETS",
    "VALID_SAMPLING_RATES",
    "WINDOW_SIZE",
    "TelemetryConfig",
    "TelemetryReading",
    "extract_window_features",
    "feature_dict_to_array",
    "should_reset_buffer",
]
