"""Shared telemetry stream continuity and window feature extraction."""

from collections.abc import Mapping, Sequence

import numpy as np

from .config import (
    CHANNEL_TO_IDX,
    CHANNELS,
    MAX_DROPPED_PACKETS,
    VALID_SAMPLING_RATES,
    WINDOW_SIZE,
    TelemetryReading,
)

_STATISTIC_FEATURE_NAMES = (
    "last",
    "mean",
    "std",
    "min",
    "max",
    "q25",
    "q50",
    "q75",
    "last_minus_mean",
)
_DYNAMIC_FEATURE_NAMES = (
    "rate_diff_mean",
    "rate_diff_std",
    "rate_diff_max",
    "rate_diff_last",
    "slope",
)
FEATURE_NAMES = (
    *_STATISTIC_FEATURE_NAMES,
    *_DYNAMIC_FEATURE_NAMES,
    "sampling",
    "buffer_fill",
    *(f"channel_{channel}" for channel in CHANNELS),
)


class StreamContinuityChecker:
    """Checks whether two telemetry readings belong to the same stream run."""

    @staticmethod
    def is_continuous(prev: TelemetryReading, curr: TelemetryReading) -> bool:
        """Return whether ``curr`` follows ``prev`` without a stream break."""
        if prev.channel != curr.channel:
            raise ValueError("Readings must belong to the same channel.")
        if prev.sampling != curr.sampling:
            return False
        return (
            curr.timestamp - prev.timestamp
            <= MAX_DROPPED_PACKETS * curr.sampling
        )


class WindowFeatureExtractor:
    """Convert telemetry readings into named fixed-length features."""

    @staticmethod
    def pad_window(values: np.ndarray) -> tuple[np.ndarray, int]:
        """Front-pad values by repeating the oldest sample up to WINDOW_SIZE."""
        values = np.asarray(values)
        if values.ndim != 1:
            raise ValueError("Window values must be a one-dimensional array.")

        buffer_fill = len(values)
        if not 1 <= buffer_fill <= WINDOW_SIZE:
            raise ValueError(
                f"Window values must contain between 1 and {WINDOW_SIZE} samples."
            )

        if buffer_fill == WINDOW_SIZE:
            return values, buffer_fill
        return (
            np.pad(
                values,
                (WINDOW_SIZE - buffer_fill, 0),
                mode="edge",
            ),
            buffer_fill,
        )

    def transform(self, buffer: Sequence[TelemetryReading]) -> dict[str, float]:
        """Extract raw-value statistics and padded-window dynamics by name."""
        if not 1 <= len(buffer) <= WINDOW_SIZE:
            raise ValueError(
                f"Buffer must contain between 1 and {WINDOW_SIZE} readings."
            )

        channel = buffer[0].channel
        if any(reading.channel != channel for reading in buffer):
            raise ValueError("All readings in a buffer must use the same channel.")
        try:
            channel_index = CHANNEL_TO_IDX[channel]
        except KeyError:
            raise ValueError(f"Unsupported telemetry channel: {channel}") from None

        sampling = buffer[0].sampling
        if sampling not in VALID_SAMPLING_RATES:
            raise ValueError(f"Unsupported sampling rate: {sampling}")
        if any(reading.sampling != sampling for reading in buffer):
            raise ValueError("All readings in a buffer must use the same sampling rate.")

        values = np.fromiter(
            (reading.value for reading in buffer),
            dtype=np.float64,
            count=len(buffer),
        )
        window, buffer_fill = self.pad_window(values)

        mean = float(np.mean(values))
        quantiles = np.quantile(values, (0.25, 0.5, 0.75))
        statistics = np.array(
            (
                values[-1],
                mean,
                np.std(values),
                np.min(values),
                np.max(values),
                *quantiles,
                values[-1] - mean,
            ),
            dtype=np.float64,
        )

        rate_differences = np.diff(window) / sampling
        sample_positions = np.arange(WINDOW_SIZE, dtype=np.float64)
        slope = np.polyfit(sample_positions, window, 1)[0] / sampling
        dynamics = np.array(
            (
                np.mean(rate_differences),
                np.std(rate_differences),
                np.max(rate_differences),
                rate_differences[-1],
                slope,
            ),
            dtype=np.float64,
        )

        channel_encoding = np.zeros(len(CHANNELS), dtype=np.float64)
        channel_encoding[channel_index] = 1.0
        feature_values = np.concatenate(
            (statistics, dynamics, (sampling, buffer_fill), channel_encoding)
        )
        return {
            name: float(value)
            for name, value in zip(FEATURE_NAMES, feature_values, strict=True)
        }


def extract_window_features(
    readings: Sequence[TelemetryReading],
) -> dict[str, float]:
    """Extract features for a chronological telemetry buffer."""
    return WindowFeatureExtractor().transform(readings)


def feature_dict_to_array(
    features: Mapping[str, float],
) -> np.ndarray:
    """Convert named features to the canonical numeric feature vector."""
    missing = tuple(name for name in FEATURE_NAMES if name not in features)
    if missing:
        raise ValueError(
            f"Missing telemetry features: {', '.join(missing)}"
        )
    return np.asarray(
        [features[name] for name in FEATURE_NAMES],
        dtype=np.float64,
    )


def should_reset_buffer(
    prev: TelemetryReading,
    curr: TelemetryReading,
) -> bool:
    """Return whether the incoming reading should start a new stream buffer."""
    return not StreamContinuityChecker.is_continuous(prev, curr)
