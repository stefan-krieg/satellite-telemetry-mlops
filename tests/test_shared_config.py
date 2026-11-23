from dataclasses import FrozenInstanceError

import pytest

from satellite_telemetry_mlops.shared.config import (
    CHANNEL_TO_IDX,
    TelemetryConfig,
    TelemetryReading,
)


def test_telemetry_config_is_frozen() -> None:
    config = TelemetryConfig()

    with pytest.raises(FrozenInstanceError):
        config.WINDOW_SIZE = 100


def test_channel_to_idx_is_read_only() -> None:
    with pytest.raises(TypeError):
        CHANNEL_TO_IDX["NEW_CHANNEL"] = 99


def test_telemetry_reading_stores_normalized_values() -> None:
    reading = TelemetryReading(
        channel="CADC0872",
        timestamp=123.5,
        value=-4.25,
        sampling=5,
    )

    assert reading.channel == "CADC0872"
    assert reading.timestamp == 123.5
    assert reading.value == -4.25
    assert reading.sampling == 5


def test_telemetry_reading_rejects_unsupported_channel() -> None:
    with pytest.raises(ValueError, match="Unsupported telemetry channel"):
        TelemetryReading(
            channel="UNKNOWN",
            timestamp=123.5,
            value=-4.25,
            sampling=5,
        )


def test_telemetry_reading_rejects_non_string_channel() -> None:
    with pytest.raises(TypeError, match="channel must be a string"):
        TelemetryReading(
            channel=None,
            timestamp=123.5,
            value=-4.25,
            sampling=5,
        )
