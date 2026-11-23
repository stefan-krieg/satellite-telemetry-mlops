import numpy as np
import pytest

from satellite_telemetry_mlops.shared import (
    CHANNELS,
    FEATURE_NAMES,
    TelemetryReading,
    WINDOW_SIZE,
    extract_window_features,
    feature_dict_to_array,
    should_reset_buffer,
)
from satellite_telemetry_mlops.shared.features import (
    StreamContinuityChecker,
    WindowFeatureExtractor,
)


def test_stream_continuity_boundaries_and_reset_behavior() -> None:
    previous = TelemetryReading("CADC0872", 0.0, 1.0, 5)
    at_limit = TelemetryReading("CADC0872", 12.5, 1.0, 5)
    beyond_limit = TelemetryReading("CADC0872", 12.6, 1.0, 5)
    different_sampling = TelemetryReading("CADC0872", 1.0, 1.0, 1)

    assert StreamContinuityChecker.is_continuous(previous, at_limit)
    assert not should_reset_buffer(previous, at_limit)
    assert not StreamContinuityChecker.is_continuous(previous, beyond_limit)
    assert should_reset_buffer(previous, beyond_limit)
    assert not StreamContinuityChecker.is_continuous(previous, different_sampling)
    assert should_reset_buffer(previous, different_sampling)

    different_channel = TelemetryReading("CADC0873", 1.0, 1.0, 5)
    with pytest.raises(ValueError, match="same channel"):
        StreamContinuityChecker.is_continuous(previous, different_channel)


def test_window_features_front_pad_and_encode_channel() -> None:
    readings = [
        TelemetryReading("CADC0873", 0.0, 2.0, 1),
        TelemetryReading("CADC0873", 1.0, 4.0, 1),
    ]

    features = extract_window_features(readings)

    assert set(features) == set(FEATURE_NAMES)
    assert len(FEATURE_NAMES) == 9 + 5 + 2 + len(CHANNELS)
    assert features["last"] == 4.0
    assert features["mean"] == 3.0
    assert features["std"] == 1.0
    assert features["q25"] == 2.5
    assert features["q50"] == 3.0
    assert features["q75"] == 3.5
    assert features["last_minus_mean"] == 1.0
    assert features["rate_diff_last"] == 2.0
    positions = np.arange(WINDOW_SIZE, dtype=np.float64)
    padded_window = np.full(WINDOW_SIZE, 2.0)
    padded_window[-2:] = [2.0, 4.0]
    centered_positions = positions - np.mean(positions)
    expected_slope = np.dot(
        centered_positions,
        padded_window - np.mean(padded_window),
    ) / np.dot(centered_positions, centered_positions)
    assert features["slope"] == pytest.approx(expected_slope)
    assert features["sampling"] == 1.0
    assert features["buffer_fill"] == 2.0
    np.testing.assert_array_equal(
        [features[f"channel_{channel}"] for channel in CHANNELS],
        [0, 1, 0, 0, 0, 0, 0, 0, 0],
    )

    feature_array = feature_dict_to_array(features)
    assert feature_array.shape == (len(FEATURE_NAMES),)
    np.testing.assert_array_equal(feature_array[:9], [4, 3, 1, 2, 4, 2.5, 3, 3.5, 1])
    np.testing.assert_array_equal(feature_array[14:16], [1, 2])
    np.testing.assert_array_equal(feature_array[16:], [0, 1, 0, 0, 0, 0, 0, 0, 0])


def test_feature_dict_to_array_rejects_missing_features() -> None:
    with pytest.raises(ValueError, match="Missing telemetry features"):
        feature_dict_to_array({})

    window, buffer_fill = WindowFeatureExtractor.pad_window(np.array([2.0, 4.0]))
    assert buffer_fill == 2
    assert window.shape == (50,)
    np.testing.assert_array_equal(window[:49], np.full(49, 2.0))
    assert window[-1] == 4.0


def test_window_features_reject_mixed_channels() -> None:
    readings = [
        TelemetryReading("CADC0872", 0.0, 1.0, 1),
        TelemetryReading("CADC0873", 1.0, 2.0, 1),
    ]

    with pytest.raises(ValueError, match="same channel"):
        extract_window_features(readings)
