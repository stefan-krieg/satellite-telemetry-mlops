"""Shared telemetry constants and lightweight reading model."""

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Final


@dataclass(frozen=True)
class TelemetryConfig:
    """Immutable defaults shared by telemetry training and serving."""

    WINDOW_SIZE: int = 50
    MAX_DROPPED_PACKETS: float = 2.5
    CHANNELS: tuple[str, ...] = (
        "CADC0872",
        "CADC0873",
        "CADC0874",
        "CADC0884",
        "CADC0886",
        "CADC0888",
        "CADC0890",
        "CADC0892",
        "CADC0894",
    )
    VALID_SAMPLING_RATES: tuple[int, ...] = (1, 5)


@dataclass(frozen=True)
class TelemetryReading:
    """A normalized telemetry sample."""

    channel: str
    timestamp: float
    value: float
    sampling: int

    def __post_init__(self) -> None:
        if not isinstance(self.channel, str):
            raise TypeError("Telemetry channel must be a string.")
        if self.channel not in CHANNELS:
            raise ValueError(f"Unsupported telemetry channel: {self.channel}")


_DEFAULT_CONFIG: Final = TelemetryConfig()
WINDOW_SIZE: Final[int] = _DEFAULT_CONFIG.WINDOW_SIZE
MAX_DROPPED_PACKETS: Final[float] = _DEFAULT_CONFIG.MAX_DROPPED_PACKETS
CHANNELS: Final[tuple[str, ...]] = _DEFAULT_CONFIG.CHANNELS
VALID_SAMPLING_RATES: Final[tuple[int, ...]] = (
    _DEFAULT_CONFIG.VALID_SAMPLING_RATES
)
CHANNEL_TO_IDX: Final[Mapping[str, int]] = MappingProxyType(
    {channel: index for index, channel in enumerate(CHANNELS)}
)