# `shared/` — Domain Configuration & Feature Engineering

## Responsibility
`shared/` is the single source of truth for telemetry domain rules, sequence continuity checks, and window feature extraction. Both `training/` and `serving/` import directly from this package so that offline training and live Redis-backed inference execute identical preprocessing code with zero training-serving skew.

**Dependency Rule:** This package must remain lightweight (standard library + `numpy` only). Never import `pandas`, `mlflow`, `fastapi`, or `redis` here so that both slim serving containers and heavy training environments can install it cleanly.


## Modules & Classes

### `config.py`
* **`TelemetryConfig` (dataclass):** Centralizes immutable pipeline constants shared across training and production:
  * `WINDOW_SIZE: int = 50` - Fixed number of sequential ticks required per model input.
  * `MAX_DROPPED_PACKETS: float = 2.5` - Multiplier defining the maximum allowed gap (`2.5 * sampling`, i.e., `2.5s` for 1 Hz and `12.5s` for 5 Hz) before resetting a sequence.
  * `CHANNELS: tuple[str, ...]` - Canonical sorted list of supported OPSSAT telemetry channels (`CADC0872` through `CADC0894`) used to guarantee deterministic one-hot encoding order.
  * `VALID_SAMPLING_RATES: tuple[int, ...] = (1, 5)` - Allowed polling intervals in seconds.
* **`TelemetryReading` (dataclass):** Lightweight internal data transfer object representing a single normalized sensor tick (`channel: str`, `timestamp: float`, `value: float`, `sampling: int`).

### `features.py`
* **`StreamContinuityChecker`:** Determines whether an incoming `TelemetryReading` belongs to the active continuous stream or starts a new block.
  * `is_continuous(prev: TelemetryReading, curr: TelemetryReading) -> bool`: Returns `False` if `sampling` changes (`1` $\leftrightarrow$ `5`), or `(curr.timestamp - prev.timestamp) > MAX_DROPPED_PACKETS * curr.sampling`. Used offline to assign `run_id` blocks and online to trigger Redis buffer resets. Throws `ValueError` if `channel` changes.
* **`WindowFeatureExtractor`:** Converts a chronological buffer of `1` to `50` `TelemetryReading` objects from a single channel into a fixed-length 1D `numpy.ndarray` feature vector.
  * `pad_window(values: np.ndarray) -> tuple[np.ndarray, int]`: Records `buffer_fill = len(values)` and front-pads buffers shorter than `WINDOW_SIZE` by repeating the oldest reading.
  * `transform(buffer: Sequence[TelemetryReading]) -> np.ndarray`: Computes statistics (`last`, `mean`, `std`, `min`, `max`, `q25`, `q50`, `q75`, `last - mean`) on the observed, unpadded values; computes rate-normalized dynamics (`np.diff(window) / sampling` mean/std/max/last and overall slope) on the padded window; then concatenates `sampling`, `buffer_fill`, and the one-hot encoded `channel` vector.