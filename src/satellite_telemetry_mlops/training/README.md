# `training/` - Offline Dataset Preparation, Model Training & Registry

## Responsibility
`training/` handles offline data ingestion from `data/segments.csv`, continuous-block window generation, leakage-free grouped cross-validation and threshold tuning on `train == 1`, final holdout evaluation on `train == 0` across warm-up and steady-state regimes, and publishing validated model artifacts to the MLflow Model Registry.

## Validation & Split Strategy
- **Strict Holdout (`train == 0`):** Kept completely untouched during feature selection, hyperparameter tuning, and threshold optimization. Used exclusively for a single final offline test evaluation and chronological replay in the live FastAPI + Redis simulator.
- **Grouped Cross-Validation (`train == 1`):** Validation folds are carved strictly out of `train == 1` using `StratifiedGroupKFold(n_splits=5)` grouped by `run_id`. Grouping by `run_id` ensures overlapping windows from the same continuous telemetry sequence never leak across training and validation folds.

## Modules & Classes

### `prepare_data.py`
- **`OpssatDatasetBuilder`:** Transforms raw `segments.csv` rows into leakage-free training and test feature matrices.
  - `load_and_sort(csv_path: str) -> pd.DataFrame`: Parses ISO timestamps, sorts rows by `(channel, timestamp)`, and computes per-channel time deltas (`dt`).
  - `assign_continuous_runs(df: pd.DataFrame) -> pd.DataFrame`: Uses `StreamContinuityChecker` rules (`channel`, `train`, or `sampling` changes, or `dt > 10s`) to increment a `run_id` per `channel`.
  - `build_windows(df: pd.DataFrame, stride: int = 1) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]`: Slides a `WINDOW_SIZE = 50` window from tick `0` of every run (using `WindowFeatureExtractor` so edge-padded cold starts are included in training), assigns each window's ground-truth label from its **last data point** (`anomaly[t]`), and returns `(X, y, buffer_fill, groups)` where `groups` stores each window's `run_id` for grouped cross-validation.

### `train.py`
- **`ModelTrainer`:** Cross-validates, tunes decision thresholds, refits, and evaluates anomaly classifiers on the prepared feature matrices.
  - `cross_validate(X_train: np.ndarray, y_train: np.ndarray, groups: np.ndarray, n_splits: int = 5) -> tuple[dict[str, float], float]`: Runs `StratifiedGroupKFold` grouped by `run_id` within `train == 1`, aggregates mean $\pm$ std validation metrics ($AUC_{PR}$, $F_1$, $AUC_{ROC}$, precision, recall, MCC), and selects the optimal probability threshold on out-of-fold predictions (`oof_probs`) to maximize $F_1$.
  - `fit(X_train: np.ndarray, y_train: np.ndarray) -> BaseEstimator`: Refits the configured model family (v1: `Random Forest`; v2: `FCNN`) on **100% of `train == 1`** once hyperparameters and the decision threshold are finalized.
  - `evaluate(model: Any, X_test: np.ndarray, y_test: np.ndarray, buffer_fill: np.ndarray, threshold: float) -> dict[str, float]`: Evaluates the refit model once on the locked `train == 0` holdout set using the CV-selected threshold, computing $F_1$, $AUC_{PR}$, $AUC_{ROC}$, precision, recall, and MCC overall, plus separate metric slices for early padded ticks (`buffer_fill < 50`) and full windows (`buffer_fill == 50`).
- **`RegistryPublisher`:** Manages experiment tracking and promotion in MLflow.
  - `log_and_register(model: Any, params: dict, cv_metrics: dict, test_metrics: dict, threshold: float, input_example: np.ndarray) -> int`: Logs hyperparameters, grouped CV metrics, holdout test metrics, the optimal decision threshold, model signature, and artifact to MLflow Tracking, then registers a new version under `opssat-anomaly`.
  - `promote_to_champion(model_name: str, version: int) -> None`: Uses `MlflowClient.set_registered_model_alias()` to point `@champion` to the newly validated model version so `serving/` pods pull it automatically on rollout.