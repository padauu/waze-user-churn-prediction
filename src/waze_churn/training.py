"""Reproducible training pipeline for the approved Waze churn model."""

from __future__ import annotations

import json
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.exceptions import FitFailedWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    fbeta_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, RobustScaler

from waze_churn.features import (
    PercentileCapper,
    add_high_usage_flags,
    add_waze_features,
)
from waze_churn.schema import (
    HIGH_USAGE_FEATURES,
    MODEL_INPUT_FEATURES,
    RAW_INPUT_FEATURES,
)

TARGET_COLUMN = "label"
TARGET_MAPPING = {"retained": 0, "churned": 1}
MODEL_NAME = "LogisticRegression_RobustCapped"
MODEL_VERSION = "1.0.0"
RANDOM_STATE = 42
FALSE_NEGATIVE_COST = 5
FALSE_POSITIVE_COST = 1


@dataclass(frozen=True)
class DatasetSplit:
    """Train, validation, and test partitions."""

    X_train: pd.DataFrame
    X_validation: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_validation: pd.Series
    y_test: pd.Series


@dataclass(frozen=True)
class TrainingResult:
    """Paths and metrics produced by one training run."""

    model_path: Path
    metadata_path: Path
    threshold_results_path: Path
    test_predictions_path: Path
    final_threshold: float
    test_metrics: dict[str, Any]


def load_training_data(data_path: str | Path) -> pd.DataFrame:
    """Load and validate the cleaned modeling dataset."""
    frame = pd.read_csv(data_path)
    required_columns = [TARGET_COLUMN, *RAW_INPUT_FEATURES]
    missing = [column for column in required_columns if column not in frame.columns]
    if missing:
        raise ValueError(f"Training data is missing required columns: {missing}")

    invalid_targets = sorted(
        set(frame[TARGET_COLUMN].dropna().unique()) - set(TARGET_MAPPING)
    )
    if invalid_targets:
        raise ValueError(f"Unsupported target labels: {invalid_targets}")
    if frame[TARGET_COLUMN].isna().any():
        raise ValueError("Training target cannot contain missing values.")

    return frame.loc[:, required_columns].copy()


def split_dataset(frame: pd.DataFrame) -> DatasetSplit:
    """Create the same stratified 60/20/20 split used in Notebook 03."""
    X = frame.loc[:, RAW_INPUT_FEATURES]
    y = frame[TARGET_COLUMN].map(TARGET_MAPPING)

    X_train_validation, X_test, y_train_validation, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=y,
    )
    X_train, X_validation, y_train, y_validation = train_test_split(
        X_train_validation,
        y_train_validation,
        test_size=0.25,
        random_state=RANDOM_STATE,
        stratify=y_train_validation,
    )

    return DatasetSplit(
        X_train=X_train,
        X_validation=X_validation,
        X_test=X_test,
        y_train=y_train,
        y_validation=y_validation,
        y_test=y_test,
    )


def build_model_pipeline(X_train: pd.DataFrame) -> Pipeline:
    """Build the approved feature, preprocessing, and classifier pipeline."""
    engineered_sample = add_waze_features(X_train.head())
    numeric_features = engineered_sample.select_dtypes(
        include=[np.number]
    ).columns.tolist()
    categorical_features = engineered_sample.select_dtypes(
        exclude=[np.number]
    ).columns.tolist()

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", RobustScaler(), numeric_features),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", drop="first"),
                categorical_features,
            ),
        ],
        remainder="drop",
    )

    return Pipeline(
        steps=[
            (
                "feature_engineering",
                FunctionTransformer(add_waze_features, validate=False),
            ),
            (
                "percentile_capping",
                PercentileCapper(lower=0.00, upper=0.99),
            ),
            ("preprocessor", preprocessor),
            (
                "model",
                LogisticRegression(
                    C=0.01,
                    penalty="l2",
                    solver="lbfgs",
                    max_iter=2000,
                    class_weight="balanced",
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )


def calibrate_prefit_model(
    fitted_model: Pipeline,
    X_validation: pd.DataFrame,
    y_validation: pd.Series,
) -> CalibratedClassifierCV:
    """Apply the same sigmoid calibration recipe used in Notebook 03."""
    calibrated_model = CalibratedClassifierCV(
        estimator=fitted_model,
        method="sigmoid",
        cv="prefit",
    )
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="The `cv='prefit'` option is deprecated",
        )
        warnings.filterwarnings("ignore", category=FitFailedWarning)
        calibrated_model.fit(X_validation, y_validation)
    return calibrated_model


def evaluate_predictions(
    y_true: pd.Series,
    probabilities: np.ndarray,
    threshold: float,
) -> dict[str, float]:
    """Calculate classification and ranking metrics at one threshold."""
    predictions = (probabilities >= threshold).astype(int)
    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true, predictions)),
        "precision": float(
            precision_score(y_true, predictions, zero_division=0)
        ),
        "recall": float(recall_score(y_true, predictions, zero_division=0)),
        "f1": float(f1_score(y_true, predictions, zero_division=0)),
        "f2": float(
            fbeta_score(y_true, predictions, beta=2, zero_division=0)
        ),
        "roc_auc": float(roc_auc_score(y_true, probabilities)),
        "pr_auc": float(average_precision_score(y_true, probabilities)),
    }


def evaluate_thresholds(
    y_true: pd.Series,
    probabilities: np.ndarray,
    thresholds: np.ndarray | None = None,
    false_negative_cost: int = FALSE_NEGATIVE_COST,
    false_positive_cost: int = FALSE_POSITIVE_COST,
) -> pd.DataFrame:
    """Evaluate business cost and classification quality by threshold."""
    if thresholds is None:
        thresholds = np.arange(0.05, 0.96, 0.01)

    rows = []
    for threshold in thresholds:
        predictions = (probabilities >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, predictions).ravel()
        rows.append(
            {
                "threshold": float(threshold),
                "tn": int(tn),
                "fp": int(fp),
                "fn": int(fn),
                "tp": int(tp),
                "precision": float(
                    precision_score(y_true, predictions, zero_division=0)
                ),
                "recall": float(
                    recall_score(y_true, predictions, zero_division=0)
                ),
                "f1": float(f1_score(y_true, predictions, zero_division=0)),
                "f2": float(
                    fbeta_score(y_true, predictions, beta=2, zero_division=0)
                ),
                "false_positive_rate": float(fp / (fp + tn)),
                "false_negative_rate": float(fn / (fn + tp)),
                "total_cost": int(
                    false_negative_cost * fn + false_positive_cost * fp
                ),
            }
        )
    return pd.DataFrame(rows)


def select_minimum_cost_threshold(threshold_results: pd.DataFrame) -> float:
    """Choose minimum cost, using F2 as the deterministic tie-breaker."""
    best_row = threshold_results.sort_values(
        ["total_cost", "f2"],
        ascending=[True, False],
    ).iloc[0]
    return float(best_row["threshold"])


def calculate_high_usage_thresholds(frame: pd.DataFrame) -> dict[str, float]:
    """Learn p95 thresholds used to create model input flags."""
    return {
        feature: float(frame[feature].quantile(0.95))
        for feature in HIGH_USAGE_FEATURES
    }


def add_training_high_usage_flags(
    split: DatasetSplit,
    thresholds: dict[str, float],
) -> DatasetSplit:
    """Add learned high-usage flags after the train/validation/test split."""
    return DatasetSplit(
        X_train=add_high_usage_flags(split.X_train, thresholds).loc[
            :, MODEL_INPUT_FEATURES
        ],
        X_validation=add_high_usage_flags(split.X_validation, thresholds).loc[
            :, MODEL_INPUT_FEATURES
        ],
        X_test=add_high_usage_flags(split.X_test, thresholds).loc[
            :, MODEL_INPUT_FEATURES
        ],
        y_train=split.y_train,
        y_validation=split.y_validation,
        y_test=split.y_test,
    )


def train_and_save(
    data_path: str | Path,
    output_dir: str | Path,
    model_version: str = MODEL_VERSION,
) -> TrainingResult:
    """Train, calibrate, evaluate, and save the approved model artifacts."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    frame = load_training_data(data_path)
    raw_split = split_dataset(frame)
    high_usage_thresholds = calculate_high_usage_thresholds(raw_split.X_train)
    split = add_training_high_usage_flags(raw_split, high_usage_thresholds)

    base_model = build_model_pipeline(split.X_train)
    base_model.fit(split.X_train, split.y_train)
    calibrated_model = calibrate_prefit_model(
        base_model,
        split.X_validation,
        split.y_validation,
    )

    validation_probabilities = calibrated_model.predict_proba(
        split.X_validation
    )[:, 1]
    threshold_results = evaluate_thresholds(
        split.y_validation,
        validation_probabilities,
    )
    final_threshold = select_minimum_cost_threshold(threshold_results)

    test_probabilities = calibrated_model.predict_proba(split.X_test)[:, 1]
    test_predictions = (test_probabilities >= final_threshold).astype(int)
    test_metrics = evaluate_predictions(
        split.y_test,
        test_probabilities,
        final_threshold,
    )
    test_metrics.update(
        {
            "model": MODEL_NAME,
            "threshold_source": "validation_min_cost",
        }
    )

    model_path = output_dir / "waze_churn_model.joblib"
    metadata_path = output_dir / "waze_churn_model_metadata.json"
    threshold_results_path = output_dir / "threshold_tuning_results.csv"
    test_predictions_path = output_dir / "waze_test_predictions.csv"

    joblib.dump(calibrated_model, model_path)
    threshold_results.to_csv(threshold_results_path, index=False)

    prediction_frame = raw_split.X_test.loc[:, RAW_INPUT_FEATURES].copy()
    prediction_frame["actual_churn"] = split.y_test.to_numpy()
    prediction_frame["predicted_churn"] = test_predictions
    prediction_frame["churn_probability"] = test_probabilities
    prediction_frame.to_csv(test_predictions_path, index=False)

    metadata = {
        "candidate_model_name": MODEL_NAME,
        "model_version": model_version,
        "artifact_file": model_path.name,
        "artifact_format": "joblib",
        "scikit_learn_version": sklearn.__version__,
        "final_threshold": final_threshold,
        "threshold_selection": "minimum validation cost",
        "false_negative_cost": FALSE_NEGATIVE_COST,
        "false_positive_cost": FALSE_POSITIVE_COST,
        "training_recipe": {
            "random_state": RANDOM_STATE,
            "train_fraction": 0.60,
            "validation_fraction": 0.20,
            "test_fraction": 0.20,
            "classifier": {
                "type": "LogisticRegression",
                "C": 0.01,
                "penalty": "l2",
                "solver": "lbfgs",
                "class_weight": "balanced",
                "max_iter": 2000,
            },
            "calibration": {
                "method": "sigmoid",
                "dataset": "validation",
            },
        },
        "split_sizes": {
            "train": len(split.X_train),
            "validation": len(split.X_validation),
            "test": len(split.X_test),
        },
        "high_usage_thresholds": high_usage_thresholds,
        "test_metrics": test_metrics,
    }
    metadata_path.write_text(
        json.dumps(metadata, indent=4) + "\n",
        encoding="utf-8",
    )

    return TrainingResult(
        model_path=model_path,
        metadata_path=metadata_path,
        threshold_results_path=threshold_results_path,
        test_predictions_path=test_predictions_path,
        final_threshold=final_threshold,
        test_metrics=test_metrics,
    )
