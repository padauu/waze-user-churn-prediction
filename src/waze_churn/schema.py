"""Input contract for churn model inference."""

from collections.abc import Mapping, Sequence
from typing import Any

import pandas as pd

RAW_INPUT_FEATURES = [
    "sessions",
    "drives",
    "total_sessions",
    "n_days_after_onboarding",
    "total_navigations_fav1",
    "total_navigations_fav2",
    "driven_km_drives",
    "duration_minutes_drives",
    "activity_days",
    "driving_days",
    "device",
]

HIGH_USAGE_FEATURES = [
    "sessions",
    "drives",
    "total_sessions",
    "driven_km_drives",
    "duration_minutes_drives",
    "total_navigations_fav1",
    "total_navigations_fav2",
]

MODEL_INPUT_FEATURES = [
    *RAW_INPUT_FEATURES,
    "is_high_sessions",
    "is_high_drives",
    "is_high_total_sessions",
    "is_high_driven_km_drives",
    "is_high_duration_minutes_drives",
    "is_high_total_navigations_fav1",
    "is_high_total_navigations_fav2",
]

NUMERIC_FEATURES = [
    feature for feature in RAW_INPUT_FEATURES if feature != "device"
]
ALLOWED_DEVICES = {"Android", "iPhone"}


class InputValidationError(ValueError):
    """Raised when inference input does not match the model contract."""


def records_to_frame(
    records: Mapping[str, Any] | Sequence[Mapping[str, Any]] | pd.DataFrame,
) -> pd.DataFrame:
    """Convert supported record formats into a validated model input frame."""
    if isinstance(records, pd.DataFrame):
        frame = records.copy()
    elif isinstance(records, Mapping):
        frame = pd.DataFrame([records])
    else:
        frame = pd.DataFrame(list(records))

    if frame.empty:
        raise InputValidationError("At least one input record is required.")

    missing = [
        column for column in RAW_INPUT_FEATURES if column not in frame.columns
    ]
    if missing:
        raise InputValidationError(f"Missing required features: {missing}")

    frame = frame.loc[:, RAW_INPUT_FEATURES].copy()

    for column in NUMERIC_FEATURES:
        try:
            frame[column] = pd.to_numeric(frame[column], errors="raise")
        except (TypeError, ValueError) as exc:
            raise InputValidationError(
                f"Feature '{column}' must contain numeric values."
            ) from exc

    if frame[NUMERIC_FEATURES].isna().any().any():
        raise InputValidationError("Numeric features cannot contain missing values.")

    negative_columns = [
        column for column in NUMERIC_FEATURES if (frame[column] < 0).any()
    ]
    if negative_columns:
        raise InputValidationError(
            f"Features cannot contain negative values: {negative_columns}"
        )

    invalid_devices = sorted(set(frame["device"]) - ALLOWED_DEVICES)
    if invalid_devices:
        raise InputValidationError(
            f"Unsupported device values: {invalid_devices}. "
            f"Allowed values: {sorted(ALLOWED_DEVICES)}"
        )

    return frame
