"""Feature engineering used by both model training and inference."""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


def add_high_usage_flags(
    X: pd.DataFrame,
    thresholds: dict[str, float],
) -> pd.DataFrame:
    """Create model input flags from thresholds learned during data preparation."""
    X = X.copy()
    for feature, threshold in thresholds.items():
        X[f"is_high_{feature}"] = (X[feature] > threshold).astype(int)
    return X


def safe_divide(
    numerator: pd.Series,
    denominator: pd.Series,
) -> pd.Series:
    """Divide two Series, replacing undefined results with zero."""
    result = numerator / denominator.replace(0, np.nan)
    return result.replace([np.inf, -np.inf], np.nan).fillna(0)


def add_waze_features(X: pd.DataFrame) -> pd.DataFrame:
    """Add behavioral features expected by the trained model pipeline."""
    X = X.copy()

    X["sessions_per_day_after_onboarding"] = safe_divide(
        X["sessions"], X["n_days_after_onboarding"]
    )
    X["total_sessions_per_day_after_onboarding"] = safe_divide(
        X["total_sessions"], X["n_days_after_onboarding"]
    )
    X["drives_per_activity_day"] = safe_divide(X["drives"], X["activity_days"])
    X["drives_per_driving_day"] = safe_divide(X["drives"], X["driving_days"])
    X["km_per_drive"] = safe_divide(X["driven_km_drives"], X["drives"])
    X["minutes_per_drive"] = safe_divide(
        X["duration_minutes_drives"], X["drives"]
    )
    X["km_per_driving_day"] = safe_divide(
        X["driven_km_drives"], X["driving_days"]
    )
    X["minutes_per_driving_day"] = safe_divide(
        X["duration_minutes_drives"], X["driving_days"]
    )
    X["percent_sessions_in_last_month"] = safe_divide(
        X["sessions"], X["total_sessions"]
    )
    X["total_favorite_navigations"] = (
        X["total_navigations_fav1"] + X["total_navigations_fav2"]
    )
    X["has_favorite_navigation"] = (
        X["total_favorite_navigations"] > 0
    ).astype(int)
    X["favorite_navigations_per_session"] = safe_divide(
        X["total_favorite_navigations"], X["total_sessions"]
    )

    return X


class PercentileCapper(BaseEstimator, TransformerMixin):
    """Clip numeric columns using bounds learned from training data."""

    def __init__(self, lower: float = 0.00, upper: float = 0.99):
        self.lower = lower
        self.upper = upper

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None):
        X_df = pd.DataFrame(X).copy()
        self.numeric_cols_ = X_df.select_dtypes(
            include=[np.number]
        ).columns.tolist()
        self.lower_bounds_ = X_df[self.numeric_cols_].quantile(self.lower)
        self.upper_bounds_ = X_df[self.numeric_cols_].quantile(self.upper)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X_df = pd.DataFrame(X).copy()
        for col in self.numeric_cols_:
            if col in X_df.columns:
                X_df[col] = X_df[col].clip(
                    lower=self.lower_bounds_[col],
                    upper=self.upper_bounds_[col],
                )
        return X_df
