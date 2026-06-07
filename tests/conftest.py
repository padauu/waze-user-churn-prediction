from pathlib import Path

import pandas as pd
import pytest

from waze_churn import ChurnPredictor

PROJECT_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def predictor() -> ChurnPredictor:
    return ChurnPredictor(
        model_path=PROJECT_ROOT / "models" / "waze_churn_model.joblib",
        metadata_path=PROJECT_ROOT / "models" / "waze_churn_model_metadata.json",
    )


@pytest.fixture()
def sample_frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "sessions": 283,
                "drives": 226,
                "total_sessions": 296.7482729,
                "n_days_after_onboarding": 2276,
                "total_navigations_fav1": 208,
                "total_navigations_fav2": 0,
                "driven_km_drives": 2628.845068,
                "duration_minutes_drives": 1985.775061,
                "activity_days": 28,
                "driving_days": 19,
                "device": "Android",
            },
            {
                "sessions": 23,
                "drives": 20,
                "total_sessions": 45.0,
                "n_days_after_onboarding": 300,
                "total_navigations_fav1": 2,
                "total_navigations_fav2": 0,
                "driven_km_drives": 500.0,
                "duration_minutes_drives": 700.0,
                "activity_days": 4,
                "driving_days": 3,
                "device": "iPhone",
            },
        ]
    )
