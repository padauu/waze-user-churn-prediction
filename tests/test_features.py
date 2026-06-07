import numpy as np
import pandas as pd

from waze_churn.features import add_waze_features


def test_feature_engineering_handles_zero_denominators():
    frame = pd.DataFrame(
        {
            "sessions": [0],
            "drives": [0],
            "total_sessions": [0],
            "n_days_after_onboarding": [0],
            "total_navigations_fav1": [0],
            "total_navigations_fav2": [0],
            "driven_km_drives": [0],
            "duration_minutes_drives": [0],
            "activity_days": [0],
            "driving_days": [0],
        }
    )

    engineered = add_waze_features(frame)
    numeric = engineered.select_dtypes(include="number")

    assert not numeric.isna().any().any()
    assert np.isfinite(numeric.to_numpy()).all()
