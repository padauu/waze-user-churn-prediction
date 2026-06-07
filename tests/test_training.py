import numpy as np
import pandas as pd

from waze_churn.training import (
    calculate_high_usage_thresholds,
    evaluate_thresholds,
    select_minimum_cost_threshold,
    split_dataset,
)


def test_split_dataset_uses_60_20_20_stratified_split():
    rows = []
    for index in range(100):
        row = {
            "label": "churned" if index % 5 == 0 else "retained",
            "sessions": index + 1,
            "drives": index + 1,
            "total_sessions": index + 1.0,
            "n_days_after_onboarding": index + 10,
            "total_navigations_fav1": index,
            "total_navigations_fav2": index,
            "driven_km_drives": index + 1.0,
            "duration_minutes_drives": index + 1.0,
            "activity_days": index % 31,
            "driving_days": index % 30,
            "device": "Android" if index % 2 == 0 else "iPhone",
            "is_high_sessions": 0,
            "is_high_drives": 0,
            "is_high_total_sessions": 0,
            "is_high_driven_km_drives": 0,
            "is_high_duration_minutes_drives": 0,
            "is_high_total_navigations_fav1": 0,
            "is_high_total_navigations_fav2": 0,
        }
        rows.append(row)

    split = split_dataset(pd.DataFrame(rows))

    assert len(split.X_train) == 60
    assert len(split.X_validation) == 20
    assert len(split.X_test) == 20
    assert split.y_train.mean() == split.y_validation.mean()
    assert split.y_validation.mean() == split.y_test.mean()


def test_threshold_selection_uses_business_cost():
    y_true = pd.Series([0, 0, 1, 1])
    probabilities = np.array([0.1, 0.4, 0.3, 0.9])
    thresholds = np.array([0.2, 0.5])

    results = evaluate_thresholds(
        y_true,
        probabilities,
        thresholds=thresholds,
        false_negative_cost=5,
        false_positive_cost=1,
    )

    assert select_minimum_cost_threshold(results) == 0.2


def test_high_usage_thresholds_are_p95(sample_frame):
    thresholds = calculate_high_usage_thresholds(sample_frame)

    assert thresholds["sessions"] == sample_frame["sessions"].quantile(0.95)
    assert thresholds["drives"] == sample_frame["drives"].quantile(0.95)
