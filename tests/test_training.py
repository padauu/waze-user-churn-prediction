import numpy as np
import pandas as pd

from waze_churn.training import (
    add_training_high_usage_flags,
    calculate_high_usage_thresholds,
    evaluate_thresholds,
    load_training_data,
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
        }
        rows.append(row)

    split = split_dataset(pd.DataFrame(rows))

    assert len(split.X_train) == 60
    assert len(split.X_validation) == 20
    assert len(split.X_test) == 20
    assert split.y_train.mean() == split.y_validation.mean()
    assert split.y_validation.mean() == split.y_test.mean()
    assert "is_high_sessions" not in split.X_train.columns


def test_training_load_accepts_raw_contract(tmp_path):
    data_path = tmp_path / "training.csv"
    pd.DataFrame(
        [
            {
                "label": "retained",
                "sessions": 10,
                "drives": 8,
                "total_sessions": 12.0,
                "n_days_after_onboarding": 100,
                "total_navigations_fav1": 1,
                "total_navigations_fav2": 0,
                "driven_km_drives": 120.0,
                "duration_minutes_drives": 80.0,
                "activity_days": 6,
                "driving_days": 5,
                "device": "Android",
            }
        ]
    ).to_csv(data_path, index=False)

    frame = load_training_data(data_path)

    assert "is_high_sessions" not in frame.columns
    assert frame.loc[0, "device"] == "Android"


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


def test_training_flags_are_added_from_learned_thresholds(sample_frame):
    split = split_dataset(
        pd.concat(
            [
                sample_frame.assign(label=["retained", "churned"]),
                sample_frame.assign(label=["retained", "churned"]),
                sample_frame.assign(label=["retained", "churned"]),
                sample_frame.assign(label=["retained", "churned"]),
                sample_frame.assign(label=["retained", "churned"]),
            ],
            ignore_index=True,
        )
    )
    thresholds = calculate_high_usage_thresholds(split.X_train)

    flagged_split = add_training_high_usage_flags(split, thresholds)

    assert "is_high_sessions" in flagged_split.X_train.columns
    assert "is_high_sessions" in flagged_split.X_validation.columns
    assert "is_high_sessions" in flagged_split.X_test.columns
