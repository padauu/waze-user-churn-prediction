"""Run one local prediction using the production inference package."""

from pathlib import Path

from waze_churn import ChurnPredictor

PROJECT_ROOT = Path(__file__).resolve().parents[1]

sample_user = {
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
    "device": "Android",
}

predictor = ChurnPredictor(
    model_path=PROJECT_ROOT / "models" / "waze_churn_model.joblib",
    metadata_path=PROJECT_ROOT / "models" / "waze_churn_model_metadata.json",
)

print(predictor.predict_one(sample_user).to_dict())
