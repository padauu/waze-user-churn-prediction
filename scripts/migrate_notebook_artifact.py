"""Migrate the notebook pickle into an importable production artifact."""

import json
import pickle
from pathlib import Path

import joblib

import __main__
from waze_churn.features import PercentileCapper, add_waze_features

PROJECT_ROOT = Path(__file__).resolve().parents[1]
LEGACY_MODEL_PATH = PROJECT_ROOT / "models" / "waze_churn_calibrated_model.pkl"
MODEL_PATH = PROJECT_ROOT / "models" / "waze_churn_model.joblib"
METADATA_PATH = PROJECT_ROOT / "models" / "waze_churn_model_metadata.json"


def main() -> None:
    """Load legacy notebook globals and persist a package-based artifact."""
    __main__.add_waze_features = add_waze_features
    __main__.PercentileCapper = PercentileCapper

    with LEGACY_MODEL_PATH.open("rb") as model_file:
        model = pickle.load(model_file)

    joblib.dump(model, MODEL_PATH)

    metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
    metadata.update(
        {
            "model_version": "1.0.0",
            "artifact_file": MODEL_PATH.name,
            "artifact_format": "joblib",
            "scikit_learn_version": "1.7.2",
            "high_usage_thresholds": {
                "sessions": 243.0,
                "drives": 200.0,
                "total_sessions": 455.4394923899998,
                "driven_km_drives": 8898.716274999999,
                "duration_minutes_drives": 4668.180091799999,
                "total_navigations_fav1": 422.0,
                "total_navigations_fav2": 124.0,
            },
        }
    )
    METADATA_PATH.write_text(
        json.dumps(metadata, indent=4) + "\n",
        encoding="utf-8",
    )

    print(f"Created {MODEL_PATH}")
    print(f"Updated {METADATA_PATH}")


if __name__ == "__main__":
    main()
