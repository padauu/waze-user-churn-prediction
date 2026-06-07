"""Application configuration loaded from environment variables."""

import os
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class Settings:
    """Runtime settings for the prediction API."""

    app_name: str = "Waze Churn Prediction API"
    app_version: str = "0.1.0"
    model_path: Path = PROJECT_ROOT / "models" / "waze_churn_model.joblib"
    metadata_path: Path = (
        PROJECT_ROOT / "models" / "waze_churn_model_metadata.json"
    )
    cors_origins: tuple[str, ...] = (
        "http://127.0.0.1:5173",
        "http://localhost:5173",
    )


def get_settings() -> Settings:
    """Return settings with optional environment variable overrides."""
    return Settings(
        app_name=os.getenv("WAZE_API_NAME", "Waze Churn Prediction API"),
        app_version=os.getenv("WAZE_API_VERSION", "0.1.0"),
        model_path=Path(
            os.getenv(
                "WAZE_MODEL_PATH",
                str(PROJECT_ROOT / "models" / "waze_churn_model.joblib"),
            )
        ),
        metadata_path=Path(
            os.getenv(
                "WAZE_METADATA_PATH",
                str(
                    PROJECT_ROOT
                    / "models"
                    / "waze_churn_model_metadata.json"
                ),
            )
        ),
        cors_origins=tuple(
            origin.strip()
            for origin in os.getenv(
                "WAZE_CORS_ORIGINS",
                "http://127.0.0.1:5173,http://localhost:5173",
            ).split(",")
            if origin.strip()
        ),
    )
