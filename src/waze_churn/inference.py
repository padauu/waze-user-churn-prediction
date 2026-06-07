"""Model loading and inference service for the Waze churn model."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from waze_churn.features import add_high_usage_flags
from waze_churn.schema import MODEL_INPUT_FEATURES, records_to_frame


@dataclass(frozen=True)
class PredictionResult:
    """Business-facing output for one user prediction."""

    churn_probability: float
    predicted_churn: bool
    predicted_label: str
    threshold: float
    model_version: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class ChurnPredictor:
    """Load one model artifact and expose single or batch prediction methods."""

    def __init__(
        self,
        model_path: str | Path,
        metadata_path: str | Path,
    ) -> None:
        self.model_path = Path(model_path)
        self.metadata_path = Path(metadata_path)
        self.model = joblib.load(self.model_path)
        self.metadata = json.loads(self.metadata_path.read_text(encoding="utf-8"))
        self.threshold = float(self.metadata["final_threshold"])
        self.model_version = str(self.metadata.get("model_version", "unknown"))
        self.high_usage_thresholds = self.metadata["high_usage_thresholds"]

    def predict_one(self, record: Mapping[str, Any]) -> PredictionResult:
        """Predict churn risk for one user."""
        return self.predict_batch(record)[0]

    def predict_batch(
        self,
        records: Mapping[str, Any]
        | Sequence[Mapping[str, Any]]
        | pd.DataFrame,
    ) -> list[PredictionResult]:
        """Predict churn risk for one or more users."""
        frame = records_to_frame(records)
        frame = add_high_usage_flags(frame, self.high_usage_thresholds)
        frame = frame.loc[:, MODEL_INPUT_FEATURES]
        probabilities = self.model.predict_proba(frame)[:, 1]

        return [
            PredictionResult(
                churn_probability=float(probability),
                predicted_churn=bool(probability >= self.threshold),
                predicted_label=(
                    "churned" if probability >= self.threshold else "retained"
                ),
                threshold=self.threshold,
                model_version=self.model_version,
            )
            for probability in probabilities
        ]
