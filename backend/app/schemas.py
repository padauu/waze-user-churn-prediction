"""Pydantic request and response schemas for the prediction API."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class UserFeatures(BaseModel):
    """Raw user features accepted by the production model."""

    model_config = ConfigDict(extra="forbid")

    sessions: int = Field(ge=0)
    drives: int = Field(ge=0)
    total_sessions: float = Field(ge=0)
    n_days_after_onboarding: int = Field(ge=0)
    total_navigations_fav1: int = Field(ge=0)
    total_navigations_fav2: int = Field(ge=0)
    driven_km_drives: float = Field(ge=0)
    duration_minutes_drives: float = Field(ge=0)
    activity_days: int = Field(ge=0, le=31)
    driving_days: int = Field(ge=0, le=31)
    device: Literal["Android", "iPhone"]


class PredictionResponse(BaseModel):
    """Prediction returned for one user."""

    churn_probability: float = Field(ge=0, le=1)
    predicted_churn: bool
    predicted_label: Literal["churned", "retained"]
    threshold: float = Field(ge=0, le=1)
    model_version: str


class BatchPredictionRequest(BaseModel):
    """Bounded collection of users for batch prediction."""

    model_config = ConfigDict(extra="forbid")

    users: list[UserFeatures] = Field(min_length=1, max_length=1000)


class BatchPredictionResponse(BaseModel):
    """Predictions returned for a batch request."""

    count: int = Field(ge=1)
    predictions: list[PredictionResponse]


class HealthResponse(BaseModel):
    """Service health status."""

    status: Literal["ok"]
    model_loaded: bool


class ModelInfoResponse(BaseModel):
    """Public metadata for the active model."""

    model_name: str
    model_version: str
    threshold: float
    threshold_selection: str
    scikit_learn_version: str
    test_metrics: dict[str, float | str]
