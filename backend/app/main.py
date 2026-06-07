"""FastAPI entry point for the Waze churn prediction service."""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Annotated

from fastapi import Depends, FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.app.config import Settings, get_settings
from backend.app.dependencies import get_predictor
from backend.app.schemas import (
    BatchPredictionRequest,
    BatchPredictionResponse,
    HealthResponse,
    ModelInfoResponse,
    PredictionResponse,
    UserFeatures,
)
from waze_churn import ChurnPredictor

PredictorDependency = Annotated[ChurnPredictor, Depends(get_predictor)]


def create_app(settings: Settings | None = None) -> FastAPI:
    """Create an application with explicit, testable runtime settings."""
    runtime_settings = settings or get_settings()

    @asynccontextmanager
    async def lifespan(application: FastAPI) -> AsyncIterator[None]:
        application.state.predictor = ChurnPredictor(
            model_path=runtime_settings.model_path,
            metadata_path=runtime_settings.metadata_path,
        )
        yield
        application.state.predictor = None

    application = FastAPI(
        title=runtime_settings.app_name,
        version=runtime_settings.app_version,
        description=(
            "Predict calibrated Waze user churn risk for single users or batches."
        ),
        lifespan=lifespan,
    )
    application.add_middleware(
        CORSMiddleware,
        allow_origins=list(runtime_settings.cors_origins),
        allow_credentials=False,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["Content-Type"],
    )

    @application.get("/health", response_model=HealthResponse, tags=["system"])
    def health(
        predictor: PredictorDependency,
    ) -> HealthResponse:
        return HealthResponse(status="ok", model_loaded=predictor is not None)

    @application.get(
        "/model/info",
        response_model=ModelInfoResponse,
        tags=["model"],
    )
    def model_info(
        predictor: PredictorDependency,
    ) -> ModelInfoResponse:
        metadata = predictor.metadata
        return ModelInfoResponse(
            model_name=metadata["candidate_model_name"],
            model_version=predictor.model_version,
            threshold=predictor.threshold,
            threshold_selection=metadata["threshold_selection"],
            scikit_learn_version=metadata["scikit_learn_version"],
            test_metrics=metadata["test_metrics"],
        )

    @application.post(
        "/predict",
        response_model=PredictionResponse,
        tags=["predictions"],
    )
    def predict(
        features: UserFeatures,
        predictor: PredictorDependency,
    ) -> PredictionResponse:
        result = predictor.predict_one(features.model_dump())
        return PredictionResponse(**result.to_dict())

    @application.post(
        "/predict/batch",
        response_model=BatchPredictionResponse,
        tags=["predictions"],
    )
    def predict_batch(
        request: BatchPredictionRequest,
        predictor: PredictorDependency,
    ) -> BatchPredictionResponse:
        records = [user.model_dump() for user in request.users]
        results = predictor.predict_batch(records)
        predictions = [
            PredictionResponse(**result.to_dict()) for result in results
        ]
        return BatchPredictionResponse(
            count=len(predictions),
            predictions=predictions,
        )

    return application


app = create_app()
