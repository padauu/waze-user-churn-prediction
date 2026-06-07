"""FastAPI dependencies shared by API routes."""

from fastapi import HTTPException, Request, status

from waze_churn import ChurnPredictor


def get_predictor(request: Request) -> ChurnPredictor:
    """Return the predictor loaded during application startup."""
    predictor = getattr(request.app.state, "predictor", None)
    if predictor is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Prediction model is not available.",
        )
    return predictor
