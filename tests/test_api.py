from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from backend.app.config import Settings
from backend.app.main import create_app

PROJECT_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def api_client():
    settings = Settings(
        model_path=PROJECT_ROOT / "models" / "waze_churn_model.joblib",
        metadata_path=PROJECT_ROOT
        / "models"
        / "waze_churn_model_metadata.json",
    )
    with TestClient(create_app(settings)) as client:
        yield client


@pytest.fixture()
def valid_payload():
    return {
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


def test_health_reports_loaded_model(api_client):
    response = api_client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok", "model_loaded": True}


def test_model_info_returns_active_metadata(api_client):
    response = api_client.get("/model/info")

    assert response.status_code == 200
    body = response.json()
    assert body["model_name"] == "LogisticRegression_RobustCapped"
    assert body["model_version"] == "1.0.0"
    assert body["threshold"] == 0.19


def test_predict_returns_calibrated_prediction(api_client, valid_payload):
    response = api_client.post("/predict", json=valid_payload)

    assert response.status_code == 200
    body = response.json()
    assert 0 <= body["churn_probability"] <= 1
    assert body["predicted_label"] in {"churned", "retained"}
    assert body["predicted_churn"] == (
        body["churn_probability"] >= body["threshold"]
    )


def test_batch_predict_preserves_order_and_count(api_client, valid_payload):
    second_payload = {**valid_payload, "device": "iPhone", "activity_days": 20}
    response = api_client.post(
        "/predict/batch",
        json={"users": [valid_payload, second_payload]},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["count"] == 2
    assert len(body["predictions"]) == 2


def test_predict_rejects_invalid_device(api_client, valid_payload):
    response = api_client.post(
        "/predict",
        json={**valid_payload, "device": "Windows Phone"},
    )

    assert response.status_code == 422


def test_predict_rejects_extra_fields(api_client, valid_payload):
    response = api_client.post(
        "/predict",
        json={**valid_payload, "unknown_feature": 1},
    )

    assert response.status_code == 422


def test_batch_rejects_empty_request(api_client):
    response = api_client.post("/predict/batch", json={"users": []})

    assert response.status_code == 422


def test_cors_allows_local_frontend(api_client):
    response = api_client.options(
        "/predict",
        headers={
            "Origin": "http://localhost:5173",
            "Access-Control-Request-Method": "POST",
        },
    )

    assert response.status_code == 200
    assert response.headers["access-control-allow-origin"] == (
        "http://localhost:5173"
    )
