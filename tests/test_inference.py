import pytest

from waze_churn.schema import InputValidationError


def test_predict_one_returns_business_output(predictor, sample_frame):
    record = sample_frame.iloc[0].to_dict()

    result = predictor.predict_one(record)

    assert 0 <= result.churn_probability <= 1
    assert result.predicted_label in {"churned", "retained"}
    assert result.predicted_churn == (
        result.churn_probability >= result.threshold
    )
    assert result.model_version == "1.0.0"


def test_predict_batch_preserves_row_count(predictor, sample_frame):
    results = predictor.predict_batch(sample_frame)

    assert len(results) == len(sample_frame)


def test_predict_rejects_missing_feature(predictor, sample_frame):
    record = sample_frame.drop(columns=["activity_days"]).iloc[0].to_dict()

    with pytest.raises(InputValidationError, match="activity_days"):
        predictor.predict_one(record)


def test_predict_rejects_unknown_device(predictor, sample_frame):
    record = sample_frame.iloc[0].to_dict()
    record["device"] = "Windows Phone"

    with pytest.raises(InputValidationError, match="Unsupported device"):
        predictor.predict_one(record)
