# Phase 1: Python Inference Pipeline

## Goal

Phase 1 turns the notebook model into a reusable Python component. A caller
provides raw user features and receives a churn probability plus a decision
made with the saved threshold.

## Install

```powershell
python -m pip install -e ".[dev]"
```

The exact scikit-learn version is pinned because persisted sklearn models are
not guaranteed to load correctly across versions.

## Run an example

```powershell
python examples/predict.py
```

## Python API

```python
from waze_churn import ChurnPredictor

predictor = ChurnPredictor(
    model_path="models/waze_churn_model.joblib",
    metadata_path="models/waze_churn_model_metadata.json",
)

result = predictor.predict_one(user_record)
print(result.to_dict())
```

For multiple users:

```python
results = predictor.predict_batch(list_of_records)
```

`predict_batch` also accepts a pandas DataFrame.

## Inference flow

1. `ChurnPredictor` loads the sklearn pipeline once.
2. `records_to_frame` validates the 11 raw features, data types, non-negative
   values, and supported devices.
3. Saved p95 thresholds create the seven internal `is_high_*` flags.
4. The saved sklearn pipeline creates engineered behavioral features.
5. The pipeline applies learned percentile caps and preprocessing.
6. The calibrated classifier returns churn probabilities.
7. The saved threshold converts each probability into a business decision.

The caller must never manually reproduce scaling, one-hot encoding, percentile
caps, or engineered features. Those transformations are part of the artifact.
