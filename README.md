# Waze User Churn Prediction

## Production App Development

Production development is taking place on the `production-app` branch.
Phase 1 extracts notebook inference into an installable Python package with:

- A versioned, importable model artifact;
- A validated 11-feature input contract;
- Single-user and batch prediction methods;
- Reusable feature engineering;
- Automated tests and linting.

Quick start:

```powershell
conda activate ml-env
python -m pip install -e ".[dev]"
python examples/predict.py
uvicorn backend.app.main:app --reload
python -m pytest -q
```

API documentation is available at `http://127.0.0.1:8000/docs` while the
development server is running.

Available endpoints:

| Method | Endpoint         | Purpose                       |
| ------ | ---------------- | ----------------------------- |
| GET    | `/health`        | Service and model health      |
| GET    | `/model/info`    | Active model metadata         |
| POST   | `/predict`       | Single-user prediction        |
| POST   | `/predict/batch` | Batch prediction up to 1,000  |

The repository includes a small demo model artifact in `models/` so the API
and frontend can run immediately after installation. Generated retraining
artifacts are ignored by default.

To retrain the model and serve that new artifact:

```powershell
python scripts/train_model.py --output-dir artifacts/training-run
$env:WAZE_MODEL_PATH = "artifacts/training-run/waze_churn_model.joblib"
$env:WAZE_METADATA_PATH = "artifacts/training-run/waze_churn_model_metadata.json"
uvicorn backend.app.main:app --reload
```

Model locations can be overridden with `WAZE_MODEL_PATH` and
`WAZE_METADATA_PATH`.

Run the frontend in a second terminal:

```powershell
cd frontend
npm install
npm run dev
```

The web interface is available at `http://127.0.0.1:5173`.

## Overview

This project builds a machine learning pipeline to predict Waze user churn risk and translate model outputs into business-facing retention recommendations.

Instead of treating churn prediction only as a binary classification task, the project frames the problem as a **churn risk prioritization system**. The goal is to help product and growth teams identify high-risk users, understand behavioral churn drivers, and design more targeted retention actions.

## Business Problem

For a navigation app like Waze, long-term value depends on repeated usage and habit formation. Users who stop using the app regularly may eventually churn. The business problem is to detect churn risk early enough so that the product or growth team can take timely action.

Key questions:

- Which users are more likely to churn?
- What behavioral patterns are associated with churn risk?
- How should users be prioritized for retention actions?
- What types of interventions may fit different high-risk user groups?

## Dataset

The dataset contains user-level Waze behavioral data with approximately 15,000 records.

Key feature groups include:

- **Usage volume:** sessions, drives, total sessions
- **Activity consistency:** activity days, driving days
- **User lifecycle:** days after onboarding
- **Driving behavior:** distance driven and driving duration
- **Favorite navigation behavior:** favorite destination navigation counts
- **Device information:** Android or iPhone
- **Target variable:** churned or retained

Rows with missing target labels were removed before modeling.

## Repository Structure

- `notebooks/01_data_cleaning_eda.ipynb`  
  Data cleaning, missing label analysis, outlier review, EDA, and Waze-specific behavioral insights.

- `notebooks/02_hypothesis_testing.ipynb`  
  Statistical testing around device type, driving behavior, and churn association.

- `notebooks/03_modeling_and_evaluation.ipynb`  
  Feature engineering, model training, model comparison, probability calibration, threshold tuning, cost-sensitive evaluation, and final test evaluation.

- `notebooks/04_model_interpretability_and_business_recommendations.ipynb`  
  Risk tiering, risk-tier validation, behavioral profiling, high-risk user clustering, model interpretation, and business recommendations.

- `src/waze_churn_preprocessing.py`  
  Reusable preprocessing utilities used for loading and interpreting the saved model pipeline.

- `models/`  
  Saved model artifacts and evaluation outputs.

- `data/processed/`  
  Cleaned data and prediction outputs used across notebooks.

## Methodology

### 1. Data Cleaning and EDA

The first notebook removes non-informative identifiers, handles missing target labels, checks duplicates and invalid values, and explores the main behavioral patterns associated with churn.

A key decision is to avoid directly capping extreme values during cleaning. In Waze, extreme usage may represent meaningful behavior, such as commuters, long-distance drivers, or power users. Outlier treatment is therefore handled later as a modeling experiment.

Main EDA findings:

- Churn is strongly associated with lower activity consistency.
- Users with fewer active days and driving days have much higher churn rates.
- Longer-tenure users tend to have lower churn risk.
- Some churned users are not low-volume users; they may use Waze intensely but irregularly.

### 2. Hypothesis Testing

The second notebook tests whether device type is associated with driving behavior and churn.

The results suggest that device type is not a strong standalone churn driver in this dataset. Churn appears to be more strongly related to behavioral engagement patterns than to whether a user is on Android or iPhone.

### 3. Modeling and Evaluation

The modeling notebook compares several models:

- Dummy Classifier
- Logistic Regression
- Logistic Regression with robust scaling and percentile capping
- Decision Tree
- Random Forest
- XGBoost
- SVM
- SVM with robust scaling and percentile capping

Models are evaluated using metrics that are more appropriate for imbalanced churn prediction:

- Precision
- Recall
- F1-score
- F2-score
- ROC-AUC
- PR-AUC

The final model is `LogisticRegression_RobustCapped`.

The final decision threshold is 0.19, selected using minimum validation cost. This approach avoids optimizing recall alone, which can create too many false positives and lead to unnecessary retention actions.

### 4. Model Interpretation and Business Recommendations

The final notebook translates model outputs into business-facing insights.

It includes:

- Calibrated churn probability analysis;
- Percentile-based risk tier assignment;
- Risk-tier validation;
- Behavioral profiling by risk tier;
- Exploratory clustering within the High Risk group;
- Permutation importance;
- Logistic Regression coefficient interpretation;
- Business recommendation matrix.

## Final Model Performance

Final model: `LogisticRegression_RobustCapped`

Final threshold: 0.19

| Metric    | Value |
| --------- | ----: |
| Precision | 0.314 |
| Recall    | 0.639 |
| F1-score  | 0.421 |
| F2-score  | 0.529 |
| ROC-AUC   | 0.742 |
| PR-AUC    | 0.368 |

The model is best interpreted as a **risk prioritization model**, not a perfect binary classifier. It captures a meaningful share of churned users while keeping false positives more controlled than a recall-only approach.

## Key Insights

### 1. Usage consistency is the strongest churn signal

The strongest churn drivers are not raw usage volume, but whether users return to Waze consistently.

Permutation importance and Logistic Regression coefficients both show that the most important behavioral signals are:

- `activity_days`
- `driving_days`
- `n_days_after_onboarding`

Users who are active and driving on more days are much less likely to churn.

### 2. Tenure is protective

Users with longer time since onboarding tend to have lower churn risk. This suggests that lifecycle maturity and product familiarity are associated with stronger retention.

### 3. High-risk users are not always low-volume users

Some High Risk users have high sessions, drives, or driving distance, but their usage is concentrated into fewer days. These users appear to be **intense but irregular users**.

### 4. Risk tiers improve business prioritization

Calibrated churn probabilities are converted into percentile-based risk tiers.

| Risk Tier   | Users | Observed Churn Rate |
| ----------- | ----: | ------------------: |
| Low Risk    | 2,002 |               10.9% |
| Medium Risk |   572 |               28.5% |
| High Risk   |   286 |               44.1% |

The observed churn rate increases clearly from Low Risk to Medium Risk to High Risk, showing that the risk tiers are useful for prioritization.

## High-Risk User Segments

High Risk users are further segmented using exploratory clustering.

| Segment               | Pattern                                     | Suggested Action                                                                                      |
| --------------------- | ------------------------------------------- | ----------------------------------------------------------------------------------------------------- |
| Intense & Irregular   | Low frequency but very high usage intensity | Encourage commute setup, saved routes, recurring-trip reminders, and personalized navigation routines |
| Newer / Medium Tenure | Shorter tenure and unstable product habit   | Provide onboarding continuation, feature education, and lightweight feedback prompts                  |
| Low Frequency         | Very low activity and driving frequency     | Use re-engagement nudges, traffic-alert reminders, and prompts to return to core navigation use cases |

This helps avoid treating all High Risk users the same way.

## Business Recommendations

The model can support a more targeted retention strategy:

- Prioritize High Risk users first.
- Use behavioral profiles to decide what type of retention action is appropriate.
- Apply lighter-touch nudges for Medium Risk users.
- Avoid unnecessary retention campaigns for Low Risk users.
- Validate recommendations through A/B testing or campaign holdout analysis.

Example actions:

| Segment                           | Suggested Action                                             |
| --------------------------------- | ------------------------------------------------------------ |
| High Risk - Intense & Irregular   | Commute setup, saved routes, recurring-trip reminders        |
| High Risk - Newer / Medium Tenure | Onboarding continuation, feature education, feedback prompts |
| High Risk - Low Frequency         | Re-engagement nudges and traffic-alert reminders             |
| Medium Risk                       | Soft nudges and monitoring                                   |
| Low Risk                          | Avoid over-targeting; maintain product experience quality    |

## Limitations

Several limitations remain:

- The dataset is a static user-level snapshot and does not capture detailed time-series behavior.
- The model cannot directly detect recent engagement decline over time.
- Risk tier cutoffs are percentile-based and should be aligned with real campaign capacity in production.
- High-risk clustering is exploratory and should be interpreted as descriptive segmentation.
- The model does not include product experience signals such as app crashes, route quality, notification engagement, user satisfaction, or support interactions.
- Business recommendations should be validated through experiments before deployment.

## Future Work

Future improvements could include:

- Using timestamped activity logs to detect recent behavior decline.
- Adding product experience signals such as app stability, route quality, and notification engagement.
- Estimating real false-positive and false-negative costs from business data.
- Validating retention actions through A/B testing.
- Building a monitoring dashboard for risk tiers and churn trends.
- Retraining and recalibrating the model periodically to handle behavior drift.
