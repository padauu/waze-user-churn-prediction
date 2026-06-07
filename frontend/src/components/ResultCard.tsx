import { ArrowUpRight, CheckCircle2, Route, ShieldAlert } from "lucide-react";
import type { PredictionResult } from "../types";

interface ResultCardProps {
  result: PredictionResult | null;
  loading: boolean;
}

function getRiskTier(probability: number, threshold: number) {
  if (probability >= threshold) {
    return {
      label: "Elevated risk",
      className: "risk-high",
      summary: "This user is above the operating churn threshold.",
      action:
        "Focus on habit-building prompts, saved routes, and timely re-engagement.",
    };
  }
  return {
    label: "Lower risk",
    className: "risk-low",
    summary: "The user is below the current churn decision threshold.",
    action:
      "Avoid over-targeting; maintain a reliable product experience instead.",
  };
}

export function ResultCard({ result, loading }: ResultCardProps) {
  if (loading) {
    return (
      <section className="result-card result-card--loading" aria-live="polite">
        <div className="result-card__loader" />
        <p>Mapping the user journey...</p>
      </section>
    );
  }

  if (!result) {
    return (
      <section className="result-card result-card--empty">
        <div className="empty-visual">
          <Route size={34} strokeWidth={1.7} />
        </div>
        <p className="eyebrow">Prediction preview</p>
        <h2>Your churn signal will appear here</h2>
        <p>
          Complete the behavioral profile and run the model to see calibrated
          risk, threshold status, and a suggested next action.
        </p>
        <div className="empty-route" aria-hidden="true">
          <span />
          <span />
          <span />
        </div>
      </section>
    );
  }

  const probability = result.churn_probability * 100;
  const threshold = result.threshold * 100;
  const tier = getRiskTier(result.churn_probability, result.threshold);
  const circumference = 2 * Math.PI * 76;
  const offset = circumference - (probability / 100) * circumference;

  return (
    <section className={`result-card result-card--filled ${tier.className}`}>
      <div className="result-card__topline">
        <span className="status-pill">
          {result.predicted_churn ? (
            <ShieldAlert size={15} />
          ) : (
            <CheckCircle2 size={15} />
          )}
          {tier.label}
        </span>
        <span className="model-chip">Model v{result.model_version}</span>
      </div>

      <div className="score-layout">
        <div className="score-ring">
          <svg viewBox="0 0 180 180" aria-hidden="true">
            <circle className="score-ring__track" cx="90" cy="90" r="76" />
            <circle
              className="score-ring__value"
              cx="90"
              cy="90"
              r="76"
              strokeDasharray={circumference}
              strokeDashoffset={offset}
            />
          </svg>
          <div className="score-ring__content">
            <strong>{probability.toFixed(1)}%</strong>
            <span>churn risk</span>
          </div>
        </div>

        <div className="score-copy">
          <p className="eyebrow">Calibrated result</p>
          <h2>{tier.summary}</h2>
          <p>
            The model flags churn at {threshold.toFixed(0)}%. This profile is{" "}
            <strong>
              {Math.abs(probability - threshold).toFixed(1)} points{" "}
              {probability >= threshold ? "above" : "below"}
            </strong>{" "}
            that operating threshold.
          </p>
        </div>
      </div>

      <div className="recommendation">
        <div>
          <span className="recommendation__label">Suggested next move</span>
          <p>{tier.action}</p>
        </div>
        <ArrowUpRight size={22} />
      </div>

      <p className="result-disclaimer">
        Use this score for prioritization, not as a certain prediction of an
        individual user's behavior.
      </p>
    </section>
  );
}
