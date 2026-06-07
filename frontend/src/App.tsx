import { useEffect, useState } from "react";
import {
  Activity,
  CarFront,
  Gauge,
  Heart,
  RotateCcw,
  Smartphone,
  Sparkles,
} from "lucide-react";
import { checkHealth, predictChurn } from "./api";
import { BrandMark } from "./components/BrandMark";
import { NumberField } from "./components/NumberField";
import { ResultCard } from "./components/ResultCard";
import type { PredictionResult, UserFeatures } from "./types";

const defaultFeatures: UserFeatures = {
  sessions: 23,
  drives: 20,
  total_sessions: 45,
  n_days_after_onboarding: 300,
  total_navigations_fav1: 2,
  total_navigations_fav2: 0,
  driven_km_drives: 500,
  duration_minutes_drives: 700,
  activity_days: 4,
  driving_days: 3,
  device: "Android",
};

const activeProfile: UserFeatures = {
  sessions: 118,
  drives: 92,
  total_sessions: 540,
  n_days_after_onboarding: 1800,
  total_navigations_fav1: 160,
  total_navigations_fav2: 42,
  driven_km_drives: 4200,
  duration_minutes_drives: 3600,
  activity_days: 25,
  driving_days: 21,
  device: "iPhone",
};

export default function App() {
  const [features, setFeatures] = useState<UserFeatures>(defaultFeatures);
  const [result, setResult] = useState<PredictionResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [serviceReady, setServiceReady] = useState<boolean | null>(null);

  useEffect(() => {
    void checkHealth().then(setServiceReady);
  }, []);

  function updateFeature<K extends keyof UserFeatures>(
    key: K,
    value: UserFeatures[K],
  ) {
    setFeatures((current) => ({ ...current, [key]: value }));
    setError(null);
  }

  async function handleSubmit(event: React.FormEvent<HTMLFormElement>) {
    event.preventDefault();
    setLoading(true);
    setError(null);
    try {
      setResult(await predictChurn(features));
    } catch (requestError) {
      setError(
        requestError instanceof Error
          ? requestError.message
          : "The service is temporarily unavailable.",
      );
    } finally {
      setLoading(false);
    }
  }

  function applyPreset(preset: UserFeatures) {
    setFeatures(preset);
    setResult(null);
    setError(null);
  }

  return (
    <div className="app-shell">
      <header className="site-header">
        <a className="brand" href="#top" aria-label="Waywise home">
          <BrandMark />
          <span>
            <strong>Waywise</strong>
            <small>Churn intelligence</small>
          </span>
        </a>
        <div
          className={`header-meta ${
            serviceReady === false ? "header-meta--offline" : ""
          }`}
        >
          <span className="live-dot" />
          {serviceReady === null
            ? "Connecting to model"
            : serviceReady
              ? "Model service ready"
              : "Model service unavailable"}
        </div>
      </header>

      <main id="top">
        <section className="hero">
          <div className="hero__copy">
            <p className="eyebrow">
              <Sparkles size={15} />
              Calibrated retention signals
            </p>
            <h1>See where a user journey may be heading.</h1>
            <p className="hero__lead">
              Turn behavioral activity into a clear churn-risk signal and a
              practical retention next step.
            </p>
            <div className="hero__stats">
              <div>
                <strong>0.742</strong>
                <span>ROC-AUC</span>
              </div>
              <div>
                <strong>63.9%</strong>
                <span>Recall</span>
              </div>
              <div>
                <strong>0.19</strong>
                <span>Decision threshold</span>
              </div>
            </div>
          </div>
          <div className="hero__route" aria-hidden="true">
            <div className="route-pin route-pin--start" />
            <div className="route-line" />
            <div className="route-car">
              <CarFront size={28} />
            </div>
            <div className="route-pin route-pin--finish" />
          </div>
        </section>

        <section className="workspace">
          <form className="profile-card" onSubmit={handleSubmit}>
            <div className="card-heading">
              <div>
                <p className="eyebrow">User profile</p>
                <h2>Behavior snapshot</h2>
                <p>Enter aggregate activity from the latest observation window.</p>
              </div>
              <button
                className="icon-button"
                type="button"
                onClick={() => applyPreset(defaultFeatures)}
                aria-label="Reset example values"
                title="Reset example values"
              >
                <RotateCcw size={18} />
              </button>
            </div>

            <div className="preset-row" aria-label="Example profiles">
              <span>Try a profile:</span>
              <button type="button" onClick={() => applyPreset(defaultFeatures)}>
                Irregular driver
              </button>
              <button type="button" onClick={() => applyPreset(activeProfile)}>
                Consistent commuter
              </button>
            </div>

            <fieldset>
              <legend>
                <Activity size={17} />
                Usage activity
              </legend>
              <div className="field-grid">
                <NumberField
                  id="sessions"
                  label="Recent sessions"
                  hint="Sessions in the latest month"
                  value={features.sessions}
                  onChange={(value) => updateFeature("sessions", value)}
                />
                <NumberField
                  id="drives"
                  label="Recent drives"
                  hint="Drives in the latest month"
                  value={features.drives}
                  onChange={(value) => updateFeature("drives", value)}
                />
                <NumberField
                  id="total-sessions"
                  label="Lifetime sessions"
                  hint="Total observed sessions"
                  step={0.1}
                  value={features.total_sessions}
                  onChange={(value) => updateFeature("total_sessions", value)}
                />
                <NumberField
                  id="onboarding-days"
                  label="Days since onboarding"
                  hint="User tenure in days"
                  value={features.n_days_after_onboarding}
                  onChange={(value) =>
                    updateFeature("n_days_after_onboarding", value)
                  }
                />
              </div>
            </fieldset>

            <fieldset>
              <legend>
                <Gauge size={17} />
                Driving consistency
              </legend>
              <div className="field-grid">
                <NumberField
                  id="activity-days"
                  label="Active days"
                  hint="Days active this month"
                  max={31}
                  value={features.activity_days}
                  onChange={(value) => updateFeature("activity_days", value)}
                />
                <NumberField
                  id="driving-days"
                  label="Driving days"
                  hint="Days with a drive this month"
                  max={31}
                  value={features.driving_days}
                  onChange={(value) => updateFeature("driving_days", value)}
                />
                <NumberField
                  id="distance"
                  label="Distance driven"
                  hint="Total driven kilometers"
                  step={0.1}
                  value={features.driven_km_drives}
                  onChange={(value) => updateFeature("driven_km_drives", value)}
                />
                <NumberField
                  id="duration"
                  label="Driving duration"
                  hint="Total driving minutes"
                  step={0.1}
                  value={features.duration_minutes_drives}
                  onChange={(value) =>
                    updateFeature("duration_minutes_drives", value)
                  }
                />
              </div>
            </fieldset>

            <fieldset>
              <legend>
                <Heart size={17} />
                Product habits
              </legend>
              <div className="field-grid field-grid--compact">
                <NumberField
                  id="favorite-one"
                  label="Favorite route 1"
                  hint="Navigation count"
                  value={features.total_navigations_fav1}
                  onChange={(value) =>
                    updateFeature("total_navigations_fav1", value)
                  }
                />
                <NumberField
                  id="favorite-two"
                  label="Favorite route 2"
                  hint="Navigation count"
                  value={features.total_navigations_fav2}
                  onChange={(value) =>
                    updateFeature("total_navigations_fav2", value)
                  }
                />
                <label className="field" htmlFor="device">
                  <span className="field__label">Device</span>
                  <div className="select-wrap">
                    <Smartphone size={17} />
                    <select
                      id="device"
                      value={features.device}
                      onChange={(event) =>
                        updateFeature(
                          "device",
                          event.target.value as UserFeatures["device"],
                        )
                      }
                    >
                      <option value="Android">Android</option>
                      <option value="iPhone">iPhone</option>
                    </select>
                  </div>
                  <span className="field__hint">Primary mobile platform</span>
                </label>
              </div>
            </fieldset>

            {error && (
              <div className="form-error" role="alert">
                <ShieldAlertIcon />
                <span>
                  <strong>Could not calculate risk.</strong>
                  {error}
                </span>
              </div>
            )}

            <button className="primary-button" type="submit" disabled={loading}>
              {loading ? "Calculating risk..." : "Calculate churn risk"}
              {!loading && <Sparkles size={18} />}
            </button>
            <p className="form-footnote">
              Results are model-assisted prioritization signals, not guarantees.
            </p>
          </form>

          <div className="result-column">
            <ResultCard result={result} loading={loading} />
            <div className="insight-strip">
              <Activity size={20} />
              <div>
                <strong>Why consistency matters</strong>
                <p>
                  Active days, driving days, and tenure are the strongest signals
                  in this model.
                </p>
              </div>
            </div>
          </div>
        </section>
      </main>

      <footer>
        <p>Waywise is an independent portfolio interface for a Waze churn model.</p>
        <span>Risk prioritization · Model v1.0.0</span>
      </footer>
    </div>
  );
}

function ShieldAlertIcon() {
  return (
    <svg viewBox="0 0 24 24" aria-hidden="true">
      <path d="M12 3 4.5 6v5.5c0 4.7 3.2 7.8 7.5 9.5 4.3-1.7 7.5-4.8 7.5-9.5V6L12 3Z" />
      <path d="M12 8v5M12 16.5v.1" />
    </svg>
  );
}
