export type Device = "Android" | "iPhone";

export interface UserFeatures {
  sessions: number;
  drives: number;
  total_sessions: number;
  n_days_after_onboarding: number;
  total_navigations_fav1: number;
  total_navigations_fav2: number;
  driven_km_drives: number;
  duration_minutes_drives: number;
  activity_days: number;
  driving_days: number;
  device: Device;
}

export interface PredictionResult {
  churn_probability: number;
  predicted_churn: boolean;
  predicted_label: "churned" | "retained";
  threshold: number;
  model_version: string;
}
