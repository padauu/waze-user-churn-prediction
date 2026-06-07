import type { PredictionResult, UserFeatures } from "./types";

const API_BASE_URL = import.meta.env.VITE_API_URL ?? "/api";

interface ApiErrorBody {
  detail?: string | Array<{ msg?: string }>;
}

function formatApiError(body: ApiErrorBody): string {
  if (typeof body.detail === "string") {
    return body.detail;
  }
  if (Array.isArray(body.detail)) {
    return body.detail
      .map((item) => item.msg)
      .filter(Boolean)
      .join(" ");
  }
  return "The prediction service could not process this request.";
}

export async function predictChurn(
  features: UserFeatures,
): Promise<PredictionResult> {
  const response = await fetch(`${API_BASE_URL}/predict`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(features),
  });

  if (!response.ok) {
    let message = `Prediction request failed (${response.status}).`;
    try {
      message = formatApiError((await response.json()) as ApiErrorBody);
    } catch {
      // Keep the status-based fallback when the server does not return JSON.
    }
    throw new Error(message);
  }

  return (await response.json()) as PredictionResult;
}

export async function checkHealth(): Promise<boolean> {
  try {
    const response = await fetch(`${API_BASE_URL}/health`);
    if (!response.ok) {
      return false;
    }
    const body = (await response.json()) as {
      status?: string;
      model_loaded?: boolean;
    };
    return body.status === "ok" && body.model_loaded === true;
  } catch {
    return false;
  }
}
