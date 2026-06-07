import "@testing-library/jest-dom/vitest";
import {
  cleanup,
  fireEvent,
  render,
  screen,
  waitFor,
} from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import App from "./App";

const { mockPredictChurn } = vi.hoisted(() => ({
  mockPredictChurn: vi.fn(),
}));

vi.mock("./api", () => ({
  checkHealth: vi.fn().mockResolvedValue(true),
  predictChurn: mockPredictChurn,
}));

afterEach(() => {
  cleanup();
  mockPredictChurn.mockReset();
});

describe("App", () => {
  it("renders the behavior form and prediction action", () => {
    render(<App />);

    expect(
      screen.getByRole("heading", {
        name: /see where a user journey may be heading/i,
      }),
    ).toBeInTheDocument();
    expect(
      screen.getByRole("button", { name: /calculate churn risk/i }),
    ).toBeInTheDocument();
    expect(screen.getByLabelText(/recent sessions/i)).toHaveValue(23);
  });

  it("shows the calibrated result after submitting the profile", async () => {
    mockPredictChurn.mockResolvedValueOnce({
      churn_probability: 0.324978,
      predicted_churn: true,
      predicted_label: "churned",
      threshold: 0.19,
      model_version: "1.0.0",
    });
    render(<App />);

    fireEvent.click(
      screen.getByRole("button", { name: /calculate churn risk/i }),
    );

    await waitFor(() => {
      expect(screen.getByText("32.5%")).toBeInTheDocument();
    });
    expect(
      screen.getByText(/above the operating churn threshold/i),
    ).toBeInTheDocument();
    expect(mockPredictChurn).toHaveBeenCalledWith(
      expect.objectContaining({ sessions: 23, device: "Android" }),
    );
  });
});
