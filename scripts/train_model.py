"""Command-line entry point for reproducible model training."""

import argparse
from pathlib import Path

from waze_churn.training import MODEL_VERSION, train_and_save

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train and save the approved Waze churn model."
    )
    parser.add_argument(
        "--data-path",
        type=Path,
        default=PROJECT_ROOT / "data" / "processed" / "waze_clean.csv",
        help="Path to the cleaned modeling CSV.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "models",
        help="Directory for model, metadata, and evaluation artifacts.",
    )
    parser.add_argument(
        "--model-version",
        default=MODEL_VERSION,
        help="Business version stored in model metadata.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = train_and_save(
        data_path=args.data_path,
        output_dir=args.output_dir,
        model_version=args.model_version,
    )

    print(f"Model: {result.model_path}")
    print(f"Metadata: {result.metadata_path}")
    print(f"Threshold results: {result.threshold_results_path}")
    print(f"Test predictions: {result.test_predictions_path}")
    print(f"Selected threshold: {result.final_threshold:.2f}")
    print("Test metrics:")
    for name, value in result.test_metrics.items():
        print(f"  {name}: {value}")


if __name__ == "__main__":
    main()
