"""Command-line entry point for the BESS load-forecasting workflow."""

import argparse
from pathlib import Path

from src.data_preparation import add_features, load_and_prepare_data
from src.evaluation import rolling_origin_validation


DEFAULT_CSV_PATH = Path(__file__).parent / "data" / "load_timeseries_2025_case_study.csv"


def main(csv_path: str | Path = DEFAULT_CSV_PATH):
    """Prepare data, run forecasts, and return predictions plus metrics."""
    df = load_and_prepare_data(csv_path)
    df, feature_columns, categorical_features = add_features(df)
    predictions, metrics = rolling_origin_validation(
        df, feature_columns, categorical_features, quantile=0.95
    )
    return {"data": df, "predictions": predictions, "metrics": metrics}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the BESS load forecast")
    parser.add_argument(
        "csv_path",
        nargs="?",
        default=DEFAULT_CSV_PATH,
        help="Path to the input CSV (defaults to data/load_timeseries_2025_case_study.csv).",
    )
    args = parser.parse_args()
    print(main(args.csv_path)["metrics"])
