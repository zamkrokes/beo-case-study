"""Command-line entry point for the BESS load-forecasting workflow."""

import argparse

from src.data_preparation import add_features, load_and_prepare_data
from src.evaluation import rolling_origin_validation


def main(csv_path="load_timeseries_2025_case_study.csv"):
    """Prepare data, run forecasts, and return predictions plus metrics."""
    df = load_and_prepare_data(csv_path)
    df, feature_columns, categorical_features = add_features(df)
    predictions, metrics = rolling_origin_validation(
        df, feature_columns, categorical_features
    )
    return {"data": df, "predictions": predictions, "metrics": metrics}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the BESS load forecast")
    parser.add_argument("csv_path", nargs="?", default="load_timeseries_2025_case_study.csv")
    args = parser.parse_args()
    print(main(args.csv_path)["metrics"])
