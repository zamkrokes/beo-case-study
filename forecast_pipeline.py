"""Reusable load-forecasting pipeline for the BESS case study."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error


def load_and_prepare_data(csv_path: str | Path, max_valid_load: float = 150.0) -> pd.DataFrame:
    """Load the semicolon-delimited file and create auditable cleaned columns."""
    df = pd.read_csv(csv_path, sep=";")
    df["timestamp"] = pd.to_datetime(df["Timestamps"], format="%d.%m.%Y %H:%M")
    df["load"] = pd.to_numeric(df["Load_kw"].str.replace(",", ".", regex=False))
    df = df[["timestamp", "load"]].set_index("timestamp").sort_index()

    full_index = pd.date_range(df.index.min(), df.index.max(), freq="15min")
    original_index = df.index.copy()
    df = df.reindex(full_index)
    df.index.name = "timestamp"
    df["was_missing"] = ~df.index.isin(original_index)
    df["is_negative"] = df["load"] < 0
    df["is_zero"] = df["load"] == 0
    df["is_high"] = df["load"] > max_valid_load

    # Non-interpolated target for validation; interpolated series is visualization-only.
    df["load_target"] = df["load"].mask(df["is_negative"] | df["is_high"])
    df["load_clean"] = df["load_target"].interpolate(limit=4)
    return df


def add_features(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str], list[str]]:
    """Add causal lags, calendar features, and model-ready categorical columns."""
    try:
        import holidays
    except ImportError as exc:
        raise ImportError("Install holidays with: pip install holidays") from exc

    years = sorted(df.index.year.unique().tolist())
    de = holidays.Germany(years=years)
    holiday_dates = pd.to_datetime(list(de.keys()))
    df["is_holiday"] = df.index.normalize().isin(holiday_dates)
    df["is_weekend"] = df.index.dayofweek >= 5
    df["is_shutdown"] = ((df.index.month == 12) & (df.index.day >= 24)) | (
        (df.index.month == 1) & (df.index.day <= 6)
    )
    df["is_workday"] = ~(df["is_weekend"] | df["is_holiday"] | df["is_shutdown"])

    df["load_model"] = df["load_target"].ffill(limit=4)
    daily_status = df["is_workday"].groupby(df.index.normalize()).first().astype(bool)
    previous_same_status: dict[pd.Timestamp, pd.Timestamp] = {}
    for current_date in daily_status.index:
        prior_dates = daily_status.index[
            (daily_status.index < current_date)
            & (daily_status == daily_status.loc[current_date])
        ]
        previous_same_status[current_date] = prior_dates[-1] if len(prior_dates) else pd.NaT

    reference_index = pd.DatetimeIndex(
        [
            previous_same_status[timestamp.normalize()]
            + (timestamp - timestamp.normalize())
            if pd.notna(previous_same_status[timestamp.normalize()])
            else pd.NaT
            for timestamp in df.index
        ]
    )
    df["lag_same_workday"] = df["load_model"].reindex(reference_index).to_numpy()
    df["lag672"] = df["load_model"].shift(672)
    # Day-ahead causality: every feature for a forecast day must be fixed
    # before 00:00.  Shift the seven-day window by one full day (96 points),
    # rather than one quarter-hour, so it cannot include today's observations.
    df["rolling_mean_week"] = df["load_model"].shift(96).rolling(672, min_periods=672).mean()
    df["rolling_max_week"] = df["load_model"].shift(96).rolling(672, min_periods=672).max()

    df["minute"] = df.index.minute
    df["hour"] = df.index.hour
    df["day"] = df.index.day
    df["month"] = df.index.month
    df["weekday"] = df.index.dayofweek
    df["weekday_name"] = df.index.day_name()

    categorical_features = [
        "minute", "hour", "month", "weekday",
        "is_weekend", "is_holiday", "is_shutdown", "is_workday",
    ]
    for column in categorical_features:
        df[column] = df[column].astype("category")

    feature_columns = categorical_features + [
        "lag_same_workday", "lag672", "rolling_mean_week", "rolling_max_week"
    ]
    return df, feature_columns, categorical_features


def rolling_origin_validation(
    df: pd.DataFrame,
    feature_columns: list[str],
    categorical_features: list[str],
    min_train_days: int = 28,
    quantile: float = 0.95,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Predict each day from an expanding history and return predictions and metrics.

    ``quantile`` controls the conservative LightGBM forecast (0.95 by
    default for the presentation workflow); pass 0.80 for a P80 variant.
    """
    if not 0 < quantile < 1:
        raise ValueError("quantile must be between 0 and 1")
    try:
        from lightgbm import LGBMRegressor
        lightgbm_available = True
    except (ImportError, OSError):
        LGBMRegressor = None
        lightgbm_available = False

    columns = feature_columns + ["load_target"]
    model_data = df[columns].dropna().copy()
    model_data["date"] = model_data.index.normalize()
    predictions: list[pd.DataFrame] = []

    for date in sorted(model_data["date"].unique()):
        train = model_data[model_data["date"] < date]
        test = model_data[model_data["date"] == date]
        if train.empty or (date - train["date"].min()).days < min_train_days or test.empty:
            continue

        result = pd.DataFrame(index=test.index)
        result["actual"] = test["load_target"]
        result["baseline_lag672"] = test["lag672"]
        result["baseline_same_workday"] = test["lag_same_workday"]
        result["day_type"] = "Weekend" if date.dayofweek >= 5 else "Weekday"
        result["is_workday"] = bool(test["is_workday"].iloc[0])
        result["is_peak"] = result["actual"] >= train["load_target"].quantile(0.95)

        if lightgbm_available:
            model = LGBMRegressor(
                n_estimators=100, learning_rate=0.05, num_leaves=31,
                random_state=42, verbosity=-1,
            )
            model.fit(train[feature_columns], train["load_target"], categorical_feature=categorical_features)
            result["lightgbm"] = model.predict(test[feature_columns])

            quantile_model = LGBMRegressor(
                objective="quantile", alpha=quantile, n_estimators=100,
                learning_rate=0.05, num_leaves=31, random_state=42, verbosity=-1,
            )
            quantile_model.fit(train[feature_columns], train["load_target"], categorical_feature=categorical_features)
            result[f"lightgbm_p{int(quantile * 100)}"] = quantile_model.predict(test[feature_columns])
        predictions.append(result)

    validation_predictions = pd.concat(predictions).dropna(subset=["baseline_lag672"])
    prediction_columns = {
        "lag-672 baseline": "baseline_lag672",
        "same-workday baseline": "baseline_same_workday",
    }
    if "lightgbm" in validation_predictions:
        quantile_label = f"P{int(quantile * 100)}"
        quantile_column = f"lightgbm_p{int(quantile * 100)}"
        prediction_columns.update({"LightGBM": "lightgbm", f"LightGBM {quantile_label}": quantile_column})

    rows = []
    groups = [("Overall", validation_predictions)]
    groups.extend(validation_predictions.groupby("day_type"))
    for day_type, group in groups:
        for scope, subset in [("all intervals", group), ("top 5% peaks", group[group["is_peak"]])]:
            for model_name, prediction_column in prediction_columns.items():
                scored = subset.dropna(subset=[prediction_column])
                if scored.empty:
                    continue
                actual, forecast = scored["actual"], scored[prediction_column]
                rows.append({
                    "day_type": day_type,
                    "scope": scope,
                    "model": model_name,
                    "MAE": mean_absolute_error(actual, forecast),
                    "RMSE": mean_squared_error(actual, forecast) ** 0.5,
                    "underforecast_rate": ((actual - forecast) > 0).mean(),
                })
    # Peak-shaving relevance: compare each model's daily maximum with the
    # actual daily maximum, restricted to genuine workdays.
    workday = validation_predictions[validation_predictions["is_workday"]]
    if not workday.empty:
        daily_peaks = workday.groupby(workday.index.normalize())
        for model_name, prediction_column in prediction_columns.items():
            daily = daily_peaks[["actual", prediction_column]].max().dropna()
            rows.append({
                "day_type": "Working days",
                "scope": "daily peak",
                "model": model_name,
                "MAE": mean_absolute_error(daily["actual"], daily[prediction_column]),
                "RMSE": mean_squared_error(daily["actual"], daily[prediction_column]) ** 0.5,
                "underforecast_rate": ((daily["actual"] - daily[prediction_column]) > 0).mean(),
            })
    return validation_predictions, pd.DataFrame(rows).set_index(["day_type", "scope", "model"]).sort_index()


def main(csv_path: str | Path = "load_timeseries_2025_case_study.csv") -> dict[str, object]:
    """Run the complete pipeline and return prepared data, predictions, and metrics."""
    df = load_and_prepare_data(csv_path)
    df, feature_columns, categorical_features = add_features(df)
    predictions, metrics = rolling_origin_validation(df, feature_columns, categorical_features)
    return {"data": df, "predictions": predictions, "metrics": metrics}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the BESS load forecasting pipeline")
    parser.add_argument("csv_path", nargs="?", default="load_timeseries_2025_case_study.csv")
    args = parser.parse_args()
    results = main(args.csv_path)
    print(results["metrics"])
