"""Forecast evaluation utilities."""

from __future__ import annotations

import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

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


