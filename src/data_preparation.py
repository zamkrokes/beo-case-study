"""Data loading, cleaning, and causal feature engineering."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


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

