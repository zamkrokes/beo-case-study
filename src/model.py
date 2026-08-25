"""Forecast model constructors."""


def build_point_model():
    """Return the main LightGBM point-forecast model."""
    from lightgbm import LGBMRegressor

    return LGBMRegressor(
        n_estimators=100,
        learning_rate=0.05,
        num_leaves=31,
        random_state=42,
        verbosity=-1,
    )


def build_quantile_model(alpha=0.95):
    """Return a conservative LightGBM quantile model."""
    from lightgbm import LGBMRegressor

    return LGBMRegressor(
        objective="quantile",
        alpha=alpha,
        n_estimators=100,
        learning_rate=0.05,
        num_leaves=31,
        random_state=42,
        verbosity=-1,
    )


def build_p80_model():
    """Return the P80 LightGBM quantile model."""
    return build_quantile_model(alpha=0.80)


def build_p95_model():
    """Return the P95 LightGBM quantile model used in the presentation."""
    return build_quantile_model(alpha=0.95)
