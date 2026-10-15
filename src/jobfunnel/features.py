"""Feature engineering for the response-prediction model."""

from __future__ import annotations

import pandas as pd


def build_features(df: pd.DataFrame, target: str = "got_response"):
    """Turn the cleaned table into a model-ready X/y pair.

    Keeps it transparent on purpose: one-hot categoricals, day-of-week and
    month from the application date, no target leakage (days_to_response
    and stage are excluded because they only exist after a response).
    """
    feat = df.copy()
    feat["applied_dow"] = feat["applied_date"].dt.dayofweek
    feat["applied_month"] = feat["applied_date"].dt.month

    categorical = ["role_type", "source", "resume_version"]
    X = pd.get_dummies(
        feat[["is_remote", "pay_usd_hr", "applied_dow", "applied_month"] + categorical],
        columns=categorical,
        drop_first=True,
        dtype=float,
    )
    y = feat[target].astype(int)
    assert X.isna().sum().sum() == 0, "Features must not contain NaNs"
    return X, y, list(X.columns)
