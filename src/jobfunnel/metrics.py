"""Descriptive analytics: the funnel numbers a job seeker actually acts on."""

from __future__ import annotations

import pandas as pd


def funnel_metrics(df: pd.DataFrame) -> dict:
    """Compute headline funnel stats from the cleaned applications table."""
    responders = df[df["got_response"] == 1]

    response_by_source = (
        df.groupby("source")["got_response"]
        .agg(applications="size", response_rate="mean")
        .sort_values("response_rate", ascending=False)
        .round(3)
    )
    response_by_role = (
        df.groupby("role_type")["got_response"]
        .agg(applications="size", response_rate="mean")
        .sort_values("response_rate", ascending=False)
        .round(3)
    )

    return {
        "total_applications": int(len(df)),
        "overall_response_rate": round(float(df["got_response"].mean()), 3),
        "median_days_to_response": (
            float(responders["days_to_response"].median())
            if len(responders)
            else None
        ),
        "stage_counts": df["stage"].value_counts().to_dict(),
        "response_by_source": response_by_source,
        "response_by_role": response_by_role,
    }
