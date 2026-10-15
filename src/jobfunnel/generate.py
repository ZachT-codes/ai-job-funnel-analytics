"""Create a realistic synthetic job-application dataset for the pipeline."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import Settings


ROLE_TYPES = [
    "AI Trainer",
    "Data Labeling",
    "Customer Support",
    "Software Dev",
    "QA Analyst",
    "Operations",
]
SOURCES = ["LinkedIn", "Indeed", "Company Site", "Referral", "Gig Platform"]
COMPANIES = [
    "Northwind AI",
    "Bluefin Labs",
    "Cedar Analytics",
    "Driftwell",
    "Foxglove Data",
    "Harborlight",
    "Kestrel Systems",
    "Lumen Works",
]


def _response_probability(df: pd.DataFrame) -> np.ndarray:
    """Hand-tuned probabilities so the data has real, learnable signal."""
    base = 0.20
    p = np.full(len(df), base)
    p += np.where(df["source"] == "Referral", 0.35, 0.0)
    p += np.where(df["source"] == "Gig Platform", 0.15, 0.0)
    p += np.where(df["is_remote"] == 1, 0.08, 0.0)
    p += np.where(df["resume_version"] == "B", 0.10, 0.0)
    p += np.where(df["role_type"].isin(["AI Trainer", "Data Labeling"]), 0.08, 0.0)
    # Very low pay looks spammy; very high pay is competitive.
    p += np.where(df["pay_usd_hr"] < 14, -0.08, 0.0)
    p += np.where(df["pay_usd_hr"] > 45, -0.05, 0.0)
    return np.clip(p, 0.02, 0.90)


def generate_sample_data(n_rows: int = 400, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n = n_rows

    df = pd.DataFrame(
        {
            "application_id": [f"APP-{i:05d}" for i in range(1, n + 1)],
            "company": rng.choice(COMPANIES, size=n),
            "role": rng.choice(
                ["Analyst", "Specialist", "Associate", "Engineer", "Reviewer"], size=n
            ),
            "role_type": rng.choice(ROLE_TYPES, size=n, p=[0.25, 0.2, 0.2, 0.15, 0.1, 0.1]),
            "source": rng.choice(
                SOURCES, size=n, p=[0.3, 0.25, 0.15, 0.1, 0.2]
            ),
            "is_remote": rng.choice([0, 1], size=n, p=[0.35, 0.65]),
            "pay_usd_hr": np.round(rng.normal(28, 9, size=n), 2),
            "resume_version": rng.choice(["A", "B"], size=n, p=[0.5, 0.5]),
        }
    )
    df["pay_usd_hr"] = df["pay_usd_hr"].clip(lower=8)

    start = np.datetime64("2026-09-01")
    df["applied_date"] = start + rng.integers(0, 44, size=n).astype("timedelta64[D]")

    probs = _response_probability(df)
    df["got_response"] = (rng.random(n) < probs).astype(int)

    days = rng.integers(1, 15, size=n).astype(float)
    df["days_to_response"] = np.where(df["got_response"] == 1, days, np.nan)

    stages = rng.choice(
        ["screening", "interview", "offer", "rejected"],
        size=n,
        p=[0.45, 0.3, 0.1, 0.15],
    )
    df["stage"] = np.where(df["got_response"] == 1, stages, "applied")

    # A little mess, on purpose: missing pay, a duplicate, messy whitespace.
    df.loc[rng.choice(n, size=8, replace=False), "pay_usd_hr"] = np.nan
    df.loc[rng.choice(n, size=5, replace=False), "company"] = (
        "  " + df["company"].sample(5, random_state=seed).values + " "
    )
    df = pd.concat([df, df.iloc[[0]]], ignore_index=True)
    return df.reset_index(drop=True)


def write_sample_csv(settings: Settings) -> None:
    df = generate_sample_data(n_rows=settings.sample_rows, seed=settings.random_state)
    df.to_csv(settings.raw_csv, index=False)
    print(f"Wrote {len(df)} rows to {settings.raw_csv}")
