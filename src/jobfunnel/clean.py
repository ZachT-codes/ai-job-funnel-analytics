"""Cleaning: the unglamorous step that decides whether the model is honest."""

from __future__ import annotations

import logging

import pandas as pd

log = logging.getLogger(__name__)

TEXT_COLUMNS = ["company", "role", "role_type", "source", "resume_version", "stage"]


def clean_applications(df: pd.DataFrame) -> pd.DataFrame:
    """Standardize types, fix mess, and drop rows we cannot trust."""
    cleaned = df.copy()

    for col in TEXT_COLUMNS:
        if col in cleaned.columns:
            cleaned[col] = cleaned[col].astype("string").str.strip()

    cleaned["applied_date"] = pd.to_datetime(cleaned["applied_date"], errors="coerce")
    cleaned["pay_usd_hr"] = pd.to_numeric(cleaned["pay_usd_hr"], errors="coerce")
    for col in ("is_remote", "got_response"):
        cleaned[col] = (
            pd.to_numeric(cleaned[col], errors="coerce").fillna(0).astype(int).clip(0, 1)
        )

    before = len(cleaned)
    cleaned = cleaned.drop_duplicates(subset=["application_id"], keep="first")
    dropped_dupes = before - len(cleaned)

    # Median imputation is a defensible default for a single numeric column.
    median_pay = cleaned["pay_usd_hr"].median()
    cleaned["pay_usd_hr"] = cleaned["pay_usd_hr"].fillna(median_pay)

    required = cleaned[["application_id", "company", "role", "applied_date"]].notna().all(
        axis=1
    )
    cleaned = cleaned[required].reset_index(drop=True)

    log.info(
        "Cleaned %d -> %d rows (dupes dropped: %d, median pay: %.2f)",
        before,
        len(cleaned),
        dropped_dupes,
        float(median_pay),
    )
    return cleaned
