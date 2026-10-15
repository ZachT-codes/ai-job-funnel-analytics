"""Ingest raw application data with light schema validation."""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

log = logging.getLogger(__name__)

REQUIRED_COLUMNS = {
    "application_id",
    "company",
    "role",
    "role_type",
    "source",
    "is_remote",
    "pay_usd_hr",
    "applied_date",
    "resume_version",
    "got_response",
}


def load_applications(csv_path: Path) -> pd.DataFrame:
    """Read the raw CSV and fail fast if the schema is wrong."""
    if not csv_path.exists():
        raise FileNotFoundError(f"Raw CSV not found: {csv_path}")

    df = pd.read_csv(csv_path, parse_dates=["applied_date"])
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    log.info("Loaded %d rows from %s", len(df), csv_path)
    return df
