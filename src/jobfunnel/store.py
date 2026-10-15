"""Persist cleaned data to SQLite: a tiny warehouse for the analysis layer."""

from __future__ import annotations

import logging
import sqlite3
from pathlib import Path

import pandas as pd

log = logging.getLogger(__name__)


def load_to_sqlite(df: pd.DataFrame, db_path: Path, table: str = "applications") -> None:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(db_path) as conn:
        df.to_sql(table, conn, if_exists="replace", index=False)
    log.info("Wrote %d rows to %s.%s", len(df), db_path, table)


def read_from_sqlite(db_path: Path, table: str = "applications") -> pd.DataFrame:
    with sqlite3.connect(db_path) as conn:
        df = pd.read_sql(f"SELECT * FROM {table}", conn, parse_dates=["applied_date"])
    log.info("Read %d rows from %s.%s", len(df), db_path, table)
    return df
