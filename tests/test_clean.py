"""Tests for the cleaning step."""

import pandas as pd

from jobfunnel.clean import clean_applications


def test_clean_removes_duplicate_application_ids(raw_df: pd.DataFrame):
    cleaned = clean_applications(raw_df)
    assert cleaned["application_id"].is_unique


def test_clean_fills_missing_pay_with_median(raw_df: pd.DataFrame):
    assert raw_df["pay_usd_hr"].isna().any()
    deduped = raw_df.drop_duplicates(subset=["application_id"], keep="first")
    cleaned = clean_applications(raw_df)
    assert cleaned["pay_usd_hr"].notna().all()
    assert cleaned["pay_usd_hr"].median() == deduped["pay_usd_hr"].median()


def test_clean_standardizes_text_and_types(raw_df: pd.DataFrame):
    cleaned = clean_applications(raw_df)
    assert (cleaned["company"].str.strip() == cleaned["company"]).all()
    assert pd.api.types.is_datetime64_any_dtype(cleaned["applied_date"])
    assert set(cleaned["got_response"].unique()) <= {0, 1}
