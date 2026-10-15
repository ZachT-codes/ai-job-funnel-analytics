"""Tests for feature engineering."""

from jobfunnel.clean import clean_applications
from jobfunnel.features import build_features


def test_build_features_returns_clean_matrix(raw_df):
    cleaned = clean_applications(raw_df)
    X, y, names = build_features(cleaned)
    assert len(X) == len(y) == len(cleaned)
    assert list(X.columns) == names
    assert X.isna().sum().sum() == 0
    assert set(y.unique()) <= {0, 1}


def test_build_features_excludes_post_response_columns(raw_df):
    """days_to_response and stage only exist after a reply: using them would leak."""
    cleaned = clean_applications(raw_df)
    X, _, _ = build_features(cleaned)
    assert "days_to_response" not in X.columns
    assert not any(c.startswith("stage") for c in X.columns)
