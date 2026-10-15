"""Tests for the response-prediction model."""

from jobfunnel.clean import clean_applications
from jobfunnel.features import build_features
from jobfunnel.generate import generate_sample_data
from jobfunnel.model import train_response_model


def test_model_trains_and_reports_metrics():
    df = clean_applications(generate_sample_data(n_rows=300, seed=11))
    X, y, _ = build_features(df)
    results = train_response_model(X, y, test_size=0.25, random_state=42)

    for key in ("accuracy", "precision", "recall", "f1", "roc_auc"):
        assert key in results["metrics"]
        assert 0.0 <= results["metrics"][key] <= 1.0
    assert not results["feature_importance"].empty
    # Synthetic data has real signal: the model should beat chance.
    assert results["metrics"]["roc_auc"] > 0.55
