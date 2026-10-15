"""Response prediction: which applications are most likely to get a reply."""

from __future__ import annotations

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split


def train_response_model(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.25,
    random_state: int = 42,
) -> dict:
    """Train a logistic regression and report honest holdout metrics."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    proba = model.predict_proba(X_test)[:, 1]
    pred = (proba >= 0.5).astype(int)

    metrics = {
        "accuracy": round(float(accuracy_score(y_test, pred)), 3),
        "precision": round(float(precision_score(y_test, pred, zero_division=0)), 3),
        "recall": round(float(recall_score(y_test, pred, zero_division=0)), 3),
        "f1": round(float(f1_score(y_test, pred, zero_division=0)), 3),
        "roc_auc": round(float(roc_auc_score(y_test, proba)), 3),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
    }

    importance = (
        pd.DataFrame({"feature": X.columns, "coefficient": model.coef_[0]})
        .assign(abs_coef=lambda d: d["coefficient"].abs())
        .sort_values("abs_coef", ascending=False)
        .drop(columns="abs_coef")
    )
    return {"model": model, "metrics": metrics, "feature_importance": importance}
