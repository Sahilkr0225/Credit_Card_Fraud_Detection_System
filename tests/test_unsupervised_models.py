"""
Unit tests for unsupervised anomaly detection (src/models/train_unsupervised.py).
"""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path
import joblib

from src.data.load_data import EXPECTED_FEATURE_COLUMNS
from src.models.train_unsupervised import (
    build_isolation_forest,
    train_and_evaluate_isolation_forest,
    save_unsupervised_artifacts,
)


@pytest.fixture
def mock_feature_data():
    """Create synthetic dataset for anomaly detection testing."""
    np.random.seed(42)
    n_samples = 300
    data = {col: np.random.randn(n_samples) for col in EXPECTED_FEATURE_COLUMNS}
    df = pd.DataFrame(data)

    # 10 synthetic anomalies
    labels = np.zeros(n_samples, dtype=int)
    labels[-10:] = 1
    # Distort last 10 samples
    df.iloc[-10:, :5] = 10.0

    return {
        "X_train": df.iloc[:200],
        "X_test": df.iloc[200:],
        "y_test": pd.Series(labels[200:]),
    }


def test_build_isolation_forest():
    """Verify Isolation Forest hyperparameter configuration."""
    model = build_isolation_forest(contamination=0.01, n_estimators=50, random_state=42)
    assert model.contamination == 0.01
    assert model.n_estimators == 50
    assert model.random_state == 42


def test_train_and_evaluate_isolation_forest(mock_feature_data):
    """Verify unsupervised fit and evaluation metrics calculation."""
    model, metrics = train_and_evaluate_isolation_forest(
        X_train=mock_feature_data["X_train"],
        X_test=mock_feature_data["X_test"],
        y_test=mock_feature_data["y_test"],
        contamination=0.05,
        random_state=42,
    )

    assert model is not None
    assert "test_roc_auc" in metrics
    assert "test_pr_auc" in metrics
    assert 0.0 <= metrics["test_roc_auc"] <= 1.0

    # Test prediction methods
    scores = model.decision_function(mock_feature_data["X_test"])
    preds = model.predict(mock_feature_data["X_test"])
    assert len(scores) == len(mock_feature_data["X_test"])
    assert set(preds).issubset({-1, 1})


def test_save_and_load_unsupervised_artifacts(tmp_path, mock_feature_data):
    """Verify joblib serialization of Isolation Forest."""
    model = build_isolation_forest(random_state=42)
    model.fit(mock_feature_data["X_train"])
    metrics = {"test_metric": 0.88}

    paths = save_unsupervised_artifacts(model, metrics, output_dir=tmp_path)
    assert Path(paths["model_path"]).exists()
    assert Path(paths["summary_path"]).exists()

    loaded = joblib.load(paths["model_path"])
    orig_scores = model.decision_function(mock_feature_data["X_test"])
    loaded_scores = loaded.decision_function(mock_feature_data["X_test"])
    np.testing.assert_array_almost_equal(orig_scores, loaded_scores)
