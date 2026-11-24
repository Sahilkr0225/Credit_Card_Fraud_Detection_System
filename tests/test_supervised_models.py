"""
Unit tests for supervised model training and evaluation (src/models/train_supervised.py).
"""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path
import joblib
from sklearn.preprocessing import RobustScaler

from src.data.load_data import EXPECTED_FEATURE_COLUMNS, EXPECTED_TARGET_COLUMN
from src.models.train_supervised import (
    get_model_candidates,
    evaluate_predictions,
    cross_validate_candidate,
    train_and_benchmark,
    save_artifacts,
)


@pytest.fixture
def mock_dataset():
    """Create reproducible synthetic dataset with sufficient minority samples for CV SMOTE."""
    np.random.seed(42)
    n_samples = 500
    n_fraud = 40  # Provides ~32 fraud samples in train set, well above SMOTE k_neighbors=5

    data = {col: np.random.randn(n_samples) for col in EXPECTED_FEATURE_COLUMNS}
    labels = np.zeros(n_samples, dtype=int)
    fraud_indices = np.random.choice(n_samples, size=n_fraud, replace=False)
    labels[fraud_indices] = 1

    df = pd.DataFrame(data)
    X = df.copy()
    y = pd.Series(labels, name=EXPECTED_TARGET_COLUMN)

    # Stratified-like 80/20 train/test split
    split_idx = int(0.8 * n_samples)
    return {
        "X_train": X.iloc[:split_idx],
        "X_test": X.iloc[split_idx:],
        "y_train": y.iloc[:split_idx],
        "y_test": y.iloc[split_idx:],
    }


def test_get_model_candidates():
    """Verify that candidate models are defined with correct identifiers."""
    candidates = get_model_candidates(random_state=42)
    expected_names = [
        "lr_class_weighted",
        "lr_smote",
        "rf_class_weighted",
        "rf_smote",
    ]
    for name in expected_names:
        assert name in candidates, f"Missing candidate: {name}"


def test_evaluate_predictions_metrics():
    """Verify evaluation metric calculations and confusion matrix bounds."""
    y_true = np.array([0, 0, 0, 0, 1, 1, 0, 1])
    y_pred = np.array([0, 0, 0, 1, 1, 1, 0, 0])
    y_prob = np.array([0.1, 0.2, 0.05, 0.6, 0.9, 0.85, 0.15, 0.4])

    metrics = evaluate_predictions(y_true, y_pred, y_prob)

    assert 0.0 <= metrics["pr_auc"] <= 1.0
    assert 0.0 <= metrics["roc_auc"] <= 1.0
    assert 0.0 <= metrics["recall"] <= 1.0
    assert 0.0 <= metrics["precision"] <= 1.0
    assert 0.0 <= metrics["f1_score"] <= 1.0

    cm = metrics["confusion_matrix"]
    assert cm["true_positives"] == 2
    assert cm["false_positives"] == 1
    assert cm["false_negatives"] == 1
    assert cm["true_negatives"] == 4


def test_cross_validate_candidate_leakage_free(mock_dataset):
    """
    Verify cross validation executes smoothly with SMOTE inside ImbPipeline.
    No exceptions or NaNs.
    """
    candidates = get_model_candidates(random_state=42)
    lr_smote_pipe = candidates["lr_smote"]

    cv_results = cross_validate_candidate(
        model=lr_smote_pipe,
        X=mock_dataset["X_train"],
        y=mock_dataset["y_train"],
        cv_splits=2,
        random_state=42,
    )

    assert "cv_pr_auc_mean" in cv_results
    assert "cv_recall_mean" in cv_results
    assert 0.0 <= cv_results["cv_pr_auc_mean"] <= 1.0


def test_train_and_benchmark_selects_best(mock_dataset):
    """Verify benchmarking completes and returns best model."""
    results, best_name, best_model = train_and_benchmark(
        data_dict=mock_dataset,
        cv_splits=2,
        random_state=42,
    )

    assert best_name in results
    assert best_model is not None
    # Check that predictions can be generated
    preds = best_model.predict(mock_dataset["X_test"])
    probs = best_model.predict_proba(mock_dataset["X_test"])
    assert len(preds) == len(mock_dataset["X_test"])
    assert probs.shape == (len(mock_dataset["X_test"]), 2)


def test_save_and_load_artifacts(tmp_path, mock_dataset):
    """Verify serialization and deserialization of model and metadata."""
    candidates = get_model_candidates(random_state=42)
    rf = candidates["rf_class_weighted"]
    rf.fit(mock_dataset["X_train"], mock_dataset["y_train"])

    scaler = RobustScaler()
    scaler.fit(mock_dataset["X_train"][["Time", "Amount"]])
    summary = {"test_metric": 0.95}

    saved = save_artifacts(
        model=rf,
        scaler=scaler,
        benchmark_summary=summary,
        best_model_name="rf_class_weighted",
        output_dir=tmp_path,
    )

    assert Path(saved["model_path"]).exists()
    assert Path(saved["scaler_path"]).exists()
    assert Path(saved["summary_path"]).exists()

    # Load and verify prediction equivalence
    loaded_model = joblib.load(saved["model_path"])
    original_preds = rf.predict(mock_dataset["X_test"])
    loaded_preds = loaded_model.predict(mock_dataset["X_test"])
    np.testing.assert_array_equal(original_preds, loaded_preds)
