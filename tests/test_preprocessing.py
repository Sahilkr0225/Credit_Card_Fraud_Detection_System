"""
Unit tests for data preprocessing module (src/data/preprocessing.py).
"""

import pytest
import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler

from src.data.load_data import EXPECTED_FEATURE_COLUMNS, EXPECTED_TARGET_COLUMN
from src.data.preprocessing import (
    separate_features_and_target,
    split_data,
    scale_features,
    apply_smote_train_only,
    compute_balanced_weights,
    build_preprocessed_pipeline,
    COLUMNS_TO_SCALE,
)


@pytest.fixture
def synthetic_dataset():
    """Create a reproducible synthetic dataset resembling the credit card schema."""
    np.random.seed(42)
    n_samples = 1000
    n_fraud = 10  # 1% fraud rate for unit test speed

    data = {col: np.random.randn(n_samples) for col in EXPECTED_FEATURE_COLUMNS}
    # Distinct ranges for Time and Amount to verify scaling
    data["Time"] = np.random.uniform(0, 172800, n_samples)
    data["Amount"] = np.random.exponential(scale=50, size=n_samples)

    labels = np.zeros(n_samples, dtype=int)
    fraud_indices = np.random.choice(n_samples, size=n_fraud, replace=False)
    labels[fraud_indices] = 1
    data[EXPECTED_TARGET_COLUMN] = labels

    return pd.DataFrame(data)


def test_separate_features_and_target(synthetic_dataset):
    """Verify target separation and feature dimension accuracy."""
    X, y = separate_features_and_target(synthetic_dataset)

    assert X.shape[1] == 30  # 28 PCA + Time + Amount
    assert EXPECTED_TARGET_COLUMN not in X.columns
    assert len(y) == len(synthetic_dataset)
    assert set(y.unique()) == {0, 1}


def test_stratified_split_preserves_ratio(synthetic_dataset):
    """Verify that train/test split maintains equal proportion of fraud cases."""
    X, y = separate_features_and_target(synthetic_dataset)
    X_train, X_test, y_train, y_test = split_data(X, y, test_size=0.2, random_state=42)

    total_ratio = y.mean()
    train_ratio = y_train.mean()
    test_ratio = y_test.mean()

    # Ratios should match within minimal rounding tolerance of stratified split
    assert pytest.approx(train_ratio, rel=1e-1) == total_ratio
    assert pytest.approx(test_ratio, rel=1e-1) == total_ratio
    assert len(X_train) + len(X_test) == len(synthetic_dataset)


def test_no_data_leakage_in_scaling(synthetic_dataset):
    """
    CRITICAL: Verify scaler is fit ONLY on X_train.
    Transforming test data with train parameters must NOT alter scaler statistics.
    """
    X, y = separate_features_and_target(synthetic_dataset)
    X_train, X_test, _, _ = split_data(X, y, test_size=0.2, random_state=42)

    X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test, scaler_type="robust")

    # Scaler center_ and scale_ should strictly match X_train medians and IQRs
    for idx, col in enumerate(COLUMNS_TO_SCALE):
        expected_median = np.median(X_train[col])
        assert pytest.approx(scaler.center_[idx], rel=1e-3) == expected_median

    # Scaled training medians for scaled columns must be approximately 0
    for col in COLUMNS_TO_SCALE:
        assert pytest.approx(np.median(X_train_scaled[col]), abs=1e-5) == 0.0

    # Ensure no NaNs produced
    assert not X_train_scaled.isnull().values.any()
    assert not X_test_scaled.isnull().values.any()


def test_smote_applied_only_to_train(synthetic_dataset):
    """Verify SMOTE resamples training data and does NOT touch testing data."""
    X, y = separate_features_and_target(synthetic_dataset)
    X_train, X_test, y_train, y_test = split_data(X, y, test_size=0.2, random_state=42)

    original_test_len = len(y_test)
    original_test_fraud_count = y_test.sum()

    X_train_res, y_train_res = apply_smote_train_only(X_train, y_train, random_state=42)

    # Training data should be balanced (or resampled)
    assert y_train_res.sum() > y_train.sum()
    assert len(X_train_res) == len(y_train_res)

    # Test set must remain completely untouched
    assert len(y_test) == original_test_len
    assert y_test.sum() == original_test_fraud_count


def test_reproducibility(synthetic_dataset):
    """Verify deterministic execution with identical random_state."""
    pipeline_1 = build_preprocessed_pipeline(synthetic_dataset, random_state=42)
    pipeline_2 = build_preprocessed_pipeline(synthetic_dataset, random_state=42)

    pd.testing.assert_frame_equal(pipeline_1["X_train"], pipeline_2["X_train"])
    pd.testing.assert_frame_equal(pipeline_1["X_test"], pipeline_2["X_test"])
    pd.testing.assert_series_equal(pipeline_1["y_train"], pipeline_2["y_train"])
    pd.testing.assert_series_equal(pipeline_1["y_test"], pipeline_2["y_test"])


def test_balanced_class_weights(synthetic_dataset):
    """Verify that minority class receives proportionally higher weight."""
    _, y = separate_features_and_target(synthetic_dataset)
    weights = compute_balanced_weights(y)

    assert 0 in weights and 1 in weights
    # Minority weight must be much higher than majority weight
    assert weights[1] > weights[0]

