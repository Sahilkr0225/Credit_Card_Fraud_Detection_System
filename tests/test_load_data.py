"""
Unit tests for data loading and validation module (src/data/load_data.py).
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from src.data.load_data import (
    load_raw_data,
    validate_schema,
    summarize_dataset,
    EXPECTED_COLUMNS,
    EXPECTED_FEATURE_COLUMNS,
    EXPECTED_TARGET_COLUMN,
)


@pytest.fixture
def valid_mock_df():
    """Create a minimal valid synthetic dataframe with correct schema."""
    n_rows = 100
    data = {col: np.random.randn(n_rows) for col in EXPECTED_FEATURE_COLUMNS}
    # 95 legitimate (0), 5 fraudulent (1)
    target = np.array([0] * 95 + [1] * 5)
    data[EXPECTED_TARGET_COLUMN] = target
    return pd.DataFrame(data)


def test_missing_file_raises_filenotfound():
    """Verify that a missing dataset file raises FileNotFoundError with helpful message."""
    with pytest.raises(FileNotFoundError) as exc_info:
        load_raw_data("data/raw/non_existent_file.csv")
    assert "Raw dataset not found" in str(exc_info.value)
    assert "https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud" in str(exc_info.value)


def test_validate_schema_success(valid_mock_df):
    """Verify that schema validation succeeds for a valid DataFrame."""
    assert validate_schema(valid_mock_df) is True


def test_validate_schema_missing_column(valid_mock_df):
    """Verify that missing required columns triggers ValueError."""
    corrupted_df = valid_mock_df.drop(columns=["V1", "Amount"])
    with pytest.raises(ValueError) as exc_info:
        validate_schema(corrupted_df)
    assert "Missing expected columns" in str(exc_info.value)


def test_validate_schema_unexpected_columns(valid_mock_df):
    """Verify that unexpected extra columns triggers ValueError."""
    corrupted_df = valid_mock_df.copy()
    corrupted_df["Unexpected_ID"] = "ID_12345"
    with pytest.raises(ValueError) as exc_info:
        validate_schema(corrupted_df)
    assert "Unexpected columns found" in str(exc_info.value)
    assert "Unexpected_ID" in str(exc_info.value)


def test_validate_schema_missing_feature_values(valid_mock_df):
    """Verify that missing/NaN values in feature columns triggers ValueError."""
    corrupted_df = valid_mock_df.copy()
    corrupted_df.loc[0, "V1"] = np.nan
    corrupted_df.loc[5, "Amount"] = np.nan
    with pytest.raises(ValueError) as exc_info:
        validate_schema(corrupted_df)
    assert "Feature columns contain missing/null values" in str(exc_info.value)
    assert "V1" in str(exc_info.value)
    assert "Amount" in str(exc_info.value)


def test_validate_schema_missing_target_values(valid_mock_df):
    """Verify that missing/NaN values in target column triggers ValueError."""
    corrupted_df = valid_mock_df.copy()
    corrupted_df.loc[2, EXPECTED_TARGET_COLUMN] = np.nan
    with pytest.raises(ValueError) as exc_info:
        validate_schema(corrupted_df)
    assert "Target column 'Class' contains 1 missing/null values" in str(exc_info.value)


def test_validate_schema_invalid_target(valid_mock_df):
    """Verify that target values other than 0 and 1 trigger ValueError."""
    corrupted_df = valid_mock_df.copy()
    corrupted_df.loc[0, EXPECTED_TARGET_COLUMN] = 99
    with pytest.raises(ValueError) as exc_info:
        validate_schema(corrupted_df)
    assert "contains invalid values" in str(exc_info.value)


def test_summarize_dataset(valid_mock_df):
    """Verify statistical summary calculations."""
    summary = summarize_dataset(valid_mock_df)
    assert summary["total_records"] == 100
    assert summary["legitimate_transactions"] == 95
    assert summary["fraudulent_transactions"] == 5
    assert summary["fraud_percentage"] == 5.0
    assert summary["total_missing_values"] == 0
    assert summary["total_duplicate_rows"] == 0
