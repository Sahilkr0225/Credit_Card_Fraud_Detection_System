"""
Data Loading and Validation Module for Credit Card Fraud Detection.

This module provides functions to safely load and validate the Kaggle
European Credit Card Fraud Detection dataset without mutating the raw files.
"""

from pathlib import Path
from typing import Dict, List, Optional, Union
import logging
import pandas as pd

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Expected dataset schema per Kaggle specification
EXPECTED_FEATURE_COLUMNS: List[str] = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount"]
EXPECTED_TARGET_COLUMN: str = "Class"
EXPECTED_COLUMNS: List[str] = EXPECTED_FEATURE_COLUMNS + [EXPECTED_TARGET_COLUMN]
EXPECTED_COLUMN_COUNT: int = 31


def validate_schema(df: pd.DataFrame) -> bool:
    """
    Validate that the DataFrame contains all expected columns, no unexpected columns,
    no missing values in features or target, and valid binary target values.

    Parameters
    ----------
    df : pd.DataFrame
        The loaded dataframe to validate.

    Returns
    -------
    bool
        True if validation passes.

    Raises
    ------
    ValueError
        If required columns are missing, unexpected columns exist, missing values are
        present, or target values are not binary {0, 1}.
    """
    # 1. Reject missing expected columns
    missing_columns = set(EXPECTED_COLUMNS) - set(df.columns)
    if missing_columns:
        raise ValueError(
            f"Dataset schema validation failed. Missing expected columns: {sorted(list(missing_columns))}"
        )

    # 2. Reject unexpected extra columns
    unexpected_columns = set(df.columns) - set(EXPECTED_COLUMNS)
    if unexpected_columns:
        raise ValueError(
            f"Dataset schema validation failed. Unexpected columns found: {sorted(list(unexpected_columns))}. "
            f"Expected exactly the {EXPECTED_COLUMN_COUNT} certified columns."
        )

    # 3. Reject missing values in target column
    if df[EXPECTED_TARGET_COLUMN].isnull().any():
        missing_target_count = int(df[EXPECTED_TARGET_COLUMN].isnull().sum())
        raise ValueError(
            f"Target column '{EXPECTED_TARGET_COLUMN}' contains {missing_target_count} missing/null values."
        )

    # 4. Validate target values are strictly binary {0, 1}
    unique_targets = set(df[EXPECTED_TARGET_COLUMN].unique())
    if not unique_targets.issubset({0, 1}):
        raise ValueError(
            f"Target column '{EXPECTED_TARGET_COLUMN}' contains invalid values: {unique_targets}. "
            f"Expected binary {0, 1}."
        )

    # 5. Reject missing values in feature columns
    missing_features = df[EXPECTED_FEATURE_COLUMNS].isnull().sum()
    cols_with_missing = missing_features[missing_features > 0].to_dict()
    if cols_with_missing:
        formatted_missing = {col: int(cnt) for col, cnt in cols_with_missing.items()}
        raise ValueError(
            f"Feature columns contain missing/null values: {formatted_missing}"
        )

    return True


def load_raw_data(
    filepath: Union[str, Path] = "data/raw/creditcard.csv"
) -> pd.DataFrame:
    """
    Safely load the raw credit card transaction dataset from CSV.

    Parameters
    ----------
    filepath : Union[str, Path], default="data/raw/creditcard.csv"
        Path to the raw CSV file.

    Returns
    -------
    pd.DataFrame
        Loaded and schema-validated DataFrame (read-only copy).

    Raises
    ------
    FileNotFoundError
        If the raw dataset does not exist at the specified path.
    ValueError
        If the dataset fails schema validation.
    """
    file_path = Path(filepath)

    if not file_path.exists():
        abs_path = file_path.resolve()
        raise FileNotFoundError(
            f"\n[ERROR] Raw dataset not found at: {abs_path}\n"
            f"Please obtain the European Credit Card Fraud Detection dataset from Kaggle:\n"
            f"  https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud\n"
            f"Extract the file 'creditcard.csv' and place it at:\n"
            f"  {abs_path}\n"
        )

    logger.info(f"Loading raw data from: {file_path}")
    df = pd.read_csv(file_path)

    # Validate schema
    validate_schema(df)

    logger.info(
        f"Data successfully loaded. Dimensions: {df.shape[0]:,} rows x {df.shape[1]} columns."
    )
    return df


def summarize_dataset(df: pd.DataFrame) -> Dict[str, Union[int, float, dict]]:
    """
    Generate basic structural and statistical summary of the raw dataset.

    Parameters
    ----------
    df : pd.DataFrame
        The credit card transaction dataframe.

    Returns
    -------
    dict
        Dictionary containing dimension, class counts, fraud ratio, and missing values.
    """
    total_records = len(df)
    class_counts = df[EXPECTED_TARGET_COLUMN].value_counts().to_dict()
    legit_count = class_counts.get(0, 0)
    fraud_count = class_counts.get(1, 0)
    fraud_pct = (fraud_count / total_records * 100) if total_records > 0 else 0.0
    missing_cells = int(df.isnull().sum().sum())
    duplicate_rows = int(df.duplicated().sum())

    summary = {
        "total_records": total_records,
        "total_columns": df.shape[1],
        "legitimate_transactions": legit_count,
        "fraudulent_transactions": fraud_count,
        "fraud_percentage": round(fraud_pct, 4),
        "total_missing_values": missing_cells,
        "total_duplicate_rows": duplicate_rows,
        "memory_usage_mb": round(df.memory_usage(deep=True).sum() / (1024 * 1024), 2),
    }
    return summary


if __name__ == "__main__":
    default_path = Path("data/raw/creditcard.csv")
    print("\n--- Credit Card Fraud Detection: Data Loader ---")
    try:
        data = load_raw_data(default_path)
        stats = summarize_dataset(data)
        print("\nDataset Summary:")
        for key, val in stats.items():
            print(f"  {key}: {val}")
    except FileNotFoundError as err:
        print(err)
