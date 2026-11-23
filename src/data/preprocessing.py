"""
Data Preprocessing and Pipeline Module for Credit Card Fraud Detection.

This module provides leakage-free data splitting, robust feature scaling,
and class imbalance handling (class weighting and train-only SMOTE).
"""

from typing import Dict, List, Optional, Tuple, Union
import logging
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.utils.class_weight import compute_class_weight
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

from src.data.load_data import (
    EXPECTED_FEATURE_COLUMNS,
    EXPECTED_TARGET_COLUMN,
    validate_schema,
)

logger = logging.getLogger(__name__)

# Features requiring scaling (V1-V28 are already standardized PCA components)
COLUMNS_TO_SCALE: List[str] = ["Time", "Amount"]


def separate_features_and_target(
    df: pd.DataFrame,
    target_col: str = EXPECTED_TARGET_COLUMN,
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Separate input DataFrame into features matrix X and target vector y.

    Parameters
    ----------
    df : pd.DataFrame
        Input transaction dataset.
    target_col : str, default='Class'
        Name of target column.

    Returns
    -------
    Tuple[pd.DataFrame, pd.Series]
        (X, y)
    """
    validate_schema(df)
    X = df.drop(columns=[target_col]).copy()
    y = df[target_col].copy()
    return X, y


def split_data(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Perform stratified train/test split preserving the true fraud ratio in both sets.

    Parameters
    ----------
    X : pd.DataFrame
        Features matrix.
    y : pd.Series
        Target vector.
    test_size : float, default=0.2
        Fraction of transactions reserved for evaluation.
    random_state : int, default=42
        Reproducibility seed.

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]
        (X_train, X_test, y_train, y_test)
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
        shuffle=True,
    )

    logger.info(
        f"Stratified split complete: "
        f"Train={len(X_train):,} ({y_train.sum():,} fraud, {y_train.mean()*100:.3f}%), "
        f"Test={len(X_test):,} ({y_test.sum():,} fraud, {y_test.mean()*100:.3f}%)"
    )
    return X_train, X_test, y_train, y_test


def scale_features(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    columns_to_scale: Optional[List[str]] = None,
    scaler_type: str = "robust",
) -> Tuple[pd.DataFrame, pd.DataFrame, Union[RobustScaler, StandardScaler]]:
    """
    Fit scaler exclusively on X_train and transform both X_train and X_test.
    Prevents data leakage.

    Parameters
    ----------
    X_train : pd.DataFrame
        Training features.
    X_test : pd.DataFrame
        Testing features.
    columns_to_scale : Optional[List[str]], default=['Time', 'Amount']
        Columns to scale.
    scaler_type : str, default='robust'
        'robust' (RobustScaler) or 'standard' (StandardScaler).
        RobustScaler is strongly recommended due to extreme transaction amount outliers.

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame, Scaler]
        (X_train_scaled, X_test_scaled, fitted_scaler)
    """
    if columns_to_scale is None:
        columns_to_scale = COLUMNS_TO_SCALE

    if scaler_type.lower() == "robust":
        scaler = RobustScaler()
    elif scaler_type.lower() == "standard":
        scaler = StandardScaler()
    else:
        raise ValueError(f"Unsupported scaler_type '{scaler_type}'. Choose 'robust' or 'standard'.")

    X_train_scaled = X_train.copy()
    X_test_scaled = X_test.copy()

    # CRITICAL: Fit ONLY on training data to prevent data leakage
    scaler.fit(X_train[columns_to_scale])

    X_train_scaled[columns_to_scale] = scaler.transform(X_train[columns_to_scale])
    X_test_scaled[columns_to_scale] = scaler.transform(X_test[columns_to_scale])

    return X_train_scaled, X_test_scaled, scaler


def apply_smote_train_only(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    sampling_strategy: Union[float, str] = "auto",
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Apply Synthetic Minority Over-sampling Technique (SMOTE) strictly to training data.
    NEVER apply to test data.

    Parameters
    ----------
    X_train : pd.DataFrame
        Training features.
    y_train : pd.Series
        Training labels.
    sampling_strategy : Union[float, str], default='auto'
        Target ratio of minority to majority class.
    random_state : int, default=42
        Reproducibility seed.

    Returns
    -------
    Tuple[pd.DataFrame, pd.Series]
        (X_train_resampled, y_train_resampled)
    """
    smote = SMOTE(sampling_strategy=sampling_strategy, random_state=random_state)
    X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

    logger.info(
        f"SMOTE applied to train set only: "
        f"Before: {len(y_train):,} (Fraud: {y_train.sum():,}) -> "
        f"After: {len(y_resampled):,} (Fraud: {y_resampled.sum():,})"
    )
    return pd.DataFrame(X_resampled, columns=X_train.columns), pd.Series(y_resampled, name=y_train.name)


def compute_balanced_weights(y_train: pd.Series) -> Dict[int, float]:
    """
    Compute inverse class weights for cost-sensitive learning.

    Parameters
    ----------
    y_train : pd.Series
        Training labels.

    Returns
    -------
    Dict[int, float]
        Mapping from class label to calculated weight.
    """
    classes = np.unique(y_train)
    weights = compute_class_weight(class_weight="balanced", classes=classes, y=y_train)
    return {cls: round(float(w), 4) for cls, w in zip(classes, weights)}


def build_preprocessed_pipeline(
    df: pd.DataFrame,
    test_size: float = 0.2,
    scaler_type: str = "robust",
    random_state: int = 42,
) -> Dict[str, Union[pd.DataFrame, pd.Series, dict, RobustScaler]]:
    """
    End-to-end reproducible preprocessing orchestration.

    Parameters
    ----------
    df : pd.DataFrame
        Raw dataset.
    test_size : float, default=0.2
        Holdout test set ratio.
    scaler_type : str, default='robust'
        Scaler selection ('robust' or 'standard').
    random_state : int, default=42
        Seed for reproducibility.

    Returns
    -------
    dict
        Dictionary containing X_train, X_test, y_train, y_test, scaler, and class weights.
    """
    X, y = separate_features_and_target(df)
    X_train, X_test, y_train, y_test = split_data(
        X, y, test_size=test_size, random_state=random_state
    )

    X_train_scaled, X_test_scaled, scaler = scale_features(
        X_train, X_test, columns_to_scale=COLUMNS_TO_SCALE, scaler_type=scaler_type
    )

    class_weights = compute_balanced_weights(y_train)

    return {
        "X_train": X_train_scaled,
        "X_test": X_test_scaled,
        "y_train": y_train,
        "y_test": y_test,
        "scaler": scaler,
        "class_weights": class_weights,
        "random_state": random_state,
    }

