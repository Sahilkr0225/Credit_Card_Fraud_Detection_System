"""
Supervised Fraud Classification Training & Evaluation Module.

Implements benchmark comparison between Logistic Regression and Random Forest
under cost-sensitive weighting and train-only SMOTE oversampling using
imbalanced-learn pipelines to strictly prevent cross-validation data leakage.
"""

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import json
import logging

# Ensure project root is in sys.path for direct script execution
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

from src.data.load_data import load_raw_data
from src.data.preprocessing import build_preprocessed_pipeline

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def get_model_candidates(
    random_state: int = 42,
    n_jobs: int = -1,
) -> Dict[str, Union[LogisticRegression, RandomForestClassifier, ImbPipeline]]:
    """
    Define supervised fraud classification candidates for benchmarking.

    Candidates:
    1. Logistic Regression (Cost-Sensitive / Balanced Class Weights)
    2. Logistic Regression (SMOTE Oversampling in ImbPipeline)
    3. Random Forest (Cost-Sensitive / Balanced Subsample Weights)
    4. Random Forest (SMOTE Oversampling in ImbPipeline)

    Parameters
    ----------
    random_state : int, default=42
        Reproducibility seed.
    n_jobs : int, default=-1
        Parallel CPU threads for ensemble estimators.

    Returns
    -------
    dict
        Dictionary of named model pipelines.
    """
    candidates = {
        "lr_class_weighted": LogisticRegression(
            class_weight="balanced",
            max_iter=1000,
            solver="lbfgs",
            random_state=random_state,
        ),
        "lr_smote": ImbPipeline([
            ("smote", SMOTE(sampling_strategy="auto", random_state=random_state)),
            ("classifier", LogisticRegression(max_iter=1000, solver="lbfgs", random_state=random_state)),
        ]),
        "rf_class_weighted": RandomForestClassifier(
            n_estimators=100,
            max_depth=12,
            min_samples_split=5,
            class_weight="balanced_subsample",
            random_state=random_state,
            n_jobs=n_jobs,
        ),
        "rf_smote": ImbPipeline([
            ("smote", SMOTE(sampling_strategy="auto", random_state=random_state)),
            ("classifier", RandomForestClassifier(
                n_estimators=100,
                max_depth=12,
                min_samples_split=5,
                random_state=random_state,
                n_jobs=n_jobs,
            )),
        ]),
    }
    return candidates


def evaluate_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
) -> Dict[str, Union[float, list]]:
    """
    Compute comprehensive fraud detection evaluation metrics.

    Parameters
    ----------
    y_true : np.ndarray
        Ground truth binary labels (0 or 1).
    y_pred : np.ndarray
        Predicted binary labels.
    y_prob : np.ndarray
        Predicted fraud probabilities (class 1).

    Returns
    -------
    dict
        Dictionary containing PR-AUC, ROC-AUC, Recall, Precision, F1, and Confusion Matrix.
    """
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()

    metrics = {
        "pr_auc": round(float(average_precision_score(y_true, y_prob)), 4),
        "roc_auc": round(float(roc_auc_score(y_true, y_prob)), 4),
        "recall": round(float(recall_score(y_true, y_pred, zero_division=0)), 4),
        "precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
        "f1_score": round(float(f1_score(y_true, y_pred, zero_division=0)), 4),
        "confusion_matrix": {
            "true_negatives": int(tn),
            "false_positives": int(fp),
            "false_negatives": int(fn),
            "true_positives": int(tp),
        },
    }
    return metrics


def cross_validate_candidate(
    model: Any,
    X: pd.DataFrame,
    y: pd.Series,
    cv_splits: int = 3,
    random_state: int = 42,
) -> Dict[str, float]:
    """
    Perform Stratified K-Fold Cross Validation.
    If using SMOTE, the ImbPipeline resamples ONLY the training fold, strictly
    preventing validation fold data leakage.

    Parameters
    ----------
    model : Any
        Estimator or ImbPipeline.
    X : pd.DataFrame
        Training features matrix.
    y : pd.Series
        Training labels.
    cv_splits : int, default=3
        Number of stratified folds.
    random_state : int, default=42
        Seed.

    Returns
    -------
    dict
        Mean and standard deviation of cross-validation PR-AUC and Recall.
    """
    skf = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=random_state)
    fold_pr_aucs = []
    fold_recalls = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        X_train_fold, X_val_fold = X.iloc[train_idx], X.iloc[val_idx]
        y_train_fold, y_val_fold = y.iloc[train_idx], y.iloc[val_idx]

        model.fit(X_train_fold, y_train_fold)

        # Probability of class 1
        y_val_prob = model.predict_proba(X_val_fold)[:, 1]
        y_val_pred = model.predict(X_val_fold)

        fold_pr_aucs.append(average_precision_score(y_val_fold, y_val_prob))
        fold_recalls.append(recall_score(y_val_fold, y_val_pred, zero_division=0))

    return {
        "cv_pr_auc_mean": round(float(np.mean(fold_pr_aucs)), 4),
        "cv_pr_auc_std": round(float(np.std(fold_pr_aucs)), 4),
        "cv_recall_mean": round(float(np.mean(fold_recalls)), 4),
        "cv_recall_std": round(float(np.std(fold_recalls)), 4),
    }


def train_and_benchmark(
    data_dict: Dict[str, Any],
    cv_splits: int = 3,
    random_state: int = 42,
) -> Tuple[Dict[str, Any], str, Any]:
    """
    Train and benchmark all candidates across Cross-Validation and Holdout Test Set.

    Parameters
    ----------
    data_dict : dict
        Output from `build_preprocessed_pipeline`.
    cv_splits : int, default=3
        Folds for cross-validation.
    random_state : int, default=42
        Seed.

    Returns
    -------
    Tuple[dict, str, Any]
        (benchmark_results_dict, best_model_name, best_fitted_model)
    """
    X_train = data_dict["X_train"]
    X_test = data_dict["X_test"]
    y_train = data_dict["y_train"]
    y_test = data_dict["y_test"]

    candidates = get_model_candidates(random_state=random_state)
    results = {}
    best_pr_auc = -1.0
    best_model_name = ""
    best_fitted_model = None

    for name, model in candidates.items():
        logger.info(f"--- Benchmarking Candidate: {name} ---")

        # 1. Stratified Cross-Validation
        cv_metrics = cross_validate_candidate(
            model=model,
            X=X_train,
            y=y_train,
            cv_splits=cv_splits,
            random_state=random_state,
        )
        logger.info(
            f"  CV PR-AUC: {cv_metrics['cv_pr_auc_mean']:.4f} (+/- {cv_metrics['cv_pr_auc_std']:.4f}), "
            f"CV Recall: {cv_metrics['cv_recall_mean']:.4f}"
        )

        # 2. Fit on full training set and evaluate on untouched holdout test set
        model.fit(X_train, y_train)
        y_test_prob = model.predict_proba(X_test)[:, 1]
        y_test_pred = model.predict(X_test)

        test_metrics = evaluate_predictions(y_test, y_test_pred, y_test_prob)
        logger.info(
            f"  Test PR-AUC: {test_metrics['pr_auc']:.4f}, "
            f"Recall: {test_metrics['recall']:.4f}, "
            f"Precision: {test_metrics['precision']:.4f}, "
            f"F1: {test_metrics['f1_score']:.4f}"
        )

        combined = {**cv_metrics, "test_metrics": test_metrics}
        results[name] = combined

        # Select best model based on PR-AUC on holdout test set
        if test_metrics["pr_auc"] > best_pr_auc:
            best_pr_auc = test_metrics["pr_auc"]
            best_model_name = name
            best_fitted_model = model

    logger.info(f"==> Best Model Selected: {best_model_name} with Test PR-AUC: {best_pr_auc:.4f}")
    return results, best_model_name, best_fitted_model


def save_artifacts(
    model: Any,
    scaler: Any,
    benchmark_summary: Dict[str, Any],
    best_model_name: str,
    output_dir: Union[str, Path] = "models",
) -> Dict[str, Path]:
    """
    Serialize trained model, fitted scaler, and benchmark summary to disk.

    Parameters
    ----------
    model : Any
        Fitted best model.
    scaler : Any
        Fitted RobustScaler.
    benchmark_summary : dict
        Complete benchmark metrics.
    best_model_name : str
        Identifier of the winning model.
    output_dir : Union[str, Path], default='models'
        Target directory.

    Returns
    -------
    dict
        Paths to saved artifacts.
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    model_path = out_path / "best_fraud_model.joblib"
    scaler_path = out_path / "robust_scaler.joblib"
    summary_path = out_path / "evaluation_summary.json"

    # Save model and scaler using joblib
    joblib.dump(model, model_path)
    joblib.dump(scaler, scaler_path)

    # Save benchmark metrics to JSON
    metadata = {
        "best_model_name": best_model_name,
        "models": benchmark_summary,
    }
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    logger.info(f"Artifacts successfully saved to {out_path.resolve()}:")
    logger.info(f"  Model : {model_path.name}")
    logger.info(f"  Scaler: {scaler_path.name}")
    logger.info(f"  Summary: {summary_path.name}")

    return {
        "model_path": model_path,
        "scaler_path": scaler_path,
        "summary_path": summary_path,
    }


def run_supervised_pipeline(
    dataset_path: Union[str, Path] = "data/raw/creditcard.csv",
    output_dir: Union[str, Path] = "models",
    cv_splits: int = 3,
    random_state: int = 42,
) -> Dict[str, Any]:
    """
    End-to-end execution of Phase 2 supervised learning pipeline.
    """
    logger.info("Starting Supervised Fraud Classification Pipeline...")
    raw_df = load_raw_data(dataset_path)

    logger.info("Executing Preprocessing Pipeline...")
    data_pipeline = build_preprocessed_pipeline(raw_df, test_size=0.2, random_state=random_state)

    results, best_name, best_model = train_and_benchmark(
        data_dict=data_pipeline,
        cv_splits=cv_splits,
        random_state=random_state,
    )

    saved_paths = save_artifacts(
        model=best_model,
        scaler=data_pipeline["scaler"],
        benchmark_summary=results,
        best_model_name=best_name,
        output_dir=output_dir,
    )

    return {
        "best_model_name": best_name,
        "benchmark_results": results,
        "artifact_paths": saved_paths,
    }


if __name__ == "__main__":
    run_supervised_pipeline()
