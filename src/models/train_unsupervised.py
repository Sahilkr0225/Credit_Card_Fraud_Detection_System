"""
Unsupervised Anomaly Detection Module using Isolation Forest.

Detects zero-day, novel, and out-of-distribution fraudulent transaction patterns
without relying on historical ground-truth labels.
"""

import sys
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union
import json
import logging
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import classification_report, roc_auc_score, average_precision_score

# Ensure project root in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.load_data import load_raw_data
from src.data.preprocessing import build_preprocessed_pipeline

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def build_isolation_forest(
    contamination: Union[float, str] = 0.0017,
    n_estimators: int = 100,
    max_samples: int = 2048,
    random_state: int = 42,
    n_jobs: int = -1,
) -> IsolationForest:
    """
    Construct an Isolation Forest model configured for extreme anomaly detection.

    Parameters
    ----------
    contamination : float or 'auto', default=0.0017
        Expected proportion of anomalies (matching the ~0.17% fraud prevalence).
    n_estimators : int, default=100
        Number of isolation trees.
    max_samples : int, default=2048
        Subsample size drawn to build each tree.
    random_state : int, default=42
        Reproducibility seed.
    n_jobs : int, default=-1
        Parallel CPU threads.

    Returns
    -------
    IsolationForest
        Unfitted Isolation Forest estimator.
    """
    return IsolationForest(
        contamination=contamination,
        n_estimators=n_estimators,
        max_samples=max_samples,
        random_state=random_state,
        n_jobs=n_jobs,
    )


def train_and_evaluate_isolation_forest(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    contamination: float = 0.0017,
    random_state: int = 42,
) -> Tuple[IsolationForest, Dict[str, Any]]:
    """
    Train Isolation Forest exclusively on features and evaluate outlier metrics.

    Parameters
    ----------
    X_train : pd.DataFrame
        Scaled training feature matrix (labels NOT used during fit).
    X_test : pd.DataFrame
        Scaled testing feature matrix.
    y_test : pd.Series
        Test ground truth labels (used exclusively for post-hoc validation).
    contamination : float, default=0.0017
        Expected outlier ratio.
    random_state : int, default=42
        Seed.

    Returns
    -------
    Tuple[IsolationForest, dict]
        Fitted IsolationForest and evaluation metrics dictionary.
    """
    logger.info("Fitting Unsupervised Isolation Forest (labels are NOT used)...")
    iforest = build_isolation_forest(
        contamination=contamination,
        random_state=random_state,
    )
    iforest.fit(X_train)

    # Scikit-learn decision_function: lower/negative numbers = more anomalous
    scores = iforest.decision_function(X_test)
    preds_raw = iforest.predict(X_test)  # -1 for anomaly, 1 for inlier
    is_anomaly = (preds_raw == -1).astype(int)

    # Inverted score: higher = more anomalous (for ROC/PR evaluation)
    anomaly_probability = -scores

    pr_auc = round(float(average_precision_score(y_test, anomaly_probability)), 4)
    roc_auc = round(float(roc_auc_score(y_test, anomaly_probability)), 4)

    logger.info(f"Isolation Forest Test Evaluation: ROC-AUC: {roc_auc:.4f}, PR-AUC: {pr_auc:.4f}")

    metrics = {
        "model_type": "IsolationForest",
        "contamination": contamination,
        "test_roc_auc": roc_auc,
        "test_pr_auc": pr_auc,
        "anomalies_flagged_count": int(is_anomaly.sum()),
        "total_test_samples": len(X_test),
    }
    return iforest, metrics


def save_unsupervised_artifacts(
    model: IsolationForest,
    metrics: Dict[str, Any],
    output_dir: Union[str, Path] = "models",
) -> Dict[str, Path]:
    """
    Persist trained Isolation Forest model to disk.
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    model_path = out_path / "isolation_forest.joblib"
    summary_path = out_path / "unsupervised_summary.json"

    joblib.dump(model, model_path)
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    logger.info(f"Saved Isolation Forest to: {model_path}")
    return {"model_path": model_path, "summary_path": summary_path}


def run_unsupervised_pipeline(
    dataset_path: Union[str, Path] = "data/raw/creditcard.csv",
    output_dir: Union[str, Path] = "models",
    random_state: int = 42,
) -> Dict[str, Any]:
    """
    End-to-end execution of Phase 3 unsupervised anomaly detection.
    """
    logger.info("Starting Phase 3 Unsupervised Anomaly Detection Pipeline...")
    raw_df = load_raw_data(dataset_path)

    data_pipeline = build_preprocessed_pipeline(raw_df, test_size=0.2, random_state=random_state)

    iforest, metrics = train_and_evaluate_isolation_forest(
        X_train=data_pipeline["X_train"],
        X_test=data_pipeline["X_test"],
        y_test=data_pipeline["y_test"],
        random_state=random_state,
    )

    saved_paths = save_unsupervised_artifacts(
        model=iforest,
        metrics=metrics,
        output_dir=output_dir,
    )

    return {
        "metrics": metrics,
        "artifact_paths": saved_paths,
    }


if __name__ == "__main__":
    run_unsupervised_pipeline()
