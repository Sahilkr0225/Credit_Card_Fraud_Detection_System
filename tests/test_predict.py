"""
Unit tests for the Real-Time ML Prediction Interface (src/models/predict.py).

Validates strict adherence to docs/API_CONTRACT.md schema, thread-safe loading,
and graceful fallback handling.
"""

import pytest
from src.models.predict import FraudPredictor, predict_transaction, get_fraud_predictor


@pytest.fixture
def sample_valid_transaction():
    """Create a sample synthetic transaction with all 30 required numerical features."""
    tx = {
        "transaction_id": "TXN_TEST_101",
        "Time": 406.0,
        "Amount": 149.62,
    }
    # Add standardized PCA features V1 through V28
    for i in range(1, 29):
        tx[f"V{i}"] = 0.05 * (-1 if i % 2 == 0 else 1)
    return tx


def test_predict_single_schema_compliance(sample_valid_transaction):
    """Verify inference output matches MLPredictionResult schema in API_CONTRACT.md."""
    predictor = get_fraud_predictor()
    result = predictor.predict_single(sample_valid_transaction)

    assert isinstance(result, dict)
    assert result["transaction_id"] == "TXN_TEST_101"
    assert "fraud_probability" in result
    assert "ml_prediction" in result
    assert "anomaly_score" in result
    assert "is_anomaly" in result
    assert "ml_risk_tier" in result
    assert "model_version" in result


def test_predict_value_ranges_and_types(sample_valid_transaction):
    """Verify predictions and anomaly scores are mathematically bounded and typed."""
    result = predict_transaction(sample_valid_transaction)

    # Probability between 0.0 and 1.0
    assert 0.0 <= result["fraud_probability"] <= 1.0
    # Binary prediction 0 or 1
    assert result["ml_prediction"] in (0, 1)
    # Anomaly score is float
    assert isinstance(result["anomaly_score"], float)
    # Anomaly flag is bool
    assert isinstance(result["is_anomaly"], bool)
    # Risk tier matches enum
    assert result["ml_risk_tier"] in ("LOW", "MEDIUM", "HIGH", "CRITICAL")


def test_predict_strict_mode_rejects_missing_features():
    """In strict mode, missing features must raise a ValueError."""
    predictor = get_fraud_predictor()
    incomplete_tx = {
        "transaction_id": "TXN_INCOMPLETE",
        "Time": 100.0,
        "Amount": 25.0,
        # Missing V1-V28
    }
    with pytest.raises(ValueError, match="Missing required transaction features"):
        predictor.predict_single(incomplete_tx, strict=True)


def test_predict_lenient_mode_handles_missing_features(sample_valid_transaction):
    """In default lenient mode, missing PCA features default gracefully without crashing."""
    predictor = get_fraud_predictor()
    minimal_tx = {
        "transaction_id": "TXN_MINIMAL",
        "amount": 50.0,
    }
    result = predictor.predict_single(minimal_tx, strict=False)
    assert result["transaction_id"] == "TXN_MINIMAL"
    assert 0.0 <= result["fraud_probability"] <= 1.0


def test_predict_fallback_on_unloaded_models(tmp_path, sample_valid_transaction):
    """Test predictor fallback behavior when models are absent from directory."""
    empty_dir = tmp_path / "empty_models"
    empty_dir.mkdir()

    predictor = FraudPredictor(model_dir=empty_dir)
    result = predictor.predict_single(sample_valid_transaction)

    assert result["transaction_id"] == "TXN_TEST_101"
    assert result["ml_prediction"] == 0
    assert result["ml_risk_tier"] == "LOW"
