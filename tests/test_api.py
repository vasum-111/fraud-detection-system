"""
API Integration Tests
"""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import MagicMock, patch

# Patch model loading before importing app
with patch("src.api.main.FraudEnsemble.load"), patch("builtins.open"):
    from src.api.main import app

client = TestClient(app)
API_KEY = "dev-secret-key"
HEADERS = {"X-API-Key": API_KEY}


SAMPLE_TXN = {
    "transaction_id": "test_txn_001",
    "card_id": "card_abc",
    "amount": 250.00,
    "merchant_category": "electronics",
    "timestamp": "2026-06-13T14:00:00Z",
    "device_id": "dev_xyz",
    "country": "US",
    "email_domain": "gmail.com",
}


class TestHealthEndpoint:
    def test_health_returns_200(self):
        resp = client.get("/health")
        assert resp.status_code == 200

    def test_health_has_required_fields(self):
        resp = client.get("/health")
        data = resp.json()
        assert "status" in data
        assert "model_version" in data
        assert "model_loaded" in data
        assert "uptime_seconds" in data


class TestPredictEndpoint:
    def test_requires_api_key(self):
        resp = client.post("/predict", json=SAMPLE_TXN)
        assert resp.status_code == 403

    def test_invalid_api_key_rejected(self):
        resp = client.post(
            "/predict", json=SAMPLE_TXN, headers={"X-API-Key": "wrong-key"}
        )
        assert resp.status_code == 403

    def test_valid_request_structure(self):
        """Test that request schema validation works."""
        bad_txn = {**SAMPLE_TXN, "amount": -100}  # Negative amount
        resp = client.post("/predict", json=bad_txn, headers=HEADERS)
        assert resp.status_code == 422  # Validation error

    def test_missing_required_field(self):
        bad_txn = {k: v for k, v in SAMPLE_TXN.items() if k != "card_id"}
        resp = client.post("/predict", json=bad_txn, headers=HEADERS)
        assert resp.status_code == 422


class TestBatchEndpoint:
    def test_batch_empty_list_rejected(self):
        resp = client.post(
            "/predict/batch",
            json={"transactions": []},
            headers=HEADERS,
        )
        assert resp.status_code == 422

    def test_batch_too_large_rejected(self):
        txns = [SAMPLE_TXN] * 1001
        resp = client.post(
            "/predict/batch",
            json={"transactions": txns},
            headers=HEADERS,
        )
        assert resp.status_code == 422


class TestMetricsEndpoint:
    def test_metrics_requires_auth(self):
        resp = client.get("/metrics")
        assert resp.status_code == 403

    def test_metrics_returns_200_with_auth(self):
        resp = client.get("/metrics", headers=HEADERS)
        assert resp.status_code == 200

    def test_metrics_fields_present(self):
        resp = client.get("/metrics", headers=HEADERS)
        data = resp.json()
        assert "total_predictions" in data
        assert "avg_latency_ms" in data
        assert "fraud_rate" in data
