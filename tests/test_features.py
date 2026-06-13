"""
Tests for Feature Engineering Pipeline
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from src.features.feature_engineering import FraudFeatureEngineer


@pytest.fixture
def sample_history():
    """Sample transaction history for a card."""
    now = datetime.now()
    return pd.DataFrame([
        {
            "card1": "card_001",
            "TransactionAmt": 50.0,
            "ProductCD": "W",
            "timestamp": now - timedelta(days=5),
        },
        {
            "card1": "card_001",
            "TransactionAmt": 75.0,
            "ProductCD": "W",
            "timestamp": now - timedelta(days=2),
        },
        {
            "card1": "card_001",
            "TransactionAmt": 60.0,
            "ProductCD": "H",
            "timestamp": now - timedelta(hours=3),
        },
    ])


@pytest.fixture
def sample_transaction():
    return {
        "card1": "card_001",
        "TransactionAmt": 500.0,  # Much higher than history
        "ProductCD": "C",          # New category
        "P_emaildomain": "gmail.com",
        "DeviceInfo": "Chrome",
        "addr1": "12345",
        "addr2": "12345",
        "timestamp": datetime.now(),
    }


@pytest.fixture
def engineer():
    return FraudFeatureEngineer()


class TestVelocityFeatures:
    def test_recent_transaction_counted(self, engineer, sample_transaction, sample_history):
        features = engineer.transform(sample_transaction, sample_history)
        # Should count the transaction 3h ago in 6h window but not 1h window
        assert features["txn_count_6h"] >= 1
        assert "txn_count_1h" in features

    def test_velocity_ratio_computed(self, engineer, sample_transaction, sample_history):
        features = engineer.transform(sample_transaction, sample_history)
        assert "velocity_ratio_1h_24h" in features
        assert features["velocity_ratio_1h_24h"] >= 0

    def test_empty_history_returns_zeros(self, engineer, sample_transaction):
        features = engineer.transform(sample_transaction, pd.DataFrame())
        assert features["txn_count_1h"] == 0
        assert features["txn_count_24h"] == 0


class TestBehavioralFeatures:
    def test_high_amount_gets_high_zscore(self, engineer, sample_transaction, sample_history):
        # $500 vs avg ~$62 should have zscore > 2
        features = engineer.transform(sample_transaction, sample_history)
        assert features["amount_zscore"] > 2.0

    def test_new_merchant_category_detected(self, engineer, sample_transaction, sample_history):
        # "C" is not in history ["W", "W", "H"]
        features = engineer.transform(sample_transaction, sample_history)
        assert features["is_new_merchant_category"] == 1

    def test_known_merchant_category_not_flagged(self, engineer, sample_history):
        txn = {
            "card1": "card_001",
            "TransactionAmt": 55.0,
            "ProductCD": "W",  # Known category
            "P_emaildomain": "gmail.com",
            "timestamp": datetime.now(),
        }
        features = engineer.transform(txn, sample_history)
        assert features["is_new_merchant_category"] == 0

    def test_new_user_no_history(self, engineer, sample_transaction):
        features = engineer.transform(sample_transaction, pd.DataFrame())
        assert features["user_txn_count_total"] == 0
        assert features["is_new_merchant_category"] == 1


class TestTimeFeatures:
    def test_night_transaction_flagged(self, engineer, sample_transaction):
        sample_transaction["timestamp"] = datetime.now().replace(hour=2)
        features = engineer.transform(sample_transaction, pd.DataFrame())
        assert features["is_night"] == 1

    def test_business_hours_flagged(self, engineer, sample_transaction):
        sample_transaction["timestamp"] = datetime.now().replace(hour=11)
        # Monday = 0, set to a weekday
        ts = sample_transaction["timestamp"]
        while ts.weekday() >= 5:  # skip to weekday
            ts = ts + timedelta(days=1)
        sample_transaction["timestamp"] = ts.replace(hour=11)
        features = engineer.transform(sample_transaction, pd.DataFrame())
        assert features["is_business_hours"] == 1

    def test_cyclical_encoding_bounds(self, engineer, sample_transaction):
        features = engineer.transform(sample_transaction, pd.DataFrame())
        assert -1 <= features["hour_sin"] <= 1
        assert -1 <= features["hour_cos"] <= 1
        assert -1 <= features["dow_sin"] <= 1
        assert -1 <= features["dow_cos"] <= 1


class TestAmountFeatures:
    def test_round_amount_flagged(self, engineer):
        txn = {
            "card1": "card_001",
            "TransactionAmt": 200.0,
            "ProductCD": "W",
            "timestamp": datetime.now(),
        }
        features = engineer.transform(txn, pd.DataFrame())
        assert features["amount_is_round"] == 1

    def test_log_transform_positive(self, engineer, sample_transaction):
        features = engineer.transform(sample_transaction, pd.DataFrame())
        assert features["amount_log"] > 0
        assert features["amount_log"] == pytest.approx(np.log1p(500.0), rel=1e-5)


class TestDeviceFeatures:
    def test_risky_email_domain_flagged(self, engineer, sample_transaction):
        sample_transaction["P_emaildomain"] = "mailinator.com"
        features = engineer.transform(sample_transaction, pd.DataFrame())
        assert features["is_high_risk_email"] == 1

    def test_safe_email_domain_not_flagged(self, engineer, sample_transaction):
        sample_transaction["P_emaildomain"] = "gmail.com"
        features = engineer.transform(sample_transaction, pd.DataFrame())
        assert features["is_high_risk_email"] == 0

    def test_missing_email_flagged(self, engineer, sample_transaction):
        sample_transaction["P_emaildomain"] = ""
        features = engineer.transform(sample_transaction, pd.DataFrame())
        assert features["email_domain_missing"] == 1
