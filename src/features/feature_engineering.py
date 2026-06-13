"""
Feature Engineering Pipeline for Fraud Detection
Generates velocity, behavioral, time, and device features from raw transactions.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Optional
import logging

logger = logging.getLogger(__name__)


class FraudFeatureEngineer:
    """
    Production feature engineering pipeline.

    Computes:
    - Velocity features (rolling counts/amounts per entity)
    - Behavioral features (deviation from historical patterns)
    - Time-based features (hour, day, recency)
    - Device/identity risk features
    """

    VELOCITY_WINDOWS = [1, 6, 24]  # hours
    HIGH_RISK_EMAIL_DOMAINS = {"guerrillamail.com", "mailinator.com", "temp-mail.org"}
    HIGH_RISK_COUNTRIES = {"NG", "RO", "UA", "ID"}  # example — update with your data

    def __init__(self, history_df: Optional[pd.DataFrame] = None):
        """
        Args:
            history_df: Historical transactions for computing behavioral baselines.
                        Columns: [card_id, amount, merchant_category, timestamp, ...]
        """
        self.history_df = history_df
        self._user_stats: dict = {}
        if history_df is not None:
            self._precompute_user_stats()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def transform(self, txn: dict, recent_txns: pd.DataFrame) -> dict:
        """
        Compute all features for a single transaction.

        Args:
            txn: Raw transaction dict
            recent_txns: Recent transactions for the same card/device (last 30 days)

        Returns:
            Feature dict ready for model inference
        """
        features = {}
        ts = pd.to_datetime(txn["timestamp"])

        features.update(self._velocity_features(txn, recent_txns, ts))
        features.update(self._behavioral_features(txn, recent_txns))
        features.update(self._time_features(ts))
        features.update(self._device_identity_features(txn))
        features.update(self._amount_features(txn, recent_txns))

        return features

    def transform_batch(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform a full DataFrame of transactions for training."""
        df = df.sort_values("TransactionDT").copy()
        df["timestamp"] = pd.to_datetime(df["TransactionDT"], unit="s", origin="2017-11-30")

        feature_rows = []
        for _, row in df.iterrows():
            card_history = df[
                (df["card1"] == row["card1"]) &
                (df["TransactionDT"] < row["TransactionDT"])
            ].tail(200)
            features = self.transform(row.to_dict(), card_history)
            feature_rows.append(features)

        features_df = pd.DataFrame(feature_rows)
        return pd.concat([df.reset_index(drop=True), features_df], axis=1)

    # ------------------------------------------------------------------
    # Velocity features
    # ------------------------------------------------------------------

    def _velocity_features(self, txn: dict, history: pd.DataFrame, ts: pd.Timestamp) -> dict:
        """Rolling transaction count and amount sums over multiple windows."""
        feats = {}
        card_id = txn.get("card1") or txn.get("card_id", "")

        for hours in self.VELOCITY_WINDOWS:
            cutoff = ts - timedelta(hours=hours)
            window = history[history["timestamp"] >= cutoff] if not history.empty else pd.DataFrame()

            feats[f"txn_count_{hours}h"] = len(window)
            feats[f"txn_amount_sum_{hours}h"] = float(window["TransactionAmt"].sum()) if not window.empty else 0.0
            feats[f"unique_merchants_{hours}h"] = int(window["ProductCD"].nunique()) if not window.empty else 0

        # Velocity ratio: compare 1h vs 24h rate
        count_1h = feats.get("txn_count_1h", 0)
        count_24h = feats.get("txn_count_24h", 0)
        feats["velocity_ratio_1h_24h"] = count_1h / max(count_24h / 24, 1e-6)

        return feats

    # ------------------------------------------------------------------
    # Behavioral features
    # ------------------------------------------------------------------

    def _behavioral_features(self, txn: dict, history: pd.DataFrame) -> dict:
        """Deviation from historical patterns."""
        feats = {}
        amount = float(txn.get("TransactionAmt", 0))

        if not history.empty and len(history) >= 3:
            hist_amounts = history["TransactionAmt"].astype(float)
            mean_amt = hist_amounts.mean()
            std_amt = hist_amounts.std() + 1e-6

            feats["amount_zscore"] = (amount - mean_amt) / std_amt
            feats["amount_vs_mean_ratio"] = amount / max(mean_amt, 1e-6)
            feats["user_mean_amount"] = mean_amt
            feats["user_std_amount"] = std_amt
            feats["user_txn_count_total"] = len(history)

            # Is this a new merchant category for this user?
            current_cat = txn.get("ProductCD", "")
            seen_cats = set(history["ProductCD"].dropna().tolist())
            feats["is_new_merchant_category"] = int(current_cat not in seen_cats)

            # Days since last transaction
            if "timestamp" in history.columns and not history.empty:
                last_ts = history["timestamp"].max()
                current_ts = pd.to_datetime(txn.get("timestamp", datetime.now()))
                feats["days_since_last_txn"] = (current_ts - last_ts).total_seconds() / 86400
            else:
                feats["days_since_last_txn"] = -1
        else:
            # New user — no history (often higher risk)
            feats["amount_zscore"] = 0.0
            feats["amount_vs_mean_ratio"] = 1.0
            feats["user_mean_amount"] = amount
            feats["user_std_amount"] = 0.0
            feats["user_txn_count_total"] = 0
            feats["is_new_merchant_category"] = 1
            feats["days_since_last_txn"] = -1

        return feats

    # ------------------------------------------------------------------
    # Time features
    # ------------------------------------------------------------------

    def _time_features(self, ts: pd.Timestamp) -> dict:
        """Cyclical and categorical time features."""
        hour = ts.hour
        dow = ts.dayofweek  # 0=Monday

        return {
            "hour_of_day": hour,
            "day_of_week": dow,
            "is_weekend": int(dow >= 5),
            "is_night": int(hour < 6 or hour >= 22),
            "is_business_hours": int(9 <= hour <= 17 and dow < 5),
            # Cyclical encoding
            "hour_sin": np.sin(2 * np.pi * hour / 24),
            "hour_cos": np.cos(2 * np.pi * hour / 24),
            "dow_sin": np.sin(2 * np.pi * dow / 7),
            "dow_cos": np.cos(2 * np.pi * dow / 7),
        }

    # ------------------------------------------------------------------
    # Device / identity features
    # ------------------------------------------------------------------

    def _device_identity_features(self, txn: dict) -> dict:
        """Risk signals from device and identity fields."""
        email = txn.get("P_emaildomain", "") or ""
        addr_match = txn.get("addr1") == txn.get("addr2") if txn.get("addr2") else True

        return {
            "is_high_risk_email": int(email in self.HIGH_RISK_EMAIL_DOMAINS),
            "email_domain_missing": int(email == ""),
            "billing_shipping_addr_match": int(addr_match),
            "device_info_missing": int(txn.get("DeviceInfo", "") in ["", None]),
        }

    # ------------------------------------------------------------------
    # Amount features
    # ------------------------------------------------------------------

    def _amount_features(self, txn: dict, history: pd.DataFrame) -> dict:
        """Raw and transformed amount features."""
        amount = float(txn.get("TransactionAmt", 0))
        return {
            "amount": amount,
            "amount_log": np.log1p(amount),
            "amount_is_round": int(amount % 1 == 0),
            "amount_cents": amount % 1,
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _precompute_user_stats(self):
        """Precompute per-user stats from historical data for fast lookup."""
        if self.history_df is None:
            return
        grouped = self.history_df.groupby("card1")["TransactionAmt"]
        self._user_stats = {
            card: {"mean": g.mean(), "std": g.std(), "count": len(g)}
            for card, g in grouped
        }
        logger.info(f"Precomputed stats for {len(self._user_stats)} cards.")
