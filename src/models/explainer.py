"""
SHAP-based Explainability for Fraud Predictions
Translates raw SHAP values into human-readable risk reasons.
"""

import shap
import numpy as np
import pandas as pd
from typing import List, Tuple
import logging

logger = logging.getLogger(__name__)

# Maps feature names to business-readable templates
FEATURE_TEMPLATES = {
    "amount_zscore": lambda v: f"Amount is {abs(v):.1f}x above your typical spending" if v > 2 else f"Amount is unusually low",
    "amount_vs_mean_ratio": lambda v: f"Amount is {v:.1f}x your historical average",
    "txn_count_1h": lambda v: f"{int(v)} transactions in the past hour",
    "txn_count_6h": lambda v: f"{int(v)} transactions in the past 6 hours",
    "velocity_ratio_1h_24h": lambda v: f"Transaction rate spiked {v:.1f}x above daily average",
    "is_new_merchant_category": lambda v: "First transaction in this merchant category" if v > 0.5 else None,
    "is_night": lambda v: "Transaction at unusual night-time hour" if v > 0.5 else None,
    "days_since_last_txn": lambda v: f"Long gap ({v:.0f} days) since last transaction" if v > 30 else f"Very quick follow-up transaction",
    "billing_shipping_addr_match": lambda v: "Billing and shipping addresses don't match" if v < 0.5 else None,
    "is_high_risk_email": lambda v: "High-risk email domain detected" if v > 0.5 else None,
    "device_info_missing": lambda v: "No device information available" if v > 0.5 else None,
    "is_weekend": lambda v: "Weekend transaction on a business-day account" if v > 0.5 else None,
    "user_txn_count_total": lambda v: "New account with no transaction history" if v == 0 else None,
    "amount_is_round": lambda v: "Suspiciously round amount" if v > 0.5 else None,
}


class FraudExplainer:
    """
    Generates SHAP-based explanations for fraud predictions.

    Usage:
        explainer = FraudExplainer(ensemble.xgb_model, feature_names)
        explainer.fit(X_background)
        reasons = explainer.explain(features_dict, top_n=3)
    """

    def __init__(self, xgb_model, feature_names: List[str]):
        self.xgb_model = xgb_model
        self.feature_names = feature_names
        self._shap_explainer = None

    def fit(self, X_background: pd.DataFrame, max_samples: int = 500):
        """
        Initialize SHAP TreeExplainer with a background dataset.

        Args:
            X_background: Background dataset for SHAP baseline (training set sample)
            max_samples: Limit background for speed
        """
        background = X_background[self.feature_names].fillna(0)
        if len(background) > max_samples:
            background = background.sample(max_samples, random_state=42)

        self._shap_explainer = shap.TreeExplainer(
            self.xgb_model,
            data=background,
            feature_perturbation="interventional",
        )
        logger.info("SHAP explainer initialized.")

    def explain(self, features: dict, top_n: int = 3) -> List[str]:
        """
        Generate top N human-readable risk reasons for a single transaction.

        Args:
            features: Feature dict (same keys as training)
            top_n: Number of reasons to return

        Returns:
            List of plain-English risk explanations
        """
        if self._shap_explainer is None:
            raise RuntimeError("Call .fit() before .explain()")

        X = pd.DataFrame([features])[self.feature_names].fillna(0)
        shap_values = self._shap_explainer.shap_values(X)

        # For XGBClassifier, shap_values is shape (1, n_features)
        if isinstance(shap_values, list):
            sv = shap_values[1][0]  # class 1 (fraud)
        else:
            sv = shap_values[0]

        # Get indices sorted by absolute SHAP value (most impactful first)
        sorted_idx = np.argsort(np.abs(sv))[::-1]
        reasons = []

        for idx in sorted_idx:
            if len(reasons) >= top_n:
                break
            feat_name = self.feature_names[idx]
            feat_val = float(X.iloc[0][feat_name])
            shap_val = sv[idx]

            # Only include features pushing toward fraud (positive SHAP)
            if shap_val <= 0:
                continue

            reason = self._feature_to_reason(feat_name, feat_val, shap_val)
            if reason:
                reasons.append(reason)

        # Fallback if no positive reasons found
        if not reasons:
            top_idx = sorted_idx[0]
            feat_name = self.feature_names[top_idx]
            feat_val = float(X.iloc[0][feat_name])
            reasons.append(f"Unusual pattern in {feat_name.replace('_', ' ')}: {feat_val:.2f}")

        return reasons

    def explain_batch(self, X: pd.DataFrame, top_n: int = 3) -> List[List[str]]:
        """Explain predictions for a batch of transactions."""
        return [
            self.explain(row.to_dict(), top_n=top_n)
            for _, row in X.iterrows()
        ]

    def shap_summary_plot(self, X: pd.DataFrame, save_path: str = None):
        """Generate SHAP summary plot (requires matplotlib)."""
        import matplotlib.pyplot as plt
        X_filled = X[self.feature_names].fillna(0)
        sv = self._shap_explainer.shap_values(X_filled)
        if isinstance(sv, list):
            sv = sv[1]
        shap.summary_plot(sv, X_filled, show=False)
        if save_path:
            plt.savefig(save_path, bbox_inches="tight", dpi=150)
            plt.close()
            logger.info(f"SHAP summary saved to {save_path}")

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    @staticmethod
    def _feature_to_reason(feat_name: str, feat_val: float, shap_val: float) -> str:
        """Convert a feature + value to a human-readable reason."""
        template_fn = FEATURE_TEMPLATES.get(feat_name)
        if template_fn:
            try:
                result = template_fn(feat_val)
                return result
            except Exception:
                pass

        # Generic fallback
        direction = "high" if shap_val > 0 else "low"
        return f"{feat_name.replace('_', ' ').title()} is unusually {direction} ({feat_val:.2f})"
