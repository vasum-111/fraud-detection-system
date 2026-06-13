"""
Data Drift Detection for Fraud Model Monitoring
Uses Population Stability Index (PSI) and KS-test to detect feature drift.
"""

import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, List, Tuple
import logging

logger = logging.getLogger(__name__)

PSI_ALERT_THRESHOLD = 0.2   # PSI > 0.2 = significant drift
PSI_WARN_THRESHOLD = 0.1    # PSI > 0.1 = moderate drift
KS_PVALUE_THRESHOLD = 0.05  # p < 0.05 = statistically significant drift


class DriftDetector:
    """
    Monitors feature distributions for data drift.

    PSI interpretation:
        < 0.1  → No significant drift
        0.1-0.2 → Moderate drift, monitor
        > 0.2  → Significant drift, investigate/retrain
    """

    def __init__(self, reference_data: pd.DataFrame, feature_cols: List[str]):
        """
        Args:
            reference_data: Training/baseline dataset
            feature_cols: Features to monitor
        """
        self.reference_data = reference_data[feature_cols].copy()
        self.feature_cols = feature_cols
        self._reference_bins: Dict[str, np.ndarray] = {}
        self._reference_dist: Dict[str, np.ndarray] = {}
        self._precompute_reference_bins()

    def _precompute_reference_bins(self, n_bins: int = 10):
        """Precompute reference distribution bins for fast PSI computation."""
        for col in self.feature_cols:
            try:
                vals = self.reference_data[col].dropna()
                if len(vals) < 10:
                    continue
                _, bins = np.histogram(vals, bins=n_bins)
                hist, _ = np.histogram(vals, bins=bins)
                dist = (hist + 0.0001) / (len(vals) + n_bins * 0.0001)
                self._reference_bins[col] = bins
                self._reference_dist[col] = dist
            except Exception as e:
                logger.debug(f"Could not precompute bins for {col}: {e}")

    def compute_psi(self, current_data: pd.DataFrame, col: str) -> float:
        """
        Compute Population Stability Index for a single feature.

        PSI = Σ (actual% - expected%) * ln(actual% / expected%)
        """
        if col not in self._reference_bins:
            return 0.0

        bins = self._reference_bins[col]
        ref_dist = self._reference_dist[col]

        curr_vals = current_data[col].dropna()
        if len(curr_vals) < 10:
            return 0.0

        curr_hist, _ = np.histogram(curr_vals, bins=bins)
        curr_dist = (curr_hist + 0.0001) / (len(curr_vals) + len(bins) * 0.0001)

        psi = np.sum((curr_dist - ref_dist) * np.log(curr_dist / ref_dist))
        return float(psi)

    def compute_ks_test(self, current_data: pd.DataFrame, col: str) -> Tuple[float, float]:
        """Kolmogorov-Smirnov test for continuous features."""
        ref_vals = self.reference_data[col].dropna().values
        curr_vals = current_data[col].dropna().values
        if len(ref_vals) < 10 or len(curr_vals) < 10:
            return 0.0, 1.0
        stat, pvalue = stats.ks_2samp(ref_vals, curr_vals)
        return float(stat), float(pvalue)

    def check_drift(self, current_data: pd.DataFrame) -> Dict:
        """
        Run full drift check across all monitored features.

        Returns:
            {
                "overall_status": "ALERT" | "WARN" | "OK",
                "features": {col: {"psi": ..., "ks_stat": ..., "status": ...}},
                "alert_features": [...],
                "warn_features": [...],
            }
        """
        results = {}
        alert_features = []
        warn_features = []

        for col in self.feature_cols:
            if col not in self._reference_bins:
                continue

            psi = self.compute_psi(current_data, col)
            ks_stat, ks_pval = self.compute_ks_test(current_data, col)

            if psi > PSI_ALERT_THRESHOLD or ks_pval < PSI_PVALUE_THRESHOLD_STRICT:
                status = "ALERT"
                alert_features.append(col)
            elif psi > PSI_WARN_THRESHOLD or ks_pval < KS_PVALUE_THRESHOLD:
                status = "WARN"
                warn_features.append(col)
            else:
                status = "OK"

            results[col] = {
                "psi": round(psi, 4),
                "ks_stat": round(ks_stat, 4),
                "ks_pvalue": round(ks_pval, 4),
                "status": status,
            }

        overall = (
            "ALERT" if alert_features
            else "WARN" if warn_features
            else "OK"
        )

        return {
            "overall_status": overall,
            "n_features_checked": len(results),
            "alert_features": alert_features,
            "warn_features": warn_features,
            "features": results,
            "recommendation": self._get_recommendation(overall, alert_features),
        }

    @staticmethod
    def _get_recommendation(status: str, alert_features: List[str]) -> str:
        if status == "ALERT":
            return (
                f"SIGNIFICANT DRIFT detected in: {', '.join(alert_features[:3])}. "
                "Recommend immediate model retraining evaluation."
            )
        elif status == "WARN":
            return "Moderate drift detected. Monitor closely and consider retraining within 2 weeks."
        return "No significant drift. Model distribution is stable."

    def score_distribution_shift(
        self, ref_scores: np.ndarray, curr_scores: np.ndarray
    ) -> Dict:
        """Check if prediction score distribution has shifted."""
        ks_stat, ks_pval = stats.ks_2samp(ref_scores, curr_scores)
        mean_shift = float(np.mean(curr_scores) - np.mean(ref_scores))
        std_shift = float(np.std(curr_scores) - np.std(ref_scores))

        return {
            "ks_stat": round(ks_stat, 4),
            "ks_pvalue": round(ks_pval, 4),
            "mean_shift": round(mean_shift, 4),
            "std_shift": round(std_shift, 4),
            "status": "ALERT" if ks_pval < 0.01 else "WARN" if ks_pval < 0.05 else "OK",
        }


# Missing constant fix
PSI_PVALUE_THRESHOLD_STRICT = 0.01
