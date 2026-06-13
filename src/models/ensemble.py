"""
Fraud Detection Ensemble Model
Combines XGBoost classifier with Isolation Forest anomaly scores.
Uses isotonic regression for probability calibration.
"""

import numpy as np
import pandas as pd
import joblib
import logging
from pathlib import Path
from typing import Optional, Tuple

from sklearn.ensemble import IsolationForest
from sklearn.calibration import CalibratedClassifierCV
from sklearn.isotonic import IsotonicRegression
from sklearn.preprocessing import StandardScaler
import xgboost as xgb

logger = logging.getLogger(__name__)


class FraudEnsemble:
    """
    Two-model ensemble for fraud detection:

    1. XGBoost: supervised, trained on labeled fraud/legit transactions
    2. Isolation Forest: unsupervised anomaly detection for novel patterns

    Final score = 0.75 * xgb_prob + 0.25 * iso_score (tunable)
    """

    XGB_PARAMS = {
        "n_estimators": 500,
        "max_depth": 6,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "scale_pos_weight": 28,   # ~1/fraud_rate for class imbalance
        "eval_metric": "aucpr",
        "use_label_encoder": False,
        "random_state": 42,
        "tree_method": "hist",    # fast training
    }

    ISO_PARAMS = {
        "n_estimators": 200,
        "contamination": 0.035,   # ~3.5% fraud rate in IEEE-CIS
        "random_state": 42,
        "n_jobs": -1,
    }

    ENSEMBLE_WEIGHT_XGB = 0.75
    ENSEMBLE_WEIGHT_ISO = 0.25

    def __init__(self):
        self.xgb_model: Optional[xgb.XGBClassifier] = None
        self.iso_model: Optional[IsolationForest] = None
        self.calibrator: Optional[IsotonicRegression] = None
        self.scaler: Optional[StandardScaler] = None
        self.feature_names: list = []
        self.threshold: float = 0.5
        self._is_fitted: bool = False

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None,
        optimize_threshold: bool = True,
    ) -> "FraudEnsemble":
        """
        Train XGBoost + Isolation Forest and calibrate probabilities.

        Args:
            X_train: Feature matrix
            y_train: Labels (0=legit, 1=fraud)
            X_val: Validation features (used for calibration + threshold tuning)
            y_val: Validation labels
            optimize_threshold: If True, find optimal F-beta threshold on val set
        """
        self.feature_names = list(X_train.columns)
        logger.info(f"Training on {len(X_train):,} samples, {y_train.sum():,} fraud cases")

        # 1. Train XGBoost
        logger.info("Training XGBoost...")
        self.xgb_model = xgb.XGBClassifier(**self.XGB_PARAMS)
        eval_set = [(X_val, y_val)] if X_val is not None else []
        self.xgb_model.fit(
            X_train, y_train,
            eval_set=eval_set,
            verbose=100,
        )

        # 2. Train Isolation Forest (on fraud-free subset for better baseline)
        logger.info("Training Isolation Forest...")
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X_train)
        self.iso_model = IsolationForest(**self.ISO_PARAMS)
        self.iso_model.fit(X_scaled)

        # 3. Calibrate on validation set
        if X_val is not None and y_val is not None:
            logger.info("Calibrating probabilities...")
            raw_scores = self._raw_ensemble_scores(X_val)
            self.calibrator = IsotonicRegression(out_of_bounds="clip")
            self.calibrator.fit(raw_scores, y_val)

            if optimize_threshold:
                calibrated = self.calibrator.predict(raw_scores)
                self.threshold = self._find_optimal_threshold(calibrated, y_val, beta=0.5)
                logger.info(f"Optimal threshold: {self.threshold:.3f}")

        self._is_fitted = True
        logger.info("Training complete.")
        return self

    def _find_optimal_threshold(
        self, scores: np.ndarray, y: pd.Series, beta: float = 0.5
    ) -> float:
        """Find threshold maximizing F-beta (beta<1 favors precision, beta>1 favors recall)."""
        from sklearn.metrics import fbeta_score
        best_t, best_f = 0.5, 0.0
        for t in np.arange(0.1, 0.95, 0.01):
            preds = (scores >= t).astype(int)
            f = fbeta_score(y, preds, beta=beta, zero_division=0)
            if f > best_f:
                best_f, best_t = f, t
        return best_t

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Return calibrated fraud probability for each row."""
        self._check_fitted()
        X = X[self.feature_names].fillna(0)
        raw = self._raw_ensemble_scores(X)
        if self.calibrator is not None:
            return self.calibrator.predict(raw)
        return raw

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Return binary labels using optimized threshold."""
        probs = self.predict_proba(X)
        return (probs >= self.threshold).astype(int)

    def predict_single(self, features: dict) -> Tuple[float, str]:
        """
        Score a single transaction.

        Returns:
            (fraud_score, risk_level)
        """
        X = pd.DataFrame([features])
        score = float(self.predict_proba(X)[0])
        risk = self._score_to_risk_level(score)
        return score, risk

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _raw_ensemble_scores(self, X: pd.DataFrame) -> np.ndarray:
        """Compute weighted combination of XGBoost and Isolation Forest scores."""
        X_filled = X[self.feature_names].fillna(0)

        # XGBoost fraud probability
        xgb_prob = self.xgb_model.predict_proba(X_filled)[:, 1]

        # Isolation Forest anomaly score → [0, 1] (higher = more anomalous)
        X_scaled = self.scaler.transform(X_filled)
        iso_raw = self.iso_model.decision_function(X_scaled)
        # Normalize: iso returns negative scores where more negative = more anomalous
        iso_score = 1 - (iso_raw - iso_raw.min()) / (iso_raw.max() - iso_raw.min() + 1e-9)

        return (
            self.ENSEMBLE_WEIGHT_XGB * xgb_prob +
            self.ENSEMBLE_WEIGHT_ISO * iso_score
        )

    @staticmethod
    def _score_to_risk_level(score: float) -> str:
        if score >= 0.8:
            return "CRITICAL"
        elif score >= 0.6:
            return "HIGH"
        elif score >= 0.4:
            return "MEDIUM"
        else:
            return "LOW"

    def _check_fitted(self):
        if not self._is_fitted:
            raise RuntimeError("Model not fitted. Call .fit() first.")

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str):
        """Save full ensemble to disk."""
        joblib.dump(self, path)
        logger.info(f"Model saved to {path}")

    @classmethod
    def load(cls, path: str) -> "FraudEnsemble":
        """Load ensemble from disk."""
        model = joblib.load(path)
        logger.info(f"Model loaded from {path}")
        return model

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def feature_importance(self, top_n: int = 20) -> pd.DataFrame:
        """Return top features by XGBoost gain."""
        self._check_fitted()
        importance = self.xgb_model.get_booster().get_score(importance_type="gain")
        df = pd.DataFrame(
            list(importance.items()), columns=["feature", "importance"]
        ).sort_values("importance", ascending=False).head(top_n)
        return df
