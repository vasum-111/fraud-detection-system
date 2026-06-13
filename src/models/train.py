"""
Training Pipeline for Fraud Detection System

Usage:
    python src/models/train.py --data data/train.csv --output models/
    python src/models/train.py --data data/train.csv --output models/ --eval data/val.csv
"""

import argparse
import logging
import os
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
from sklearn.metrics import (
    classification_report,
    roc_auc_score,
    average_precision_score,
)
from sklearn.model_selection import train_test_split

from src.features.feature_engineering import FraudFeatureEngineer
from src.models.ensemble import FraudEnsemble
from src.models.explainer import FraudExplainer

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


FEATURE_COLS = [
    # Velocity
    "txn_count_1h", "txn_count_6h", "txn_count_24h",
    "txn_amount_sum_1h", "txn_amount_sum_6h", "txn_amount_sum_24h",
    "unique_merchants_1h", "unique_merchants_6h",
    "velocity_ratio_1h_24h",
    # Behavioral
    "amount_zscore", "amount_vs_mean_ratio", "user_mean_amount",
    "user_std_amount", "user_txn_count_total",
    "is_new_merchant_category", "days_since_last_txn",
    # Time
    "hour_of_day", "day_of_week", "is_weekend", "is_night",
    "is_business_hours", "hour_sin", "hour_cos", "dow_sin", "dow_cos",
    # Device/Identity
    "is_high_risk_email", "email_domain_missing",
    "billing_shipping_addr_match", "device_info_missing",
    # Amount
    "amount", "amount_log", "amount_is_round", "amount_cents",
    # Raw IEEE-CIS features (passthrough)
    "ProductCD", "card4", "card6", "P_emaildomain",
    "dist1", "C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8",
    "C9", "C10", "C11", "C12", "C13", "C14",
    "D1", "D2", "D3", "D4", "D5",
    "V1", "V2", "V3", "V4", "V5",
]


def load_and_prepare(data_path: str) -> tuple:
    """Load IEEE-CIS dataset and prepare features + labels."""
    logger.info(f"Loading data from {data_path}")
    df = pd.read_csv(data_path)

    # Feature engineering
    engineer = FraudFeatureEngineer(history_df=df)
    logger.info("Running feature engineering...")
    df_features = engineer.transform_batch(df)

    # Select available feature columns
    available = [c for c in FEATURE_COLS if c in df_features.columns]
    logger.info(f"Using {len(available)} features")

    # Encode categoricals
    cat_cols = df_features[available].select_dtypes(include=["object"]).columns.tolist()
    for col in cat_cols:
        df_features[col] = pd.Categorical(df_features[col]).codes

    X = df_features[available].fillna(-999)
    y = df_features["isFraud"].astype(int)

    return X, y, available


def evaluate(model: FraudEnsemble, X: pd.DataFrame, y: pd.Series) -> dict:
    """Compute full evaluation metrics."""
    probs = model.predict_proba(X)
    preds = model.predict(X)

    metrics = {
        "roc_auc": roc_auc_score(y, probs),
        "avg_precision": average_precision_score(y, probs),
    }

    report = classification_report(y, preds, output_dict=True)
    metrics["precision_fraud"] = report.get("1", {}).get("precision", 0)
    metrics["recall_fraud"] = report.get("1", {}).get("recall", 0)
    metrics["f1_fraud"] = report.get("1", {}).get("f1-score", 0)

    return metrics


def main(args):
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    mlflow.set_experiment("fraud-detection")

    with mlflow.start_run():
        # Load data
        X, y, feature_names = load_and_prepare(args.data)

        logger.info(f"Dataset: {len(X):,} samples, {y.sum():,} fraud ({y.mean()*100:.2f}%)")

        # Train/val split (time-based for fraud — no future leakage)
        split = int(len(X) * 0.8)
        X_train, X_val = X.iloc[:split], X.iloc[split:]
        y_train, y_val = y.iloc[:split], y.iloc[split:]

        # Train
        model = FraudEnsemble()
        model.fit(X_train, y_train, X_val, y_val, optimize_threshold=True)

        # Evaluate
        metrics = evaluate(model, X_val, y_val)
        for k, v in metrics.items():
            logger.info(f"  {k}: {v:.4f}")
            mlflow.log_metric(k, v)

        # Print report
        preds = model.predict(X_val)
        print("\n" + classification_report(y_val, preds, target_names=["Legit", "Fraud"]))

        # Feature importance
        fi = model.feature_importance(top_n=15)
        logger.info("\nTop features:\n" + fi.to_string())

        # Train explainer
        explainer = FraudExplainer(model.xgb_model, feature_names)
        explainer.fit(X_train, max_samples=500)

        # Save artifacts
        model_path = output_dir / "fraud_model.pkl"
        explainer_path = output_dir / "fraud_explainer.pkl"
        fi_path = output_dir / "feature_importance.csv"

        model.save(str(model_path))
        import joblib
        joblib.dump(explainer, explainer_path)
        fi.to_csv(fi_path, index=False)

        mlflow.log_artifact(str(model_path))
        mlflow.log_artifact(str(fi_path))
        mlflow.log_params({
            "xgb_n_estimators": FraudEnsemble.XGB_PARAMS["n_estimators"],
            "xgb_max_depth": FraudEnsemble.XGB_PARAMS["max_depth"],
            "ensemble_weight_xgb": FraudEnsemble.ENSEMBLE_WEIGHT_XGB,
            "threshold": model.threshold,
        })

        logger.info(f"\nModel saved to {model_path}")
        logger.info(f"ROC-AUC: {metrics['roc_auc']:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train fraud detection model")
    parser.add_argument("--data", required=True, help="Path to training CSV")
    parser.add_argument("--output", default="models/", help="Output directory")
    args = parser.parse_args()
    main(args)
