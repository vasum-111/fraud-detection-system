"""
Real Fraud Detection API
Loads the trained model and serves predictions via FastAPI.

Run:
    uvicorn serve:app --reload --port 8000

Test:
    http://localhost:8000/docs
"""

import pickle
import numpy as np
from pathlib import Path
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
import time

# ── Load model ─────────────────────────────────────────────────
MODEL_PATH = Path("models/fraud_model.pkl")

print("Loading fraud model...")
with open(MODEL_PATH, "rb") as f:
    artifacts = pickle.load(f)

xgb        = artifacts["xgb"]
iso        = artifacts["iso"]
scaler     = artifacts["scaler"]
calibrator = artifacts["calibrator"]
FEATURE_COLS = artifacts["feature_cols"]
THRESHOLD  = artifacts["threshold"]
ISO_MIN    = artifacts["iso_min"]
ISO_MAX    = artifacts["iso_max"]
XGB_W      = artifacts["xgb_weight"]
ISO_W      = artifacts["iso_weight"]
METRICS    = artifacts["metrics"]

print(f"Model loaded. ROC-AUC: {METRICS['roc_auc']} | Precision: {METRICS['precision']}")

# ── FastAPI app ────────────────────────────────────────────────
app = FastAPI(
    title="Fraud Detection API",
    description="Real-time credit card fraud scoring using XGBoost + Isolation Forest ensemble",
    version="1.0.0"
)


# ── Schemas ────────────────────────────────────────────────────
class TransactionRequest(BaseModel):
    # V1-V28 PCA features from the card processor
    V1: float; V2: float; V3: float; V4: float; V5: float
    V6: float; V7: float; V8: float; V9: float; V10: float
    V11: float; V12: float; V13: float; V14: float; V15: float
    V16: float; V17: float; V18: float; V19: float; V20: float
    V21: float; V22: float; V23: float; V24: float; V25: float
    V26: float; V27: float; V28: float
    Amount: float
    Time: float = 0.0  # seconds since first transaction in batch

    model_config = {"json_schema_extra": {"example": {
        "V1": -1.36, "V2": -0.07, "V3": 2.54, "V4": 1.38, "V5": -0.34,
        "V6": -0.47, "V7": 0.21, "V8": 0.10, "V9": 0.14, "V10": -0.09,
        "V11": -0.26, "V12": -0.17, "V13": 0.06, "V14": -0.22, "V15": -0.17,
        "V16": -0.27, "V17": -0.16, "V18": -0.16, "V19": -0.01, "V20": 0.07,
        "V21": 0.13, "V22": -0.04, "V23": -0.06, "V24": -0.09, "V25": -0.06,
        "V26": 0.13, "V27": -0.02, "V28": 0.01,
        "Amount": 149.62,
        "Time": 0.0
    }}}


class FraudResponse(BaseModel):
    fraud_score: float          # 0.0 - 1.0 probability
    is_fraud: bool              # True if above threshold
    risk_level: str             # LOW / MEDIUM / HIGH / CRITICAL
    confidence: str             # how confident the model is
    threshold_used: float
    model_metrics: dict


class BatchRequest(BaseModel):
    transactions: list[TransactionRequest]


class BatchResponse(BaseModel):
    results: list[FraudResponse]
    total: int
    fraud_count: int
    processing_time_ms: float


# ── Feature builder ────────────────────────────────────────────
def build_features(txn: TransactionRequest) -> np.ndarray:
    hour = (txn.Time / 3600) % 24
    log_amount    = np.log1p(txn.Amount)
    amount_zscore = (txn.Amount - 88.35) / 250.12  # dataset mean/std
    is_round      = 1 if txn.Amount % 1 == 0 else 0
    is_large      = 1 if txn.Amount > 1000 else 0
    hour_sin      = np.sin(2 * np.pi * hour / 24)
    hour_cos      = np.cos(2 * np.pi * hour / 24)

    v_features = [getattr(txn, f"V{i}") for i in range(1, 29)]
    return np.array(v_features + [
        log_amount, amount_zscore, is_round, is_large, hour_sin, hour_cos
    ]).reshape(1, -1)


def score_transaction(txn: TransactionRequest) -> FraudResponse:
    features = build_features(txn)
    features_scaled = scaler.transform(features)

    # XGBoost score
    xgb_score = xgb.predict_proba(features_scaled)[0, 1]

    # Isolation Forest score
    iso_raw = -iso.score_samples(features_scaled)[0]
    iso_norm = np.clip((iso_raw - ISO_MIN) / (ISO_MAX - ISO_MIN), 0, 1)

    # Ensemble
    ensemble = XGB_W * xgb_score + ISO_W * iso_norm
    fraud_score = float(calibrator.transform([ensemble])[0])

    is_fraud = fraud_score >= THRESHOLD

    if fraud_score >= 0.80:
        risk_level = "CRITICAL"
    elif fraud_score >= 0.60:
        risk_level = "HIGH"
    elif fraud_score >= 0.35:
        risk_level = "MEDIUM"
    else:
        risk_level = "LOW"

    confidence = "HIGH" if fraud_score > 0.85 or fraud_score < 0.10 else \
                 "MEDIUM" if fraud_score > 0.65 or fraud_score < 0.25 else "LOW"

    return FraudResponse(
        fraud_score=round(fraud_score, 4),
        is_fraud=is_fraud,
        risk_level=risk_level,
        confidence=confidence,
        threshold_used=THRESHOLD,
        model_metrics=METRICS,
    )


# ── Endpoints ──────────────────────────────────────────────────
@app.get("/")
def root():
    return {
        "service": "Fraud Detection API",
        "version": "1.0.0",
        "status": "running",
        "model_metrics": METRICS,
        "docs": "/docs"
    }


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model_loaded": True,
        "roc_auc": METRICS["roc_auc"],
        "precision": METRICS["precision"],
        "recall": METRICS["recall"],
    }


@app.post("/predict", response_model=FraudResponse)
def predict(txn: TransactionRequest):
    """Score a single transaction for fraud probability."""
    try:
        return score_transaction(txn)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/batch", response_model=BatchResponse)
def predict_batch(req: BatchRequest):
    """Score up to 500 transactions at once."""
    if len(req.transactions) > 500:
        raise HTTPException(status_code=400, detail="Max 500 transactions per batch")

    t0 = time.time()
    results = [score_transaction(txn) for txn in req.transactions]
    elapsed = (time.time() - t0) * 1000

    return BatchResponse(
        results=results,
        total=len(results),
        fraud_count=sum(1 for r in results if r.is_fraud),
        processing_time_ms=round(elapsed, 2),
    )
