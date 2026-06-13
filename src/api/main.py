"""
Fraud Detection API — FastAPI Application

Endpoints:
    POST /predict          — Score a single transaction
    POST /predict/batch    — Score up to 1000 transactions
    GET  /health           — Health check
    GET  /metrics          — Rolling performance metrics
"""

import time
import logging
import os
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Optional

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, Depends, Security
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security.api_key import APIKeyHeader

from src.api.schemas import (
    TransactionRequest,
    FraudPredictionResponse,
    BatchTransactionRequest,
    BatchFraudPredictionResponse,
    HealthResponse,
    MetricsResponse,
    RiskLevel,
)
from src.models.ensemble import FraudEnsemble
from src.models.explainer import FraudExplainer
from src.features.feature_engineering import FraudFeatureEngineer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MODEL_VERSION = os.getenv("MODEL_VERSION", "v1.0.0")
MODEL_PATH = os.getenv("MODEL_PATH", "models/fraud_model.pkl")
EXPLAINER_PATH = os.getenv("EXPLAINER_PATH", "models/fraud_explainer.pkl")
API_KEY = os.getenv("API_KEY", "dev-secret-key")

# --- Global state ---
_model: Optional[FraudEnsemble] = None
_explainer: Optional[FraudExplainer] = None
_engineer = FraudFeatureEngineer()
_start_time = time.time()
_prediction_log: list = []  # In production, use a proper time-series store


# --- Lifespan ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    global _model, _explainer
    logger.info("Loading model artifacts...")
    try:
        _model = FraudEnsemble.load(MODEL_PATH)
        _explainer = joblib.load(EXPLAINER_PATH)
        logger.info(f"Model {MODEL_VERSION} loaded successfully.")
    except FileNotFoundError:
        logger.warning(
            "Model files not found. Run 'python src/models/train.py' first. "
            "API will return 503 until models are loaded."
        )
    yield
    logger.info("Shutting down.")


# --- App ---
app = FastAPI(
    title="Fraud Detection API",
    description="Real-time transaction fraud scoring with SHAP explanations",
    version=MODEL_VERSION,
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


def verify_api_key(key: str = Security(api_key_header)):
    if key != API_KEY:
        raise HTTPException(status_code=403, detail="Invalid API key")
    return key


def require_model():
    if _model is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Run training first.",
        )


# --- Feature extraction helper ---
def txn_to_features(txn: TransactionRequest) -> dict:
    """Convert API request to model feature dict."""
    raw = {
        "card1": txn.card_id,
        "TransactionAmt": txn.amount,
        "ProductCD": txn.merchant_category,
        "DeviceInfo": "present" if txn.device_id else "",
        "P_emaildomain": txn.email_domain or "",
        "addr1": txn.billing_addr,
        "addr2": txn.shipping_addr,
        "timestamp": txn.timestamp,
    }
    # Use empty history for real-time (history lookup would come from feature store)
    features = _engineer.transform(raw, pd.DataFrame())
    return features


# --- Routes ---
@app.get("/health", response_model=HealthResponse)
def health():
    last_pred = None
    if _prediction_log:
        last_pred = _prediction_log[-1]["timestamp"]
    return HealthResponse(
        status="ok" if _model is not None else "degraded",
        model_version=MODEL_VERSION,
        model_loaded=_model is not None,
        uptime_seconds=round(time.time() - _start_time, 1),
        last_prediction_at=last_pred,
    )


@app.post("/predict", response_model=FraudPredictionResponse)
def predict(txn: TransactionRequest, _: str = Depends(verify_api_key)):
    require_model()
    t0 = time.time()

    try:
        features = txn_to_features(txn)
        score, risk_level = _model.predict_single(features)

        # Explainability
        reasons = []
        if _explainer is not None:
            try:
                reasons = _explainer.explain(features, top_n=3)
            except Exception as e:
                logger.warning(f"Explainer failed: {e}")
                reasons = ["Score based on transaction patterns"]

        latency_ms = round((time.time() - t0) * 1000, 2)

        # Log for metrics
        _prediction_log.append({
            "transaction_id": txn.transaction_id,
            "fraud_score": score,
            "risk_level": risk_level,
            "latency_ms": latency_ms,
            "timestamp": datetime.utcnow().isoformat(),
        })
        # Keep only last 10k in memory
        if len(_prediction_log) > 10000:
            _prediction_log.pop(0)

        return FraudPredictionResponse(
            transaction_id=txn.transaction_id,
            fraud_score=round(score, 4),
            risk_level=RiskLevel(risk_level),
            top_reasons=reasons,
            model_version=MODEL_VERSION,
            processing_time_ms=latency_ms,
        )

    except Exception as e:
        logger.error(f"Prediction error for {txn.transaction_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/batch", response_model=BatchFraudPredictionResponse)
def predict_batch(req: BatchTransactionRequest, _: str = Depends(verify_api_key)):
    require_model()
    t0 = time.time()

    predictions = []
    for txn in req.transactions:
        try:
            features = txn_to_features(txn)
            score, risk_level = _model.predict_single(features)
            reasons = []
            if _explainer:
                try:
                    reasons = _explainer.explain(features, top_n=3)
                except Exception:
                    reasons = ["Score based on transaction patterns"]
            predictions.append(
                FraudPredictionResponse(
                    transaction_id=txn.transaction_id,
                    fraud_score=round(score, 4),
                    risk_level=RiskLevel(risk_level),
                    top_reasons=reasons,
                    model_version=MODEL_VERSION,
                    processing_time_ms=0.0,
                )
            )
        except Exception as e:
            logger.error(f"Batch error for {txn.transaction_id}: {e}")

    total_ms = round((time.time() - t0) * 1000, 2)
    high_risk = sum(1 for p in predictions if p.risk_level in (RiskLevel.HIGH, RiskLevel.CRITICAL))

    return BatchFraudPredictionResponse(
        predictions=predictions,
        total_processed=len(predictions),
        high_risk_count=high_risk,
        processing_time_ms=total_ms,
    )


@app.get("/metrics", response_model=MetricsResponse)
def metrics(hours: int = 24, _: str = Depends(verify_api_key)):
    """Rolling metrics over the last N hours."""
    if not _prediction_log:
        return MetricsResponse(
            period_hours=hours,
            total_predictions=0,
            fraud_rate=0.0,
            avg_fraud_score=0.0,
            avg_latency_ms=0.0,
            p95_latency_ms=0.0,
            high_risk_rate=0.0,
        )

    import numpy as np
    from datetime import timezone

    cutoff = datetime.utcnow().replace(tzinfo=timezone.utc).timestamp() - hours * 3600
    recent = [
        p for p in _prediction_log
        if datetime.fromisoformat(p["timestamp"]).timestamp() > cutoff
    ]

    if not recent:
        recent = _prediction_log[-100:]

    scores = [p["fraud_score"] for p in recent]
    latencies = [p["latency_ms"] for p in recent]
    high_risk = [p for p in recent if p["risk_level"] in ("HIGH", "CRITICAL")]

    return MetricsResponse(
        period_hours=hours,
        total_predictions=len(recent),
        fraud_rate=sum(1 for s in scores if s > 0.5) / len(scores),
        avg_fraud_score=float(np.mean(scores)),
        avg_latency_ms=float(np.mean(latencies)),
        p95_latency_ms=float(np.percentile(latencies, 95)),
        high_risk_rate=len(high_risk) / len(recent),
    )
