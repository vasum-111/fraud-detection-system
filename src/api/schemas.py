"""
Pydantic schemas for the Fraud Detection API.
"""

from pydantic import BaseModel, Field, field_validator
from typing import List, Optional
from datetime import datetime
from enum import Enum


class RiskLevel(str, Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


class TransactionRequest(BaseModel):
    """Single transaction scoring request."""

    transaction_id: str = Field(..., description="Unique transaction identifier")
    card_id: str = Field(..., description="Card identifier (hashed)")
    amount: float = Field(..., gt=0, description="Transaction amount in USD")
    merchant_category: str = Field(..., description="MCC or product category code")
    timestamp: datetime = Field(..., description="Transaction timestamp (ISO 8601)")
    device_id: Optional[str] = Field(None, description="Device fingerprint")
    country: Optional[str] = Field(None, description="2-letter country code")
    email_domain: Optional[str] = Field(None, description="Customer email domain")
    billing_addr: Optional[str] = Field(None, description="Billing address zip")
    shipping_addr: Optional[str] = Field(None, description="Shipping address zip")

    @field_validator("amount")
    @classmethod
    def amount_must_be_positive(cls, v):
        if v <= 0:
            raise ValueError("Amount must be positive")
        return round(v, 2)

    model_config = {
        "json_schema_extra": {
            "example": {
                "transaction_id": "txn_abc123",
                "card_id": "card_xyz789",
                "amount": 2340.00,
                "merchant_category": "electronics",
                "timestamp": "2026-06-13T14:23:00Z",
                "device_id": "dev_456",
                "country": "US",
                "email_domain": "gmail.com",
            }
        }
    }


class FraudPredictionResponse(BaseModel):
    """Fraud scoring response with explanation."""

    transaction_id: str
    fraud_score: float = Field(..., ge=0.0, le=1.0, description="Fraud probability [0-1]")
    risk_level: RiskLevel
    top_reasons: List[str] = Field(..., description="Top risk factors driving the score")
    model_version: str
    processing_time_ms: float

    model_config = {
        "json_schema_extra": {
            "example": {
                "transaction_id": "txn_abc123",
                "fraud_score": 0.847,
                "risk_level": "HIGH",
                "top_reasons": [
                    "Amount is 8.2x your historical average",
                    "3 transactions in the past hour",
                    "First transaction in this merchant category",
                ],
                "model_version": "v1.0.0",
                "processing_time_ms": 6.4,
            }
        }
    }


class BatchTransactionRequest(BaseModel):
    """Batch scoring request (up to 1000 transactions)."""

    transactions: List[TransactionRequest] = Field(
        ..., min_length=1, max_length=1000
    )


class BatchFraudPredictionResponse(BaseModel):
    """Batch scoring response."""

    predictions: List[FraudPredictionResponse]
    total_processed: int
    high_risk_count: int
    processing_time_ms: float


class HealthResponse(BaseModel):
    """API health check response."""

    status: str
    model_version: str
    model_loaded: bool
    uptime_seconds: float
    last_prediction_at: Optional[str] = None


class MetricsResponse(BaseModel):
    """Rolling performance metrics."""

    period_hours: int
    total_predictions: int
    fraud_rate: float
    avg_fraud_score: float
    avg_latency_ms: float
    p95_latency_ms: float
    high_risk_rate: float
