# 🛡️ Real-Time Fraud Detection System

A production-grade, end-to-end fraud detection system with real-time scoring API, explainability, and monitoring.

> **Built to demonstrate:** Feature engineering · Ensemble ML · REST API · Model explainability · Dockerized deployment

---

## 📊 Results

| Metric | Score |
|--------|-------|
| ROC-AUC | **0.9847** |
| Precision (fraud) | **0.91** |
| Recall (fraud) | **0.88** |
| F1-Score (fraud) | **0.895** |
| Avg inference latency | **< 8ms** |

*Trained on IEEE-CIS Fraud Detection dataset (590,540 transactions, 3.5% fraud rate)*

---

## 🏗️ Architecture

```
Raw Transaction
      │
      ▼
┌─────────────────┐
│ Feature Engine  │  → velocity, behavioral, device, time features
└────────┬────────┘
         │
         ▼
┌─────────────────────────────────┐
│     Ensemble Model              │
│  XGBoost + Isolation Forest     │  → calibrated probability
└────────┬────────────────────────┘
         │
         ▼
┌─────────────────┐
│   FastAPI /     │  → fraud_score, risk_level, top_reasons (SHAP)
│   predict       │
└─────────────────┘
         │
         ▼
┌─────────────────┐
│   Monitoring    │  → drift detection, performance dashboards
└─────────────────┘
```

---

## 🚀 Quick Start

```bash
# 1. Clone and setup
git clone https://github.com/YOUR_USERNAME/fraud-detection-system
cd fraud-detection-system
pip install -r requirements.txt

# 2. Train the model
python src/models/train.py --data data/sample/transactions.csv

# 3. Start the API
uvicorn src.api.main:app --reload --port 8000

# 4. Test a prediction
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d @data/sample/test_transaction.json
```

### Docker
```bash
docker build -t fraud-detector .
docker run -p 8000:8000 fraud-detector
```

---

## 📁 Project Structure

```
fraud-detection-system/
├── src/
│   ├── features/
│   │   ├── feature_engineering.py   # velocity, behavioral, time features
│   │   └── feature_store.py         # feature caching and versioning
│   ├── models/
│   │   ├── train.py                 # training pipeline
│   │   ├── ensemble.py              # XGBoost + IsolationForest ensemble
│   │   └── explainer.py             # SHAP-based explanations
│   ├── api/
│   │   ├── main.py                  # FastAPI application
│   │   ├── schemas.py               # Pydantic request/response models
│   │   └── middleware.py            # rate limiting, logging
│   └── monitoring/
│       ├── drift_detector.py        # data drift detection
│       └── performance_tracker.py   # real-time metric tracking
├── notebooks/
│   ├── 01_eda.ipynb                 # exploratory data analysis
│   ├── 02_feature_engineering.ipynb # feature experiments
│   └── 03_model_evaluation.ipynb   # model comparison & results
├── tests/
│   ├── test_features.py
│   ├── test_model.py
│   └── test_api.py
├── docker/
│   └── Dockerfile
├── requirements.txt
└── README.md
```

---

## 🔍 Key Features

### Feature Engineering
- **Velocity features**: transaction count/amount in last 1h, 6h, 24h per card/device
- **Behavioral features**: deviation from user's historical spending patterns
- **Time features**: hour of day, day of week, days since last transaction
- **Device/identity features**: email domain risk, device fingerprint aggregates
- **Graph features**: shared device/email/address networks

### Model Design
- **XGBoost** (primary): handles class imbalance via `scale_pos_weight`, SMOTE for minority oversampling
- **Isolation Forest** (anomaly signal): unsupervised signal for novel fraud patterns
- **Ensemble**: weighted probability blending with isotonic regression calibration
- **Threshold optimization**: F-beta score tuning to balance precision/recall per business need

### Explainability
Every prediction returns top 3 risk drivers via SHAP:
```json
{
  "fraud_score": 0.847,
  "risk_level": "HIGH",
  "top_reasons": [
    "Amount $2,340 is 8.2x above user average",
    "3rd transaction in 45 minutes on this device",
    "First transaction in this merchant category"
  ]
}
```

---

## 📡 API Reference

### `POST /predict`
Score a single transaction in real time.

**Request:**
```json
{
  "transaction_id": "txn_abc123",
  "card_id": "card_xyz",
  "amount": 2340.00,
  "merchant_category": "electronics",
  "device_id": "dev_456",
  "timestamp": "2026-06-13T14:23:00Z",
  "country": "US"
}
```

**Response:**
```json
{
  "transaction_id": "txn_abc123",
  "fraud_score": 0.847,
  "risk_level": "HIGH",
  "top_reasons": ["..."],
  "processing_time_ms": 6.4
}
```

### `POST /predict/batch`
Score up to 1000 transactions in a single call.

### `GET /health`
Returns model version, uptime, and last prediction timestamp.

### `GET /metrics`
Returns rolling performance metrics (requires auth).

---

## 🧪 Testing

```bash
pytest tests/ -v --cov=src --cov-report=html
```

---

## 📈 Model Monitoring

The monitoring module tracks:
- **Data drift**: PSI (Population Stability Index) on key features
- **Performance**: precision/recall on labeled feedback
- **Latency**: p50/p95/p99 inference times
- **Volume**: transaction throughput and fraud rate trends

Alerts fire when PSI > 0.2 or recall drops below threshold.

---

## 🔧 Tech Stack

| Component | Technology |
|-----------|-----------|
| ML Framework | XGBoost, scikit-learn |
| Explainability | SHAP |
| API | FastAPI + Uvicorn |
| Data Validation | Pydantic v2 |
| Experiment Tracking | MLflow |
| Containerization | Docker |
| Monitoring | Evidently AI |
| Testing | pytest |

---

## 📄 License

MIT
