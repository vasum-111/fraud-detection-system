"""
Real Training Script — Credit Card Fraud Detection
Dataset: Kaggle Credit Card Fraud (284,807 transactions, 492 fraud)

Run:
    python train_real.py
"""

import os
import time
import pickle
import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.metrics import (
    roc_auc_score, precision_score, recall_score,
    f1_score, classification_report, confusion_matrix
)
from sklearn.isotonic import IsotonicRegression
from xgboost import XGBClassifier

# ── Paths ──────────────────────────────────────────────────────
DATA_PATH  = Path("data/raw/creditcard.csv")
MODEL_DIR  = Path("models")
MODEL_DIR.mkdir(exist_ok=True)

print("=" * 60)
print("  FRAUD DETECTION — REAL TRAINING")
print("=" * 60)

# ── 1. Load Data ───────────────────────────────────────────────
print("\n[1/6] Loading dataset...")
df = pd.read_csv(DATA_PATH)
print(f"  Shape: {df.shape}")
print(f"  Fraud rate: {df['Class'].mean():.4%} ({df['Class'].sum()} fraud / {len(df):,} total)")

# ── 2. Feature Engineering ─────────────────────────────────────
print("\n[2/6] Engineering features...")

# Time features — convert seconds to hour of day
df["hour_of_day"] = (df["Time"] / 3600) % 24
df["hour_sin"]    = np.sin(2 * np.pi * df["hour_of_day"] / 24)
df["hour_cos"]    = np.cos(2 * np.pi * df["hour_of_day"] / 24)

# Amount features
df["log_amount"]    = np.log1p(df["Amount"])
df["amount_zscore"] = (df["Amount"] - df["Amount"].mean()) / df["Amount"].std()
df["is_round"]      = (df["Amount"] % 1 == 0).astype(int)
df["is_large"]      = (df["Amount"] > df["Amount"].quantile(0.95)).astype(int)

# Feature columns — V1-V28 + engineered features
V_COLS = [f"V{i}" for i in range(1, 29)]
FEATURE_COLS = V_COLS + [
    "log_amount", "amount_zscore", "is_round", "is_large",
    "hour_sin", "hour_cos"
]

X = df[FEATURE_COLS].values
y = df["Class"].values

print(f"  Features: {len(FEATURE_COLS)}")
print(f"  Class balance: {np.bincount(y)}")

# ── 3. Train / Test Split ──────────────────────────────────────
print("\n[3/6] Splitting data (80/20 time-based)...")
# Use time-based split — more realistic for fraud
df_sorted = df.sort_values("Time")
split_idx  = int(len(df_sorted) * 0.8)
train_idx  = df_sorted.index[:split_idx]
test_idx   = df_sorted.index[split_idx:]

X_train, X_test = X[df.index.isin(train_idx)], X[df.index.isin(test_idx)]
y_train, y_test = y[df.index.isin(train_idx)], y[df.index.isin(test_idx)]

print(f"  Train: {len(X_train):,} | Test: {len(X_test):,}")
print(f"  Train fraud: {y_train.sum()} | Test fraud: {y_test.sum()}")

# Scale
scaler  = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test  = scaler.transform(X_test)

# ── 4. Train Models ────────────────────────────────────────────
print("\n[4/6] Training models...")

# XGBoost
scale_pos = (y_train == 0).sum() / (y_train == 1).sum()
print(f"  XGBoost scale_pos_weight = {scale_pos:.1f}")
t0 = time.time()
xgb = XGBClassifier(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.05,
    scale_pos_weight=scale_pos,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric="auc",
    random_state=42,
    n_jobs=-1,
    verbosity=0,
)
xgb.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
print(f"  XGBoost trained in {time.time()-t0:.1f}s")

# Isolation Forest (on normal transactions only)
t0 = time.time()
iso = IsolationForest(
    n_estimators=200,
    contamination=0.002,
    random_state=42,
    n_jobs=-1,
)
iso.fit(X_train[y_train == 0])
print(f"  Isolation Forest trained in {time.time()-t0:.1f}s")

# ── 5. Ensemble Scoring ────────────────────────────────────────
print("\n[5/6] Evaluating ensemble...")

# XGBoost scores
xgb_scores_train = xgb.predict_proba(X_train)[:, 1]
xgb_scores_test  = xgb.predict_proba(X_test)[:, 1]

# ISO scores (flip sign: more negative = more anomalous)
iso_scores_train = -iso.score_samples(X_train)
iso_scores_test  = -iso.score_samples(X_test)

# Normalise ISO to [0,1]
iso_min, iso_max = iso_scores_train.min(), iso_scores_train.max()
iso_scores_train_n = (iso_scores_train - iso_min) / (iso_max - iso_min)
iso_scores_test_n  = (iso_scores_test  - iso_min) / (iso_max - iso_min)

# Weighted ensemble
XGB_W, ISO_W = 0.80, 0.20
ensemble_train = XGB_W * xgb_scores_train + ISO_W * iso_scores_train_n
ensemble_test  = XGB_W * xgb_scores_test  + ISO_W * iso_scores_test_n

# Calibrate with isotonic regression
calibrator = IsotonicRegression(out_of_bounds="clip")
calibrator.fit(ensemble_train, y_train)
final_scores = calibrator.transform(ensemble_test)

# Find best threshold (maximise F1)
best_f1, best_thresh = 0, 0.5
for thresh in np.arange(0.1, 0.9, 0.01):
    preds = (final_scores >= thresh).astype(int)
    f1 = f1_score(y_test, preds, zero_division=0)
    if f1 > best_f1:
        best_f1, best_thresh = f1, thresh

y_pred = (final_scores >= best_thresh).astype(int)

# Metrics
auc       = roc_auc_score(y_test, final_scores)
precision = precision_score(y_test, y_pred, zero_division=0)
recall    = recall_score(y_test, y_pred, zero_division=0)
f1        = f1_score(y_test, y_pred, zero_division=0)
tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

print(f"\n  {'='*40}")
print(f"  RESULTS")
print(f"  {'='*40}")
print(f"  ROC-AUC   : {auc:.4f}")
print(f"  Precision : {precision:.4f}")
print(f"  Recall    : {recall:.4f}")
print(f"  F1-Score  : {f1:.4f}")
print(f"  Threshold : {best_thresh:.2f}")
print(f"  {'='*40}")
print(f"  True Positives  (caught fraud)  : {tp}")
print(f"  False Positives (false alarms)  : {fp}")
print(f"  False Negatives (missed fraud)  : {fn}")
print(f"  True Negatives  (correct legit) : {tn}")
print(f"  {'='*40}")

# ── 6. Save Everything ─────────────────────────────────────────
print("\n[6/6] Saving model artifacts...")

artifacts = {
    "xgb": xgb,
    "iso": iso,
    "scaler": scaler,
    "calibrator": calibrator,
    "feature_cols": FEATURE_COLS,
    "threshold": best_thresh,
    "iso_min": iso_min,
    "iso_max": iso_max,
    "xgb_weight": XGB_W,
    "iso_weight": ISO_W,
    "metrics": {
        "roc_auc": round(auc, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "tp": int(tp), "fp": int(fp),
        "fn": int(fn), "tn": int(tn),
    }
}

with open(MODEL_DIR / "fraud_model.pkl", "wb") as f:
    pickle.dump(artifacts, f)

print(f"  Saved: models/fraud_model.pkl")
print(f"\n✅ Training complete!")
print(f"   ROC-AUC {auc:.4f} | Precision {precision:.4f} | Recall {recall:.4f}")
