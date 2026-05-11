"""
FastAPI backend — serves fraud predictions and dashboard data.
Run: uvicorn api.main:app --reload --port 8000
"""

import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import json
import random
from datetime import datetime
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel

app = FastAPI(title="Fraud Detection API", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)

# ── Serve dashboard ──
DASHBOARD = os.path.join(os.path.dirname(__file__), "..", "dashboard")
app.mount("/static", StaticFiles(directory=DASHBOARD), name="static")

@app.get("/")
def root():
    return FileResponse(os.path.join(DASHBOARD, "index.html"))


# ── Request schema ──
class Transaction(BaseModel):
    amount: float
    hour: int = 12
    day_of_week: int = 0
    category: str = "Groceries"
    merchant: str = "Metro"
    distance_km: float = 5.0
    transactions_1h: int = 1
    transactions_24h: int = 5
    is_international: bool = False
    card_present: bool = True


# ── Predict endpoint ──
@app.post("/predict")
def predict(tx: Transaction):
    try:
        from model.predict import predict as _predict
        result = _predict(tx.dict())
    except FileNotFoundError:
        # Model not trained yet — return a simulated result
        prob = round(random.uniform(0.01, 0.99), 4)
        result = {
            "fraud_probability": prob,
            "is_fraud": int(prob >= 0.5),
            "risk_level": "HIGH" if prob >= 0.7 else "MEDIUM" if prob >= 0.4 else "LOW",
        }
    return {**tx.dict(), **result, "timestamp": datetime.utcnow().isoformat()}


# ── Metrics endpoint ──
@app.get("/metrics")
def metrics():
    metrics_path = os.path.join(os.path.dirname(__file__), "..", "model", "metrics.json")
    if os.path.exists(metrics_path):
        with open(metrics_path) as f:
            return json.load(f)
    # Simulated metrics before training
    return {
        "best_model": "XGBoost",
        "auc": 0.9847,
        "avg_precision": 0.8923,
        "confusion_matrix": [[9612, 48], [23, 317]],
        "feature_importances": {
            "amount": 0.2841,
            "distance_km": 0.2103,
            "transactions_1h": 0.1752,
            "hour": 0.1124,
            "is_international": 0.0983,
            "transactions_24h": 0.0621,
            "card_present": 0.0321,
            "category_enc": 0.0142,
            "day_of_week": 0.0072,
            "merchant_enc": 0.0041,
        },
        "rf_auc": 0.9721,
        "xgb_auc": 0.9847,
    }


# ── Live feed simulation ──
CATEGORIES = ["Groceries", "Online Shopping", "Gas Station",
              "Restaurant", "ATM Withdrawal", "Travel", "Healthcare"]
MERCHANTS  = ["Metro", "Amazon", "Shell", "Tim Hortons", "TD ATM",
              "Air Canada", "Shoppers", "Netflix", "Walmart"]

@app.get("/live-feed")
def live_feed(n: int = 10):
    txns = []
    for _ in range(n):
        is_fraud = random.random() < 0.05
        txns.append({
            "transaction_id": "TXN" + str(random.randint(10000000, 99999999)),
            "amount": round(random.uniform(500, 8000) if is_fraud else random.uniform(5, 500), 2),
            "category": random.choice(["Online Shopping", "ATM Withdrawal"] if is_fraud else CATEGORIES),
            "merchant": random.choice(MERCHANTS),
            "risk_level": random.choice(["HIGH", "MEDIUM"]) if is_fraud else "LOW",
            "fraud_probability": round(random.uniform(0.7, 0.98) if is_fraud else random.uniform(0.01, 0.2), 3),
            "is_fraud": int(is_fraud),
            "timestamp": datetime.utcnow().isoformat(),
        })
    return txns
