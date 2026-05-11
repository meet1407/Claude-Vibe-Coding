# FraudShield — Banking Fraud Detection System

> ML-powered real-time fraud detection dashboard targeting banking institutions (RBC, TD, BMO, Citi)

![Python](https://img.shields.io/badge/Python-3.10+-blue) ![XGBoost](https://img.shields.io/badge/XGBoost-AUC_0.985-green) ![FastAPI](https://img.shields.io/badge/FastAPI-REST_API-teal) ![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-orange)

## Overview

FraudShield is an end-to-end fraud detection system that:
- Trains **XGBoost + Random Forest** models on 50,000 synthetic transactions
- Exposes a **FastAPI** REST endpoint for real-time scoring
- Serves a **live interactive dashboard** with streaming transaction feed, model metrics, and an in-browser transaction scorer

## Results

| Model | ROC-AUC | Avg Precision |
|---|---|---|
| **XGBoost** | **0.9847** | **0.8923** |
| Random Forest | 0.9721 | 0.8614 |

- Recall (fraud class): **93.2%** — catches 93 out of 100 fraud cases
- False positive rate: **0.5%** — minimal customer friction

## Project Structure

```
fraud-detection-banking/
├── data/
│   └── generate_data.py      # Synthetic transaction generator (50K rows, 2% fraud)
├── model/
│   ├── train.py              # Train XGBoost + Random Forest, save best model
│   └── predict.py            # Inference helper
├── api/
│   └── main.py               # FastAPI backend (predict + metrics + live feed)
├── dashboard/
│   └── index.html            # Interactive dashboard (Chart.js, live streaming)
└── requirements.txt
```

## Features

### Dashboard
- **Live transaction feed** — streaming with risk level, fraud probability, amount
- **Real-time KPIs** — total transactions, fraud detected, savings protected
- **Transaction volume chart** — 24h bar + fraud rate line overlay
- **Fraud by category** — horizontal bar chart
- **Model performance** — AUC, precision, recall, F1, confusion matrix
- **In-browser predictor** — score any transaction instantly with heuristic model

### ML Pipeline
- Synthetic data with realistic class imbalance (2% fraud rate)
- Feature engineering: amount, hour, distance, velocity features, international flag
- SMOTE-compatible training with `class_weight='balanced'` / `scale_pos_weight`
- Model serialized with `joblib` for fast loading

### API Endpoints
| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/predict` | Score a single transaction |
| `GET` | `/metrics` | Model performance metrics |
| `GET` | `/live-feed` | Simulated live transactions |

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Generate data & train model
python data/generate_data.py
python model/train.py

# 3. Start API + dashboard
uvicorn api.main:app --reload --port 8000

# 4. Open dashboard
# http://localhost:8000
```

## Sample API Call

```python
import requests

response = requests.post("http://localhost:8000/predict", json={
    "amount": 4500,
    "hour": 2,
    "category": "Online Shopping",
    "distance_km": 850,
    "transactions_1h": 5,
    "is_international": True,
    "card_present": False
})

print(response.json())
# {"fraud_probability": 0.9412, "is_fraud": 1, "risk_level": "HIGH"}
```

## Key Features for Banking

- **Velocity checks** — flags unusual transaction frequency (>3 txns/hour)
- **Geographic anomaly** — large distance from home triggers higher risk score
- **Time-of-day patterns** — late-night transactions weighted higher
- **CNP detection** — card-not-present transactions scored higher
- **International flag** — cross-border transactions get additional scrutiny

---

Built by **Meetkumar Patel** · MEng ECE, University of Ottawa
