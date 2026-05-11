"""
Inference helper — loads the saved model and exposes a predict() function.
"""

import os
import joblib
import numpy as np

_BASE = os.path.dirname(__file__)

_model  = None
_scaler = None
_le_cat = None
_le_merc= None


def _load():
    global _model, _scaler, _le_cat, _le_merc
    if _model is None:
        _model  = joblib.load(os.path.join(_BASE, "fraud_model.pkl"))
        _scaler = joblib.load(os.path.join(_BASE, "scaler.pkl"))
        _le_cat = joblib.load(os.path.join(_BASE, "le_cat.pkl"))
        _le_merc= joblib.load(os.path.join(_BASE, "le_merc.pkl"))


def _safe_encode(le, value):
    if value in le.classes_:
        return le.transform([value])[0]
    return 0


def predict(transaction: dict) -> dict:
    _load()

    cat_enc  = _safe_encode(_le_cat,  transaction.get("category", ""))
    merc_enc = _safe_encode(_le_merc, transaction.get("merchant", ""))

    features = np.array([[
        transaction.get("amount", 0),
        transaction.get("hour", 12),
        transaction.get("day_of_week", 0),
        transaction.get("distance_km", 0),
        transaction.get("transactions_1h", 0),
        transaction.get("transactions_24h", 0),
        int(transaction.get("is_international", False)),
        int(transaction.get("card_present", True)),
        cat_enc,
        merc_enc,
    ]])

    features_scaled = _scaler.transform(features)
    prob  = float(_model.predict_proba(features_scaled)[0][1])
    label = int(prob >= 0.5)

    risk = "HIGH" if prob >= 0.7 else "MEDIUM" if prob >= 0.4 else "LOW"

    return {
        "fraud_probability": round(prob, 4),
        "is_fraud": label,
        "risk_level": risk,
    }
