"""
Inference — loads the trained model and scores a new weekly record.
"""

import os, joblib
import numpy as np

_BASE = os.path.dirname(__file__)
_model, _scaler, _feature_cols = None, None, None


def _load():
    global _model, _scaler, _feature_cols
    if _model is None:
        _model        = joblib.load(os.path.join(_BASE, "outbreak_classifier.pkl"))
        _scaler       = joblib.load(os.path.join(_BASE, "scaler.pkl"))
        _feature_cols = joblib.load(os.path.join(_BASE, "feature_cols.pkl"))


def predict(record: dict) -> dict:
    _load()
    row = np.array([[record.get(c, 0) for c in _feature_cols]])
    row_s = _scaler.transform(row)
    prob  = float(_model.predict_proba(row_s)[0][1])
    risk  = "HIGH" if prob >= 0.70 else "MEDIUM" if prob >= 0.40 else "LOW"
    return {
        "outbreak_probability": round(prob, 4),
        "is_outbreak":          int(prob >= 0.50),
        "risk_level":           risk,
    }
