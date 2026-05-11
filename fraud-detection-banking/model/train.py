"""
Trains Random Forest and XGBoost fraud detection models,
evaluates them, and saves the best one with its scaler.
"""

import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import pandas as pd
import numpy as np
import joblib
import json
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (classification_report, roc_auc_score,
                             confusion_matrix, average_precision_score)
from xgboost import XGBClassifier
from data.generate_data import generate

MODEL_DIR = os.path.dirname(__file__)


def prepare_features(df):
    df = df.copy()
    le_cat  = LabelEncoder()
    le_merc = LabelEncoder()
    df["category_enc"] = le_cat.fit_transform(df["category"])
    df["merchant_enc"] = le_merc.fit_transform(df["merchant"])

    features = ["amount", "hour", "day_of_week", "distance_km",
                "transactions_1h", "transactions_24h",
                "is_international", "card_present",
                "category_enc", "merchant_enc"]
    return df[features], le_cat, le_merc


def evaluate(name, model, X_test, y_test):
    y_pred  = model.predict(X_test)
    y_prob  = model.predict_proba(X_test)[:, 1]
    auc     = roc_auc_score(y_test, y_prob)
    ap      = average_precision_score(y_test, y_prob)
    cm      = confusion_matrix(y_test, y_pred).tolist()
    report  = classification_report(y_test, y_pred, output_dict=True)

    print(f"\n{'='*50}")
    print(f" {name}")
    print(f"{'='*50}")
    print(f"  ROC-AUC : {auc:.4f}")
    print(f"  Avg Prec: {ap:.4f}")
    print(classification_report(y_test, y_pred))
    return {"auc": auc, "avg_precision": ap, "confusion_matrix": cm, "report": report}


def train():
    print("Generating data...")
    df = generate(n=50_000)

    X, le_cat, le_merc = prepare_features(df)
    y = df["is_fraud"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)

    scaler  = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)

    # ── Random Forest ──
    rf = RandomForestClassifier(
        n_estimators=200, max_depth=12,
        class_weight="balanced", random_state=42, n_jobs=-1)
    rf.fit(X_train_s, y_train)
    rf_metrics = evaluate("Random Forest", rf, X_test_s, y_test)

    # ── XGBoost ──
    scale_pos = (y_train == 0).sum() / (y_train == 1).sum()
    xgb = XGBClassifier(
        n_estimators=300, max_depth=6, learning_rate=0.05,
        scale_pos_weight=scale_pos, eval_metric="aucpr",
        random_state=42, n_jobs=-1, verbosity=0)
    xgb.fit(X_train_s, y_train,
            eval_set=[(X_test_s, y_test)], verbose=False)
    xgb_metrics = evaluate("XGBoost", xgb, X_test_s, y_test)

    # ── Save best model ──
    best_name   = "XGBoost" if xgb_metrics["auc"] >= rf_metrics["auc"] else "RandomForest"
    best_model  = xgb        if best_name == "XGBoost" else rf
    best_metrics= xgb_metrics if best_name == "XGBoost" else rf_metrics

    joblib.dump(best_model, os.path.join(MODEL_DIR, "fraud_model.pkl"))
    joblib.dump(scaler,     os.path.join(MODEL_DIR, "scaler.pkl"))
    joblib.dump(le_cat,     os.path.join(MODEL_DIR, "le_cat.pkl"))
    joblib.dump(le_merc,    os.path.join(MODEL_DIR, "le_merc.pkl"))

    feature_names = ["amount", "hour", "day_of_week", "distance_km",
                     "transactions_1h", "transactions_24h",
                     "is_international", "card_present",
                     "category_enc", "merchant_enc"]

    importances = dict(zip(feature_names,
                           best_model.feature_importances_.round(4).tolist()))

    metadata = {
        "best_model": best_name,
        "auc":        round(best_metrics["auc"], 4),
        "avg_precision": round(best_metrics["avg_precision"], 4),
        "confusion_matrix": best_metrics["confusion_matrix"],
        "feature_importances": importances,
        "rf_auc":  round(rf_metrics["auc"], 4),
        "xgb_auc": round(xgb_metrics["auc"], 4),
    }
    with open(os.path.join(MODEL_DIR, "metrics.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\n✅ Best model: {best_name} (AUC={best_metrics['auc']:.4f})")
    print(f"   Saved to model/fraud_model.pkl")


if __name__ == "__main__":
    train()
