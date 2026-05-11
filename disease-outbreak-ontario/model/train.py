"""
Trains an XGBoost outbreak classifier on Ontario disease surveillance data.
Also fits a Prophet forecasting model per disease for the top PHU (Toronto).

Outputs:
  model/outbreak_classifier.pkl
  model/scaler.pkl
  model/feature_cols.json
  model/metrics.json
  model/forecast_toronto.csv
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import json
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (roc_auc_score, classification_report,
                             average_precision_score, confusion_matrix)
from xgboost import XGBClassifier

MODEL_DIR = os.path.dirname(__file__)
DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "raw", "ontario_surveillance.csv")


def load_data():
    if not os.path.exists(DATA_PATH):
        print("Data not found — generating synthetic dataset...")
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "data"))
        from generate_synthetic import generate_and_save
        generate_and_save()
    return pd.read_csv(DATA_PATH, parse_dates=["week"])


def get_feature_cols(df):
    base = ["incidence_rate", "roll4_mean", "roll8_mean", "roll4_std", "roll4_max",
            "lag1", "lag2", "lag4", "wow_change", "mom_change",
            "epidemic_ratio", "sin_week", "cos_week", "incidence_lag1",
            "week_num", "year"]
    extra = [c for c in df.columns if c.startswith("dis_") or c.startswith("reg_")]
    return [c for c in base + extra if c in df.columns]


def add_features(df):
    df = df.copy().sort_values(["phu_id", "disease", "week"]).reset_index(drop=True)
    grp = df.groupby(["phu_id", "disease"])
    df["roll4_mean"]     = grp["cases"].transform(lambda x: x.rolling(4,  min_periods=1).mean())
    df["roll8_mean"]     = grp["cases"].transform(lambda x: x.rolling(8,  min_periods=1).mean())
    df["roll4_std"]      = grp["cases"].transform(lambda x: x.rolling(4,  min_periods=1).std().fillna(0))
    df["roll4_max"]      = grp["cases"].transform(lambda x: x.rolling(4,  min_periods=1).max())
    df["lag1"]           = grp["cases"].transform(lambda x: x.shift(1).fillna(0))
    df["lag2"]           = grp["cases"].transform(lambda x: x.shift(2).fillna(0))
    df["lag4"]           = grp["cases"].transform(lambda x: x.shift(4).fillna(0))
    df["wow_change"]     = (df["cases"] - df["lag1"]).fillna(0)
    df["mom_change"]     = (df["cases"] - df["lag4"]).fillna(0)
    df["epidemic_ratio"] = df["cases"] / (df["roll4_mean"] + 1e-5)
    df["sin_week"]       = np.sin(2 * np.pi * df["week_num"] / 52)
    df["cos_week"]       = np.cos(2 * np.pi * df["week_num"] / 52)
    df["incidence_lag1"] = df["lag1"] / df["population"] * 100_000
    df = pd.get_dummies(df, columns=["disease"], prefix="dis")
    df = pd.get_dummies(df, columns=["region"],  prefix="reg")
    return df


def train():
    print("Loading surveillance data...")
    raw = load_data()
    print(f"  {len(raw):,} weekly records | {raw['phu_id'].nunique()} PHUs | {raw['week'].nunique()} weeks")

    df = add_features(raw)
    feature_cols = get_feature_cols(df)

    X = df[feature_cols].fillna(0)
    y = df["is_outbreak"]

    # Time-series cross-validation (no data leakage)
    tscv = TimeSeriesSplit(n_splits=5)
    aucs, aps = [], []

    scale_pos = (y == 0).sum() / (y == 1).sum()

    model = XGBClassifier(
        n_estimators=400, max_depth=6, learning_rate=0.04,
        scale_pos_weight=scale_pos, subsample=0.8, colsample_bytree=0.8,
        eval_metric="aucpr", random_state=42, n_jobs=-1, verbosity=0,
    )

    print("\nTime-series cross-validation (5 folds)...")
    scaler = StandardScaler()

    for fold, (tr_idx, val_idx) in enumerate(tscv.split(X), 1):
        X_tr, X_val = X.iloc[tr_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[tr_idx], y.iloc[val_idx]

        X_tr_s  = scaler.fit_transform(X_tr)
        X_val_s = scaler.transform(X_val)

        model.fit(X_tr_s, y_tr, eval_set=[(X_val_s, y_val)], verbose=False)
        prob = model.predict_proba(X_val_s)[:, 1]
        auc = roc_auc_score(y_val, prob)
        ap  = average_precision_score(y_val, prob)
        aucs.append(auc); aps.append(ap)
        print(f"  Fold {fold}: AUC={auc:.4f}  AvgPrecision={ap:.4f}")

    print(f"\nMean AUC: {np.mean(aucs):.4f} ± {np.std(aucs):.4f}")

    # Final fit on all data
    print("\nFitting final model on full dataset...")
    X_all = scaler.fit_transform(X)
    model.fit(X_all, y, verbose=False)

    y_pred = model.predict(X_all)
    y_prob = model.predict_proba(X_all)[:, 1]
    cm = confusion_matrix(y, y_pred).tolist()

    importances = dict(zip(feature_cols,
                           model.feature_importances_.round(4).tolist()))
    top_features = dict(sorted(importances.items(), key=lambda x: -x[1])[:10])

    metrics = {
        "mean_cv_auc":   round(float(np.mean(aucs)), 4),
        "std_cv_auc":    round(float(np.std(aucs)), 4),
        "mean_cv_ap":    round(float(np.mean(aps)), 4),
        "final_auc":     round(float(roc_auc_score(y, y_prob)), 4),
        "final_ap":      round(float(average_precision_score(y, y_prob)), 4),
        "confusion_matrix": cm,
        "top_features":  top_features,
        "n_features":    len(feature_cols),
        "n_samples":     len(df),
        "outbreak_rate": round(float(y.mean()), 4),
    }

    # Save
    joblib.dump(model,        os.path.join(MODEL_DIR, "outbreak_classifier.pkl"))
    joblib.dump(scaler,       os.path.join(MODEL_DIR, "scaler.pkl"))
    joblib.dump(feature_cols, os.path.join(MODEL_DIR, "feature_cols.pkl"))
    with open(os.path.join(MODEL_DIR, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\n✅ Model saved — Final AUC: {metrics['final_auc']}")
    print(f"   Top feature: {list(top_features.items())[0]}")

    # Prophet forecast for Toronto Influenza A
    try:
        _run_prophet(raw)
    except Exception as e:
        print(f"Prophet forecast skipped: {e}")


def _run_prophet(raw):
    from prophet import Prophet
    tph_flu = (raw[(raw["phu_id"] == "TPH") & (raw["disease"] == "Influenza A")]
               .rename(columns={"week": "ds", "cases": "y"})
               [["ds", "y"]].copy())

    m = Prophet(yearly_seasonality=True, weekly_seasonality=False,
                seasonality_mode="multiplicative", interval_width=0.95)
    m.fit(tph_flu)

    future = m.make_future_dataframe(periods=26, freq="W")
    forecast = m.predict(future)
    forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].to_csv(
        os.path.join(MODEL_DIR, "forecast_toronto_influenza.csv"), index=False)
    print("   Prophet forecast saved → model/forecast_toronto_influenza.csv")


if __name__ == "__main__":
    train()
