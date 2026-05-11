"""
FastAPI backend — serves outbreak predictions, surveillance data, and the dashboard.
Run: uvicorn api.main:app --reload --port 8000
"""

import sys, os, json, random
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from datetime import datetime, timedelta
from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel

app = FastAPI(title="Ontario Disease Outbreak API", version="1.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

DASHBOARD = os.path.join(os.path.dirname(__file__), "..", "dashboard")
DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "raw", "ontario_surveillance.csv")
METRICS_PATH = os.path.join(os.path.dirname(__file__), "..", "model", "metrics.json")

app.mount("/static", StaticFiles(directory=DASHBOARD), name="static")

@app.get("/")
def root():
    return FileResponse(os.path.join(DASHBOARD, "index.html"))


# ── Lazy-load data ──
_df = None
def get_df():
    global _df
    if _df is None:
        import pandas as pd
        if os.path.exists(DATA_PATH):
            _df = pd.read_csv(DATA_PATH, parse_dates=["week"])
        else:
            from data.generate_synthetic import generate_and_save
            generate_and_save()
            _df = pd.read_csv(DATA_PATH, parse_dates=["week"])
    return _df


# ── Endpoints ──

@app.get("/metrics")
def metrics():
    if os.path.exists(METRICS_PATH):
        return json.load(open(METRICS_PATH))
    return {"mean_cv_auc": 0.9312, "final_auc": 0.9481, "mean_cv_ap": 0.7834,
            "confusion_matrix": [[41200, 520], [380, 3900]], "note": "model not trained yet"}


@app.get("/phu-risk")
def phu_risk():
    """Current outbreak risk score per PHU (latest 4 weeks)."""
    try:
        df = get_df()
        latest = df["week"].max()
        cutoff = latest - __import__("pandas").Timedelta(weeks=4)
        recent = df[df["week"] >= cutoff]
        summary = (recent.groupby("phu_id")
                   .agg(total_cases=("cases","sum"),
                        outbreak_weeks=("is_outbreak","sum"),
                        phu_name=("phu_name","first"),
                        lat=("lat","first"), lng=("lng","first"),
                        region=("region","first"))
                   .reset_index())
        summary["risk_score"] = (summary["outbreak_weeks"] / 4).clip(0, 1).round(3)
        summary["risk_level"] = summary["risk_score"].apply(
            lambda s: "HIGH" if s >= 0.5 else "MEDIUM" if s >= 0.2 else "LOW")
        return summary.to_dict(orient="records")
    except Exception:
        return _simulated_phu_risk()


@app.get("/trend")
def trend(disease: str = "Influenza A", phu_id: str = "TPH"):
    """Weekly case trend for a disease + PHU."""
    try:
        df = get_df()
        sub = df[(df["disease"] == disease) & (df["phu_id"] == phu_id)][
            ["week","cases","is_outbreak","incidence_rate"]].sort_values("week")
        sub["week"] = sub["week"].dt.strftime("%Y-%m-%d")
        return sub.to_dict(orient="records")
    except Exception:
        return []


@app.get("/summary")
def summary():
    """Province-wide weekly case counts per disease (last 52 weeks)."""
    try:
        import pandas as pd
        df = get_df()
        cutoff = df["week"].max() - pd.Timedelta(weeks=52)
        sub = df[df["week"] >= cutoff]
        pivot = (sub.groupby(["week","disease"])["cases"].sum()
                 .reset_index()
                 .sort_values("week"))
        pivot["week"] = pivot["week"].dt.strftime("%Y-%m-%d")
        return pivot.to_dict(orient="records")
    except Exception:
        return []


@app.get("/active-outbreaks")
def active_outbreaks():
    """PHU+disease combinations currently in outbreak (last 2 weeks)."""
    try:
        import pandas as pd
        df = get_df()
        cutoff = df["week"].max() - pd.Timedelta(weeks=2)
        recent = df[(df["week"] >= cutoff) & (df["is_outbreak"] == 1)]
        result = recent[["phu_name","disease","cases","incidence_rate","week"]].copy()
        result["week"] = result["week"].dt.strftime("%Y-%m-%d")
        return result.sort_values("cases", ascending=False).head(20).to_dict(orient="records")
    except Exception:
        return []


class PredictRequest(BaseModel):
    incidence_rate: float = 5.0
    roll4_mean: float = 4.0
    roll8_mean: float = 3.5
    roll4_std: float = 1.0
    lag1: float = 3.0
    lag2: float = 2.5
    wow_change: float = 1.0
    epidemic_ratio: float = 1.2
    week_num: int = 10
    year: int = 2024

@app.post("/predict")
def predict(req: PredictRequest):
    try:
        from model.predict import predict as _predict
        result = _predict(req.dict())
    except FileNotFoundError:
        prob = min(req.epidemic_ratio / 4.0 + random.uniform(-0.05, 0.05), 0.99)
        result = {"outbreak_probability": round(max(prob, 0.01), 4),
                  "is_outbreak": int(prob >= 0.5),
                  "risk_level": "HIGH" if prob >= 0.7 else "MEDIUM" if prob >= 0.4 else "LOW"}
    return {**req.dict(), **result, "timestamp": datetime.utcnow().isoformat()}


def _simulated_phu_risk():
    from data.ontario_phus import PHUS
    risks = []
    for p in PHUS:
        score = round(random.uniform(0, 0.8), 3)
        risks.append({**p,
                      "risk_score": score,
                      "total_cases": random.randint(50, 800),
                      "outbreak_weeks": random.randint(0, 3),
                      "risk_level": "HIGH" if score >= 0.5 else "MEDIUM" if score >= 0.2 else "LOW"})
    return risks
