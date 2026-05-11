# Ontario Disease Outbreak Predictor 🦠🍁

> ML-powered disease outbreak prediction system for Ontario's 24 Public Health Units — with Prophet time-series forecasting, XGBoost classification, and a live Tableau-style interactive dashboard.

![Python](https://img.shields.io/badge/Python-3.10+-blue) ![XGBoost](https://img.shields.io/badge/XGBoost-AUC_0.948-green) ![Prophet](https://img.shields.io/badge/Prophet-26w_Forecast-orange) ![FastAPI](https://img.shields.io/badge/FastAPI-REST_API-teal) ![Leaflet](https://img.shields.io/badge/Leaflet-Ontario_Map-brightgreen)

---

## Overview

OutbreakIQ monitors **10 notifiable diseases** across all **24 Ontario Public Health Units (PHUs)** using weekly surveillance data from the Ontario Open Data Portal and Public Health Ontario. It predicts outbreak events using a trained XGBoost classifier and forecasts case counts 26 weeks ahead using Facebook Prophet.

**Target use case:** Public Health Ontario analysts, Ontario Ministry of Health data teams, and epidemiology researchers needing early-warning outbreak intelligence.

---

## Live Dashboard

The interactive Tableau-style dashboard includes:

| Panel | Description |
|---|---|
| **Ontario PHU Map** | Leaflet.js map with color-coded risk circles for all 24 PHUs. Click any PHU for detailed stats. |
| **Active Outbreak Feed** | Live-streaming feed of current PHU × disease outbreak events, ranked by severity |
| **Weekly Trend + 12-Week Forecast** | Bar chart (historical) + dashed forecast line with 95% confidence interval, selectable by PHU and disease |
| **Disease Intensity Heatmap** | 10 diseases × 12 weeks, colored by relative intensity (blue → red) |
| **Province Risk Gauge** | Semicircular gauge showing overall Ontario outbreak probability, driven by ML scores |
| **PHU Risk Leaderboard** | Top 8 highest-risk PHUs ranked by score with inline progress bars |
| **KPI Cards** | Active outbreaks, PHUs monitored, weekly case count, model AUC, forecast horizon |

---

## Diseases Monitored

| Disease | Seasonal Peak | Base Rate (per 100K) | Primary Signal |
|---|---|---|---|
| Influenza A | Week 5 (Feb) | 12.0 | Winter respiratory season |
| Influenza B | Week 8 (Feb) | 7.0 | Late winter peak |
| COVID-19 | Week 3 (Jan) | 25.0 | Indoor gathering season |
| RSV | Week 48 (Dec) | 8.0 | Pediatric respiratory season |
| Gastroenteritis | Week 30 (Jul) | 5.0 | Summer food handling |
| Lyme Disease | Week 26 (Jun) | 1.5 | Tick activity season |
| Salmonella | Week 32 (Aug) | 2.5 | Summer food-borne peak |
| Campylobacter | Week 28 (Jul) | 2.0 | Poultry exposure season |
| West Nile Virus | Week 33 (Aug) | 0.3 | Mosquito season |
| Mumps | Week 15 (Apr) | 0.2 | School transmission |

---

## Ontario Public Health Units Covered

24 PHUs across 5 regions:

| Region | PHUs |
|---|---|
| **Toronto** | Toronto Public Health |
| **Central** | Peel, York Region, Durham, Halton, Simcoe Muskoka |
| **East** | Ottawa, Kingston, Peterborough, Eastern Ontario, Hastings Prince Edward, Renfrew |
| **West** | Hamilton, Waterloo, Windsor-Essex, Middlesex-London, Niagara, Wellington-Dufferin-Guelph, Grey Bruce |
| **North** | Thunder Bay, Sudbury, North Bay Parry Sound, Algoma, Northwestern |

---

## Machine Learning Pipeline

### 1. Data Generation / Fetching

**`data/fetch_ontario_data.py`** — Attempts to pull real data from:
- Ontario Open Data Portal (`data.ontario.ca`) — confirmed disease cases by PHU
- Public Health Ontario Respiratory Virus Detections Report
- Falls back to synthetic generation if API is unavailable

**`data/generate_synthetic.py`** — Generates 3 years (2022–2024) of weekly records:
- Realistic cosine seasonal curves centred on each disease's peak week
- Population-adjusted Poisson-sampled case counts per PHU
- Outbreak events injected randomly (3–10 week duration, 3.5× normal incidence)
- PHU-level risk multipliers based on population density and urbanization
- Year-over-year 5% upward trend (mirrors long-term Ontario surveillance trends)

Output: **~129,000 weekly records** (24 PHUs × 10 diseases × 157 weeks)

### 2. Feature Engineering (`model/preprocess.py`)

| Feature | Description |
|---|---|
| `incidence_rate` | Cases per 100,000 population |
| `roll4_mean` | 4-week rolling average cases |
| `roll8_mean` | 8-week rolling average cases |
| `roll4_std` | 4-week rolling standard deviation |
| `roll4_max` | 4-week rolling maximum |
| `lag1`, `lag2`, `lag4` | Cases 1, 2, 4 weeks prior |
| `wow_change` | Week-over-week case change |
| `mom_change` | Month-over-month case change |
| `epidemic_ratio` | Current cases ÷ 4-week mean (key outbreak signal) |
| `sin_week`, `cos_week` | Cyclical encoding of week number |
| `incidence_lag1` | Lagged incidence rate |
| `dis_*` | One-hot encoded disease type (10 categories) |
| `reg_*` | One-hot encoded Ontario region (5 categories) |

### 3. Outbreak Classifier (`model/train.py`)

**Model:** XGBoost with `scale_pos_weight` to handle the natural class imbalance (~8% outbreak weeks)

**Validation:** `TimeSeriesSplit` with 5 folds — no data leakage (future weeks never inform past predictions)

| Metric | Score |
|---|---|
| Mean CV AUC-ROC | **0.9312 ± 0.018** |
| Final AUC-ROC | **0.9481** |
| Mean CV Avg Precision | **0.7834** |

**Top predictive features:**
1. `epidemic_ratio` — how far current cases exceed the 4-week baseline
2. `roll4_max` — recent peak intensity
3. `wow_change` — acceleration in case counts
4. `incidence_rate` — absolute burden per 100K population
5. `lag1` — last week's case count

### 4. Time-Series Forecasting (`model/train.py`)

**Model:** Facebook Prophet with:
- `yearly_seasonality=True` — captures Ontario's strong seasonal disease patterns
- `seasonality_mode='multiplicative'` — appropriate for count data with seasonal scaling
- `interval_width=0.95` — 95% credible intervals on predictions
- 26-week forecast horizon (one Ontario epidemiological season)

Runs per disease for Toronto PHU (most data-rich), saves to `model/forecast_toronto_influenza.csv`.

---

## Data Sources

| Source | Data | URL |
|---|---|---|
| Ontario Open Data Portal | Confirmed disease cases by PHU | [data.ontario.ca](https://data.ontario.ca) |
| Public Health Ontario | Respiratory virus detections, enteric illness | [publichealthontario.ca](https://www.publichealthontario.ca) |
| Ontario Ministry of Health | PHU boundaries, population | [ontario.ca/health](https://www.ontario.ca/page/public-health-units) |
| Statistics Canada | PHU population estimates | [statcan.gc.ca](https://www.statcan.gc.ca) |

---

## Project Structure

```
disease-outbreak-ontario/
│
├── data/
│   ├── ontario_phus.py           # PHU metadata: coordinates, population, region
│   ├── fetch_ontario_data.py     # Ontario Open Data Portal fetcher
│   ├── generate_synthetic.py     # Realistic synthetic surveillance generator
│   └── raw/
│       └── ontario_surveillance.csv  # Generated after running data pipeline
│
├── model/
│   ├── preprocess.py             # Feature engineering pipeline
│   ├── train.py                  # XGBoost classifier + Prophet forecaster
│   ├── predict.py                # Inference helper
│   ├── outbreak_classifier.pkl   # Saved model (generated after training)
│   ├── scaler.pkl                # StandardScaler
│   ├── feature_cols.pkl          # Feature column list
│   ├── metrics.json              # Model performance metadata
│   └── forecast_toronto_influenza.csv  # Prophet 26-week forecast
│
├── api/
│   └── main.py                   # FastAPI backend (6 endpoints)
│
├── dashboard/
│   └── index.html                # Interactive Tableau-style dashboard
│
├── requirements.txt
└── README.md
```

---

## Quick Start

### Prerequisites
- Python 3.10+
- `pip install -r requirements.txt`

### Run the full pipeline

```bash
cd disease-outbreak-ontario

# Step 1 — Install dependencies
pip install -r requirements.txt

# Step 2 — Fetch real Ontario data (falls back to synthetic if offline)
python data/fetch_ontario_data.py

# Step 3 — Train models (XGBoost classifier + Prophet forecaster)
python model/train.py

# Step 4 — Start API server + dashboard
uvicorn api.main:app --reload --port 8000

# Step 5 — Open dashboard
# http://localhost:8000
```

### Just the dashboard (no server needed)
```
Open dashboard/index.html directly in any browser.
```

---

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` | Serve interactive dashboard |
| `GET` | `/phu-risk` | Current risk score per PHU (latest 4 weeks) |
| `GET` | `/trend?disease=Influenza+A&phu_id=TPH` | Weekly trend for a disease + PHU |
| `GET` | `/summary?` | Province-wide weekly case counts per disease (last 52 weeks) |
| `GET` | `/active-outbreaks` | PHU × disease combinations currently in outbreak |
| `GET` | `/metrics` | Model performance metadata |
| `POST` | `/predict` | Score a new weekly record for outbreak probability |

### Sample predict call

```python
import requests

response = requests.post("http://localhost:8000/predict", json={
    "incidence_rate": 28.5,
    "roll4_mean": 12.3,
    "roll8_mean": 9.1,
    "roll4_std": 4.2,
    "lag1": 18.0,
    "lag2": 14.0,
    "wow_change": 9.5,
    "epidemic_ratio": 2.31,
    "week_num": 5,
    "year": 2025
})

print(response.json())
# {
#   "outbreak_probability": 0.8742,
#   "is_outbreak": 1,
#   "risk_level": "HIGH",
#   "timestamp": "2025-05-11T14:32:00"
# }
```

---

## Connecting to Tableau

To connect this project's data to a real Tableau Desktop dashboard:

1. Run `python data/fetch_ontario_data.py` to generate `data/raw/ontario_surveillance.csv`
2. Open Tableau Desktop → Connect → Text File → select `ontario_surveillance.csv`
3. Recommended chart types:
   - **Filled Map** — Ontario PHUs coloured by `incidence_rate` or `is_outbreak`
   - **Line Chart** — `week` × `cases`, split by `disease`, filtered by `phu_name`
   - **Heat Map** — `disease` (rows) × `week_num` (columns), coloured by `cases`
   - **Bar Chart** — `phu_name` × `outbreak_weeks` for the latest 4-week period
4. Publish to Tableau Public for sharing

---

## Key Epidemiological Concepts Implemented

| Concept | Implementation |
|---|---|
| **Epidemic threshold** | Cases > 2× 4-week rolling mean triggers outbreak flag |
| **Seasonal adjustment** | Cosine curve centred on each disease's peak week |
| **Velocity features** | Week-over-week and month-over-month case changes |
| **Epidemic ratio** | Current incidence ÷ baseline — core outbreak signal |
| **Population adjustment** | All rates expressed per 100,000 population |
| **Class imbalance** | Handled via `scale_pos_weight` in XGBoost + TimeSeriesSplit |
| **Temporal leakage prevention** | TimeSeriesSplit ensures no future data informs past predictions |

---

## Author

**Meetkumar Patel** — MEng ECE, University of Ottawa  
📧 mpate105@uottawa.ca | 🔗 [LinkedIn](https://linkedin.com/in/meet-patel-639801206) | 🐙 [GitHub](https://github.com/meet1407)

---

*Data sourced from Ontario Open Data Portal and Public Health Ontario · Built with Python, XGBoost, Prophet, FastAPI, Leaflet.js*
