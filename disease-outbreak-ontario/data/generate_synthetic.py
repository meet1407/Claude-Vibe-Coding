"""
Generates realistic Ontario disease surveillance data for all 24 PHUs,
10 diseases, over 3 years (2022-2024) — weekly resolution.

Seasonal patterns, outbreak events, and population-adjusted rates
all mirror real Public Health Ontario surveillance reports.
"""

import numpy as np
import pandas as pd
import os
from ontario_phus import PHUS, DISEASES

np.random.seed(42)

DATA_DIR = os.path.join(os.path.dirname(__file__), "raw")

# Seasonal peak week (1–52) and relative intensity per disease
DISEASE_PROFILES = {
    "Influenza A":      {"peak_week": 5,  "season_strength": 0.90, "base_rate": 12.0, "outbreak_prob": 0.08},
    "Influenza B":      {"peak_week": 8,  "season_strength": 0.85, "base_rate": 7.0,  "outbreak_prob": 0.06},
    "COVID-19":         {"peak_week": 3,  "season_strength": 0.70, "base_rate": 25.0, "outbreak_prob": 0.10},
    "RSV":              {"peak_week": 48, "season_strength": 0.80, "base_rate": 8.0,  "outbreak_prob": 0.06},
    "Gastroenteritis":  {"peak_week": 30, "season_strength": 0.40, "base_rate": 5.0,  "outbreak_prob": 0.05},
    "Lyme Disease":     {"peak_week": 26, "season_strength": 0.95, "base_rate": 1.5,  "outbreak_prob": 0.03},
    "Salmonella":       {"peak_week": 32, "season_strength": 0.60, "base_rate": 2.5,  "outbreak_prob": 0.04},
    "Campylobacter":    {"peak_week": 28, "season_strength": 0.55, "base_rate": 2.0,  "outbreak_prob": 0.03},
    "West Nile Virus":  {"peak_week": 33, "season_strength": 0.98, "base_rate": 0.3,  "outbreak_prob": 0.02},
    "Mumps":            {"peak_week": 15, "season_strength": 0.30, "base_rate": 0.2,  "outbreak_prob": 0.02},
}

# PHU-level risk multipliers (denser/southern PHUs tend to have higher rates)
PHU_RISK = {
    "TPH":   1.30, "PPH":  1.20, "YPH":  1.15, "DPH":  1.10,
    "OPH":   1.10, "HPH":  1.05, "HPPH": 1.00, "WRPH": 1.00,
    "WEPH":  0.95, "MLHU": 0.95, "SMDHU":0.90, "NPH":  0.95,
    "KFL":   0.85, "PDH":  0.85, "WDGPH":0.88, "GBHU": 0.80,
    "EOHU":  0.80, "HPEPH":0.82, "RCDHU":0.78,
    "TBDHU": 0.75, "SDHU": 0.72, "NBPSDHU":0.70,
    "APH":   0.65, "NWHU": 0.60,
}


def seasonal_factor(week: int, peak_week: int, strength: float) -> float:
    """Cosine seasonal curve centred on peak_week."""
    diff = min(abs(week - peak_week), 52 - abs(week - peak_week))
    return 1 + strength * np.cos(np.pi * diff / 26)


def generate_surveillance(start="2022-01-01", end="2024-12-31") -> pd.DataFrame:
    weeks = pd.date_range(start, end, freq="W-MON")
    records = []

    for phu in PHUS:
        phu_id   = phu["id"]
        pop      = phu["population"]
        risk_mul = PHU_RISK.get(phu_id, 0.85)

        for disease, profile in DISEASE_PROFILES.items():
            outbreak_active = False
            outbreak_weeks_left = 0

            for date in weeks:
                week_num = date.isocalendar()[1]
                year     = date.year

                sf = seasonal_factor(week_num, profile["peak_week"], profile["season_strength"])

                # Year-over-year trend (slight increase each year)
                yr_factor = 1.0 + 0.05 * (year - 2022)

                # Outbreak event trigger
                if not outbreak_active and np.random.random() < profile["outbreak_prob"] / 52:
                    outbreak_active = True
                    outbreak_weeks_left = np.random.randint(3, 10)

                outbreak_mul = 3.5 if outbreak_active else 1.0

                # Base incidence rate per 100K, Poisson-sampled to cases
                rate = profile["base_rate"] * sf * yr_factor * risk_mul * outbreak_mul
                expected_cases = max(0, rate * pop / 100_000)
                cases = int(np.random.poisson(max(0.01, expected_cases)))

                incidence_rate = round(cases / pop * 100_000, 4)

                # Outbreak flag: cases > 2× rolling 4-week mean (approximated)
                threshold = profile["base_rate"] * sf * risk_mul * pop / 100_000 * 2
                is_outbreak = int(outbreak_active or cases > threshold)

                records.append({
                    "week":          date.strftime("%Y-%m-%d"),
                    "year":          year,
                    "week_num":      week_num,
                    "phu_id":        phu_id,
                    "phu_name":      phu["name"],
                    "region":        phu["region"],
                    "disease":       disease,
                    "cases":         cases,
                    "population":    pop,
                    "incidence_rate":incidence_rate,
                    "is_outbreak":   is_outbreak,
                    "lat":           phu["lat"],
                    "lng":           phu["lng"],
                })

                if outbreak_active:
                    outbreak_weeks_left -= 1
                    if outbreak_weeks_left <= 0:
                        outbreak_active = False

    return pd.DataFrame(records)


def generate_and_save():
    os.makedirs(DATA_DIR, exist_ok=True)
    print("Generating Ontario disease surveillance data (2022–2024)...")
    df = generate_surveillance()
    out = os.path.join(DATA_DIR, "ontario_surveillance.csv")
    df.to_csv(out, index=False)
    print(f"Saved {len(df):,} weekly records → {out}")
    print(f"  PHUs: {df['phu_id'].nunique()}  |  Diseases: {df['disease'].nunique()}")
    print(f"  Outbreak weeks: {df['is_outbreak'].sum():,} ({df['is_outbreak'].mean()*100:.1f}%)")
    return df


if __name__ == "__main__":
    generate_and_save()
