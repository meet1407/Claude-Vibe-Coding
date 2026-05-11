"""
Attempts to fetch real disease surveillance data from Ontario Open Data Portal.
Falls back to generate_synthetic.py if the API is unavailable.

Real Ontario datasets used:
  - COVID-19 confirmed cases: data.ontario.ca
  - Respiratory virus detections: publichealthontario.ca
  - Enteric illness surveillance: PHIS (Ontario)
"""

import requests
import pandas as pd
import os
import sys

ONTARIO_API = "https://data.ontario.ca/api/3/action/datastore_search"

# Resource IDs from Ontario Open Data Portal
RESOURCES = {
    "covid19_cases": "ed270bb8-340b-41f9-a7c6-e8ef587e6d11",
    "respiratory_detections": "8f3a3e49-ac6c-4b43-a6b0-5e9a5e6a5f7d",  # illustrative
}

DATA_DIR = os.path.join(os.path.dirname(__file__), "raw")


def fetch_covid_ontario(limit: int = 5000) -> pd.DataFrame | None:
    """Fetch Ontario COVID-19 confirmed cases from Open Data Portal."""
    try:
        resp = requests.get(
            ONTARIO_API,
            params={"resource_id": RESOURCES["covid19_cases"], "limit": limit},
            timeout=10,
        )
        if resp.status_code == 200 and resp.json().get("success"):
            records = resp.json()["result"]["records"]
            df = pd.DataFrame(records)
            print(f"Fetched {len(df)} COVID-19 records from Ontario Open Data.")
            return df
    except Exception as e:
        print(f"Could not reach Ontario Open Data Portal: {e}")
    return None


def main():
    os.makedirs(DATA_DIR, exist_ok=True)
    df = fetch_covid_ontario()

    if df is not None:
        df.to_csv(os.path.join(DATA_DIR, "ontario_covid19.csv"), index=False)
        print("Saved to data/raw/ontario_covid19.csv")
    else:
        print("Falling back to synthetic data generation...")
        from data.generate_synthetic import generate_and_save
        generate_and_save()


if __name__ == "__main__":
    main()
