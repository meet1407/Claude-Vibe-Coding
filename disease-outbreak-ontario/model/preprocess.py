"""
Feature engineering for the outbreak prediction model.
Builds rolling statistics, seasonal signals, and lag features
from the weekly surveillance dataframe.
"""

import pandas as pd
import numpy as np


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["week"] = pd.to_datetime(df["week"])
    df = df.sort_values(["phu_id", "disease", "week"]).reset_index(drop=True)

    grp = df.groupby(["phu_id", "disease"])

    # Rolling statistics (4-week and 8-week windows)
    df["roll4_mean"]  = grp["cases"].transform(lambda x: x.rolling(4,  min_periods=1).mean())
    df["roll8_mean"]  = grp["cases"].transform(lambda x: x.rolling(8,  min_periods=1).mean())
    df["roll4_std"]   = grp["cases"].transform(lambda x: x.rolling(4,  min_periods=1).std().fillna(0))
    df["roll4_max"]   = grp["cases"].transform(lambda x: x.rolling(4,  min_periods=1).max())

    # Lag features
    df["lag1"]  = grp["cases"].transform(lambda x: x.shift(1).fillna(0))
    df["lag2"]  = grp["cases"].transform(lambda x: x.shift(2).fillna(0))
    df["lag4"]  = grp["cases"].transform(lambda x: x.shift(4).fillna(0))

    # Week-over-week and month-over-month change
    df["wow_change"]  = (df["cases"] - df["lag1"]).fillna(0)
    df["mom_change"]  = (df["cases"] - df["lag4"]).fillna(0)

    # Epidemic ratio: current vs 4-week mean
    df["epidemic_ratio"] = df["cases"] / (df["roll4_mean"] + 1e-5)

    # Seasonal signals
    df["sin_week"] = np.sin(2 * np.pi * df["week_num"] / 52)
    df["cos_week"] = np.cos(2 * np.pi * df["week_num"] / 52)

    # Population-adjusted features
    df["incidence_lag1"] = df["lag1"] / df["population"] * 100_000

    # Disease one-hot encoding
    df = pd.get_dummies(df, columns=["disease"], prefix="dis", drop_first=False)

    # Region one-hot encoding
    df = pd.get_dummies(df, columns=["region"], prefix="reg", drop_first=False)

    return df


FEATURE_COLS = [
    "incidence_rate", "roll4_mean", "roll8_mean", "roll4_std", "roll4_max",
    "lag1", "lag2", "lag4", "wow_change", "mom_change",
    "epidemic_ratio", "sin_week", "cos_week", "incidence_lag1",
    "week_num", "year",
]
