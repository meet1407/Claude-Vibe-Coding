"""
Generates a realistic synthetic credit card transaction dataset
with ~2% fraud rate, mimicking real-world class imbalance.
"""

import numpy as np
import pandas as pd

np.random.seed(42)

N = 50_000
FRAUD_RATE = 0.02

categories = ["Groceries", "Online Shopping", "Gas Station",
              "Restaurant", "ATM Withdrawal", "Travel", "Healthcare", "Entertainment"]
merchants  = ["Metro", "Amazon", "Shell", "Tim Hortons", "TD ATM",
              "Air Canada", "Shoppers", "Netflix", "Walmart", "Uber"]

def generate(n=N, fraud_rate=FRAUD_RATE):
    n_fraud = int(n * fraud_rate)
    n_legit = n - n_fraud

    def legit_records(size):
        return {
            "amount":          np.random.lognormal(3.5, 1.2, size).clip(1, 3000),
            "hour":            np.random.choice(range(7, 23), size),
            "day_of_week":     np.random.randint(0, 7, size),
            "category":        np.random.choice(categories, size),
            "merchant":        np.random.choice(merchants, size),
            "distance_km":     np.abs(np.random.normal(5, 8, size)).clip(0, 200),
            "transactions_1h": np.random.poisson(1.5, size).clip(0, 10),
            "transactions_24h":np.random.poisson(5, size).clip(0, 30),
            "is_international":np.random.choice([0, 1], size, p=[0.92, 0.08]),
            "card_present":    np.random.choice([0, 1], size, p=[0.25, 0.75]),
            "is_fraud":        np.zeros(size, dtype=int),
        }

    def fraud_records(size):
        return {
            "amount":          np.random.lognormal(5.5, 1.8, size).clip(50, 15000),
            "hour":            np.random.choice([0, 1, 2, 3, 22, 23], size),
            "day_of_week":     np.random.randint(0, 7, size),
            "category":        np.random.choice(["Online Shopping", "ATM Withdrawal", "Travel"], size),
            "merchant":        np.random.choice(merchants, size),
            "distance_km":     np.abs(np.random.normal(300, 200, size)).clip(50, 5000),
            "transactions_1h": np.random.poisson(6, size).clip(1, 20),
            "transactions_24h":np.random.poisson(15, size).clip(3, 50),
            "is_international":np.random.choice([0, 1], size, p=[0.3, 0.7]),
            "card_present":    np.random.choice([0, 1], size, p=[0.85, 0.15]),
            "is_fraud":        np.ones(size, dtype=int),
        }

    df = pd.concat([
        pd.DataFrame(legit_records(n_legit)),
        pd.DataFrame(fraud_records(n_fraud)),
    ]).sample(frac=1).reset_index(drop=True)

    df["transaction_id"] = ["TXN" + str(i).zfill(8) for i in range(len(df))]
    df["timestamp"] = pd.date_range("2024-01-01", periods=len(df), freq="1min")
    return df


if __name__ == "__main__":
    df = generate()
    df.to_csv("data/transactions.csv", index=False)
    print(f"Generated {len(df):,} transactions — {df['is_fraud'].sum()} fraud ({df['is_fraud'].mean()*100:.1f}%)")
