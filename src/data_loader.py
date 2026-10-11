"""
data_loader.py - synthetic OHLC price series (deterministic) for demo/testing,
plus a plain CSV loader for real data (columns: date,open,high,low,close).
"""
import random
import pandas as pd


def generate_synthetic_series(
    bars: int = 500,
    start_price: float = 1.1000,
    drift: float = 0.0001,
    volatility: float = 0.004,
    seed: int = 42,
) -> pd.DataFrame:
    """Random-walk-with-drift OHLC generator, reproducible via `seed`."""
    rng = random.Random(seed)
    rows = []
    price = start_price
    for i in range(bars):
        o = price
        change = rng.gauss(drift, volatility)
        c = max(0.0001, o * (1 + change))
        h = max(o, c) * (1 + abs(rng.gauss(0, volatility / 3)))
        l = min(o, c) * (1 - abs(rng.gauss(0, volatility / 3)))
        rows.append({"open": o, "high": h, "low": l, "close": c})
        price = c
    df = pd.DataFrame(rows)
    df.index.name = "bar"
    return df


def load_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"open", "high", "low", "close"}
    missing = required - set(df.columns.str.lower())
    if missing:
        raise ValueError(f"CSV missing required columns: {missing}")
    df.columns = [c.lower() for c in df.columns]
    return df
