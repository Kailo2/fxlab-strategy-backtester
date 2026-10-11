"""
strategies.py - signal generators. Each strategy takes a price DataFrame
(columns: open, high, low, close) and returns a pandas Series of positions:
  1  = long
  0  = flat
 -1  = short (only used by strategies that take shorts)
indexed identically to the input.
"""
import pandas as pd

from .indicators import sma, rsi


def ma_crossover(df: pd.DataFrame, fast: int = 10, slow: int = 30, allow_short: bool = True) -> pd.Series:
    if fast >= slow:
        raise ValueError("fast window must be smaller than slow window")
    fast_ma = sma(df["close"], fast)
    slow_ma = sma(df["close"], slow)
    position = pd.Series(0, index=df.index, dtype=int)
    position[fast_ma > slow_ma] = 1
    if allow_short:
        position[fast_ma < slow_ma] = -1
    # no signal until both MAs are defined
    position[fast_ma.isna() | slow_ma.isna()] = 0
    return position


def rsi_mean_reversion(df: pd.DataFrame, period: int = 14, oversold: float = 30,
                        overbought: float = 70, allow_short: bool = True) -> pd.Series:
    if not (0 < oversold < overbought < 100):
        raise ValueError("require 0 < oversold < overbought < 100")
    r = rsi(df["close"], period)
    position = pd.Series(0, index=df.index, dtype=int)

    state = 0
    out = []
    for val in r:
        if pd.isna(val):
            out.append(0)
            continue
        if val < oversold:
            state = 1
        elif val > overbought:
            state = -1 if allow_short else 0
        elif oversold <= val <= overbought and (val > 50 and state == 1):
            state = 0  # exit long once RSI recovers past midline
        elif oversold <= val <= overbought and (val < 50 and state == -1):
            state = 0  # exit short once RSI drops past midline
        out.append(state)
    position = pd.Series(out, index=df.index, dtype=int)
    return position


STRATEGIES = {
    "ma_crossover": ma_crossover,
    "rsi_mean_reversion": rsi_mean_reversion,
}
