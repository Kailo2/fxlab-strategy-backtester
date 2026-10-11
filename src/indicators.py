"""
indicators.py - technical indicators used by the strategies, built on pandas.
"""
import pandas as pd


def sma(series: pd.Series, window: int) -> pd.Series:
    if window < 1:
        raise ValueError("window must be >= 1")
    return series.rolling(window=window, min_periods=window).mean()


def ema(series: pd.Series, span: int) -> pd.Series:
    if span < 1:
        raise ValueError("span must be >= 1")
    return series.ewm(span=span, adjust=False, min_periods=span).mean()


def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """
    Wilder's RSI. Uses an exponential (Wilder) smoothing of average
    gains/losses, which is the standard definition (matches most
    charting platforms, not the naive simple-average variant).
    """
    if period < 1:
        raise ValueError("period must be >= 1")
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()

    rs = avg_gain / avg_loss.replace(0, pd.NA)
    result = 100 - (100 / (1 + rs))
    # where avg_loss is 0 and avg_gain > 0 -> RSI is 100; where both 0 -> RSI undefined (NaN handled upstream)
    result = result.where(avg_loss != 0, 100.0)
    return result


def bollinger_bands(series: pd.Series, window: int = 20, num_std: float = 2.0):
    mid = sma(series, window)
    std = series.rolling(window=window, min_periods=window).std()
    upper = mid + num_std * std
    lower = mid - num_std * std
    return mid, upper, lower
