"""
backtest.py - core backtest engine. Turns a position series into an
equity curve and a trade log, with no look-ahead bias: a position
decided from bar t's close only affects bar t+1's return.
"""
import pandas as pd
from dataclasses import dataclass, field


@dataclass
class Trade:
    entry_index: int
    exit_index: int
    entry_price: float
    exit_price: float
    direction: int  # 1 = long, -1 = short
    pnl_pct: float


@dataclass
class BacktestResult:
    equity_curve: pd.Series
    strategy_returns: pd.Series
    position: pd.Series
    trades: list = field(default_factory=list)
    initial_capital: float = 10000.0

    @property
    def final_equity(self) -> float:
        return float(self.equity_curve.iloc[-1])


def run_backtest(
    df: pd.DataFrame,
    position: pd.Series,
    initial_capital: float = 10000.0,
    fee_pct: float = 0.0,
) -> BacktestResult:
    """
    df: must have a 'close' column, index aligned with `position`.
    position: 1 (long) / 0 (flat) / -1 (short) per bar, as of that bar's close.
    fee_pct: round-trip-agnostic per-bar fee applied whenever position changes
             (e.g. 0.0005 = 5 bps), to roughly approximate spread/commission.
    """
    if len(df) != len(position):
        raise ValueError("df and position must be the same length")

    price_returns = df["close"].pct_change()
    # shift(1): the position known at bar t's close earns bar (t+1)'s return
    applied_position = position.shift(1).fillna(0)

    strategy_returns = applied_position * price_returns

    if fee_pct > 0:
        position_change = position.diff().abs().fillna(0)
        fee_drag = position_change.shift(1).fillna(0) * fee_pct
        strategy_returns = strategy_returns - fee_drag

    strategy_returns = strategy_returns.fillna(0)
    equity_curve = (1 + strategy_returns).cumprod() * initial_capital
    equity_curve.iloc[0] = initial_capital

    trades = _extract_trades(df, position)

    return BacktestResult(
        equity_curve=equity_curve,
        strategy_returns=strategy_returns,
        position=position,
        trades=trades,
        initial_capital=initial_capital,
    )


def _extract_trades(df: pd.DataFrame, position: pd.Series) -> list:
    trades = []
    pos_values = position.values
    prices = df["close"].values
    entry_idx = None
    entry_dir = 0

    for i in range(len(pos_values)):
        current = pos_values[i]
        if entry_idx is None and current != 0:
            entry_idx = i
            entry_dir = current
        elif entry_idx is not None and current != entry_dir:
            exit_idx = i
            entry_price = prices[entry_idx]
            exit_price = prices[exit_idx]
            pnl_pct = entry_dir * (exit_price - entry_price) / entry_price * 100
            trades.append(Trade(entry_idx, exit_idx, entry_price, exit_price, entry_dir, pnl_pct))
            if current != 0:
                entry_idx = i
                entry_dir = current
            else:
                entry_idx = None
                entry_dir = 0

    if entry_idx is not None:
        exit_idx = len(pos_values) - 1
        entry_price = prices[entry_idx]
        exit_price = prices[exit_idx]
        pnl_pct = entry_dir * (exit_price - entry_price) / entry_price * 100
        trades.append(Trade(entry_idx, exit_idx, entry_price, exit_price, entry_dir, pnl_pct))

    return trades
