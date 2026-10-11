"""
metrics.py - standard performance metrics computed from a BacktestResult.
"""
import math
import pandas as pd


def total_return_pct(equity_curve: pd.Series) -> float:
    return (equity_curve.iloc[-1] / equity_curve.iloc[0] - 1) * 100


def cagr_pct(equity_curve: pd.Series, periods_per_year: int = 252) -> float:
    n = len(equity_curve)
    if n < 2:
        return 0.0
    total_growth = equity_curve.iloc[-1] / equity_curve.iloc[0]
    years = n / periods_per_year
    if years <= 0 or total_growth <= 0:
        return 0.0
    return (total_growth ** (1 / years) - 1) * 100


def sharpe_ratio(returns: pd.Series, periods_per_year: int = 252, risk_free_rate: float = 0.0) -> float:
    excess = returns - (risk_free_rate / periods_per_year)
    std = excess.std()
    if std == 0 or pd.isna(std):
        return 0.0
    return float((excess.mean() / std) * math.sqrt(periods_per_year))


def max_drawdown_pct(equity_curve: pd.Series) -> float:
    running_max = equity_curve.cummax()
    drawdown = (equity_curve - running_max) / running_max
    return float(drawdown.min() * 100)


def win_rate_pct(trades: list) -> float:
    if not trades:
        return 0.0
    wins = sum(1 for t in trades if t.pnl_pct > 0)
    return wins / len(trades) * 100


def profit_factor(trades: list) -> float:
    gross_profit = sum(t.pnl_pct for t in trades if t.pnl_pct > 0)
    gross_loss = -sum(t.pnl_pct for t in trades if t.pnl_pct < 0)
    if gross_loss == 0:
        return float("inf") if gross_profit > 0 else 0.0
    return gross_profit / gross_loss


def summarize(result, periods_per_year: int = 252) -> dict:
    return {
        "total_return_pct": round(total_return_pct(result.equity_curve), 2),
        "cagr_pct": round(cagr_pct(result.equity_curve, periods_per_year), 2),
        "sharpe_ratio": round(sharpe_ratio(result.strategy_returns, periods_per_year), 3),
        "max_drawdown_pct": round(max_drawdown_pct(result.equity_curve), 2),
        "win_rate_pct": round(win_rate_pct(result.trades), 1),
        "profit_factor": round(profit_factor(result.trades), 2) if result.trades else 0.0,
        "num_trades": len(result.trades),
        "final_equity": round(result.final_equity, 2),
        "initial_capital": result.initial_capital,
    }
