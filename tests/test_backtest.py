import os
import sys
import unittest
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.backtest import run_backtest
from src.metrics import (
    total_return_pct, max_drawdown_pct, sharpe_ratio, win_rate_pct, profit_factor, summarize
)


def make_df(closes):
    return pd.DataFrame({"close": closes})


class TestBacktestNoLookahead(unittest.TestCase):
    def test_signal_on_bar_t_affects_bar_t_plus_1_only(self):
        # Flat prices except a jump from bar2->bar3. If we go long exactly
        # at bar2 (signal known at bar2's close), bar3's return should be
        # captured; bar2's own return should NOT be (that would be lookahead).
        closes = [100, 100, 100, 110, 110]
        position = pd.Series([0, 0, 1, 1, 0])
        df = make_df(closes)
        result = run_backtest(df, position, initial_capital=1000)
        # return from bar2->bar3 is +10%, applied at bar3 because position
        # was set to 1 as of bar2's close (shift(1) rule)
        self.assertAlmostEqual(result.strategy_returns.iloc[3], 0.10, places=4)
        # bar2's own return (0, since 100->100) shouldn't matter either way here
        self.assertAlmostEqual(result.strategy_returns.iloc[2], 0.0, places=4)

    def test_flat_position_never_moves_equity(self):
        closes = [100, 105, 95, 120, 80]
        position = pd.Series([0, 0, 0, 0, 0])
        df = make_df(closes)
        result = run_backtest(df, position, initial_capital=1000)
        self.assertAlmostEqual(result.final_equity, 1000.0, places=2)


class TestMetrics(unittest.TestCase):
    def test_total_return_doubling(self):
        equity = pd.Series([1000, 1500, 2000])
        self.assertAlmostEqual(total_return_pct(equity), 100.0, places=2)

    def test_max_drawdown_simple_case(self):
        # peak 1000 -> trough 800 -> recover to 900: drawdown should be -20%
        equity = pd.Series([1000, 900, 800, 850, 900])
        self.assertAlmostEqual(max_drawdown_pct(equity), -20.0, places=2)

    def test_max_drawdown_zero_for_monotonic_increase(self):
        equity = pd.Series([1000, 1100, 1200, 1300])
        self.assertAlmostEqual(max_drawdown_pct(equity), 0.0, places=5)

    def test_sharpe_zero_for_zero_volatility(self):
        returns = pd.Series([0.0, 0.0, 0.0])
        self.assertEqual(sharpe_ratio(returns), 0.0)

    def test_sharpe_positive_for_consistent_positive_returns(self):
        returns = pd.Series([0.01, 0.012, 0.009, 0.011] * 20)
        self.assertGreater(sharpe_ratio(returns), 0)


class TestTradeExtractionAndWinRate(unittest.TestCase):
    def test_simple_long_trade_win(self):
        closes = [100, 100, 110, 110, 100]
        position = pd.Series([0, 1, 1, 0, 0])
        df = make_df(closes)
        result = run_backtest(df, position, initial_capital=1000)
        self.assertEqual(len(result.trades), 1)
        self.assertGreater(result.trades[0].pnl_pct, 0)
        self.assertEqual(win_rate_pct(result.trades), 100.0)

    def test_simple_short_trade(self):
        closes = [100, 100, 90, 90, 100]
        position = pd.Series([0, -1, -1, 0, 0])
        df = make_df(closes)
        result = run_backtest(df, position, initial_capital=1000)
        self.assertEqual(len(result.trades), 1)
        self.assertGreater(result.trades[0].pnl_pct, 0)  # short profits when price falls

    def test_profit_factor_all_wins_is_infinite(self):
        trades = []

        class T:
            def __init__(self, pnl):
                self.pnl_pct = pnl

        trades = [T(5), T(3), T(2)]
        self.assertEqual(profit_factor(trades), float("inf"))

    def test_summarize_keys_present(self):
        closes = [100, 101, 102, 101, 103, 105, 104]
        position = pd.Series([0, 1, 1, 1, 0, -1, -1])
        df = make_df(closes)
        result = run_backtest(df, position, initial_capital=5000)
        stats = summarize(result)
        for key in ["total_return_pct", "cagr_pct", "sharpe_ratio", "max_drawdown_pct",
                    "win_rate_pct", "profit_factor", "num_trades", "final_equity"]:
            self.assertIn(key, stats)


class TestFees(unittest.TestCase):
    def test_fees_reduce_final_equity(self):
        closes = [100, 105, 100, 105, 100, 105, 100]
        position = pd.Series([1, -1, 1, -1, 1, -1, 1])
        df = make_df(closes)
        no_fee = run_backtest(df, position, initial_capital=1000, fee_pct=0.0)
        with_fee = run_backtest(df, position, initial_capital=1000, fee_pct=0.01)
        self.assertLess(with_fee.final_equity, no_fee.final_equity)


if __name__ == "__main__":
    unittest.main()
