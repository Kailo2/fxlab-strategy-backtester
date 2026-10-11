import os
import sys
import unittest
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.indicators import sma, ema, rsi


class TestSMA(unittest.TestCase):
    def test_simple_average(self):
        s = pd.Series([1, 2, 3, 4, 5])
        result = sma(s, 3)
        self.assertTrue(pd.isna(result.iloc[0]))
        self.assertTrue(pd.isna(result.iloc[1]))
        self.assertAlmostEqual(result.iloc[2], 2.0)  # mean(1,2,3)
        self.assertAlmostEqual(result.iloc[4], 4.0)  # mean(3,4,5)

    def test_invalid_window_raises(self):
        with self.assertRaises(ValueError):
            sma(pd.Series([1, 2, 3]), 0)


class TestEMA(unittest.TestCase):
    def test_ema_converges_toward_recent_values(self):
        s = pd.Series([1.0] * 10 + [2.0] * 10)
        result = ema(s, span=5)
        # after the jump to 2.0, EMA should trend upward and end close to 2.0
        self.assertGreater(result.iloc[-1], result.iloc[10])
        self.assertAlmostEqual(result.iloc[-1], 2.0, delta=0.15)


class TestRSI(unittest.TestCase):
    def test_all_gains_gives_rsi_100(self):
        s = pd.Series([float(i) for i in range(1, 30)])  # strictly increasing
        result = rsi(s, period=14)
        self.assertAlmostEqual(result.iloc[-1], 100.0, places=1)

    def test_all_losses_gives_rsi_0(self):
        s = pd.Series([float(i) for i in range(30, 1, -1)])  # strictly decreasing
        result = rsi(s, period=14)
        self.assertAlmostEqual(result.iloc[-1], 0.0, places=1)

    def test_flat_series_has_no_defined_direction_but_no_crash(self):
        s = pd.Series([1.0] * 30)
        result = rsi(s, period=14)
        self.assertFalse(result.isna().all())

    def test_mixed_series_rsi_between_0_and_100(self):
        s = pd.Series([1, 2, 1.5, 3, 2.5, 4, 3.5, 5, 4.5, 6, 5, 7, 6.5, 8, 7, 9])
        result = rsi(s, period=5)
        valid = result.dropna()
        self.assertTrue((valid >= 0).all() and (valid <= 100).all())


if __name__ == "__main__":
    unittest.main()
