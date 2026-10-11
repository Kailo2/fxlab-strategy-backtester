import os
import sys
import unittest
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.strategies import ma_crossover, rsi_mean_reversion
from src.data_loader import generate_synthetic_series


class TestMACrossover(unittest.TestCase):
    def test_fast_must_be_less_than_slow(self):
        df = generate_synthetic_series(bars=50)
        with self.assertRaises(ValueError):
            ma_crossover(df, fast=30, slow=10)

    def test_output_length_matches_input(self):
        df = generate_synthetic_series(bars=100)
        pos = ma_crossover(df, fast=5, slow=20)
        self.assertEqual(len(pos), len(df))

    def test_values_are_within_allowed_set(self):
        df = generate_synthetic_series(bars=100)
        pos = ma_crossover(df, fast=5, slow=20)
        self.assertTrue(set(pos.unique()).issubset({-1, 0, 1}))

    def test_no_short_mode_never_goes_negative(self):
        df = generate_synthetic_series(bars=100)
        pos = ma_crossover(df, fast=5, slow=20, allow_short=False)
        self.assertTrue((pos >= 0).all())

    def test_clean_uptrend_goes_long(self):
        df = pd.DataFrame({"close": [100 + i * 0.5 for i in range(60)]})
        pos = ma_crossover(df, fast=5, slow=20)
        self.assertEqual(pos.iloc[-1], 1)

    def test_clean_downtrend_goes_short(self):
        df = pd.DataFrame({"close": [100 - i * 0.5 for i in range(60)]})
        pos = ma_crossover(df, fast=5, slow=20)
        self.assertEqual(pos.iloc[-1], -1)


class TestRSIMeanReversion(unittest.TestCase):
    def test_invalid_thresholds_raise(self):
        df = generate_synthetic_series(bars=50)
        with self.assertRaises(ValueError):
            rsi_mean_reversion(df, oversold=80, overbought=20)

    def test_output_length_matches_input(self):
        df = generate_synthetic_series(bars=100)
        pos = rsi_mean_reversion(df)
        self.assertEqual(len(pos), len(df))

    def test_values_within_allowed_set(self):
        df = generate_synthetic_series(bars=100)
        pos = rsi_mean_reversion(df)
        self.assertTrue(set(pos.unique()).issubset({-1, 0, 1}))

    def test_no_short_mode_never_negative(self):
        df = generate_synthetic_series(bars=100)
        pos = rsi_mean_reversion(df, allow_short=False)
        self.assertTrue((pos >= 0).all())


if __name__ == "__main__":
    unittest.main()
