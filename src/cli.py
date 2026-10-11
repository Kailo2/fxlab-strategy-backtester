"""
cli.py - run a backtest from the command line.

Examples:
    python -m src.cli --strategy ma_crossover --fast 10 --slow 30
    python -m src.cli --strategy rsi_mean_reversion --period 14 --csv data/mydata.csv
"""
import argparse
import json
import sys

from .data_loader import generate_synthetic_series, load_csv
from .strategies import STRATEGIES
from .backtest import run_backtest
from .metrics import summarize


def build_parser():
    p = argparse.ArgumentParser(description="FX strategy backtester")
    p.add_argument("--strategy", choices=list(STRATEGIES.keys()), default="ma_crossover")
    p.add_argument("--csv", help="path to OHLC CSV; omit to use synthetic demo data")
    p.add_argument("--bars", type=int, default=500, help="synthetic data: number of bars")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--capital", type=float, default=10000.0)
    p.add_argument("--fee-pct", type=float, default=0.0, help="per-side fee, e.g. 0.0005 = 5bps")
    p.add_argument("--periods-per-year", type=int, default=252)
    p.add_argument("--no-short", action="store_true", help="disable short positions")
    # ma_crossover params
    p.add_argument("--fast", type=int, default=10)
    p.add_argument("--slow", type=int, default=30)
    # rsi params
    p.add_argument("--period", type=int, default=14)
    p.add_argument("--oversold", type=float, default=30)
    p.add_argument("--overbought", type=float, default=70)
    p.add_argument("--plot", metavar="PATH", help="save equity curve PNG")
    p.add_argument("--json", action="store_true")
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)

    if args.csv:
        df = load_csv(args.csv)
    else:
        df = generate_synthetic_series(bars=args.bars, seed=args.seed)

    strategy_fn = STRATEGIES[args.strategy]
    allow_short = not args.no_short

    if args.strategy == "ma_crossover":
        position = strategy_fn(df, fast=args.fast, slow=args.slow, allow_short=allow_short)
    else:
        position = strategy_fn(df, period=args.period, oversold=args.oversold,
                                overbought=args.overbought, allow_short=allow_short)

    result = run_backtest(df, position, initial_capital=args.capital, fee_pct=args.fee_pct)
    stats = summarize(result, periods_per_year=args.periods_per_year)

    if args.json:
        print(json.dumps(stats, indent=2))
    else:
        print(f"Strategy:          {args.strategy}")
        print(f"Bars:              {len(df)}")
        print(f"Initial capital:   {stats['initial_capital']}")
        print(f"Final equity:      {stats['final_equity']}")
        print(f"Total return:      {stats['total_return_pct']}%")
        print(f"CAGR:              {stats['cagr_pct']}%")
        print(f"Sharpe ratio:      {stats['sharpe_ratio']}")
        print(f"Max drawdown:      {stats['max_drawdown_pct']}%")
        print(f"Trades:            {stats['num_trades']}  (win rate {stats['win_rate_pct']}%)")
        print(f"Profit factor:     {stats['profit_factor']}")

    if args.plot:
        from .plot import plot_equity_curve
        path = plot_equity_curve(result, args.plot, title=f"{args.strategy} equity curve")
        print(f"Chart saved to {path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
