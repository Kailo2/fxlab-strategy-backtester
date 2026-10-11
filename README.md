# fxlab-strategy-backtester

A small but correct **event-driven backtesting engine** for FX strategies,
built on pandas. Includes two reference strategies (MA crossover, RSI mean
reversion), Wilder's RSI/SMA/EMA indicators, and the standard performance
metrics (Sharpe, CAGR, max drawdown, win rate, profit factor).

## Why this exists

A lot of toy backtesters apply a signal to the *same bar* it was computed
on, which silently bakes in look-ahead bias and makes every strategy look
better than it is. This engine shifts position by one bar before applying
returns — explicitly tested (`test_signal_on_bar_t_affects_bar_t_plus_1_only`)
— so a signal known at bar *t*'s close only affects bar *t+1*'s return.

## Install

```bash
git clone https://github.com/Kailo2/fxlab-strategy-backtester.git
cd fxlab-strategy-backtester
pip install -r requirements.txt
```

## Usage

```bash
# MA crossover on synthetic demo data:
python -m src.cli --strategy ma_crossover --fast 10 --slow 30 --bars 500

# RSI mean reversion, with an equity curve chart:
python -m src.cli --strategy rsi_mean_reversion --period 14 --plot equity.png

# on your own CSV (columns: open,high,low,close):
python -m src.cli --strategy ma_crossover --csv data/eurusd_daily.csv --json

# long-only, with a 5bp per-side fee:
python -m src.cli --strategy ma_crossover --no-short --fee-pct 0.0005
```

Sample output:

```
Strategy:          ma_crossover
Bars:              500
Initial capital:   10000.0
Final equity:      10238.64
Total return:      2.39%
CAGR:              1.2%
Sharpe ratio:      0.226
Max drawdown:      -8.0%
Trades:            19  (win rate 52.6%)
Profit factor:     1.25
```

### As a library

```python
from src.data_loader import generate_synthetic_series
from src.strategies import ma_crossover
from src.backtest import run_backtest
from src.metrics import summarize

df = generate_synthetic_series(bars=500)
position = ma_crossover(df, fast=10, slow=30)
result = run_backtest(df, position, initial_capital=10000)
print(summarize(result))
```

## Writing your own strategy

A strategy is any function `(df, **params) -> pd.Series` returning a
position of `1` (long) / `0` (flat) / `-1` (short) per bar. Add it to
`STRATEGIES` in `src/strategies.py` and it's immediately usable from the CLI.

## Metrics

| Metric | Definition |
|---|---|
| CAGR | annualized compound growth, assumes `periods_per_year` bars/year (default 252) |
| Sharpe | mean(excess returns) / std(returns) × √(periods_per_year) |
| Max drawdown | largest peak-to-trough decline in the equity curve |
| Win rate | % of closed trades with positive PnL |
| Profit factor | gross profit ÷ gross loss across closed trades |

## Tests

```bash
python -m unittest discover -s tests -v
```

29 tests: indicator correctness (RSI against known monotonic series, SMA
against hand-computed windows), no-look-ahead verification, trade
extraction for both long and short trades, drawdown/Sharpe/fee math, and
strategy signal-shape validation.

## Disclaimer

Educational tool, synthetic demo data by default. Past performance
(simulated or real) does not guarantee future results. Not financial advice.

## License

MIT — see [LICENSE](LICENSE).
