# Ensemble Meta-Labeled Strategy Usage Guide

This repository snapshot already contains the full scaffold for the two-layer
ensemble trading workflow described in the plan (mean-reversion + breakout
signals filtered by a calibrated meta-label and optional RL sizing). The notes
below walk through the practical steps to run it end-to-end with Freqtrade.

## 1. Prepare your environment

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -U freqtrade[hyperopt]
```

All helper scripts live under `user_data/`. Activate the virtual environment
before executing any of the commands that follow.

## 2. Download historical candles

Pull the data that covers the January 2024 – June 2025 target period (plus some
warmup):

```bash
freqtrade download-data \
  --exchange binance \
  -t 5m 15m 1h \
  --days 650
```

The data is cached under `user_data/data/` and reused by the feature builder.
Adjust the exchange or fee assumptions if you intend to trade elsewhere.

## 3. Run baseline backtests for the sub-strategies

Before the ensemble is enabled, create a trade log for each component strategy.
Export the trades so they can be converted into meta-label features.

```bash
freqtrade backtesting \
  --config user_data/config.json \
  --strategy StratScalperMR \
  --timerange 20240101-20250630 \
  --export trades \
  --export-filename user_data/backtest_results/scalper_trades.json

freqtrade backtesting \
  --config user_data/config.json \
  --strategy StratBreakoutTF \
  --timerange 20240101-20250630 \
  --export trades \
  --export-filename user_data/backtest_results/breakout_trades.json
```

You can merge the JSON exports (they share the same schema) or build features
from each file independently.

## 4. Build feature datasets for meta-label training

Convert an exported trade file into the engineered feature table required by the
meta-label model. The script will fetch OHLCV data from `user_data/data/` for the
base and informative timeframes and align the candles with each trade.

```bash
python user_data/features/make_features.py \
  --trades user_data/backtest_results/scalper_trades.json \
  --base-timeframe 5m \
  --informative-timeframes 15m 1h \
  --output user_data/features/trade_features.parquet
```

Repeat the command for any additional trade exports you want to include, then
concatenate the resulting parquet files if needed. The default location matches
the path expected by the training script.

## 5. Train and calibrate the meta-label model

Fit the XGBoost classifier with isotonic calibration on the engineered feature
set. This produces both the calibrated model and the feature scaler consumed by
`EnsembleMetaStrategy` at runtime.

```bash
python user_data/scripts/train_meta.py \
  --features user_data/features/trade_features.parquet \
  --model-output user_data/models/meta_label_xgb.pkl \
  --scaler-output user_data/models/scaler.pkl
```

Inspect the printed classification report to ensure the model is learning the
win/lose separation. The resulting pickle files are loaded automatically by the
ensemble strategy.

## 6. Backtest the ensemble strategy

With the meta-label artifacts in place, run a standard backtest to verify the
combined behaviour. The provided `config.json` already includes the recommended
pairs, fee assumptions, and base protections.

```bash
freqtrade backtesting \
  --config user_data/config.json \
  --strategy EnsembleMetaStrategy \
  --timerange 20240101-20250630 \
  --export trades \
  --export-filename user_data/backtest_results/ensemble_trades.json
```

Adjust the strategy’s risk appetite by editing `p_threshold` inside
`user_data/strategies/ensemble_meta_strategy.py` (higher values tighten the
meta-label filter for a higher win-rate) or by using freqtrade’s strategy
parameter overrides.

## 7. Hyperopt with the custom loss

To tune ROI tables, stoploss, and indicator thresholds while enforcing the
per-month win-rate requirement, run hyperopt with the supplied loss function.

```bash
freqtrade hyperopt \
  --config user_data/config.json \
  --strategy EnsembleMetaStrategy \
  --hyperopt-loss MonthlyPosAndHighWR \
  --spaces buy sell roi stoploss trailing \
  --timerange 20240101-20250630 \
  -e 200
```

Increase the number of epochs (`-e`) or expand the search spaces as you collect
more data. Review the hyperopt results before applying them to the strategy.

## 8. Enforce the monthly WR > 80% check

After tuning, validate every month individually with the automation helper. The
script re-runs backtests month-by-month and aborts if any month violates the
win-rate or profitability constraints.

```bash
python user_data/scripts/run_monthly_backtests.py \
  --strategy EnsembleMetaStrategy \
  --config user_data/config.json \
  --start 2024-01 \
  --end 2025-06 \
  --min-winrate 0.80 \
  --min-roi 0.10
```

Successful runs create JSON exports per month and a
`user_data/backtest_results/monthly/monthly_summary.json` file with the metrics.
Adjust `--min-roi` if you want a stricter or looser profitability target.

## 9. Optional: RL overlay for position sizing

If you want to experiment with reinforcement-learning-based sizing or exits,
review the scaffolding under `user_data/rl/`. The environment stub in
`env_freqtrade_like.py` and the `train_rl.py` script give you the placeholders to
integrate PPO/SAC agents once the base ensemble meets your targets.

## 10. Additional tuning tips

- **Meta-label strictness:** raise `p_threshold` toward 0.75–0.8 to prioritise
  win-rate; reduce it for more trade frequency.
- **Mean-reversion aggressiveness:** tweak the hyperopt parameters `buy_rsi`,
  `buy_z`, and `trade_timeout_minutes` in `StratScalperMR`.
- **Trend-follow bias:** adjust the breakout ROI table or `slope`/`adx`
  thresholds if strong trends are under-represented.
- **Protections:** customise the shared protections in
  `user_data/strategies/protections_config.py` to match your risk tolerance.
- **Fees and slippage:** update `fee` (and, if needed, the bid/ask strategies)
  inside `user_data/config.json` to reflect the live exchange.

Following the sequence above reproduces the pipeline needed to target the
80 %+ monthly win-rate objective while keeping each month profitable.
