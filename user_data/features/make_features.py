"""Utility script to build meta-labelling features from backtest trades."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Iterable

import numpy as np
import pandas as pd
import talib as ta
from freqtrade.data.btanalysis import load_trades
from freqtrade.data.history import load_pair_history
from freqtrade.enums import CandleType
from freqtrade.misc import timeframe_to_minutes

BASE_COLUMNS = [
    "pair",
    "open_date",
    "close_date",
    "open_rate",
    "close_rate",
    "profit_abs",
    "profit_ratio",
    "trade_duration",
]


def timeframe_to_pandas_freq(timeframe: str) -> str:
    """Map freqtrade timeframe string to pandas frequency string."""

    if timeframe.endswith("m"):
        return f"{int(timeframe[:-1])}T"
    if timeframe.endswith("h"):
        return f"{int(timeframe[:-1])}H"
    if timeframe.endswith("d"):
        return f"{int(timeframe[:-1])}D"
    raise ValueError(f"Unsupported timeframe: {timeframe}")


def compute_mr_features(dataframe: pd.DataFrame) -> pd.DataFrame:
    result = pd.DataFrame(index=dataframe.index)
    result["rsi"] = ta.RSI(dataframe["close"], timeperiod=14)
    log_ret = np.log(dataframe["close"]).diff()
    mean = log_ret.rolling(96, min_periods=20).mean()
    std = log_ret.rolling(96, min_periods=20).std().replace(0, np.nan)
    result["zret"] = ((log_ret - mean) / std).fillna(0.0)
    atr = ta.ATR(dataframe["high"], dataframe["low"], dataframe["close"], timeperiod=14)
    atr_mean = atr.rolling(96, min_periods=20).mean()
    atr_std = atr.rolling(96, min_periods=20).std().replace(0, np.nan)
    result["atr_z"] = ((atr - atr_mean) / atr_std).fillna(0.0)
    return result


def compute_breakout_features(dataframe: pd.DataFrame) -> pd.DataFrame:
    result = pd.DataFrame(index=dataframe.index)
    ema200 = ta.EMA(dataframe["close"], timeperiod=200)
    ema50 = ta.EMA(dataframe["close"], timeperiod=50)
    result["slope"] = (ema50 / ema200) - 1.0
    result["adx"] = ta.ADX(dataframe["high"], dataframe["low"], dataframe["close"], timeperiod=14)
    return result


def compute_regime_features(dataframe: pd.DataFrame) -> pd.DataFrame:
    result = pd.DataFrame(index=dataframe.index)
    ema200 = dataframe["close"].ewm(span=200).mean()
    result["slope_1h"] = dataframe["close"] / ema200 - 1.0
    return result


def enrich_with_time_features(dataframe: pd.DataFrame) -> pd.DataFrame:
    enriched = dataframe.copy()
    enriched["hour"] = enriched.index.hour
    enriched["weekday"] = enriched.index.weekday
    return enriched


def load_price_frame(pair: str, timeframe: str, data_dir: Path, candle_type: CandleType) -> pd.DataFrame:
    frame = load_pair_history(
        pair=pair,
        timeframe=timeframe,
        datadir=data_dir,
        timerange=None,
        data_format=None,
        candle_type=candle_type,
    )
    frame = frame.copy()
    frame["date"] = pd.to_datetime(frame["date"], utc=True)
    frame.set_index("date", inplace=True)
    return frame


def build_feature_frame(
    pair: str,
    base_timeframe: str,
    informative_timeframes: Iterable[str],
    data_dir: Path,
    candle_type: CandleType,
) -> pd.DataFrame:
    base = load_price_frame(pair, base_timeframe, data_dir, candle_type)
    features = enrich_with_time_features(compute_mr_features(base).join(base[["close"]]))

    for tf in informative_timeframes:
        informative = load_price_frame(pair, tf, data_dir, candle_type)
        if timeframe_to_minutes(base_timeframe) < timeframe_to_minutes(tf):
            informative = informative.reindex(features.index, method="ffill")
        if tf.endswith("m"):
            sub_features = compute_breakout_features(informative)
        else:
            sub_features = compute_regime_features(informative)
        suffix = "" if tf == base_timeframe else f"_{tf}"
        features = features.join(sub_features.add_suffix(suffix), how="left")

    features["pair"] = pair
    features.fillna(0.0, inplace=True)
    return features


def merge_trades_with_features(
    trades: pd.DataFrame,
    feature_frames: Dict[str, pd.DataFrame],
    base_timeframe: str,
) -> pd.DataFrame:
    freq = timeframe_to_pandas_freq(base_timeframe)
    trades = trades.copy()
    trades["open_candle"] = trades["open_date"].dt.floor(freq)

    rows = []
    feature_columns: set[str] = set()
    for pair, group in trades.groupby("pair"):
        features = feature_frames.get(pair)
        if features is None:
            continue
        feature_indexed = features.set_index("pair", append=True)
        feature_indexed.index.names = ["date", "pair"]
        feature_columns.update([col for col in features.columns if col != "pair"])
        joined = group.join(
            feature_indexed,
            on=["open_candle", "pair"],
            how="left",
        )
        rows.append(joined)

    if not rows:
        return pd.DataFrame()

    merged = pd.concat(rows, ignore_index=True)
    if feature_columns:
        cols = sorted(feature_columns)
        merged[cols] = merged[cols].fillna(0.0)
    merged["label"] = (merged["profit_ratio"] > 0).astype(int)
    return merged


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build ML features for meta-labelling.")
    parser.add_argument("--trades", type=Path, required=True, help="Path to exported trades JSON file")
    parser.add_argument("--data-dir", type=Path, default=Path("user_data/data"), help="Directory with cached OHLCV data")
    parser.add_argument("--base-timeframe", default="5m", help="Base strategy timeframe")
    parser.add_argument(
        "--informative-timeframes",
        nargs="*",
        default=["15m", "1h"],
        help="Additional timeframes to compute features for",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("user_data/features/trade_features.parquet"),
        help="Output parquet path",
    )
    parser.add_argument(
        "--candle-type",
        default="spot",
        choices=["spot", "futures"],
        help="Type of candles to load",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()
    trades = load_trades(
        source="file",
        db_url="",
        exportfilename=args.trades,
        no_trades=False,
        strategy=None,
    )
    if trades.empty:
        raise SystemExit("No trades found in export file")

    trades = trades[BASE_COLUMNS].copy()
    trades["open_date"] = pd.to_datetime(trades["open_date"], utc=True)
    trades["close_date"] = pd.to_datetime(trades["close_date"], utc=True)

    candle_type = CandleType.from_string(args.candle_type)

    pairs = trades["pair"].unique().tolist()
    feature_frames = {
        pair: build_feature_frame(
            pair,
            args.base_timeframe,
            args.informative_timeframes,
            args.data_dir,
            candle_type,
        )
        for pair in pairs
    }

    dataset = merge_trades_with_features(trades, feature_frames, args.base_timeframe)
    if dataset.empty:
        raise SystemExit("Failed to merge trades with features")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_parquet(args.output, index=False)
    print(f"Saved {len(dataset)} rows to {args.output}")


if __name__ == "__main__":
    main()
