"""FreqAI futures strategy with adaptive risk targeting.

This strategy expects to be used together with FreqAI using a regression model
(e.g. CatboostRegressor). It forecasts a risk-adjusted return several hours into
the future and trades perpetual futures in hedge mode based on that signal.
"""

from __future__ import annotations

import logging
from functools import reduce

import numpy as np
import talib.abstract as ta
from pandas import DataFrame

from freqtrade.exchange import timeframe_to_minutes
from freqtrade.strategy import IStrategy


logger = logging.getLogger(__name__)


class FreqaiFuturesAdaptiveStrategy(IStrategy):
    """Adaptive FreqAI strategy tuned for perpetual futures."""

    timeframe = "5m"
    can_short = False
    process_only_new_candles = True
    startup_candle_count = 400
    ignore_roi_if_entry_signal = True
    minimal_roi = {"0": 10}  # rely on exit signals / FreqAI guidance
    stoploss = -0.045
    use_exit_signal = True
    exit_profit_only = False
    ignore_buy_signal = False
    trailing_stop = False

    # Strategy-level parameters
    future_horizon_hours = 8  # long horizon target
    short_horizon_hours = 2  # quick horizon target to gate trades
    long_threshold = 0.0011
    short_threshold = -0.0011
    short_horizon_threshold = 0.00065
    short_horizon_threshold_short = -0.00065
    volume_ratio_threshold = 0.72
    std_threshold = 0.018
    edge_ratio_threshold = 0.026
    realized_vol_threshold = 0.027

    protections = [
        {
            "method": "CooldownPeriod",
            "stop_duration_candles": 12,
        },
        {
            "method": "MaxDrawdown",
            "lookback_period_candles": 1440,
            "trade_limit": 5,
            "max_allowed_drawdown": 0.25,
            "stop_duration_candles": 240,
        },
    ]

    plot_config = {
        "main_plot": {
            "ema_fast": {"color": "orange"},
            "ema_slow": {"color": "blue"},
        },
        "subplots": {
            "RiskReward": {
                "&-risk_reward_short": {"color": "orange"},
                "&-risk_reward_long": {"color": "green"},
            },
            "Volatility": {
                "atr_pct": {"color": "red"},
                "vol_ratio": {"color": "purple"},
            },
        },
    }

    def leverage(
        self,
        pair: str,
        current_time,
        current_rate: float,
        proposed_leverage: float,
        max_leverage: float,
        entry_tag: str | None,
        side: str,
        **kwargs,
    ) -> float:
        """Use modest fixed leverage, capped by exchange limits."""

        base_leverage = 2.0
        return float(min(max(base_leverage, 1.0), max_leverage))

    # -------- FreqAI Hooks -------------------------------------------------
    def feature_engineering_expand_all(
        self, dataframe: DataFrame, period: int, metadata: dict, **kwargs
    ) -> DataFrame:
        """Features that should be expanded across indicator periods / timeframes."""

        dataframe[f"%-atr-{period}"] = ta.ATR(dataframe, timeperiod=period)
        dataframe[f"%-ema-{period}"] = ta.EMA(dataframe, timeperiod=period)
        dataframe[f"%-roc-{period}"] = ta.ROC(dataframe, timeperiod=period)
        dataframe[f"%-mfi-{period}"] = ta.MFI(dataframe, timeperiod=period)
        dataframe[f"%-willr-{period}"] = ta.WILLR(dataframe, timeperiod=period)

        volume_ema = ta.EMA(dataframe["volume"], timeperiod=period)
        dataframe[f"%-volume_ema-{period}"] = volume_ema

        price_range = (dataframe["high"] - dataframe["low"]) / dataframe["close"]
        dataframe[f"%-price_range-{period}"] = price_range.rolling(period).mean()

        return dataframe

    def feature_engineering_expand_basic(
        self, dataframe: DataFrame, metadata: dict, **kwargs
    ) -> DataFrame:
        """Features replicated to supplementary timeframes but not indicator periods."""

        dataframe["%-log_close"] = np.log(dataframe["close"])
        dataframe["%-return_1"] = dataframe["close"].pct_change()
        dataframe["%-return_3"] = dataframe["close"].pct_change(3)
        dataframe["%-range"] = (dataframe["high"] - dataframe["low"]) / dataframe["close"]
        dataframe["%-volume"] = dataframe["volume"]

        return dataframe

    def feature_engineering_standard(
        self, dataframe: DataFrame, metadata: dict, **kwargs
    ) -> DataFrame:
        """Features that should only be added to the base timeframe."""

        dataframe["%-day_of_week"] = dataframe["date"].dt.dayofweek
        dataframe["%-hour_of_day"] = dataframe["date"].dt.hour

        atr_pct = ta.ATR(dataframe, timeperiod=14) / dataframe["close"]
        dataframe["%-atr_pct"] = atr_pct

        ema_fast = ta.EMA(dataframe, timeperiod=21)
        ema_slow = ta.EMA(dataframe, timeperiod=55)
        dataframe["%-ema_ratio"] = ema_fast / ema_slow

        returns = dataframe["close"].pct_change()
        dataframe["%-log_return"] = returns.fillna(0.0)

        realized_vol = returns.rolling(96).std().fillna(0.0) * np.sqrt(96)
        dataframe["%-realized_vol"] = realized_vol

        vol_mean = dataframe["volume"].rolling(288).mean()
        vol_std = dataframe["volume"].rolling(288).std().replace(0.0, np.nan)
        dataframe["%-volume_zscore"] = ((dataframe["volume"] - vol_mean) / vol_std).fillna(0.0)

        dataframe["%-session_hour_sin"] = np.sin(2 * np.pi * dataframe["%-hour_of_day"] / 24)
        dataframe["%-session_hour_cos"] = np.cos(2 * np.pi * dataframe["%-hour_of_day"] / 24)
        dataframe["%-day_of_week_sin"] = np.sin(2 * np.pi * dataframe["%-day_of_week"] / 7)
        dataframe["%-day_of_week_cos"] = np.cos(2 * np.pi * dataframe["%-day_of_week"] / 7)

        dataframe["%-return_overnight"] = dataframe["open"].pct_change().fillna(0.0)

        dataframe["%-trend_velocity"] = (ema_fast - ema_slow) / dataframe["close"]

        return dataframe

    def set_freqai_targets(self, dataframe: DataFrame, metadata: dict, **kwargs) -> DataFrame:
        """Define the regression target for the ML model."""

        timeframe_minutes = max(1, timeframe_to_minutes(self.timeframe))

        long_horizon_candles = max(
            1, int(round((self.future_horizon_hours * 60) / timeframe_minutes))
        )
        short_horizon_candles = max(
            1, int(round((self.short_horizon_hours * 60) / timeframe_minutes))
        )

        future_close_long = dataframe["close"].shift(-long_horizon_candles)
        future_close_short = dataframe["close"].shift(-short_horizon_candles)

        future_return_long = (future_close_long / dataframe["close"]) - 1.0
        future_return_short = (future_close_short / dataframe["close"]) - 1.0

        atr_norm = ta.ATR(dataframe, timeperiod=14) / dataframe["close"]
        atr_future_long = atr_norm.shift(-long_horizon_candles).ffill()
        atr_future_short = atr_norm.shift(-short_horizon_candles).ffill()

        risk_adjusted_long = (future_return_long - 0.6 * atr_future_long).clip(
            lower=-0.2, upper=0.2
        )
        risk_adjusted_short = (future_return_short - 0.4 * atr_future_short).clip(
            lower=-0.15, upper=0.15
        )

        dataframe["&-risk_reward_long"] = risk_adjusted_long
        dataframe["&-risk_reward_short"] = risk_adjusted_short

        return dataframe

    # -------- Core Strategy -------------------------------------------------
    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe = self.freqai.start(dataframe, metadata, self)

        dataframe["ema_fast"] = ta.EMA(dataframe, timeperiod=21)
        dataframe["ema_slow"] = ta.EMA(dataframe, timeperiod=55)
        dataframe["trend_strength"] = (dataframe["ema_fast"] / dataframe["ema_slow"]) - 1.0

        dataframe["atr"] = ta.ATR(dataframe, timeperiod=14)
        dataframe["atr_pct"] = dataframe["atr"] / dataframe["close"]

        dataframe["vol_ma"] = dataframe["volume"].rolling(96).mean()
        dataframe["vol_ratio"] = dataframe["volume"] / dataframe["vol_ma"]

        dataframe["rsi"] = ta.RSI(dataframe, timeperiod=14)
        dataframe["mfi"] = ta.MFI(dataframe, timeperiod=14)

        dataframe["ret_1h"] = dataframe["close"].pct_change(12)
        dataframe["ret_1d"] = dataframe["close"].pct_change(288)

        dataframe.ffill(inplace=True)
        dataframe.replace([np.inf, -np.inf], np.nan, inplace=True)
        dataframe.fillna(value=0.0, inplace=True)

        return dataframe

    def populate_entry_trend(self, df: DataFrame, metadata: dict) -> DataFrame:  # noqa: C901
        required_cols = {"&-risk_reward_short", "&-risk_reward_long"}
        if not required_cols.issubset(df.columns):
            return df

        short_pred = df["&-risk_reward_short"]
        long_pred = df["&-risk_reward_long"]
        short_std = df.get("&-risk_reward_short_std", 0.0)
        long_std = df.get("&-risk_reward_long_std", 0.0)
        atr_pct = df["atr_pct"].replace(0.0, np.nan)
        edge_ratio = (short_pred / atr_pct).replace([np.inf, -np.inf], np.nan).fillna(0.0)

        realized_vol = df.get("%-realized_vol", 0.0)
        confidence_long = (short_std < self.std_threshold) & (long_std < self.std_threshold * 1.3)
        alignment_long = (
            (short_pred > self.short_horizon_threshold) & (long_pred > self.long_threshold)
        ) | (long_pred > self.long_threshold * 1.25)
        bull_regime = (df["ret_1d"] > 0.003) & (df["ret_1h"] > 0.0015)
        bear_regime = (df["ret_1d"] < -0.003) & (df["ret_1h"] < -0.0015)

        edge_long = (edge_ratio > self.edge_ratio_threshold) | (
            long_pred > self.long_threshold * 1.05
        )

        long_conditions = [
            df["do_predict"] == 1,
            confidence_long,
            alignment_long,
            edge_long,
            df["ema_fast"] > df["ema_slow"],
            df["trend_strength"] > 0.01,
            df["vol_ratio"] > self.volume_ratio_threshold,
            df["atr_pct"] < 0.07,
            realized_vol < self.realized_vol_threshold,
            df["rsi"] < 69,
            bull_regime,
        ]

        if long_conditions:
            df.loc[
                reduce(lambda x, y: x & y, long_conditions),
                ["enter_long", "enter_tag"],
            ] = (1, "freqai_long")

        confidence_short = (short_std < self.std_threshold) & (long_std < self.std_threshold * 1.4)
        alignment_short = (
            (short_pred < self.short_horizon_threshold_short) & (long_pred < self.short_threshold)
        ) | (long_pred < self.short_threshold * 1.4)
        edge_short = ((-edge_ratio) > self.edge_ratio_threshold) | (
            long_pred < self.short_threshold * 1.1
        )

        short_conditions = [
            df["do_predict"] == 1,
            confidence_short,
            alignment_short,
            edge_short,
            df["ema_fast"] < df["ema_slow"],
            df["trend_strength"] < -0.01,
            df["vol_ratio"] > self.volume_ratio_threshold,
            df["atr_pct"] < 0.07,
            realized_vol < self.realized_vol_threshold * 1.2,
            bear_regime,
        ]

        if short_conditions and self.can_short:
            df.loc[
                reduce(lambda x, y: x & y, short_conditions),
                ["enter_short", "enter_tag"],
            ] = (1, "freqai_short")

        return df

    def populate_exit_trend(self, df: DataFrame, metadata: dict) -> DataFrame:
        if "&-risk_reward_long" not in df.columns:
            return df

        short_pred = df.get("&-risk_reward_short", 0.0)
        long_pred = df["&-risk_reward_long"]
        short_std = df.get("&-risk_reward_short_std", self.std_threshold)
        atr_pct = df["atr_pct"].replace(0.0, np.nan)
        edge_ratio = (short_pred / atr_pct).replace([np.inf, -np.inf], np.nan).fillna(0.0)

        exit_long_conditions = [
            (df["do_predict"] <= 0)
            | (short_std > self.std_threshold * 1.35)
            | (edge_ratio < -self.edge_ratio_threshold * 0.2),
            (long_pred < 0) | (short_pred < 0) | (df["ret_1h"] < -0.0005),
            df["ema_fast"] < df["ema_slow"],
            df["rsi"] > 72,
        ]

        if exit_long_conditions:
            df.loc[
                reduce(lambda x, y: x & y, exit_long_conditions),
                ["exit_long", "exit_tag"],
            ] = (1, "rr_flip")

        exit_short_conditions = [
            self.can_short,
            (df["do_predict"] <= 0) | (short_std > self.std_threshold * 1.5) | (edge_ratio > 0),
            (long_pred > 0) | (short_pred > 0) | (df["ret_1h"] > 0),
            df["ema_fast"] > df["ema_slow"],
            df["rsi"] < 45,
        ]

        if exit_short_conditions:
            df.loc[
                reduce(lambda x, y: x & y, exit_short_conditions),
                ["exit_short", "exit_tag"],
            ] = (1, "rr_flip")

        return df
