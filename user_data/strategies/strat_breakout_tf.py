"""Trend-following breakout strategy redesigned for high RR trades."""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import pandas as pd
import talib as ta
from freqtrade.persistence import Trade
from freqtrade.strategy import IStrategy

from custom_stoploss import profit_protect_stoploss, time_since_open_minutes


class StratBreakoutTF(IStrategy):
    """15m trend-following breakout with multi-timeframe confirmation."""

    timeframe = "15m"
    informative_timeframe = "1h"
    can_short = False

    startup_candle_count = 200

    stoploss = -0.03
    minimal_roi = {
        "0": 0.12,
        "360": 0.06,
        "720": 0.03,
        "1080": 0.0,
    }
    use_custom_stoploss = True

    def informative_pairs(self):
        return []

    def _get_informative(self, pair: str) -> pd.DataFrame:
        if not self.dp:
            return pd.DataFrame()
        informative = self.dp.get_pair_dataframe(pair=pair, timeframe=self.informative_timeframe)
        informative = informative.copy()
        informative["ema200_1h"] = ta.EMA(informative["close"], timeperiod=200)
        informative["ema50_1h"] = ta.EMA(informative["close"], timeperiod=50)
        informative["adx_1h"] = ta.ADX(
            informative["high"], informative["low"], informative["close"], timeperiod=14
        )
        informative["rsi_1h"] = ta.RSI(informative["close"], timeperiod=14)
        informative["atr_1h"] = ta.ATR(
            informative["high"], informative["low"], informative["close"], timeperiod=14
        )
        informative["atr_pct_1h"] = informative["atr_1h"] / informative["close"]
        informative["trend_strength"] = informative["ema50_1h"] / informative["ema200_1h"] - 1.0
        informative = informative[
            [
                "close",
                "ema200_1h",
                "ema50_1h",
                "adx_1h",
                "rsi_1h",
                "atr_pct_1h",
                "trend_strength",
            ]
        ]
        informative.rename(columns={"close": "close_1h"}, inplace=True)
        informative = informative.resample("15T").ffill()
        return informative

    def populate_indicators(self, dataframe: pd.DataFrame, metadata: Dict) -> pd.DataFrame:
        dataframe = dataframe.copy()
        dataframe["ema21"] = ta.EMA(dataframe["close"], timeperiod=21)
        dataframe["ema55"] = ta.EMA(dataframe["close"], timeperiod=55)
        dataframe["ema200"] = ta.EMA(dataframe["close"], timeperiod=200)
        dataframe["slope"] = dataframe["ema21"] / dataframe["ema200"] - 1.0

        dataframe["adx"] = ta.ADX(
            dataframe["high"], dataframe["low"], dataframe["close"], timeperiod=14
        )
        dataframe["atr"] = ta.ATR(
            dataframe["high"], dataframe["low"], dataframe["close"], timeperiod=14
        )
        dataframe["atr_pct"] = dataframe["atr"] / dataframe["close"]

        swing_high = dataframe["high"].rolling(48, min_periods=10).max()
        swing_low = dataframe["low"].rolling(48, min_periods=10).min()
        dataframe["swing_high"] = swing_high
        dataframe["swing_low"] = swing_low

        dataframe["momentum"] = dataframe["close"] / dataframe["close"].shift(4) - 1.0

        informative = self._get_informative(metadata["pair"])
        if not informative.empty:
            informative.index = informative.index.tz_localize(None)
            dataframe = dataframe.join(informative, how="left")
        dataframe[
            [
                "close_1h",
                "ema200_1h",
                "ema50_1h",
                "adx_1h",
                "rsi_1h",
                "atr_pct_1h",
                "trend_strength",
            ]
        ] = dataframe[
            [
                "close_1h",
                "ema200_1h",
                "ema50_1h",
                "adx_1h",
                "rsi_1h",
                "atr_pct_1h",
                "trend_strength",
            ]
        ].ffill()
        dataframe.fillna(method="ffill", inplace=True)
        dataframe.fillna(method="bfill", inplace=True)
        return dataframe

    def populate_buy_trend(self, dataframe: pd.DataFrame, metadata: Dict) -> pd.DataFrame:
        dataframe["buy"] = 0

        uptrend = (
            (dataframe["close_1h"] > dataframe["ema200_1h"] * 1.015)
            & (dataframe["trend_strength"] > 0.02)
            & (dataframe["adx_1h"] > 22)
            & (dataframe["rsi_1h"].between(55, 80))
        )

        volatility_ok = dataframe["atr_pct"].between(0.006, 0.025)

        breakout_signal = (
            (dataframe["close"] > dataframe["swing_high"].shift(1))
            & (dataframe["ema21"] > dataframe["ema55"])
            & (dataframe["momentum"] > 0.015)
            & (dataframe["adx"] > 25)
        )

        dataframe.loc[uptrend & volatility_ok & breakout_signal, "buy"] = 1
        return dataframe

    def populate_sell_trend(self, dataframe: pd.DataFrame, metadata: Dict) -> pd.DataFrame:
        dataframe["sell"] = 0
        return dataframe

    def custom_stoploss(
        self,
        pair: str,
        trade: Trade,
        current_time: pd.Timestamp,
        current_rate: float,
        current_profit: float,
        **kwargs,
    ) -> float:
        tightened = profit_protect_stoploss(
            current_profit,
            {
                0.25: 0.12,
                0.18: 0.08,
                0.10: 0.04,
            },
        )
        if tightened is not None:
            return tightened

        if time_since_open_minutes(trade, current_time) > 720:
            return -0.01
        return self.stoploss

    order_types: Dict[str, object] = {
        "entry": "limit",
        "exit": "limit",
        "stoploss": "market",
        "stoploss_on_exchange": False,
    }

    def protections(self) -> Optional[list]:
        try:
            from protections_config import get_protections

            return get_protections()
        except ImportError:
            return None
