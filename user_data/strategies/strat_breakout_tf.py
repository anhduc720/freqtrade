"""Trend-following breakout strategy for maintaining positive expectancy."""
from __future__ import annotations

from typing import Dict, Optional

import pandas as pd
import talib as ta
from freqtrade.persistence import Trade
from freqtrade.strategy import IStrategy

from .custom_stoploss import profit_protect_stoploss, time_since_open_minutes


class StratBreakoutTF(IStrategy):
    timeframe = "15m"
    can_short = False

    stoploss = -0.02
    minimal_roi = {"0": 0.01, "120": 0.005, "360": 0}
    use_custom_stoploss = True

    startup_candle_count: int = 240

    def populate_indicators(self, dataframe: pd.DataFrame, metadata: Dict) -> pd.DataFrame:
        dataframe["ema200"] = ta.EMA(dataframe["close"], timeperiod=200)
        dataframe["ema50"] = ta.EMA(dataframe["close"], timeperiod=50)
        dataframe["adx"] = ta.ADX(
            dataframe["high"], dataframe["low"], dataframe["close"], timeperiod=14
        )
        dataframe["hh"] = dataframe["high"].rolling(96, min_periods=20).max()
        dataframe["ll"] = dataframe["low"].rolling(96, min_periods=20).min()
        dataframe["slope"] = (dataframe["ema50"] / dataframe["ema200"]) - 1.0
        dataframe["atr"] = ta.ATR(
            dataframe["high"], dataframe["low"], dataframe["close"], timeperiod=14
        )
        return dataframe

    def populate_buy_trend(self, dataframe: pd.DataFrame, metadata: Dict) -> pd.DataFrame:
        dataframe.loc[
            (
                (dataframe["close"] > dataframe["hh"].shift(1))
                & (dataframe["adx"] > 20)
                & (dataframe["slope"] > 0)
                & (dataframe["close"] > dataframe["ema200"])
            ),
            "buy",
        ] = 1
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
        """Let winners run with a progressive trailing stop."""

        tightened = profit_protect_stoploss(
            current_profit,
            {
                0.12: 0.04,
                0.06: 0.02,
                0.03: 0.01,
            },
        )
        if tightened is not None:
            return tightened

        # Fail-safe: tighten stop if trade drags for too long
        if time_since_open_minutes(trade, current_time) > 360:
            return -0.005

        return self.stoploss

    @staticmethod
    def order_types() -> Dict[str, str]:
        return {
            "buy": "limit",
            "sell": "limit",
            "stoploss": "market",
            "stoploss_on_exchange": "market",
        }

    def protections(self) -> Optional[list]:
        try:
            from .protections_config import get_protections

            return get_protections()
        except ImportError:
            return None
