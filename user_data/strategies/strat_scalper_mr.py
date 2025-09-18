"""Mean-reversion scalper redesigned to target larger reversals within uptrends."""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import pandas as pd
import talib as ta
from freqtrade.persistence import Trade
from freqtrade.strategy import IntParameter, DecimalParameter, IStrategy

from custom_stoploss import profit_protect_stoploss, time_since_open_minutes


class StratScalperMR(IStrategy):
    """Aggressive dip-buying strategy that reacts to deep pullbacks inside a bullish regime."""

    timeframe = "5m"
    informative_timeframe = "1h"
    can_short = False

    startup_candle_count = 300

    stoploss = -0.02
    minimal_roi = {
        "0": 0.07,
        "180": 0.04,
        "360": 0.02,
        "540": 0.0,
    }
    use_custom_stoploss = True

    buy_rsi = IntParameter(8, 30, default=18, space="buy", optimize=True)
    buy_z = DecimalParameter(1.5, 3.5, default=2.2, decimals=1, space="buy", optimize=True)

    trade_timeout_minutes = IntParameter(90, 360, default=180, space="sell", optimize=True)

    plot_config = {
        "main_plot": {
            "ema50": {"color": "blue"},
            "ema200": {"color": "orange"},
            "ema21": {"color": "purple"},
            "kc_lower": {"color": "red"},
        },
        "subplots": {
            "RSI": {"rsi": {"color": "green"}},
            "Z-Score": {"zret": {"color": "red"}},
        },
    }

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
        informative["slope_1h"] = informative["ema50_1h"] / informative["ema200_1h"] - 1.0
        informative = informative[["close", "ema200_1h", "ema50_1h", "adx_1h", "slope_1h"]]
        informative.rename(columns={"close": "close_1h"}, inplace=True)
        informative = informative.resample("5T").ffill()
        return informative

    def populate_indicators(self, dataframe: pd.DataFrame, metadata: Dict) -> pd.DataFrame:
        dataframe = dataframe.copy()
        dataframe["ema21"] = ta.EMA(dataframe["close"], timeperiod=21)
        dataframe["ema50"] = ta.EMA(dataframe["close"], timeperiod=55)
        dataframe["ema200"] = ta.EMA(dataframe["close"], timeperiod=200)
        dataframe["ema_trend"] = dataframe["ema21"].pct_change(periods=12)

        dataframe["rsi"] = ta.RSI(dataframe["close"], timeperiod=int(self.buy_rsi.value))
        returns = np.log(dataframe["close"]).diff()
        mean = returns.rolling(96, min_periods=20).mean()
        std = returns.rolling(96, min_periods=20).std().replace(0, np.nan)
        dataframe["zret"] = ((returns - mean) / std).fillna(0.0)

        atr = ta.ATR(dataframe["high"], dataframe["low"], dataframe["close"], timeperiod=14)
        dataframe["atr_pct"] = atr / dataframe["close"]

        kc_middle = ta.EMA(dataframe["close"], timeperiod=34)
        kc_range = atr * 1.5
        dataframe["kc_lower"] = kc_middle - kc_range

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
                "slope_1h",
            ]
        ] = dataframe[
            [
                "close_1h",
                "ema200_1h",
                "ema50_1h",
                "adx_1h",
                "slope_1h",
            ]
        ].ffill()

        dataframe.fillna(method="ffill", inplace=True)
        dataframe.fillna(method="bfill", inplace=True)
        return dataframe

    def populate_buy_trend(self, dataframe: pd.DataFrame, metadata: Dict) -> pd.DataFrame:
        dataframe["buy"] = 0
        cond_uptrend = (
            (dataframe["close_1h"] > dataframe["ema200_1h"] * 1.01)
            & (dataframe["ema50_1h"] > dataframe["ema200_1h"])
            & (dataframe["adx_1h"] > 20)
            & (dataframe["slope_1h"] > 0)
        )

        cond_pullback = (
            (dataframe["rsi"] < float(self.buy_rsi.value) - 2)
            & (dataframe["zret"] < -float(self.buy_z.value))
            & (dataframe["close"] < dataframe["kc_lower"])
            & (dataframe["ema_trend"] > -0.02)
            & (dataframe["atr_pct"].between(0.008, 0.035))
            & (dataframe["close"] > dataframe["ema200"] * 0.96)
        )

        dataframe.loc[cond_uptrend & cond_pullback, "buy"] = 1
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
                0.16: 0.08,
                0.10: 0.05,
                0.06: 0.03,
            },
        )
        if tightened is not None:
            return tightened

        open_minutes = time_since_open_minutes(trade, current_time)
        timeout = int(self.trade_timeout_minutes.value)
        if open_minutes > timeout:
            return -0.005

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
