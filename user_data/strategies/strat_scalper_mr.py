"""Mean-reversion scalping strategy designed for high win-rate."""
from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import pandas as pd
import talib as ta
from freqtrade.persistence import Trade
from freqtrade.strategy import DecimalParameter, IntParameter, IStrategy

from .custom_stoploss import profit_protect_stoploss, time_since_open_minutes


class StratScalperMR(IStrategy):
    """High-frequency mean-reversion strategy."""

    timeframe = "5m"
    can_short = False

    stoploss = -0.005
    minimal_roi = {"0": 0.003, "30": 0.001, "60": 0}
    use_custom_stoploss = True

    # Hyperoptable parameters for aggressiveness tuning.
    buy_rsi = IntParameter(5, 25, default=14, space="buy", optimize=True)
    buy_z = DecimalParameter(1.0, 3.5, default=2.0, decimals=1, space="buy", optimize=True)

    # Time-based exit once the trade stalls.
    trade_timeout_minutes = IntParameter(15, 60, default=30, space="sell", optimize=True)

    plot_config = {
        "main_plot": {
            "ema_fast": {"color": "blue"},
            "ema_slow": {"color": "orange"},
        },
        "subplots": {
            "RSI": {
                "rsi": {"color": "green"},
            },
            "Z-Score": {
                "zret": {"color": "red"},
            },
        },
    }

    def populate_indicators(self, dataframe: pd.DataFrame, metadata: Dict) -> pd.DataFrame:
        dataframe["rsi"] = ta.RSI(dataframe["close"], timeperiod=int(self.buy_rsi.value))
        returns = np.log(dataframe["close"]).diff()
        mean = returns.rolling(96, min_periods=20).mean()
        std = returns.rolling(96, min_periods=20).std().replace(0, np.nan)
        dataframe["zret"] = (returns - mean) / std
        dataframe["zret"].fillna(0.0, inplace=True)

        dataframe["atr"] = ta.ATR(
            dataframe["high"], dataframe["low"], dataframe["close"], timeperiod=14
        )
        atr_mean = dataframe["atr"].rolling(96, min_periods=20).mean()
        atr_std = dataframe["atr"].rolling(96, min_periods=20).std().replace(0, np.nan)
        dataframe["atr_z"] = (dataframe["atr"] - atr_mean) / atr_std
        dataframe["atr_z"].fillna(0.0, inplace=True)

        dataframe["ema_fast"] = ta.EMA(dataframe["close"], timeperiod=21)
        dataframe["ema_slow"] = ta.EMA(dataframe["close"], timeperiod=50)
        return dataframe

    def populate_buy_trend(self, dataframe: pd.DataFrame, metadata: Dict) -> pd.DataFrame:
        dataframe.loc[
            (
                (dataframe["rsi"] < 30)
                & (dataframe["zret"] < -float(self.buy_z.value))
                & (dataframe["atr_z"] < 1.5)
                & (dataframe["close"] < dataframe["ema_fast"])
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
        """Implement adaptive stop based on elapsed time and reversion strength."""

        tightened = profit_protect_stoploss(current_profit, {0.004: -0.0005})
        if tightened is not None:
            return tightened

        # Time-based exit to avoid capital lock-up in sideways regimes.
        open_minutes = time_since_open_minutes(trade, current_time)
        timeout = int(self.trade_timeout_minutes.value)
        if open_minutes > timeout:
            return -0.0001

        # Fallback to initial stoploss.
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
        """Delegate protections to external configuration if available."""
        try:
            from .protections_config import get_protections

            return get_protections()
        except ImportError:
            return None

