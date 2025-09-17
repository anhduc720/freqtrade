"""Ensemble strategy combining mean-reversion and breakout signals with meta-labelling."""
from __future__ import annotations

import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from freqtrade.strategy import IStrategy

from .strat_breakout_tf import StratBreakoutTF
from .strat_scalper_mr import StratScalperMR


class EnsembleMetaStrategy(IStrategy):
    timeframe = "5m"
    informative_timeframe = "1h"
    can_short = False
    startup_candle_count = 400

    # Meta-model configuration.
    p_threshold: float = 0.65
    model_path = Path("user_data/models/meta_label_xgb.pkl")
    scaler_path = Path("user_data/models/scaler.pkl")

    process_only_new_candles = True

    def __init__(self, config: Dict) -> None:
        super().__init__(config)
        self.strat_mr = StratScalperMR(config)
        self.strat_tf = StratBreakoutTF(config)
        self.model = self._load_pickle(self.model_path)
        self.scaler = self._load_pickle(self.scaler_path)

    @staticmethod
    def _load_pickle(path: Path):
        if path.exists():
            with open(path, "rb") as handle:
                return pickle.load(handle)
        return None

    def informative_pairs(self) -> List[Tuple[str, str]]:
        if not self.dp:
            return []
        return [(pair, self.informative_timeframe) for pair in self.dp.current_whitelist()]

    def _get_informative(self, pair: str) -> Optional[pd.DataFrame]:
        if not self.dp:
            return None
        return self.dp.get_pair_dataframe(pair=pair, timeframe=self.informative_timeframe)

    def populate_indicators(self, dataframe: pd.DataFrame, metadata: Dict) -> pd.DataFrame:
        dataframe = self.strat_mr.populate_indicators(dataframe, metadata)
        dataframe = self.strat_tf.populate_indicators(dataframe, metadata)

        informative = self._get_informative(metadata["pair"])
        if informative is not None:
            informative = informative[["close"]].copy()
            informative.rename(columns={"close": "close_1h"}, inplace=True)
            informative["ema200_1h"] = informative["close_1h"].ewm(span=200).mean()
            informative["slope_1h"] = informative["close_1h"] / informative["ema200_1h"] - 1.0
            dataframe = dataframe.join(informative[["slope_1h"]], how="left")
        dataframe["slope_1h"].fillna(0.0, inplace=True)

        dataframe["hour"] = dataframe.index.hour
        dataframe["weekday"] = dataframe.index.weekday
        return dataframe

    def _collect_features(self, row: pd.Series) -> np.ndarray:
        features = np.array(
            [
                row.get("rsi", 50),
                row.get("zret", 0.0),
                row.get("atr_z", 0.0),
                row.get("adx", 20),
                row.get("slope", 0.0),
                row.get("slope_1h", 0.0),
                row.get("hour", 12),
                row.get("weekday", 2),
            ],
            dtype=np.float32,
        ).reshape(1, -1)
        if self.scaler is not None:
            features = self.scaler.transform(features)
        return features

    def _ml_pass(self, row: pd.Series) -> bool:
        if self.model is None:
            return True
        proba = float(self.model.predict_proba(self._collect_features(row))[:, 1][0])
        row["meta_proba"] = proba
        return proba >= self.p_threshold

    def populate_buy_trend(self, dataframe: pd.DataFrame, metadata: Dict) -> pd.DataFrame:
        dataframe["buy"] = 0
        # combine underlying strategies
        sub_a = self.strat_mr.populate_buy_trend(dataframe.copy(), metadata)
        sub_b = self.strat_tf.populate_buy_trend(dataframe.copy(), metadata)
        candidate_index = dataframe.index[(sub_a.get("buy", 0) == 1) | (sub_b.get("buy", 0) == 1)]

        for idx in candidate_index:
            row = dataframe.loc[idx]
            if self._ml_pass(row):
                dataframe.at[idx, "buy"] = 1

        return dataframe

    def populate_sell_trend(self, dataframe: pd.DataFrame, metadata: Dict) -> pd.DataFrame:
        dataframe["sell"] = 0
        return dataframe

    @staticmethod
    def order_types() -> Dict[str, str]:
        return {
            "buy": "limit",
            "sell": "limit",
            "stoploss": "market",
            "stoploss_on_exchange": "market",
        }

    def protections(self) -> Optional[List[dict]]:
        try:
            from .protections_config import get_protections

            return get_protections()
        except ImportError:
            return None
