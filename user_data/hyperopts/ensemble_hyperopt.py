"""Custom hyperopt loss emphasising monthly win-rate constraints."""
from __future__ import annotations

import pandas as pd
from freqtrade.optimize.hyperopt import IHyperOptLoss


class MonthlyPosAndHighWR(IHyperOptLoss):
    """Penalise configurations that violate monthly targets."""

    @staticmethod
    def hyperopt_loss_function(results, trade_count, *args, **kwargs):
        if results is None or results.empty:
            return 10_000.0
        if trade_count < 100:
            return 10_000.0

        df = results.copy()
        df["close_date"] = pd.to_datetime(df["close_date"], utc=True, errors="coerce")
        df["is_win"] = df["profit_ratio"] > 0
        df["month"] = df["close_date"].dt.to_period("M")

        penalty = 0.0
        for _, group in df.groupby("month"):
            winrate = group["is_win"].mean()
            roi = group["profit_ratio"].sum()
            if winrate < 0.80:
                penalty += (0.80 - winrate) * 200.0
            if roi <= 0:
                penalty += 200.0

        total_roi = df["profit_ratio"].sum()
        loss = -total_roi + penalty
        return float(loss)
