"""Utility helpers for dynamic stoploss management."""
from __future__ import annotations

from datetime import datetime

from freqtrade.persistence import Trade


def time_since_open_minutes(trade: Trade, current_time: datetime) -> float:
    """Return minutes elapsed since trade open."""

    return (current_time - trade.open_date_utc).total_seconds() / 60.0


def profit_protect_stoploss(current_profit: float, thresholds: dict[float, float]) -> float | None:
    """Return tightened stoploss based on profit brackets."""

    for profit, stop in sorted(thresholds.items(), reverse=True):
        if current_profit >= profit:
            return stop
    return None
