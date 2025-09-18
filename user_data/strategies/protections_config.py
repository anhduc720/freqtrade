"""Reusable protection configuration helpers."""
from __future__ import annotations

from typing import List


def get_protections() -> List[dict]:
    """Return default protections shared across ensemble strategies."""

    return [
        {
            "method": "CooldownPeriod",
            "stop_duration_candles": 12,
        },
        {
            "method": "StoplossGuard",
            "lookback_period_candles": 288,
            "trade_limit": 2,
            "stop_duration_candles": 288,
            "only_per_pair": True,
        },
        {
            "method": "MaxDrawdown",
            "lookback_period_candles": 720,
            "trade_limit": 3,
            "stop_duration_candles": 360,
            "max_allowed_drawdown": 0.1,
        },
    ]
