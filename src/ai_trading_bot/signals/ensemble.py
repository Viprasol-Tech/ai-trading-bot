"""Combine multiple signals into one score via a weighted average.

This is the heart of the "AI" layer: any mix of indicator signals, ML models, or
sentiment scores that implement :class:`Signal` can be blended. Swap in your own
weights (or learn them) without touching the execution code.

Part of AI Trading Bot by Viprasol Tech Private Limited (https://viprasol.com).
"""

from __future__ import annotations

from collections.abc import Sequence

from ai_trading_bot.models import Bar
from ai_trading_bot.signals.base import Signal


class SignalEnsemble(Signal):
    """Weighted average of child signals (weights default to equal)."""

    name = "ensemble"

    def __init__(self, signals: Sequence[Signal], weights: Sequence[float] | None = None) -> None:
        if not signals:
            raise ValueError("ensemble needs at least one signal")
        if weights is not None and len(weights) != len(signals):
            raise ValueError("weights length must match signals length")
        self.signals = list(signals)
        self.weights = list(weights) if weights is not None else [1.0] * len(signals)

    def evaluate(self, bars: Sequence[Bar]) -> float:
        total_weight = sum(self.weights)
        if total_weight == 0:
            return 0.0
        pairs = zip(self.signals, self.weights, strict=True)
        weighted = sum(s.evaluate(bars) * w for s, w in pairs)
        return self._clamp(weighted / total_weight)
