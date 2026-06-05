"""Indicator-based signals: moving-average crossover and RSI.

These are deliberately simple, well-understood baselines you can combine or
replace with ML models that implement the same :class:`Signal` interface.

Part of AI Trading Bot by Viprasol Tech Private Limited (https://viprasol.com).
"""

from __future__ import annotations

from collections.abc import Sequence

from ai_trading_bot.models import Bar
from ai_trading_bot.signals.base import Signal


def _sma(values: Sequence[float], window: int) -> float:
    """Simple moving average of the last ``window`` values."""
    return sum(values[-window:]) / window


class MovingAverageCrossover(Signal):
    """Fast/slow SMA crossover.

    Score is the normalised gap between the fast and slow SMA, so a fast MA well
    above the slow MA gives a strongly bullish (+) score and vice versa.
    """

    name = "ma_crossover"

    def __init__(self, fast: int = 10, slow: int = 30) -> None:
        if fast >= slow:
            raise ValueError("fast window must be shorter than slow window")
        self.fast = fast
        self.slow = slow

    def evaluate(self, bars: Sequence[Bar]) -> float:
        if len(bars) < self.slow:
            return 0.0
        closes = [b.close for b in bars]
        fast = _sma(closes, self.fast)
        slow = _sma(closes, self.slow)
        if slow == 0:
            return 0.0
        # Gap as a fraction of the slow MA, scaled so a few-percent gap saturates.
        return self._clamp((fast - slow) / slow * 20.0)


class RSISignal(Signal):
    """Relative Strength Index mean-reversion signal.

    Oversold (RSI < lower) → bullish; overbought (RSI > upper) → bearish. The
    score scales linearly with how far past the threshold the RSI sits.
    """

    name = "rsi"

    def __init__(self, period: int = 14, lower: float = 30.0, upper: float = 70.0) -> None:
        self.period = period
        self.lower = lower
        self.upper = upper

    def _rsi(self, closes: Sequence[float]) -> float:
        gains = 0.0
        losses = 0.0
        for prev, curr in zip(closes[-self.period - 1 :], closes[-self.period :], strict=False):
            change = curr - prev
            if change >= 0:
                gains += change
            else:
                losses -= change
        if losses == 0:
            return 100.0
        rs = (gains / self.period) / (losses / self.period)
        return 100.0 - (100.0 / (1.0 + rs))

    def evaluate(self, bars: Sequence[Bar]) -> float:
        if len(bars) < self.period + 1:
            return 0.0
        rsi = self._rsi([b.close for b in bars])
        if rsi < self.lower:
            return self._clamp((self.lower - rsi) / self.lower)
        if rsi > self.upper:
            return -self._clamp((rsi - self.upper) / (100.0 - self.upper))
        return 0.0
