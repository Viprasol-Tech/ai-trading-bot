"""Indicator-based signals: MA crossover, RSI, MACD, Bollinger, breakout.

These are deliberately simple, well-understood baselines you can combine or
replace with ML models that implement the same :class:`Signal` interface.

Part of AI Trading Bot by Viprasol Tech Private Limited (https://viprasol.com).
"""

from __future__ import annotations

import math
from collections.abc import Sequence

from ai_trading_bot.models import Bar
from ai_trading_bot.signals.base import Signal


def _sma(values: Sequence[float], window: int) -> float:
    """Simple moving average of the last ``window`` values."""
    return sum(values[-window:]) / window


def _ema_series(values: Sequence[float], window: int) -> list[float]:
    """Exponential moving average over the whole series (seeded with the SMA)."""
    if len(values) < window:
        return []
    alpha = 2.0 / (window + 1.0)
    ema = sum(values[:window]) / window
    out = [ema]
    for value in values[window:]:
        ema = alpha * value + (1.0 - alpha) * ema
        out.append(ema)
    return out


def _stdev(values: Sequence[float], window: int) -> float:
    """Population standard deviation of the last ``window`` values."""
    window_vals = values[-window:]
    mean = sum(window_vals) / window
    variance = sum((v - mean) ** 2 for v in window_vals) / window
    return math.sqrt(variance)


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


class MACDSignal(Signal):
    """Moving Average Convergence Divergence trend/momentum signal.

    The MACD line is ``EMA(fast) - EMA(slow)``; the signal line is the EMA of the
    MACD line. The histogram (MACD - signal) drives the score: a positive,
    widening histogram is bullish, a negative one bearish. The score normalises
    the histogram by the latest price so it is comparable across instruments.
    """

    name = "macd"

    def __init__(self, fast: int = 12, slow: int = 26, signal: int = 9) -> None:
        if fast >= slow:
            raise ValueError("fast window must be shorter than slow window")
        if min(fast, slow, signal) < 1:
            raise ValueError("all windows must be >= 1")
        self.fast = fast
        self.slow = slow
        self.signal = signal

    def evaluate(self, bars: Sequence[Bar]) -> float:
        if len(bars) < self.slow + self.signal:
            return 0.0
        closes = [b.close for b in bars]
        fast_ema = _ema_series(closes, self.fast)
        slow_ema = _ema_series(closes, self.slow)
        # Align the two EMA series on their common (most recent) tail.
        tail = min(len(fast_ema), len(slow_ema))
        macd_line = [f - s for f, s in zip(fast_ema[-tail:], slow_ema[-tail:], strict=True)]
        signal_line = _ema_series(macd_line, self.signal)
        if not signal_line:
            return 0.0
        histogram = macd_line[-1] - signal_line[-1]
        price = closes[-1]
        if price <= 0:
            return 0.0
        # Scale: a histogram worth ~1% of price saturates the score.
        return self._clamp(histogram / price * 100.0)


class BollingerBandsSignal(Signal):
    """Bollinger Bands mean-reversion signal.

    Bands sit ``num_std`` standard deviations around the SMA. Price below the
    lower band is oversold (bullish); above the upper band is overbought
    (bearish). The score scales with how far past the band price has travelled.
    """

    name = "bollinger"

    def __init__(self, period: int = 20, num_std: float = 2.0) -> None:
        if period < 2:
            raise ValueError("period must be >= 2")
        if num_std <= 0:
            raise ValueError("num_std must be positive")
        self.period = period
        self.num_std = num_std

    def evaluate(self, bars: Sequence[Bar]) -> float:
        if len(bars) < self.period:
            return 0.0
        closes = [b.close for b in bars]
        mid = _sma(closes, self.period)
        std = _stdev(closes, self.period)
        if std == 0:
            return 0.0
        upper = mid + self.num_std * std
        lower = mid - self.num_std * std
        price = closes[-1]
        if price < lower:
            return self._clamp((lower - price) / (self.num_std * std))
        if price > upper:
            return -self._clamp((price - upper) / (self.num_std * std))
        return 0.0


class BreakoutSignal(Signal):
    """Donchian-channel breakout (trend-following) signal.

    Looks at the highest high and lowest low over a lookback window (excluding the
    current bar). A close above the prior high is a bullish breakout; a close
    below the prior low is a bearish breakdown. The score scales with how far the
    close has cleared the channel edge relative to the channel width.
    """

    name = "breakout"

    def __init__(self, lookback: int = 20) -> None:
        if lookback < 2:
            raise ValueError("lookback must be >= 2")
        self.lookback = lookback

    def evaluate(self, bars: Sequence[Bar]) -> float:
        if len(bars) < self.lookback + 1:
            return 0.0
        window = bars[-self.lookback - 1 : -1]
        highest = max(b.high for b in window)
        lowest = min(b.low for b in window)
        width = highest - lowest
        if width <= 0:
            return 0.0
        close = bars[-1].close
        if close > highest:
            return self._clamp((close - highest) / width)
        if close < lowest:
            return -self._clamp((lowest - close) / width)
        return 0.0
