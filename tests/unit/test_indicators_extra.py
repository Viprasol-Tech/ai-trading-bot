"""Tests for the MACD, Bollinger, and breakout signals added in 0.2.0."""

from __future__ import annotations

from ai_trading_bot.models import Bar
from ai_trading_bot.signals.indicators import (
    BollingerBandsSignal,
    BreakoutSignal,
    MACDSignal,
)


def _bars(prices: list[float]) -> list[Bar]:
    return [Bar(timestamp=i, open=p, high=p + 1, low=p - 1, close=p) for i, p in enumerate(prices)]


def test_macd_neutral_without_history() -> None:
    sig = MACDSignal(3, 6, 3)
    assert sig.evaluate(_bars([1, 2, 3])) == 0.0


def test_macd_bullish_on_accelerating_uptrend() -> None:
    sig = MACDSignal(3, 6, 3)
    prices = [100 + i * i * 0.05 for i in range(40)]  # convex, accelerating up
    score = sig.evaluate(_bars(prices))
    assert score > 0
    assert -1.0 <= score <= 1.0


def test_macd_bearish_on_downtrend() -> None:
    sig = MACDSignal(3, 6, 3)
    prices = [200 - i * i * 0.05 for i in range(40)]
    score = sig.evaluate(_bars(prices))
    assert score < 0


def test_macd_rejects_bad_windows() -> None:
    try:
        MACDSignal(26, 12, 9)
    except ValueError:
        return
    raise AssertionError("expected ValueError for fast >= slow")


def test_bollinger_bullish_below_lower_band() -> None:
    sig = BollingerBandsSignal(period=10, num_std=2.0)
    prices = [100.0] * 12 + [80.0]  # sharp drop below the lower band
    score = sig.evaluate(_bars(prices))
    assert score > 0


def test_bollinger_bearish_above_upper_band() -> None:
    sig = BollingerBandsSignal(period=10, num_std=2.0)
    prices = [100.0] * 12 + [130.0]
    score = sig.evaluate(_bars(prices))
    assert score < 0


def test_bollinger_neutral_inside_bands() -> None:
    sig = BollingerBandsSignal(period=10, num_std=2.0)
    prices = [100.0, 101.0, 99.0, 100.5, 99.5, 100.2, 99.8, 100.1, 99.9, 100.0, 100.3]
    score = sig.evaluate(_bars(prices))
    assert score == 0.0


def test_bollinger_rejects_bad_params() -> None:
    try:
        BollingerBandsSignal(period=1)
    except ValueError:
        return
    raise AssertionError("expected ValueError for period < 2")


def test_breakout_bullish_on_new_high() -> None:
    sig = BreakoutSignal(lookback=5)
    bars = [Bar(i, 100, 101, 99, 100) for i in range(5)]
    bars.append(Bar(5, 100, 110, 100, 109))  # close clears the prior high
    assert sig.evaluate(bars) > 0


def test_breakout_bearish_on_new_low() -> None:
    sig = BreakoutSignal(lookback=5)
    bars = [Bar(i, 100, 101, 99, 100) for i in range(5)]
    bars.append(Bar(5, 100, 100, 90, 91))
    assert sig.evaluate(bars) < 0


def test_breakout_neutral_in_range() -> None:
    sig = BreakoutSignal(lookback=5)
    bars = [Bar(i, 100, 105, 95, 100) for i in range(6)]
    assert sig.evaluate(bars) == 0.0


def test_breakout_rejects_bad_lookback() -> None:
    try:
        BreakoutSignal(lookback=1)
    except ValueError:
        return
    raise AssertionError("expected ValueError for lookback < 2")
