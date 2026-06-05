"""Tests for signals and the ensemble."""

from __future__ import annotations

from ai_trading_bot.models import Bar
from ai_trading_bot.signals.ensemble import SignalEnsemble
from ai_trading_bot.signals.indicators import MovingAverageCrossover, RSISignal


def _bars(prices: list[float]) -> list[Bar]:
    return [Bar(timestamp=i, open=p, high=p, low=p, close=p) for i, p in enumerate(prices)]


def test_ma_crossover_bullish_on_uptrend() -> None:
    sig = MovingAverageCrossover(3, 6)
    score = sig.evaluate(_bars([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]))
    assert score > 0


def test_ma_crossover_neutral_without_history() -> None:
    sig = MovingAverageCrossover(3, 6)
    assert sig.evaluate(_bars([1, 2])) == 0.0


def test_ma_crossover_rejects_bad_windows() -> None:
    try:
        MovingAverageCrossover(30, 10)
    except ValueError:
        return
    raise AssertionError("expected ValueError for fast >= slow")


def test_rsi_bounds() -> None:
    sig = RSISignal(period=5)
    score = sig.evaluate(_bars([1, 2, 3, 4, 5, 6, 7]))  # strong uptrend -> overbought
    assert -1.0 <= score <= 1.0


def test_ensemble_averages_and_clamps() -> None:
    bars = _bars([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
    ens = SignalEnsemble([MovingAverageCrossover(3, 6), RSISignal(5)])
    score = ens.evaluate(bars)
    assert -1.0 <= score <= 1.0


def test_ensemble_requires_signals() -> None:
    try:
        SignalEnsemble([])
    except ValueError:
        return
    raise AssertionError("expected ValueError for empty ensemble")
