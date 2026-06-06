"""Tests for the grid scan and walk-forward optimiser."""

from __future__ import annotations

import math
from collections.abc import Mapping

import pytest

from ai_trading_bot.models import Bar
from ai_trading_bot.optimize.scan import grid_scan, walk_forward
from ai_trading_bot.signals.base import Signal
from ai_trading_bot.signals.indicators import MovingAverageCrossover


def _trend_bars(n: int) -> list[Bar]:
    out: list[Bar] = []
    for i in range(n):
        price = 100.0 + i * 0.2 + 6.0 * math.sin(i / 10.0)
        out.append(Bar(i, price, price + 1, price - 1, price))
    return out


def _ma_builder(params: Mapping[str, float]) -> Signal:
    return MovingAverageCrossover(int(params["fast"]), int(params["slow"]))


def test_grid_scan_expands_and_ranks() -> None:
    bars = _trend_bars(200)
    grid = {"fast": [5, 10], "slow": [30, 40]}
    ranked = grid_scan("T", _ma_builder, bars, grid, metric="sharpe")
    assert len(ranked) == 4  # 2 x 2 combinations
    # Sorted best-first.
    assert ranked[0].score >= ranked[-1].score
    assert set(ranked[0].params) == {"fast", "slow"}


def test_grid_scan_unknown_metric_raises() -> None:
    bars = _trend_bars(100)
    with pytest.raises(KeyError):
        grid_scan("T", _ma_builder, bars, {"fast": [5], "slow": [20]}, metric="nope")


def test_walk_forward_produces_folds() -> None:
    bars = _trend_bars(500)
    grid = {"fast": [5, 10], "slow": [30, 40]}
    wf = walk_forward("T", _ma_builder, bars, grid, folds=4, metric="sharpe")
    assert len(wf.folds) == 4
    for f in wf.folds:
        assert set(f.best_params) == {"fast", "slow"}
        assert len(f.out_of_sample.equity_curve) > 0
    assert isinstance(wf.avg_out_of_sample, float)
    assert wf.efficiency >= 0.0


def test_walk_forward_rejects_too_many_folds() -> None:
    bars = _trend_bars(10)
    with pytest.raises(ValueError):
        walk_forward("T", _ma_builder, bars, {"fast": [5], "slow": [20]}, folds=20)


def test_walk_forward_rejects_zero_folds() -> None:
    bars = _trend_bars(100)
    with pytest.raises(ValueError):
        walk_forward("T", _ma_builder, bars, {"fast": [5], "slow": [20]}, folds=0)
