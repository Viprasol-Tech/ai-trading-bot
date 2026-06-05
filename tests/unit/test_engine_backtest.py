"""Tests for the paper broker, sizing, bot engine, and backtester."""

from __future__ import annotations

import math

from ai_trading_bot.backtest.engine import run_backtest
from ai_trading_bot.brokers.paper import PaperBroker
from ai_trading_bot.models import Bar, Order, Side
from ai_trading_bot.risk.sizing import PositionSizer, RiskConfig
from ai_trading_bot.signals.indicators import MovingAverageCrossover


def test_paper_broker_buy_and_sell() -> None:
    b = PaperBroker(starting_cash=1_000.0)
    b.submit(Order("X", Side.BUY, 5), price=10.0)
    assert b.position("X").quantity == 5
    assert math.isclose(b.cash, 950.0)
    b.submit(Order("X", Side.SELL, 5), price=12.0)
    assert b.position("X").quantity == 0
    assert math.isclose(b.cash, 1_010.0)  # +60 from sale


def test_sizer_flat_below_threshold() -> None:
    sizer = PositionSizer(RiskConfig(entry_threshold=0.2))
    assert sizer.target_quantity(score=0.1, equity=10_000, price=100) == 0.0


def test_sizer_caps_at_fraction() -> None:
    sizer = PositionSizer(RiskConfig(max_position_fraction=0.25, entry_threshold=0.0))
    qty = sizer.target_quantity(score=1.0, equity=10_000, price=100)
    assert math.isclose(qty * 100, 2_500.0)  # 25% of equity


def test_backtest_runs_and_reports() -> None:
    bars = [Bar(i, p, p, p, p) for i, p in enumerate(100 + 0.3 * i for i in range(80))]
    result = run_backtest("T", MovingAverageCrossover(5, 15), bars, starting_cash=10_000)
    assert len(result.equity_curve) == len(bars)
    assert result.max_drawdown_pct <= 0.0
    assert isinstance(result.total_return_pct, float)
