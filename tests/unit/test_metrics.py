"""Tests for the richer backtest metrics and trade tracking."""

from __future__ import annotations

import math

from ai_trading_bot.backtest.engine import BacktestResult
from ai_trading_bot.brokers.paper import PaperBroker
from ai_trading_bot.models import Order, Side


def _result(curve: list[float], **kw: float) -> BacktestResult:
    return BacktestResult(
        starting_equity=curve[0],
        final_equity=curve[-1],
        equity_curve=curve,
        periods_per_year=252,
        **kw,  # type: ignore[arg-type]
    )


def test_total_return_and_cagr() -> None:
    curve = [100.0 * (1.001**i) for i in range(253)]  # ~1y of steady growth
    r = _result(curve)
    assert r.total_return_pct > 0
    assert r.cagr_pct > 0


def test_sharpe_positive_for_smooth_uptrend() -> None:
    curve = [100.0 + i for i in range(60)]
    r = _result(curve)
    assert r.sharpe > 0


def test_volatility_zero_for_flat_curve() -> None:
    r = _result([100.0] * 30)
    assert r.volatility_pct == 0.0
    assert r.sharpe == 0.0
    assert r.sortino == 0.0


def test_max_drawdown_negative_on_dip() -> None:
    r = _result([100.0, 120.0, 90.0, 110.0])
    assert r.max_drawdown_pct < 0
    assert math.isclose(r.max_drawdown_pct, -25.0)  # 120 -> 90


def test_win_rate_and_as_dict() -> None:
    r = _result([100.0, 110.0], trades=4, wins=3)
    assert r.win_rate_pct == 75.0
    d = r.as_dict()
    assert d["trades"] == 4.0
    assert "sharpe" in d and "sortino" in d


def test_broker_counts_winning_trade() -> None:
    b = PaperBroker(starting_cash=1_000.0)
    b.submit(Order("X", Side.BUY, 10), price=10.0)
    b.submit(Order("X", Side.SELL, 10), price=12.0)  # profit
    assert b.closed_trades == 1
    assert b.winning_trades == 1


def test_broker_counts_losing_trade() -> None:
    b = PaperBroker(starting_cash=1_000.0)
    b.submit(Order("X", Side.BUY, 10), price=10.0)
    b.submit(Order("X", Side.SELL, 10), price=8.0)  # loss
    assert b.closed_trades == 1
    assert b.winning_trades == 0
