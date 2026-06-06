"""Backtester: run a bot over historical bars and report performance.

Records the equity curve each bar and computes a rich set of summary metrics:
total return, CAGR, annualised volatility, Sharpe and Sortino ratios, max
drawdown, and the realised win rate. Intentionally a clear reference engine, not
a high-fidelity matching engine.

Part of AI Trading Bot by Viprasol Tech Private Limited (https://viprasol.com).
"""

from __future__ import annotations

import itertools
import math
from collections.abc import Sequence
from dataclasses import dataclass, field

from ai_trading_bot.brokers.paper import PaperBroker
from ai_trading_bot.models import Bar
from ai_trading_bot.risk.sizing import PositionSizer
from ai_trading_bot.signals.base import Signal
from ai_trading_bot.strategy.engine import TradingBot

#: Bars-per-year assumption used to annualise return and risk metrics.
#: Defaults to daily bars; override via ``run_backtest(periods_per_year=...)``.
DEFAULT_PERIODS_PER_YEAR = 252


def _returns(curve: Sequence[float]) -> list[float]:
    """Per-bar simple returns from an equity curve."""
    out: list[float] = []
    for prev, curr in itertools.pairwise(curve):
        if prev > 0:
            out.append(curr / prev - 1.0)
    return out


@dataclass(slots=True)
class BacktestResult:
    """Summary of a backtest, with metrics computed lazily from the curve."""

    starting_equity: float
    final_equity: float
    equity_curve: list[float] = field(default_factory=list)
    trades: int = 0
    wins: int = 0
    periods_per_year: int = DEFAULT_PERIODS_PER_YEAR

    @property
    def total_return_pct(self) -> float:
        """Total return as a percentage of starting equity."""
        if self.starting_equity == 0:
            return 0.0
        return (self.final_equity / self.starting_equity - 1.0) * 100.0

    @property
    def max_drawdown_pct(self) -> float:
        """Largest peak-to-trough drawdown as a percentage (<= 0)."""
        peak = float("-inf")
        worst = 0.0
        for value in self.equity_curve:
            peak = max(peak, value)
            if peak > 0:
                worst = min(worst, (value - peak) / peak)
        return worst * 100.0

    @property
    def cagr_pct(self) -> float:
        """Compound annual growth rate as a percentage."""
        n = len(self.equity_curve)
        if n < 2 or self.starting_equity <= 0 or self.final_equity <= 0:
            return 0.0
        years = (n - 1) / self.periods_per_year
        if years <= 0:
            return 0.0
        growth = (self.final_equity / self.starting_equity) ** (1.0 / years)
        return (float(growth) - 1.0) * 100.0

    @property
    def volatility_pct(self) -> float:
        """Annualised standard deviation of per-bar returns, as a percentage."""
        rets = _returns(self.equity_curve)
        if len(rets) < 2:
            return 0.0
        mean = sum(rets) / len(rets)
        var = sum((r - mean) ** 2 for r in rets) / (len(rets) - 1)
        return math.sqrt(var) * math.sqrt(self.periods_per_year) * 100.0

    @property
    def sharpe(self) -> float:
        """Annualised Sharpe ratio (risk-free rate assumed zero)."""
        rets = _returns(self.equity_curve)
        if len(rets) < 2:
            return 0.0
        mean = sum(rets) / len(rets)
        var = sum((r - mean) ** 2 for r in rets) / (len(rets) - 1)
        std = math.sqrt(var)
        if std == 0:
            return 0.0
        return mean / std * math.sqrt(self.periods_per_year)

    @property
    def sortino(self) -> float:
        """Annualised Sortino ratio (downside deviation only)."""
        rets = _returns(self.equity_curve)
        if len(rets) < 2:
            return 0.0
        mean = sum(rets) / len(rets)
        downside = [r for r in rets if r < 0]
        if not downside:
            return 0.0
        dd = math.sqrt(sum(r * r for r in downside) / len(downside))
        if dd == 0:
            return 0.0
        return mean / dd * math.sqrt(self.periods_per_year)

    @property
    def win_rate_pct(self) -> float:
        """Fraction of closed trades that were profitable, as a percentage."""
        if self.trades == 0:
            return 0.0
        return self.wins / self.trades * 100.0

    def as_dict(self) -> dict[str, float]:
        """Flat dict of headline metrics (handy for tables and reports)."""
        return {
            "starting_equity": self.starting_equity,
            "final_equity": self.final_equity,
            "total_return_pct": self.total_return_pct,
            "cagr_pct": self.cagr_pct,
            "volatility_pct": self.volatility_pct,
            "sharpe": self.sharpe,
            "sortino": self.sortino,
            "max_drawdown_pct": self.max_drawdown_pct,
            "win_rate_pct": self.win_rate_pct,
            "trades": float(self.trades),
        }


def run_backtest(
    symbol: str,
    signal: Signal,
    bars: Sequence[Bar],
    starting_cash: float = 10_000.0,
    fee_rate: float = 0.0,
    sizer: PositionSizer | None = None,
    periods_per_year: int = DEFAULT_PERIODS_PER_YEAR,
) -> BacktestResult:
    """Run ``signal`` over ``bars`` with a paper broker and return metrics."""
    broker = PaperBroker(starting_cash=starting_cash, fee_rate=fee_rate)
    bot = TradingBot(symbol, signal, broker, sizer)
    result = BacktestResult(
        starting_equity=starting_cash,
        final_equity=starting_cash,
        periods_per_year=periods_per_year,
    )
    for bar in bars:
        bot.on_bar(bar)
        result.equity_curve.append(broker.equity({symbol: bar.close}))
    result.final_equity = result.equity_curve[-1] if result.equity_curve else starting_cash
    result.trades = broker.closed_trades
    result.wins = broker.winning_trades
    return result
