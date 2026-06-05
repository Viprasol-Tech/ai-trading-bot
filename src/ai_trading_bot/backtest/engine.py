"""Backtester: run a bot over historical bars and report performance.

Records the equity curve each bar and computes summary metrics (total return,
max drawdown). Intentionally simple — a clear starting point, not a high-fidelity
matching engine.

Part of AI Trading Bot by Viprasol Tech Private Limited (https://viprasol.com).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

from ai_trading_bot.brokers.paper import PaperBroker
from ai_trading_bot.models import Bar
from ai_trading_bot.risk.sizing import PositionSizer
from ai_trading_bot.signals.base import Signal
from ai_trading_bot.strategy.engine import TradingBot


@dataclass(slots=True)
class BacktestResult:
    """Summary of a backtest."""

    starting_equity: float
    final_equity: float
    equity_curve: list[float] = field(default_factory=list)

    @property
    def total_return_pct(self) -> float:
        """Total return as a percentage of starting equity."""
        if self.starting_equity == 0:
            return 0.0
        return (self.final_equity / self.starting_equity - 1.0) * 100.0

    @property
    def max_drawdown_pct(self) -> float:
        """Largest peak-to-trough drawdown as a percentage."""
        peak = float("-inf")
        worst = 0.0
        for value in self.equity_curve:
            peak = max(peak, value)
            if peak > 0:
                worst = min(worst, (value - peak) / peak)
        return worst * 100.0


def run_backtest(
    symbol: str,
    signal: Signal,
    bars: Sequence[Bar],
    starting_cash: float = 10_000.0,
    fee_rate: float = 0.0,
    sizer: PositionSizer | None = None,
) -> BacktestResult:
    """Run ``signal`` over ``bars`` with a paper broker and return metrics."""
    broker = PaperBroker(starting_cash=starting_cash, fee_rate=fee_rate)
    bot = TradingBot(symbol, signal, broker, sizer)
    result = BacktestResult(starting_equity=starting_cash, final_equity=starting_cash)
    for bar in bars:
        bot.on_bar(bar)
        result.equity_curve.append(broker.equity({symbol: bar.close}))
    result.final_equity = result.equity_curve[-1] if result.equity_curve else starting_cash
    return result
