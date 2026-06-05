"""The trading bot: glue between signals, risk sizing, and a broker.

On each new bar the bot evaluates its signal, computes a target position via the
sizer, and submits the delta order to the broker. Long-only by default (it buys
toward the target and sells down to zero), which keeps the example safe and clear.

Part of AI Trading Bot by Viprasol Tech Private Limited (https://viprasol.com).
"""

from __future__ import annotations

from collections.abc import Sequence

from ai_trading_bot.brokers.base import Broker
from ai_trading_bot.models import Bar, Order, Side
from ai_trading_bot.risk.sizing import PositionSizer
from ai_trading_bot.signals.base import Signal


class TradingBot:
    """Drive a single symbol with one signal, one sizer, one broker."""

    def __init__(
        self,
        symbol: str,
        signal: Signal,
        broker: Broker,
        sizer: PositionSizer | None = None,
    ) -> None:
        self.symbol = symbol
        self.signal = signal
        self.broker = broker
        self.sizer = sizer or PositionSizer()
        self._history: list[Bar] = []

    def on_bar(self, bar: Bar) -> None:
        """Process one new bar and rebalance toward the target position."""
        self._history.append(bar)
        score = self.signal.evaluate(self._history)
        equity = self.broker.equity({self.symbol: bar.close})
        target = self.sizer.target_quantity(score, equity, bar.close)
        current = self.broker.position(self.symbol).quantity
        delta = target - current

        if abs(delta) * bar.close < 1e-6:
            return
        side = Side.BUY if delta > 0 else Side.SELL
        self.broker.submit(Order(self.symbol, side, abs(delta)), bar.close)
        # Selling never goes short here: the broker clamps quantity at zero.

    def run(self, bars: Sequence[Bar]) -> None:
        """Process a sequence of bars in order."""
        for bar in bars:
            self.on_bar(bar)
