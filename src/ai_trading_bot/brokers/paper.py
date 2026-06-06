"""A simple paper-trading broker that simulates fills at the reference price.

Tracks cash, positions, and a volume-weighted average entry price, and counts
realised round-trip trades (and how many were profitable) so the backtester can
report a win rate. Good enough to validate strategy logic end to end.

Part of AI Trading Bot by Viprasol Tech Private Limited (https://viprasol.com).
"""

from __future__ import annotations

from ai_trading_bot.brokers.base import Broker
from ai_trading_bot.models import Order, Position, Side


class PaperBroker(Broker):
    """In-memory broker for paper trading and backtests."""

    def __init__(self, starting_cash: float = 10_000.0, fee_rate: float = 0.0) -> None:
        self.cash = starting_cash
        self.fee_rate = fee_rate
        self._positions: dict[str, Position] = {}
        #: Number of sells that reduced a long position (realised trades).
        self.closed_trades = 0
        #: Subset of ``closed_trades`` that booked a positive P&L after fees.
        self.winning_trades = 0

    def submit(self, order: Order, price: float) -> None:
        if price <= 0 or order.quantity <= 0:
            return
        pos = self._positions.setdefault(order.symbol, Position(order.symbol))
        notional = price * order.quantity
        fee = notional * self.fee_rate

        if order.side is Side.BUY:
            self.cash -= notional + fee
            new_qty = pos.quantity + order.quantity
            # Volume-weighted average price on the long side.
            pos.avg_price = (
                (pos.avg_price * pos.quantity + notional) / new_qty if new_qty > 0 else 0.0
            )
            pos.quantity = new_qty
        else:
            closed_qty = min(order.quantity, pos.quantity)
            if closed_qty > 0:
                # Realised P&L on the portion that closes an existing long.
                pnl = (price - pos.avg_price) * closed_qty - notional * self.fee_rate
                self.closed_trades += 1
                if pnl > 0:
                    self.winning_trades += 1
            self.cash += notional - fee
            pos.quantity -= order.quantity
            if pos.quantity <= 0:
                pos.quantity = 0.0
                pos.avg_price = 0.0

    def position(self, symbol: str) -> Position:
        return self._positions.get(symbol, Position(symbol))

    def equity(self, mark_prices: dict[str, float]) -> float:
        holdings = sum(
            pos.quantity * mark_prices.get(sym, pos.avg_price)
            for sym, pos in self._positions.items()
        )
        return self.cash + holdings
