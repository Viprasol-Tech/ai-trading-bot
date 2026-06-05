"""Broker interface — the boundary between strategy logic and execution.

Implement this to connect any exchange or broker. The bundled
:class:`~ai_trading_bot.brokers.paper.PaperBroker` simulates fills so you can run
end-to-end without risking capital.

Part of AI Trading Bot by Viprasol Tech Private Limited (https://viprasol.com).
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from ai_trading_bot.models import Order, Position


class Broker(ABC):
    """Abstract trading venue."""

    @abstractmethod
    def submit(self, order: Order, price: float) -> None:
        """Execute an order at the given reference price."""

    @abstractmethod
    def position(self, symbol: str) -> Position:
        """Return the current position for a symbol."""

    @abstractmethod
    def equity(self, mark_prices: dict[str, float]) -> float:
        """Total account equity given current mark prices."""
