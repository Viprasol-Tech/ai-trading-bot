"""Core data types shared across the framework.

Part of AI Trading Bot by Viprasol Tech Private Limited (https://viprasol.com).
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class Side(str, Enum):
    """Order side."""

    BUY = "buy"
    SELL = "sell"


@dataclass(slots=True, frozen=True)
class Bar:
    """A single OHLCV candle."""

    timestamp: int
    open: float
    high: float
    low: float
    close: float
    volume: float = 0.0


@dataclass(slots=True, frozen=True)
class Order:
    """A market order request."""

    symbol: str
    side: Side
    quantity: float


@dataclass(slots=True)
class Position:
    """An open position in a symbol."""

    symbol: str
    quantity: float = 0.0
    avg_price: float = 0.0
