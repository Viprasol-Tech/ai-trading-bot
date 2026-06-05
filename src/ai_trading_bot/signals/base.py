"""Signal interface.

A signal looks at recent price history and emits a score in [-1.0, 1.0]:
positive = bullish (go long), negative = bearish (go short / exit), 0 = neutral.
This normalised score lets heterogeneous signals (indicators, ML models, sentiment)
be combined in an ensemble.

Part of AI Trading Bot by Viprasol Tech Private Limited (https://viprasol.com).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence

from ai_trading_bot.models import Bar


class Signal(ABC):
    """Base class for all trading signals."""

    #: Human-readable signal name.
    name: str = "signal"

    @abstractmethod
    def evaluate(self, bars: Sequence[Bar]) -> float:
        """Return a score in [-1.0, 1.0] for the most recent bar.

        Implementations should return 0.0 when there is not enough history.
        """
        raise NotImplementedError

    @staticmethod
    def _clamp(value: float) -> float:
        """Clamp a raw score into [-1.0, 1.0]."""
        return max(-1.0, min(1.0, value))
