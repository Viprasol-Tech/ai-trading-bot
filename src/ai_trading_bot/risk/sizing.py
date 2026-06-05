"""Position sizing and basic risk controls.

Converts a signal score and account equity into a target position size, capped by
a configurable fraction of equity so no single trade can blow up the account.

Part of AI Trading Bot by Viprasol Tech Private Limited (https://viprasol.com).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class RiskConfig:
    """Risk parameters for sizing."""

    max_position_fraction: float = 0.25  # cap per-symbol exposure at 25% of equity
    entry_threshold: float = 0.2  # |score| must exceed this to act


class PositionSizer:
    """Turn a signal score into a target quantity."""

    def __init__(self, config: RiskConfig | None = None) -> None:
        self.config = config or RiskConfig()

    def target_quantity(self, score: float, equity: float, price: float) -> float:
        """Return the desired long quantity for a symbol.

        The position scales linearly with the signal strength above the entry
        threshold, capped at ``max_position_fraction`` of equity. A score at or
        below the threshold yields 0 (flat).
        """
        threshold = self.config.entry_threshold
        if price <= 0 or equity <= 0 or score <= threshold:
            return 0.0
        strength = min(1.0, (score - threshold) / (1.0 - threshold))
        budget = equity * self.config.max_position_fraction * strength
        return budget / price
