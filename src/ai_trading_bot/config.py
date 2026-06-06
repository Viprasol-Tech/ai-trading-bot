"""Typed configuration for the bot, powered by pydantic.

Everything that tunes behaviour lives here so a run is fully described by one
object: the signal mix, their weights, risk limits, and broker assumptions.
:class:`BotSettings` also reads environment variables prefixed with ``ATB_``
(e.g. ``ATB_STARTING_CASH=25000``), which makes it easy to drive the bot from a
``.env`` file or a container without code changes.

Part of AI Trading Bot by Viprasol Tech Private Limited (https://viprasol.com).
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from ai_trading_bot.risk.sizing import PositionSizer, RiskConfig
from ai_trading_bot.signals.base import Signal
from ai_trading_bot.signals.ensemble import SignalEnsemble
from ai_trading_bot.signals.indicators import (
    BollingerBandsSignal,
    BreakoutSignal,
    MACDSignal,
    MovingAverageCrossover,
    RSISignal,
)


class SignalKind(str, Enum):
    """Built-in signal types that can be configured by name."""

    MA_CROSSOVER = "ma_crossover"
    RSI = "rsi"
    MACD = "macd"
    BOLLINGER = "bollinger"
    BREAKOUT = "breakout"


class SignalConfig(BaseModel):
    """One configured signal: its kind, blend weight, and keyword parameters."""

    kind: SignalKind
    weight: float = Field(default=1.0, ge=0.0)
    params: dict[str, float] = Field(default_factory=dict)

    def build(self) -> Signal:
        """Instantiate the concrete :class:`Signal` for this config."""
        builders: dict[SignalKind, type[Signal]] = {
            SignalKind.MA_CROSSOVER: MovingAverageCrossover,
            SignalKind.RSI: RSISignal,
            SignalKind.MACD: MACDSignal,
            SignalKind.BOLLINGER: BollingerBandsSignal,
            SignalKind.BREAKOUT: BreakoutSignal,
        }
        cls = builders[self.kind]
        # Integer-typed windows must be passed as ints, not floats.
        coerced = {k: (int(v) if v.is_integer() else v) for k, v in self.params.items()}
        signal: Signal = cls(**coerced)
        return signal


def _default_signals() -> list[SignalConfig]:
    return [
        SignalConfig(kind=SignalKind.MA_CROSSOVER, weight=1.0),
        SignalConfig(kind=SignalKind.RSI, weight=1.0),
        SignalConfig(kind=SignalKind.MACD, weight=1.0),
    ]


class BotSettings(BaseSettings):
    """Top-level run configuration (env-overridable with the ``ATB_`` prefix)."""

    model_config = SettingsConfigDict(env_prefix="ATB_", env_file=".env", extra="ignore")

    symbol: str = "DEMO"
    starting_cash: float = Field(default=10_000.0, gt=0.0)
    fee_rate: float = Field(default=0.0, ge=0.0, le=0.1)
    max_position_fraction: float = Field(default=0.25, gt=0.0, le=1.0)
    entry_threshold: float = Field(default=0.2, ge=0.0, lt=1.0)
    signals: list[SignalConfig] = Field(default_factory=_default_signals)

    @field_validator("signals")
    @classmethod
    def _non_empty(cls, value: list[SignalConfig]) -> list[SignalConfig]:
        if not value:
            raise ValueError("at least one signal must be configured")
        return value

    def build_ensemble(self) -> SignalEnsemble:
        """Construct the weighted :class:`SignalEnsemble` from this config."""
        built = [s.build() for s in self.signals]
        weights = [s.weight for s in self.signals]
        return SignalEnsemble(built, weights)

    def build_sizer(self) -> PositionSizer:
        """Construct the :class:`PositionSizer` from this config's risk limits."""
        return PositionSizer(
            RiskConfig(
                max_position_fraction=self.max_position_fraction,
                entry_threshold=self.entry_threshold,
            )
        )
