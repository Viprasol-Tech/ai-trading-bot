"""Tests for the pydantic configuration module."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from ai_trading_bot.config import BotSettings, SignalConfig, SignalKind
from ai_trading_bot.signals.ensemble import SignalEnsemble
from ai_trading_bot.signals.indicators import MACDSignal, MovingAverageCrossover


def test_default_settings_build_ensemble_and_sizer() -> None:
    settings = BotSettings()
    ensemble = settings.build_ensemble()
    assert isinstance(ensemble, SignalEnsemble)
    assert len(ensemble.signals) == 3
    sizer = settings.build_sizer()
    assert sizer.config.max_position_fraction == 0.25


def test_signal_config_builds_with_params() -> None:
    cfg = SignalConfig(kind=SignalKind.MA_CROSSOVER, params={"fast": 5, "slow": 20})
    sig = cfg.build()
    assert isinstance(sig, MovingAverageCrossover)
    assert sig.fast == 5 and sig.slow == 20


def test_macd_config_builds() -> None:
    cfg = SignalConfig(kind=SignalKind.MACD, params={"fast": 12, "slow": 26, "signal": 9})
    assert isinstance(cfg.build(), MACDSignal)


def test_settings_reject_empty_signals() -> None:
    with pytest.raises(ValidationError):
        BotSettings(signals=[])


def test_settings_reject_bad_fraction() -> None:
    with pytest.raises(ValidationError):
        BotSettings(max_position_fraction=1.5)


def test_settings_env_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ATB_STARTING_CASH", "25000")
    monkeypatch.setenv("ATB_SYMBOL", "BTC")
    settings = BotSettings()
    assert settings.starting_cash == 25000.0
    assert settings.symbol == "BTC"
