"""Smoke tests for the Typer CLI subcommands."""

from __future__ import annotations

from typer.testing import CliRunner

from ai_trading_bot import __version__
from ai_trading_bot.cli import app

runner = CliRunner()


def test_version_command() -> None:
    result = runner.invoke(app, ["version"])
    assert result.exit_code == 0
    assert __version__ in result.stdout


def test_demo_command() -> None:
    result = runner.invoke(app, ["demo", "--starting-cash", "5000"])
    assert result.exit_code == 0
    assert "Final equity" in result.stdout


def test_backtest_command_macd() -> None:
    result = runner.invoke(app, ["backtest", "--signal", "macd", "--bars", "120"])
    assert result.exit_code == 0
    assert "Sharpe" in result.stdout


def test_backtest_rejects_unknown_signal() -> None:
    result = runner.invoke(app, ["backtest", "--signal", "nope"])
    assert result.exit_code != 0


def test_scan_command() -> None:
    result = runner.invoke(app, ["scan", "--metric", "sharpe", "--top", "3"])
    assert result.exit_code == 0
    assert "Parameter scan" in result.stdout


def test_walkforward_command() -> None:
    result = runner.invoke(app, ["walkforward", "--folds", "3", "--bars", "400"])
    assert result.exit_code == 0
    assert "Walk-forward" in result.stdout
