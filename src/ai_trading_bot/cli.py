"""Command-line interface for AI Trading Bot.

Run ``ai-trading-bot --help`` after installing, or ``python -m ai_trading_bot``.
The ``demo`` command runs a backtest on synthetic data so you can see the whole
pipeline (signals → ensemble → sizing → paper broker → metrics) end to end.

Part of AI Trading Bot by Viprasol Tech Private Limited (https://viprasol.com).
"""

from __future__ import annotations

import math

import typer
from rich.console import Console

from ai_trading_bot import __version__
from ai_trading_bot.backtest.engine import run_backtest
from ai_trading_bot.models import Bar
from ai_trading_bot.signals.ensemble import SignalEnsemble
from ai_trading_bot.signals.indicators import MovingAverageCrossover, RSISignal

app = typer.Typer(add_completion=False, help="AI Trading Bot — by Viprasol Tech.")
console = Console()


def _synthetic_bars(n: int = 300) -> list[Bar]:
    """Generate a deterministic trending+wavy price series for demos/tests."""
    bars: list[Bar] = []
    price = 100.0
    for i in range(n):
        price = 100.0 + i * 0.2 + 8.0 * math.sin(i / 12.0)
        bars.append(Bar(timestamp=i, open=price, high=price, low=price, close=price))
    return bars


@app.command()
def version() -> None:
    """Print the installed version."""
    console.print(f"ai-trading-bot [bold cyan]{__version__}[/] — by Viprasol Tech")


@app.command()
def demo(starting_cash: float = 10_000.0) -> None:
    """Run a backtest on synthetic data using an MA-crossover + RSI ensemble."""
    bars = _synthetic_bars()
    ensemble = SignalEnsemble([MovingAverageCrossover(10, 30), RSISignal(14)])
    result = run_backtest("DEMO", ensemble, bars, starting_cash=starting_cash)
    console.print(f"Bars:           {len(bars)}")
    console.print(f"Start equity:   [bold]${result.starting_equity:,.2f}[/]")
    console.print(f"Final equity:   [bold]${result.final_equity:,.2f}[/]")
    console.print(f"Total return:   [bold green]{result.total_return_pct:+.2f}%[/]")
    console.print(f"Max drawdown:   [bold red]{result.max_drawdown_pct:.2f}%[/]")


if __name__ == "__main__":
    app()
