"""Command-line interface for AI Trading Bot.

Run ``ai-trading-bot --help`` after installing, or ``python -m ai_trading_bot``.
Subcommands:

* ``demo``       — run a backtest on synthetic data (signals -> ensemble -> sizing
  -> paper broker -> metrics) end to end.
* ``backtest``   — backtest a chosen built-in signal and print a metrics table.
* ``scan``       — grid-scan a signal's parameters and rank by a metric.
* ``walkforward``— rolling in-sample/out-of-sample validation of a parameter grid.
* ``version``    — print the installed version.

Part of AI Trading Bot by Viprasol Tech Private Limited (https://viprasol.com).
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

import typer
from rich.console import Console
from rich.table import Table

from ai_trading_bot import __version__
from ai_trading_bot.backtest.engine import BacktestResult, run_backtest
from ai_trading_bot.config import BotSettings
from ai_trading_bot.models import Bar
from ai_trading_bot.optimize.scan import grid_scan, walk_forward
from ai_trading_bot.signals.base import Signal
from ai_trading_bot.signals.indicators import (
    BollingerBandsSignal,
    BreakoutSignal,
    MACDSignal,
    MovingAverageCrossover,
    RSISignal,
)

app = typer.Typer(add_completion=False, help="AI Trading Bot -- by Viprasol Tech.")
console = Console()


def _synthetic_bars(n: int = 300) -> list[Bar]:
    """Generate a deterministic trending+wavy price series for demos/tests."""
    bars: list[Bar] = []
    for i in range(n):
        price = 100.0 + i * 0.2 + 8.0 * math.sin(i / 12.0)
        high = price + 1.5
        low = price - 1.5
        bars.append(Bar(timestamp=i, open=price, high=high, low=low, close=price))
    return bars


def _build_signal(name: str) -> Signal:
    """Map a CLI signal name to a concrete signal with sensible defaults."""
    table: dict[str, Signal] = {
        "ma": MovingAverageCrossover(10, 30),
        "rsi": RSISignal(14),
        "macd": MACDSignal(12, 26, 9),
        "bollinger": BollingerBandsSignal(20, 2.0),
        "breakout": BreakoutSignal(20),
    }
    if name not in table:
        raise typer.BadParameter(f"unknown signal {name!r}; choose from {sorted(table)}")
    return table[name]


def _metrics_table(title: str, result: BacktestResult) -> Table:
    """Render a backtest result as a rich metrics table."""
    table = Table(title=title, show_header=True, header_style="bold cyan")
    table.add_column("Metric")
    table.add_column("Value", justify="right")
    table.add_row("Bars", f"{len(result.equity_curve)}")
    table.add_row("Start equity", f"${result.starting_equity:,.2f}")
    table.add_row("Final equity", f"${result.final_equity:,.2f}")
    table.add_row("Total return", f"{result.total_return_pct:+.2f}%")
    table.add_row("CAGR", f"{result.cagr_pct:+.2f}%")
    table.add_row("Volatility (ann.)", f"{result.volatility_pct:.2f}%")
    table.add_row("Sharpe", f"{result.sharpe:.2f}")
    table.add_row("Sortino", f"{result.sortino:.2f}")
    table.add_row("Max drawdown", f"{result.max_drawdown_pct:.2f}%")
    table.add_row("Trades", f"{result.trades}")
    table.add_row("Win rate", f"{result.win_rate_pct:.1f}%")
    return table


@app.command()
def version() -> None:
    """Print the installed version."""
    console.print(f"ai-trading-bot [bold cyan]{__version__}[/] -- by Viprasol Tech")


@app.command()
def demo(starting_cash: float = 10_000.0) -> None:
    """Run a backtest on synthetic data using the default configured ensemble."""
    settings = BotSettings(starting_cash=starting_cash)
    bars = _synthetic_bars()
    result = run_backtest(
        settings.symbol,
        settings.build_ensemble(),
        bars,
        starting_cash=settings.starting_cash,
        fee_rate=settings.fee_rate,
        sizer=settings.build_sizer(),
    )
    console.print(_metrics_table("Demo ensemble backtest", result))


@app.command()
def backtest(
    signal: str = typer.Option("ma", help="ma | rsi | macd | bollinger | breakout"),
    starting_cash: float = 10_000.0,
    fee_rate: float = 0.0,
    bars: int = 300,
) -> None:
    """Backtest a single built-in signal and print a metrics table."""
    series = _synthetic_bars(bars)
    result = run_backtest(
        "DEMO",
        _build_signal(signal),
        series,
        starting_cash=starting_cash,
        fee_rate=fee_rate,
    )
    console.print(_metrics_table(f"Backtest -- {signal}", result))


def _ma_builder(params: Mapping[str, float]) -> Signal:
    return MovingAverageCrossover(int(params["fast"]), int(params["slow"]))


@app.command()
def scan(
    metric: str = typer.Option("sharpe", help="sharpe | sortino | total_return_pct | cagr_pct"),
    top: int = typer.Option(5, help="how many leaders to show"),
    bars: int = 300,
) -> None:
    """Grid-scan an MA-crossover's fast/slow windows, ranked by a metric."""
    series = _synthetic_bars(bars)
    grid: dict[str, Sequence[float]] = {
        "fast": [5, 10, 15, 20],
        "slow": [30, 40, 50, 60],
    }
    ranked = grid_scan("DEMO", _ma_builder, series, grid, metric=metric)
    table = Table(title=f"Parameter scan -- ranked by {metric}", header_style="bold cyan")
    table.add_column("#", justify="right")
    table.add_column("fast", justify="right")
    table.add_column("slow", justify="right")
    table.add_column(metric, justify="right")
    table.add_column("return %", justify="right")
    table.add_column("max dd %", justify="right")
    for rank, s in enumerate(ranked[:top], start=1):
        table.add_row(
            str(rank),
            f"{int(s.params['fast'])}",
            f"{int(s.params['slow'])}",
            f"{s.score:.2f}",
            f"{s.result.total_return_pct:+.2f}",
            f"{s.result.max_drawdown_pct:.2f}",
        )
    console.print(table)


@app.command()
def walkforward(
    folds: int = typer.Option(4, help="number of rolling folds"),
    metric: str = typer.Option("sharpe", help="metric to optimise / report"),
    bars: int = 600,
) -> None:
    """Walk-forward validate an MA-crossover grid (in-sample vs out-of-sample)."""
    series = _synthetic_bars(bars)
    grid: dict[str, Sequence[float]] = {
        "fast": [5, 10, 15],
        "slow": [30, 40, 50],
    }
    wf = walk_forward("DEMO", _ma_builder, series, grid, folds=folds, metric=metric)
    table = Table(title=f"Walk-forward -- {metric}", header_style="bold cyan")
    table.add_column("Fold", justify="right")
    table.add_column("Best params")
    table.add_column("IS score", justify="right")
    table.add_column("OOS return %", justify="right")
    table.add_column("OOS max dd %", justify="right")
    for f in wf.folds:
        params = f"fast={int(f.best_params['fast'])}, slow={int(f.best_params['slow'])}"
        table.add_row(
            str(f.fold),
            params,
            f"{f.in_sample_score:.2f}",
            f"{f.out_of_sample.total_return_pct:+.2f}",
            f"{f.out_of_sample.max_drawdown_pct:.2f}",
        )
    console.print(table)
    console.print(f"Avg out-of-sample {metric}: [bold]{wf.avg_out_of_sample:.2f}[/]")
    console.print(f"Walk-forward efficiency:    [bold]{wf.efficiency:.2f}[/]")


if __name__ == "__main__":
    app()
