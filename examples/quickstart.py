"""End-to-end example: build an ensemble, backtest it, then tune it.

Run from the repo root with::

    PYTHONPATH=src python examples/quickstart.py

It shows the three things you will do most often:

1. Blend several signals into one ensemble and backtest it.
2. Read the rich metrics off the result.
3. Grid-scan a parameter space and walk-forward validate the winner.

Part of AI Trading Bot by Viprasol Tech Private Limited (https://viprasol.com).
"""

from __future__ import annotations

import math
from collections.abc import Mapping

from ai_trading_bot.backtest.engine import run_backtest
from ai_trading_bot.config import BotSettings
from ai_trading_bot.models import Bar
from ai_trading_bot.optimize.scan import grid_scan, walk_forward
from ai_trading_bot.signals.base import Signal
from ai_trading_bot.signals.indicators import MovingAverageCrossover


def make_bars(n: int = 600) -> list[Bar]:
    """A deterministic trending-with-noise series so the example is reproducible."""
    bars: list[Bar] = []
    for i in range(n):
        price = 100.0 + i * 0.2 + 8.0 * math.sin(i / 12.0)
        bars.append(Bar(timestamp=i, open=price, high=price + 1.5, low=price - 1.5, close=price))
    return bars


def main() -> None:
    bars = make_bars()

    # 1) Config-driven ensemble (MA + RSI + MACD by default) and a backtest.
    settings = BotSettings(starting_cash=10_000.0)
    result = run_backtest(
        settings.symbol,
        settings.build_ensemble(),
        bars,
        starting_cash=settings.starting_cash,
        sizer=settings.build_sizer(),
    )
    print("Ensemble backtest")
    for key, value in result.as_dict().items():
        print(f"  {key:>18}: {value:,.2f}")

    # 2) Grid-scan an MA-crossover's windows, ranked by Sharpe.
    def build_ma(params: Mapping[str, float]) -> Signal:
        return MovingAverageCrossover(int(params["fast"]), int(params["slow"]))

    grid = {"fast": [5, 10, 15], "slow": [30, 40, 50]}
    ranked = grid_scan(settings.symbol, build_ma, bars, grid, metric="sharpe")
    best = ranked[0]
    print("\nBest grid params:", best.params, f"(sharpe={best.score:.2f})")

    # 3) Walk-forward validation of that same grid.
    wf = walk_forward(settings.symbol, build_ma, bars, grid, folds=4, metric="sharpe")
    print(f"Walk-forward avg OOS sharpe: {wf.avg_out_of_sample:.2f}")
    print(f"Walk-forward efficiency:     {wf.efficiency:.2f}")


if __name__ == "__main__":
    main()
