"""Parameter search: exhaustive grid scan and walk-forward analysis.

``grid_scan`` evaluates every combination of a parameter grid by re-running the
backtest and ranking the results by a chosen metric. ``walk_forward`` splits the
data into rolling in-sample/out-of-sample folds, optimises on each in-sample
window, then measures the chosen parameters on the *next* out-of-sample window —
the honest way to estimate how a strategy would have generalised.

A "build" callable maps a parameter dict to a :class:`Signal`, which keeps this
module agnostic to which signal (or ensemble) you are tuning.

Part of AI Trading Bot by Viprasol Tech Private Limited (https://viprasol.com).
"""

from __future__ import annotations

import itertools
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field

from ai_trading_bot.backtest.engine import BacktestResult, run_backtest
from ai_trading_bot.models import Bar
from ai_trading_bot.signals.base import Signal

#: A factory turning a concrete parameter set into a ready-to-run signal.
SignalBuilder = Callable[[Mapping[str, float]], Signal]


def _expand_grid(grid: Mapping[str, Sequence[float]]) -> list[dict[str, float]]:
    """Cartesian product of a parameter grid into a list of param dicts."""
    if not grid:
        return [{}]
    keys = list(grid.keys())
    return [dict(zip(keys, combo, strict=True)) for combo in itertools.product(*grid.values())]


def _metric_value(result: BacktestResult, metric: str) -> float:
    """Pull a named metric off a result, raising for unknown names."""
    values = result.as_dict()
    if metric not in values:
        raise KeyError(f"unknown metric {metric!r}; choose one of {sorted(values)}")
    return values[metric]


@dataclass(slots=True)
class ScanResult:
    """One evaluated point in a parameter scan."""

    params: dict[str, float]
    result: BacktestResult
    score: float


def grid_scan(
    symbol: str,
    builder: SignalBuilder,
    bars: Sequence[Bar],
    grid: Mapping[str, Sequence[float]],
    metric: str = "sharpe",
    starting_cash: float = 10_000.0,
    fee_rate: float = 0.0,
) -> list[ScanResult]:
    """Backtest every grid combination, ranked best-first by ``metric``."""
    combos = _expand_grid(grid)
    scanned: list[ScanResult] = []
    for params in combos:
        signal = builder(params)
        result = run_backtest(symbol, signal, bars, starting_cash=starting_cash, fee_rate=fee_rate)
        score = _metric_value(result, metric)
        scanned.append(ScanResult(params=params, result=result, score=score))
    scanned.sort(key=lambda s: s.score, reverse=True)
    return scanned


@dataclass(slots=True)
class WalkForwardFold:
    """A single in-sample/out-of-sample fold of a walk-forward run."""

    fold: int
    best_params: dict[str, float]
    in_sample_score: float
    out_of_sample: BacktestResult


@dataclass(slots=True)
class WalkForwardResult:
    """Aggregated walk-forward analysis across all folds."""

    metric: str
    folds: list[WalkForwardFold] = field(default_factory=list)

    @property
    def avg_out_of_sample(self) -> float:
        """Mean of the chosen metric across out-of-sample windows."""
        if not self.folds:
            return 0.0
        scores = [_metric_value(f.out_of_sample, self.metric) for f in self.folds]
        return sum(scores) / len(scores)

    @property
    def efficiency(self) -> float:
        """OOS / IS score ratio: how well in-sample edge survived (clamped >= 0)."""
        if not self.folds:
            return 0.0
        is_total = sum(max(0.0, f.in_sample_score) for f in self.folds)
        if is_total == 0:
            return 0.0
        oos_total = sum(max(0.0, _metric_value(f.out_of_sample, self.metric)) for f in self.folds)
        return oos_total / is_total


def walk_forward(
    symbol: str,
    builder: SignalBuilder,
    bars: Sequence[Bar],
    grid: Mapping[str, Sequence[float]],
    folds: int = 4,
    metric: str = "sharpe",
    starting_cash: float = 10_000.0,
    fee_rate: float = 0.0,
) -> WalkForwardResult:
    """Rolling-window walk-forward: optimise in-sample, score out-of-sample.

    The series is split into ``folds + 1`` equal segments. For fold *i* the bot
    is optimised on segment *i* (in-sample) and evaluated on segment *i + 1*
    (out-of-sample), so every evaluation uses parameters chosen without seeing
    that data.
    """
    if folds < 1:
        raise ValueError("folds must be >= 1")
    n = len(bars)
    if n < (folds + 1) * 2:
        raise ValueError("not enough bars for the requested number of folds")

    seg = n // (folds + 1)
    wf = WalkForwardResult(metric=metric)
    for i in range(folds):
        in_sample = bars[i * seg : (i + 1) * seg]
        out_sample = bars[(i + 1) * seg : (i + 2) * seg]
        ranked = grid_scan(
            symbol,
            builder,
            in_sample,
            grid,
            metric=metric,
            starting_cash=starting_cash,
            fee_rate=fee_rate,
        )
        best = ranked[0]
        oos = run_backtest(
            symbol,
            builder(best.params),
            out_sample,
            starting_cash=starting_cash,
            fee_rate=fee_rate,
        )
        wf.folds.append(
            WalkForwardFold(
                fold=i,
                best_params=best.params,
                in_sample_score=best.score,
                out_of_sample=oos,
            )
        )
    return wf
