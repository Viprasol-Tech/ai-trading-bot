# Changelog

All notable changes to this project are documented here. Format based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); versioning follows
[SemVer](https://semver.org/).

## [0.2.0] - 2025

### Added
- **Three new signals**: `MACDSignal` (trend/momentum), `BollingerBandsSignal`
  (mean reversion), and `BreakoutSignal` (Donchian-channel breakout), all
  implementing the same `Signal` interface so they slot straight into an ensemble.
- **Parameter optimisation** (`ai_trading_bot.optimize`): `grid_scan` for an
  exhaustive, metric-ranked grid search, and `walk_forward` for rolling
  in-sample/out-of-sample validation with a walk-forward efficiency score.
- **Richer backtest metrics**: CAGR, annualised volatility, Sharpe ratio,
  Sortino ratio, realised win rate, and a `BacktestResult.as_dict()` helper.
  The paper broker now tracks realised round-trip trades and how many were wins.
- **Typed configuration** (`ai_trading_bot.config.BotSettings`): a pydantic model
  describing the full run (signal mix, weights, risk limits, broker assumptions),
  overridable via `ATB_`-prefixed environment variables or a `.env` file.
- **New CLI subcommands**: `backtest` (single signal + metrics table),
  `scan` (grid search), and `walkforward` (rolling validation), alongside the
  existing `demo` and `version`. Output now renders as Rich tables.
- **Examples**: `examples/quickstart.py` demonstrating ensemble backtest, grid
  scan, and walk-forward validation end to end.

### Changed
- `demo` now builds its ensemble from `BotSettings`, keeping the CLI and the
  programmatic config path in sync.
- Synthetic demo bars now carry distinct high/low values so breakout-style
  signals behave realistically.

## [0.1.0] - 2025

### Added
- Initial release of ai-trading-bot: Broker-agnostic AI/ML signal-driven algorithmic trading bot framework.
