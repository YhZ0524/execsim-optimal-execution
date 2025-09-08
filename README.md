# Optimal Execution with Market and Limit Orders (Python)

This repo implements an intraday execution simulator with order-book microstructure and casts execution as policy optimization. Prices follow a diffusion-like process, aggressive order flow arrives via a Poisson process, and limit-order fills depend on posting depth via an exponential fill model. A parametric policy trades off price improvement from resting bids and progress via market orders under a deviation band from a target schedule. A grid search with vectorized Monte Carlo selects policy hyperparameters.

Data are synthetic and mimic China A-share microstructure: U-shaped aggressive order arrival intensity over time, discrete ticks, and a one-tick quoted spread. Benchmarks include market-only TWAP and a front-loaded market schedule.

## Quickstart

python -m src.execsim.main

Artifacts are written to `results/`: `grid_search.csv`, `summary.csv`, and PNG plots.

## Updated resume bullets (synthetic data)

Built an intraday execution simulator with diffusion prices, Poisson order flow, and an exponential depth–fill curve; framed execution as policy optimization that balances shortfall and progress under deviation bands, tuned by Monte Carlo on synthetic order-book traces mimicking China A-share microstructure.

Benchmarked against market-only Almgren–Chriss/TWAP baselines; at matched completion the feature-driven policy reduced implementation shortfall on synthetic U-shaped intensity days while relying on limit orders for price improvement and market orders for on-schedule catch-up.
