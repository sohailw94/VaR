# VaR
Quantitative Trading Strategies: Monte Carlo VaR & Pair Trading

This repository contains two end-to-end quantitative trading strategies built using Python and real financial data:

- Monte Carlo Simulation for Value at Risk (VaR)**
- Pair Trading Strategy using Cointegration (Engle-Granger Test)**

Each module is designed to be self-contained, easy to run, and informative for learning or demonstrating quant finance skills.

---

Strategy 1: Monte Carlo Simulation for VaR

This script performs a Monte Carlo simulation to estimate the potential downside risk (VaR) of a 5-stock portfolio over a 20-day horizon.

Features:
- Downloads historical data for selected tickers using `yfinance`
- Calculates log returns, mean vector, and covariance matrix
- Applies Cholesky decomposition to simulate correlated asset paths
- Computes simulated portfolio P&L
- Visualizes the distribution and highlights the 1% and 5% Value at Risk (VaR)

**Portfolio:**
- AAPL, TSLA, MSFT, GOOG, NVDA (equal-weighted)

---

Strategy 2: Cointegration-Based Pair Trading

This script implements a classic mean-reverting strategy based on identifying cointegrated stock pairs.

Features:
- Tests all stock pairs using the Engle-Granger cointegration test
- Selects the best pair based on lowest p-value
- Calculates the Z-score of the residual spread
- Generates long/short/exit signals based on Z-score thresholds
- Tracks cumulative strategy PnL (equity curve)
- Prints the current trading signal

**Signal Thresholds:**
- Long if Z < -1, Short if Z > +1, Exit when Z crosses zero

---

