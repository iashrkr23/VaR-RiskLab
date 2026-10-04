# VaR-RiskLab

<div align="center">

**Portfolio Risk Analytics & VaR Backtesting Engine**

*An educational market-risk analytics project demonstrating Historical, Parametric, and Monte Carlo VaR*

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-red?logo=streamlit)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-Educational-orange)

</div>

---

## Overview

**VaR-RiskLab** is a Python-based portfolio risk analytics engine built to demonstrate the core concepts of market risk measurement. It implements three widely-used VaR methodologies, Expected Shortfall, rolling backtesting, and scenario-based stress testing — all exposed through an interactive Streamlit dashboard.

The project is designed to be:
- **Simple** — every formula is documented and explainable
- **Correct** — implementations follow standard risk management methodology
- **Portable** — runs with a single `streamlit run app.py` command

---

## Motivation

Understanding Value at Risk is fundamental to any quantitative role in market risk. This project was built to explore and implement the core techniques used by risk teams at investment banks:

- How do we measure the potential loss of a portfolio?
- How do we validate that our risk model is well-calibrated?
- What happens to our portfolio under extreme market conditions?

By implementing each methodology from scratch — rather than using a black-box library — the project builds genuine understanding of the assumptions, strengths, and limitations of each approach.

---

## Key Features

| Feature | Description |
|---|---|
| 📉 **Historical VaR** | Empirical quantile of historical returns — no distribution assumption |
| 📐 **Parametric VaR** | Variance-covariance method assuming normality |
| 🎲 **Monte Carlo VaR** | 10,000+ simulated scenarios from multivariate normal |
| 📊 **Expected Shortfall** | Average tail loss beyond the VaR threshold |
| 🔍 **VaR Backtesting** | Rolling one-day-ahead backtest with exception counting |
| ⚡ **Stress Testing** | Scenario shocks: -10%, -20%, -30%, custom |
| 🖥️ **Streamlit Dashboard** | Interactive, dark-themed dashboard with Plotly charts |
| 🔄 **Synthetic Fallback** | Auto-generates synthetic data if yfinance is unavailable |

---

## Architecture

```text
Market Data (yfinance)
        ↓
 Adjusted Closing Prices
        ↓
 Daily Returns  (pct_change)
        ↓
 Weighted Portfolio Returns
        ↓
 ┌────────────────────────────┐
 │     Risk Models            │
 │  ┌─────────────────────┐   │
 │  │  Historical VaR     │   │
 │  │  Parametric VaR     │   │
 │  │  Monte Carlo VaR    │   │
 │  └─────────────────────┘   │
 └────────────────────────────┘
        ↓
 VaR & Expected Shortfall
        ↓
 Rolling Backtesting  →  Exceptions + Kupiec Test
        ↓
 Stress Testing  →  Scenario Losses
        ↓
 Streamlit Dashboard
