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

**VaR-RiskLab** is a Python-based portfolio risk analytics engine built to demonstrate the core
concepts of market risk measurement. It implements three widely-used VaR methodologies, Expected
Shortfall, rolling backtesting, and scenario-based stress testing — all exposed through an
interactive Streamlit dashboard.

The project is designed to be:
- **Simple** — every formula is documented and explainable
- **Correct** — implementations follow standard risk management methodology
- **Portable** — runs with a single `streamlit run app.py` command
- **Interview-ready** — the code and README directly address common Risk Methodology interview questions

> ⚠️ **Disclaimer:** This is a simplified educational implementation. It is **not** intended for
> production risk management, regulatory capital calculation, or any real financial decision-making.

---

## Motivation

Understanding Value at Risk is fundamental to any quantitative role in market risk. This project was
built to explore and implement the core techniques used by risk teams at investment banks:

- How do we measure the potential loss of a portfolio?
- How do we validate that our risk model is well-calibrated?
- What happens to our portfolio under extreme market conditions?

By implementing each methodology from scratch — rather than using a black-box library — the project
builds genuine understanding of the assumptions, strengths, and limitations of each approach.

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

```
Market Data (yfinance)
        ↓
  Adjusted Closing Prices
        ↓
  Daily Returns  (pct_change)
        ↓
  Weighted Portfolio Returns
        ↓
  ┌────────────────────────────┐
  │      Risk Models           │
  │  ┌─────────────────────┐  │
  │  │  Historical VaR      │  │
  │  │  Parametric VaR      │  │
  │  │  Monte Carlo VaR     │  │
  │  └─────────────────────┘  │
  └────────────────────────────┘
        ↓
  VaR & Expected Shortfall
        ↓
  Rolling Backtesting  →  Exceptions + Kupiec Test
        ↓
  Stress Testing  →  Scenario Losses
        ↓
  Streamlit Dashboard
```

---

## Project Structure

```
VaR-RiskLab/
│
├── app.py                  # Streamlit dashboard — main entry point
├── requirements.txt        # Python dependencies
├── README.md               # This file
├── .gitignore
│
├── src/
│   ├── __init__.py
│   ├── data_loader.py      # yfinance download, returns, synthetic fallback
│   ├── portfolio.py        # Weighted returns, covariance, stats
│   ├── var_models.py       # Historical, Parametric, Monte Carlo VaR + ES
│   ├── backtesting.py      # Rolling backtest + Kupiec test
│   └── stress_testing.py   # Scenario stress tests
│
├── data/
│   └── README.md           # Data directory notes
│
└── screenshots/
    └── README.md           # Screenshots directory
```

---

## Default Portfolio

| Ticker | Company    | Weight |
|--------|------------|--------|
| AAPL   | Apple      | 30%    |
| MSFT   | Microsoft  | 25%    |
| GOOGL  | Alphabet   | 20%    |
| AMZN   | Amazon     | 15%    |
| JPM    | JPMorgan   | 10%    |

All weights are configurable via the Streamlit sidebar.

---

## Installation & Setup

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/VaR-RiskLab.git
cd VaR-RiskLab
```

### 2. Create a virtual environment (recommended)

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the dashboard

```bash
streamlit run app.py
```

The dashboard will open in your browser at `http://localhost:8501`.

---

## Mathematical Methodology

### Portfolio Return

The daily portfolio return is a weighted linear combination of individual asset returns:

$$R_{p,t} = \sum_{i=1}^{N} w_i \cdot R_{i,t}$$

where $w_i$ is the weight of asset $i$ and $R_{i,t}$ is its daily return on day $t$.

This assumes the portfolio is **rebalanced daily** back to the target weights.

---

### Portfolio Variance from Covariance Matrix

$$\sigma_p^2 = \mathbf{w}^\top \Sigma \mathbf{w}$$

$$\sigma_p = \sqrt{\sigma_p^2}$$

where:
- $\mathbf{w}$ is the $(N \times 1)$ weight vector
- $\Sigma$ is the $(N \times N)$ sample covariance matrix of daily asset returns

---

### Historical Simulation VaR

$$\text{VaR}_\alpha = -Q_{1-\alpha}(\{R_{p,t}\}) \times V$$

where:
- $\alpha$ is the confidence level (e.g. 0.99)
- $Q_{1-\alpha}$ is the empirical quantile at the $(1-\alpha)$ level
- $V$ is the portfolio value

*No distributional assumptions are made — the historical return distribution is used directly.*

---

### Parametric (Variance-Covariance) VaR

Assuming $R_p \sim \mathcal{N}(\mu_p, \sigma_p^2)$:

$$\text{VaR}_\alpha = (z_\alpha \cdot \sigma_p - \mu_p) \times V$$

where $z_\alpha = \Phi^{-1}(1-\alpha)$ is the left-tail normal quantile.

For 99% VaR: $z_{0.01} \approx -2.326$

---

### Expected Shortfall

$$\text{ES}_\alpha = \mathbb{E}[\text{Loss} \mid \text{Loss} > \text{VaR}_\alpha]$$

For the historical model, this is the average of losses exceeding the VaR threshold.

For the parametric normal model, the analytical expression is:

$$\text{ES}_\alpha = \left( -\mu_p + \sigma_p \cdot \frac{\phi(z_\alpha)}{1-\alpha} \right) \times V$$

where $\phi(\cdot)$ is the standard normal PDF.

---

### Monte Carlo VaR

1. Calibrate $\boldsymbol{\mu}$ and $\Sigma$ from historical data.
2. Sample $N = 10{,}000$ scenarios: $\mathbf{r}^{(i)} \sim \mathcal{N}(\boldsymbol{\mu}, \Sigma)$
3. Compute simulated portfolio returns: $r_p^{(i)} = \mathbf{w}^\top \mathbf{r}^{(i)}$
4. $\text{VaR}_\alpha = Q_{1-\alpha}(\{-r_p^{(i)} \cdot V\})$

*Model Note:* The current implementation uses a multivariate normal model; the Monte Carlo framework can be extended to other distributions (e.g. Student's t, copulas) and nonlinear payoffs (e.g. options full revaluation).

---

### VaR Backtesting — Kupiec Test

The **Kupiec (1995) unconditional coverage test** tests whether the observed exception rate is
statistically consistent with the expected rate.

$$H_0: p = p_0$$

$$LR_{uc} = -2 \ln\!\left[\frac{p_0^x (1-p_0)^{n-x}}{\hat{p}^x (1-\hat{p})^{n-x}}\right] \sim \chi^2_1$$

where $x$ = observed exceptions, $n$ = total observations, $p_0 = 1 - \alpha$.

---

## Results

When you run the dashboard you will see:

- **VaR estimates** from all three models — they should be broadly similar, with the parametric
  model diverging if the actual return distribution is non-normal.
- **ES values** that are consistently higher than VaR, capturing average tail severity.
- **Backtest chart** showing how the rolling VaR compares to actual daily P&L over time.
- **Exception count** that you can compare against the expected rate (1% or 5%).
- **Stress scenario losses** that are typically many multiples of the 1-day VaR.

Exact numbers will vary depending on the market data period and selected parameters.

---

## Limitations

1. **Simplified educational implementation** — not a production risk system.
2. **Historical VaR** depends on the data window; it cannot anticipate crises not in the sample.
3. **Parametric VaR** assumes normally distributed returns — real equity returns exhibit fat tails and skewness.
4. **Monte Carlo VaR** calibrated to a multivariate normal also inherits the normality limitation.
5. **Uniform stress shocks** are a gross simplification; real stress tests use factor-level sensitivities.
6. **Constant weights** — the backtest assumes daily rebalancing, which is not realistic in practice.
7. **1-day horizon only** — extending to multi-day VaR requires scaling assumptions (e.g. $\sqrt{T}$ rule).

---

## Future Improvements

The following extensions would be natural next steps for a more advanced implementation:

- **GARCH volatility models** — allow volatility to evolve over time (e.g. GARCH(1,1))
- **Non-normal distributions** — Student's t, skewed normal for fatter tails
- **FRTB / Expected Shortfall** — full Fundamental Review of the Trading Book framework
- **Christoffersen conditional coverage test** — tests for exception clustering, not just count
- **Filtered Historical Simulation** — rescale historical shocks by current vs. historical volatility
- **Richer stress scenarios** — historical crisis scenarios (2008, COVID), regulatory scenarios
- **Counterparty credit risk** — Credit VaR, CVA
- **Options / non-linear payoffs** — Delta-Gamma approximation in parametric VaR
- **Regulatory capital calculation** — Basel III/IV capital charge estimates

---

## Interview Notes

*Concise answers to common Risk Methodology / Market Risk interview questions.*

---

**1. What is VaR?**
Value at Risk (VaR) is a quantile-based estimate of portfolio loss over a specified time horizon
and confidence level, under normal market conditions. Intuitively, a 99%, 1-day VaR of \$2,000 means
that on 99% of trading days, portfolio losses are expected to be \$2,000 or less (or equivalently,
there is a 1% probability of losing more than \$2,000 in a single day). Crucially, VaR is a loss
threshold quantile — it is not the maximum possible loss, and it says nothing about the severity
of losses beyond the threshold.

---

**2. Why implement three VaR methodologies?**
Each method makes different assumptions: Historical Simulation uses the empirical return
distribution with no parametric assumptions; Parametric VaR assumes normality and provides a closed-form
analytical solution; Monte Carlo explicitly simulates scenario-by-scenario outcomes.
While our baseline Monte Carlo implementation uses a multivariate normal model, the Monte Carlo
framework can be extended to non-normal distributions and non-linear option payoffs.
Comparing all three allows us to evaluate model risk and validate consistency.

---

**3. What is the difference between Historical and Parametric VaR?**
Historical VaR uses the actual observed distribution of past returns, with no assumption about
its shape — it naturally captures fat tails. Parametric VaR assumes returns are normally
distributed and computes VaR analytically using the portfolio mean and volatility. Parametric VaR
is computationally faster and allows risk decomposition, but it will understate VaR when tails are
fatter than a normal distribution.

---

**4. How does Monte Carlo VaR work?**
We calibrate a multivariate normal distribution to historical returns (estimating mean and
covariance), then simulate thousands of hypothetical one-day portfolio returns. The loss
distribution is built from these simulated scenarios, and VaR is read off as the empirical quantile of the simulated losses.

**Key Interview Distinction — Parametric vs. Monte Carlo VaR:**
- **Parametric VaR** is a closed-form analytical calculation ($z_\alpha \cdot \sigma_p - \mu_p$).
- **Monte Carlo VaR** explicitly generates simulated scenario paths.
- *Framework Flexibility:* The current implementation uses a multivariate normal model; the Monte Carlo framework can be extended to other distributions (e.g. Student's t, copulas) and nonlinear payoffs (e.g. options full revaluation).

---

**5. Why is Expected Shortfall useful?**
VaR identifies a loss threshold but ignores the severity of losses beyond it. Expected Shortfall
(ES) measures the *average* loss conditional on exceeding VaR. Two portfolios can have identical
VaR but very different ES if one has a thicker tail. ES is also a coherent risk measure (VaR is
not), and Basel IV / FRTB mandates ES for internal model regulatory capital.

---

**6. What is VaR backtesting?**
Backtesting compares ex-ante VaR predictions to ex-post actual P&L outcomes. For each day in the
test period, we ask: "Did the actual loss exceed yesterday's VaR prediction?" If so, it is an
exception. We count exceptions and compare to the expected rate (e.g. 1% for 99% VaR).

---

**7. What is a VaR exception?**
A VaR exception (or breach) occurs when the actual trading-day loss exceeds the VaR prediction
made the previous day. Under a well-calibrated 99% VaR model, we expect ~1% of days to be
exceptions — roughly 2–3 per year of trading. More frequent exceptions suggest the model
under-estimates risk.

---

**8. Why can VaR fail?**
VaR can fail because: (1) Historical data does not represent future crises; (2) The normality
assumption underestimates fat tails; (3) Correlations between assets spike during market stress
("correlation breakdown"), making the diversification benefit in the model unrealistic; (4) VaR
is a point estimate — it says nothing about what happens beyond the threshold; (5) Model parameters
(covariance, mean) are estimated with statistical error.

---

**9. Why do we perform stress testing?**
VaR is calibrated to normal market conditions. Stress tests explore scenarios that are outside the
historical data window — extreme tail events, regulatory scenarios, or hypothetical crises. They
answer "what would happen to our portfolio if the market fell 20% today?" and are a necessary
complement to VaR-based risk management.

---

**10. What happens when volatility increases?**
Higher volatility increases the spread of the return distribution. Both Parametric and Monte Carlo
VaR (which depend on $\sigma_p$ directly) will immediately produce a higher VaR estimate. Historical
VaR will also increase, but only after the high-volatility period is reflected in the lookback window.
This lagged response is one of the limitations of Historical Simulation.

---

**11. What happens when correlations between assets increase?**
Higher correlations reduce the benefit of diversification. The portfolio variance
$\sigma_p^2 = \mathbf{w}^\top \Sigma \mathbf{w}$ increases when off-diagonal elements of $\Sigma$
are larger, leading to higher portfolio volatility and higher VaR. In a crisis, correlations tend
to spike toward 1, so the diversification assumed in the model often evaporates exactly when it is
needed most — this is the "correlation breakdown" problem.

---

**12. What are the limitations of Historical VaR?**
(1) It depends entirely on the historical sample — if the lookback window does not include a crisis
period, the model will understate risk. (2) All historical returns are equally weighted — a
1000-day lookback treats the observation from 4 years ago the same as yesterday. (3) It cannot
model events not observed in the historical window. (4) The lookback window choice is arbitrary
and can significantly affect the VaR estimate.

---

**13. Why might Parametric VaR be inaccurate?**
The normality assumption is the main weakness. Real equity returns exhibit negative skewness and
excess kurtosis (fat tails). The normal distribution systematically underestimates the probability
of extreme losses. Additionally, the covariance matrix is estimated from historical data and may
not represent the stressed correlation structure during a market crisis.

---

**14. Why do Monte Carlo results depend on simulation assumptions?**
Monte Carlo VaR is only as good as the distribution it simulates from. If we simulate from a
multivariate normal (as in this implementation), we inherit the normality limitation. If we use
too short a calibration period, the covariance matrix may not be representative. The random seed
affects results (though fixing it ensures reproducibility), and a small number of simulations
increases sampling noise. Model risk — the risk of using the wrong model — applies to Monte Carlo
just as it does to other methods.

---

**15. What does model performance monitoring mean?**
Model performance monitoring (or model validation) is the ongoing process of assessing whether a
risk model remains fit for purpose. For VaR models, this includes: (1) regular backtesting to check
exception rates; (2) statistical tests like Kupiec and Christoffersen; (3) P&L attribution — comparing
model-predicted P&L to actual P&L; (4) sensitivity analysis — checking how the model responds to
changes in inputs; (5) comparison to alternative models (benchmark testing). Banks are required by
regulators to maintain model risk management frameworks for all models used in capital calculation.

---

## License

MIT License — see [LICENSE](LICENSE) for details.

---

<div align="center">
Built for educational purposes · Not for production use
</div>
