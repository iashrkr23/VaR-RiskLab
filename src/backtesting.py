"""
backtesting.py
==============
Rolling one-day-ahead VaR backtesting using Historical Simulation.

Methodology
-----------
For each trading day *t* after the initial lookback window:

    1. Use the previous `window` trading days as the historical sample.
    2. Compute the Historical VaR at the chosen confidence level.
    3. Record the *actual* portfolio return on day *t*.
    4. An **exception** (breach) occurs when the actual loss > predicted VaR.
       i.e.  -actual_return_t * portfolio_value  >  VaR_t

The exception rate should be approximately equal to (1 - confidence_level):
    - 99% VaR → ~1%  expected exceptions
    - 95% VaR → ~5%  expected exceptions

If the observed exception rate is significantly higher, the model may be
under-estimating risk.

Optional: Kupiec Unconditional Coverage Test
--------------------------------------------
A likelihood-ratio test that checks whether the observed number of
exceptions is statistically consistent with the model's expected rate.
"""

import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, Tuple


# ---------------------------------------------------------------------------
# Rolling backtest
# ---------------------------------------------------------------------------

def run_backtest(
    portfolio_returns: pd.Series,
    confidence_level: float,
    portfolio_value: float,
    window: int = 250,
) -> pd.DataFrame:
    """
    Perform rolling one-day-ahead Historical VaR backtesting.

    Parameters
    ----------
    portfolio_returns : pd.Series  — full history of daily portfolio returns
    confidence_level  : float      — e.g. 0.95 or 0.99
    portfolio_value   : float      — notional value ($)
    window            : int        — lookback window in trading days (default 250)

    Returns
    -------
    pd.DataFrame with columns:
        date          : index (pd.DatetimeIndex)
        actual_return : actual portfolio return on that day
        actual_loss   : -actual_return * portfolio_value  (positive = loss)
        predicted_var : VaR predicted using the prior `window` days
        exception     : bool — True if actual_loss > predicted_var
    """
    n = len(portfolio_returns)
    if n <= window:
        raise ValueError(
            f"Not enough data for backtesting. "
            f"Need > {window} observations, got {n}."
        )

    dates         = []
    actual_returns = []
    actual_losses  = []
    predicted_vars = []
    exceptions     = []

    tail_quantile = 1.0 - confidence_level

    for i in range(window, n):
        # Historical window: days [i-window, i-1]
        historical_window = portfolio_returns.iloc[i - window : i]

        # Predict VaR using the historical window
        var_return = float(np.quantile(historical_window, tail_quantile))
        var_dollar = max(-var_return * portfolio_value, 0.0)

        # Actual portfolio return/loss on day i (the *next* day)
        actual_return = float(portfolio_returns.iloc[i])
        actual_loss   = -actual_return * portfolio_value

        # Exception: actual loss exceeds the VaR prediction
        is_exception = actual_loss > var_dollar

        dates.append(portfolio_returns.index[i])
        actual_returns.append(actual_return)
        actual_losses.append(actual_loss)
        predicted_vars.append(var_dollar)
        exceptions.append(is_exception)

    backtest_df = pd.DataFrame(
        {
            "actual_return" : actual_returns,
            "actual_loss"   : actual_losses,
            "predicted_var" : predicted_vars,
            "exception"     : exceptions,
        },
        index=pd.DatetimeIndex(dates),
    )
    return backtest_df


# ---------------------------------------------------------------------------
# Summary statistics
# ---------------------------------------------------------------------------

def compute_backtest_stats(
    backtest_df: pd.DataFrame,
    confidence_level: float,
) -> Dict[str, float]:
    """
    Compute summary statistics from the backtest results.

    Parameters
    ----------
    backtest_df      : pd.DataFrame — output from run_backtest()
    confidence_level : float

    Returns
    -------
    dict with:
        n_observations    : total number of backtest days
        n_exceptions      : observed number of VaR breaches
        exception_rate    : n_exceptions / n_observations
        expected_rate     : 1 - confidence_level
        expected_n_exceptions : expected_rate * n_observations
        kupiec_pvalue     : p-value from Kupiec LR test (or NaN if unavailable)
    """
    n_obs        = len(backtest_df)
    n_exc        = int(backtest_df["exception"].sum())
    exc_rate     = n_exc / n_obs if n_obs > 0 else 0.0
    expected_rate = 1.0 - confidence_level
    expected_n   = expected_rate * n_obs

    kupiec_p = _kupiec_pvalue(n_exc, n_obs, expected_rate)

    return {
        "n_observations"       : n_obs,
        "n_exceptions"         : n_exc,
        "exception_rate"       : exc_rate,
        "expected_rate"        : expected_rate,
        "expected_n_exceptions": expected_n,
        "kupiec_pvalue"        : kupiec_p,
    }


# ---------------------------------------------------------------------------
# Kupiec Unconditional Coverage Test
# ---------------------------------------------------------------------------

def _kupiec_pvalue(
    n_exceptions: int,
    n_observations: int,
    expected_rate: float,
) -> float:
    """
    Perform the Kupiec (1995) unconditional coverage likelihood-ratio test.

    H₀: The true exception rate equals the model's expected rate.
    H₁: The true exception rate differs from the expected rate.

    The test statistic is:

    LR_uc = -2 * ln [ (p̂^x * (1-p̂)^(n-x)) / (p₀^x * (1-p₀)^(n-x)) ]

    where:
        p₀ = expected_rate  (model exception probability)
        p̂  = n_exceptions / n_observations  (observed rate)
        x  = n_exceptions
        n  = n_observations

    Under H₀, LR_uc ~ χ²(1).

    A p-value > 0.05 means we cannot reject H₀ — the model's exception
    rate is statistically consistent with the observed rate.

    Parameters
    ----------
    n_exceptions   : int
    n_observations : int
    expected_rate  : float  — e.g. 0.01 for 99% VaR

    Returns
    -------
    float : p-value (NaN if the test cannot be computed)
    """
    x  = n_exceptions
    n  = n_observations
    p0 = expected_rate
    p_hat = x / n if n > 0 else 0.0

    # Edge cases: perfect or zero exception rate
    if x == 0 or x == n:
        return float("nan")
    if p0 <= 0.0 or p0 >= 1.0:
        return float("nan")

    try:
        # Log-likelihood under null (model) vs alternative (observed rate)
        ll_null = x * np.log(p0)       + (n - x) * np.log(1.0 - p0)
        ll_alt  = x * np.log(p_hat)    + (n - x) * np.log(1.0 - p_hat)

        lr_stat = -2.0 * (ll_null - ll_alt)      # always >= 0

        # Chi-squared test with 1 degree of freedom
        p_value = float(1.0 - stats.chi2.cdf(lr_stat, df=1))
        return p_value
    except Exception:
        return float("nan")
