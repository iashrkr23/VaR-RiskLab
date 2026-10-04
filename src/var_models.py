"""
var_models.py
=============
Implements three Value-at-Risk (VaR) methodologies and Expected Shortfall (ES).

Methodologies
-------------
1. Historical Simulation VaR
   — Uses the empirical distribution of past portfolio returns.
   — No distributional assumptions.

2. Parametric (Variance-Covariance) VaR
   — Assumes portfolio returns are normally distributed.
   — Uses the portfolio mean, volatility, and a normal quantile (z-score).

3. Monte Carlo VaR
   — Simulates future returns by drawing from a multivariate normal
     distribution calibrated to historical means and covariances.
   — The loss distribution is built from 10,000+ simulated scenarios.

Sign convention
---------------
VaR is expressed as a *positive* monetary loss.
A VaR of $2,000 means the estimated potential loss is $2,000.
"""

import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, Tuple

MONTE_CARLO_SEED: int = 42     # Fixed seed for reproducibility


# ---------------------------------------------------------------------------
# 1. Historical Simulation VaR
# ---------------------------------------------------------------------------

def historical_var(
    portfolio_returns: pd.Series,
    confidence_level: float,
    portfolio_value: float,
) -> float:
    """
    Calculate Historical Simulation VaR.

    Method
    ------
    Sort historical portfolio returns and identify the (1 - confidence_level)
    quantile of the loss distribution.

    For 99% VaR:
        quantile = 1 - 0.99 = 0.01  (bottom 1% of returns)

    The quantile return is negative (a loss); we multiply by -1 and by the
    portfolio value to express it as a positive dollar loss.

    VaR = -quantile(return, 1 - confidence_level) * portfolio_value

    Parameters
    ----------
    portfolio_returns  : pd.Series   — daily portfolio returns (decimal)
    confidence_level   : float       — e.g. 0.95 or 0.99
    portfolio_value    : float       — notional portfolio value ($)

    Returns
    -------
    float : VaR expressed as a positive dollar loss.
    """
    # The quantile at the left tail (e.g. 1st percentile for 99% VaR)
    tail_quantile = 1.0 - confidence_level
    var_return = float(np.quantile(portfolio_returns, tail_quantile))

    # Convert negative return to a positive dollar loss
    var_dollar = -var_return * portfolio_value
    return max(var_dollar, 0.0)    # Ensure non-negative


def historical_es(
    portfolio_returns: pd.Series,
    confidence_level: float,
    portfolio_value: float,
) -> Tuple[float, float]:
    """
    Calculate Historical VaR and Expected Shortfall (Conditional VaR).

    Expected Shortfall (ES) answers the question:
    "Given that we exceed VaR, what is the *average* loss?"

    ES = average of losses that exceed the VaR threshold
       = -mean( returns | return < VaR_return_threshold ) * portfolio_value

    ES is always >= VaR because it averages over the entire tail,
    not just the boundary.

    Parameters
    ----------
    portfolio_returns  : pd.Series
    confidence_level   : float
    portfolio_value    : float

    Returns
    -------
    (var_dollar, es_dollar) : (float, float)
        Both expressed as positive dollar losses.
    """
    tail_quantile = 1.0 - confidence_level
    var_return    = float(np.quantile(portfolio_returns, tail_quantile))

    # Tail returns: all returns worse than the VaR threshold
    tail_returns = portfolio_returns[portfolio_returns <= var_return]

    if len(tail_returns) == 0:
        # Fallback: use the single worst return
        es_return = float(portfolio_returns.min())
    else:
        es_return = float(tail_returns.mean())

    var_dollar = -var_return  * portfolio_value
    es_dollar  = -es_return   * portfolio_value

    return max(var_dollar, 0.0), max(es_dollar, 0.0)


# ---------------------------------------------------------------------------
# 2. Parametric (Variance-Covariance) VaR
# ---------------------------------------------------------------------------

def parametric_var(
    portfolio_mean: float,
    portfolio_volatility: float,
    confidence_level: float,
    portfolio_value: float,
) -> float:
    """
    Calculate Parametric (Variance-Covariance) VaR.

    Assumption
    ----------
    Portfolio returns are normally distributed: R ~ N(μ, σ²)

    Formula
    -------
    z   = norm.ppf(1 - confidence_level)   [negative left-tail z-score]

    VaR = ( -z * σ - μ ) * portfolio_value

    Since z is negative for left-tail quantiles, -z is positive,
    giving a positive VaR figure.

    Example: 99% VaR → z = norm.ppf(0.01) ≈ -2.326
    VaR = (2.326 * σ - μ) * portfolio_value

    Parameters
    ----------
    portfolio_mean       : float — daily mean return (decimal)
    portfolio_volatility : float — daily standard deviation (decimal)
    confidence_level     : float — e.g. 0.95 or 0.99
    portfolio_value      : float — notional portfolio value ($)

    Returns
    -------
    float : VaR expressed as a positive dollar loss.
    """
    # Left-tail z-score (e.g. -2.326 for 99% VaR)
    z = stats.norm.ppf(1.0 - confidence_level)

    # Parametric VaR formula
    # -z converts from negative quantile to positive loss direction
    var_return = z * portfolio_volatility + portfolio_mean
    var_dollar = -var_return * portfolio_value

    return max(var_dollar, 0.0)


def parametric_es(
    portfolio_mean: float,
    portfolio_volatility: float,
    confidence_level: float,
    portfolio_value: float,
) -> Tuple[float, float]:
    """
    Calculate Parametric VaR and Expected Shortfall.

    For a normal distribution, the analytical ES formula is:

    ES = ( φ(z) / (1 - confidence_level) * σ - μ ) * portfolio_value

    where φ(z) is the standard normal PDF at the quantile z.

    Parameters
    ----------
    portfolio_mean       : float
    portfolio_volatility : float
    confidence_level     : float
    portfolio_value      : float

    Returns
    -------
    (var_dollar, es_dollar) : (float, float)
    """
    alpha = 1.0 - confidence_level      # tail probability, e.g. 0.01
    z     = stats.norm.ppf(alpha)       # left-tail quantile, e.g. -2.326

    # Parametric VaR
    var_return = z * portfolio_volatility + portfolio_mean
    var_dollar = -var_return * portfolio_value

    # Analytical ES for normal distribution
    # ES = μ_p + σ_p * φ(z_α) / α
    # expressed as a loss: -(ES_return) * portfolio_value
    phi_z     = stats.norm.pdf(z)                   # PDF at z-score
    es_return = portfolio_mean + portfolio_volatility * (-phi_z / alpha)
    es_dollar = -es_return * portfolio_value

    return max(var_dollar, 0.0), max(es_dollar, 0.0)


# ---------------------------------------------------------------------------
# 3. Monte Carlo VaR
# ---------------------------------------------------------------------------

def monte_carlo_var(
    asset_returns: pd.DataFrame,
    weights: Dict[str, float],
    confidence_level: float,
    portfolio_value: float,
    n_simulations: int = 10_000,
    seed: int = MONTE_CARLO_SEED,
) -> Tuple[float, float, np.ndarray]:
    """
    Calculate Monte Carlo VaR using a multivariate normal model.

    Methodology
    -----------
    1. Calibrate the multivariate normal distribution to historical data:
       - μ  = vector of historical mean daily returns per asset
       - Σ  = historical covariance matrix of daily returns
    2. Draw n_simulations samples from N(μ, Σ).
    3. Compute simulated one-day portfolio returns:
       R_sim = simulated_asset_returns @ weights
    4. The VaR is the (1 - confidence_level) quantile of the
       *loss* distribution (i.e. -R_sim * portfolio_value).

    A fixed random seed ensures reproducibility across runs.

    Parameters
    ----------
    asset_returns    : pd.DataFrame — historical daily asset returns
    weights          : dict {ticker: weight}
    confidence_level : float
    portfolio_value  : float
    n_simulations    : int          — number of scenarios (default 10,000)
    seed             : int          — random seed

    Returns
    -------
    (var_dollar, es_dollar, simulated_portfolio_returns)
        var_dollar  : float    — VaR as a positive dollar loss
        es_dollar   : float    — ES as a positive dollar loss
        sim_returns : np.ndarray — array of simulated portfolio returns
    """
    rng = np.random.default_rng(seed)

    tickers      = asset_returns.columns.tolist()
    weight_array = np.array([weights.get(t, 0.0) for t in tickers])

    # Calibrate distribution to historical data
    mu_vec  = asset_returns.mean().values                  # shape: (n_assets,)
    cov_mat = asset_returns.cov().values                   # shape: (n_assets, n_assets)

    # Draw correlated asset returns from multivariate normal
    # shape: (n_simulations, n_assets)
    simulated_asset_returns = rng.multivariate_normal(
        mean=mu_vec,
        cov=cov_mat,
        size=n_simulations,
    )

    # Simulated one-day portfolio returns
    # Portfolio Return = weighted sum of asset returns
    sim_portfolio_returns = simulated_asset_returns @ weight_array  # (n_simulations,)

    # Convert returns to dollar losses (negative return = positive loss)
    sim_losses = -sim_portfolio_returns * portfolio_value

    # VaR = confidence_level quantile of the loss distribution
    var_dollar = float(np.quantile(sim_losses, confidence_level))

    # ES = average loss beyond VaR
    tail_losses = sim_losses[sim_losses >= var_dollar]
    es_dollar   = float(tail_losses.mean()) if len(tail_losses) > 0 else var_dollar

    return max(var_dollar, 0.0), max(es_dollar, 0.0), sim_portfolio_returns


# ---------------------------------------------------------------------------
# Convenience: run all three models and return a summary dict
# ---------------------------------------------------------------------------

def compute_all_var(
    portfolio_returns: pd.Series,
    asset_returns: pd.DataFrame,
    weights: Dict[str, float],
    portfolio_mean: float,
    portfolio_volatility: float,
    confidence_level: float,
    portfolio_value: float,
    n_simulations: int = 10_000,
) -> Dict[str, float]:
    """
    Run all three VaR models and return results in a single dictionary.

    Keys returned
    -------------
    hist_var, hist_es,
    param_var, param_es,
    mc_var, mc_es

    Parameters
    ----------
    See individual function docstrings for parameter descriptions.

    Returns
    -------
    dict[str, float]
    """
    hist_var, hist_es = historical_es(
        portfolio_returns, confidence_level, portfolio_value
    )
    param_var, param_es = parametric_es(
        portfolio_mean, portfolio_volatility, confidence_level, portfolio_value
    )
    mc_var, mc_es, _ = monte_carlo_var(
        asset_returns, weights, confidence_level, portfolio_value, n_simulations
    )

    return {
        "hist_var"  : hist_var,
        "hist_es"   : hist_es,
        "param_var" : param_var,
        "param_es"  : param_es,
        "mc_var"    : mc_var,
        "mc_es"     : mc_es,
    }
