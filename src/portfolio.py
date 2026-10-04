"""
portfolio.py
============
Portfolio construction and summary statistics.

Computes portfolio-level returns, covariance matrices, and key
performance metrics such as annualised return, volatility, Sharpe ratio,
and maximum drawdown.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple

TRADING_DAYS_PER_YEAR: int = 252
RISK_FREE_RATE_ANNUAL: float = 0.04   # Approximate 1-year US T-bill rate


# ---------------------------------------------------------------------------
# Portfolio return calculation
# ---------------------------------------------------------------------------

def compute_portfolio_returns(
    asset_returns: pd.DataFrame,
    weights: Dict[str, float],
) -> pd.Series:
    """
    Calculate daily portfolio returns as a weighted sum of asset returns.

    Formula
    -------
    Portfolio Return_t = sum_i ( weight_i * return_{i,t} )

    This is a simple weighted linear combination — it assumes the portfolio
    is rebalanced daily back to the target weights.

    Parameters
    ----------
    asset_returns : pd.DataFrame
        Daily asset returns (decimal form), one column per ticker.
    weights       : dict {ticker: weight}
        Portfolio weights. Must sum to 1.0.

    Returns
    -------
    pd.Series
        Daily portfolio returns indexed by date.
    """
    # Align weights to the DataFrame's column order
    tickers = asset_returns.columns.tolist()
    weight_array = np.array([weights.get(t, 0.0) for t in tickers])

    # Weighted sum across assets for each day
    # shape: (n_days, n_assets) @ (n_assets,) → (n_days,)
    portfolio_returns = asset_returns.values @ weight_array
    return pd.Series(portfolio_returns, index=asset_returns.index, name="Portfolio")


def compute_covariance_matrix(asset_returns: pd.DataFrame) -> pd.DataFrame:
    """
    Compute the sample covariance matrix of daily asset returns.

    This is used for Parametric (Variance-Covariance) VaR.

    Parameters
    ----------
    asset_returns : pd.DataFrame

    Returns
    -------
    pd.DataFrame  — (n_assets × n_assets) covariance matrix
    """
    return asset_returns.cov()


def compute_portfolio_volatility(
    weights: Dict[str, float],
    cov_matrix: pd.DataFrame,
) -> float:
    """
    Compute daily portfolio volatility from the covariance matrix.

    Formula
    -------
    σ_p² = w^T · Σ · w
    σ_p  = sqrt(σ_p²)

    where:
        w  = weight vector   (n_assets,)
        Σ  = covariance matrix  (n_assets × n_assets)

    Parameters
    ----------
    weights    : dict {ticker: weight}
    cov_matrix : pd.DataFrame

    Returns
    -------
    float  — daily portfolio volatility (standard deviation)
    """
    tickers = cov_matrix.columns.tolist()
    w = np.array([weights.get(t, 0.0) for t in tickers])

    # Portfolio variance: w^T Σ w
    portfolio_variance = w.T @ cov_matrix.values @ w
    portfolio_volatility = np.sqrt(portfolio_variance)
    return float(portfolio_volatility)


# ---------------------------------------------------------------------------
# Summary statistics
# ---------------------------------------------------------------------------

def compute_portfolio_stats(
    portfolio_returns: pd.Series,
    portfolio_value: float,
) -> Dict[str, float]:
    """
    Compute key portfolio summary statistics.

    Metrics returned
    ----------------
    annualized_return      : Geometric annualised return.
    annualized_volatility  : Annualised standard deviation.
    sharpe_ratio           : (annualised_return - risk_free) / annualised_vol.
    max_drawdown           : Peak-to-trough percentage drawdown (negative).
    total_return           : Total compounded return over the period.
    daily_mean             : Average daily return.
    daily_std              : Daily standard deviation.

    Parameters
    ----------
    portfolio_returns : pd.Series  — daily returns (decimal)
    portfolio_value   : float      — current portfolio notional ($)

    Returns
    -------
    dict of metric_name → float
    """
    n = len(portfolio_returns)

    # Daily statistics
    daily_mean = float(portfolio_returns.mean())
    daily_std  = float(portfolio_returns.std())

    # Annualised return: compound daily mean over 252 trading days
    annualized_return = float((1 + daily_mean) ** TRADING_DAYS_PER_YEAR - 1)

    # Annualised volatility: scale daily std by sqrt(252)
    annualized_volatility = float(daily_std * np.sqrt(TRADING_DAYS_PER_YEAR))

    # Sharpe ratio: excess return over risk-free rate, per unit of risk
    daily_rf = RISK_FREE_RATE_ANNUAL / TRADING_DAYS_PER_YEAR
    sharpe_ratio = (
        (daily_mean - daily_rf) / daily_std * np.sqrt(TRADING_DAYS_PER_YEAR)
        if daily_std > 0 else 0.0
    )

    # Maximum drawdown: largest peak-to-trough decline in cumulative returns
    cumulative = (1 + portfolio_returns).cumprod()
    rolling_max = cumulative.cummax()
    drawdown = (cumulative - rolling_max) / rolling_max
    max_drawdown = float(drawdown.min())

    # Total compounded return over the full data period
    total_return = float(cumulative.iloc[-1] - 1)

    return {
        "annualized_return"     : annualized_return,
        "annualized_volatility" : annualized_volatility,
        "sharpe_ratio"          : sharpe_ratio,
        "max_drawdown"          : max_drawdown,
        "total_return"          : total_return,
        "daily_mean"            : daily_mean,
        "daily_std"             : daily_std,
        "n_days"                : n,
    }
