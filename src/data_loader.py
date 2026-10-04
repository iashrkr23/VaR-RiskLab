"""
data_loader.py
==============
Handles downloading and processing of historical market price data.

Uses yfinance to fetch adjusted closing prices. Falls back to a
synthetically-generated dataset if network access is unavailable.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
import warnings
warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DEFAULT_TICKERS: List[str] = ["AAPL", "MSFT", "GOOGL", "AMZN", "JPM"]
DEFAULT_PERIOD: str = "3y"          # 3 years of daily history
TRADING_DAYS_PER_YEAR: int = 252


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_price_data(
    tickers: List[str] = DEFAULT_TICKERS,
    period: str = DEFAULT_PERIOD,
) -> pd.DataFrame:
    """
    Download adjusted closing prices for the given tickers.

    Parameters
    ----------
    tickers : list of str
        Ticker symbols to download.
    period  : str
        yfinance period string, e.g. '3y' for three years.

    Returns
    -------
    pd.DataFrame
        DataFrame of adjusted closing prices, indexed by date,
        one column per ticker.  NaNs are forward-filled then back-filled.
    """
    try:
        import yfinance as yf
        raw = yf.download(
            tickers,
            period=period,
            auto_adjust=True,   # gives adjusted close in 'Close' column
            progress=False,
        )

        # yfinance returns multi-level columns when multiple tickers are given
        if isinstance(raw.columns, pd.MultiIndex):
            prices = raw["Close"]
        else:
            # single-ticker edge case — shouldn't happen with the default list
            prices = raw[["Close"]].rename(columns={"Close": tickers[0]})

        # Keep only requested tickers (yfinance may silently drop delisted ones)
        available = [t for t in tickers if t in prices.columns]
        if not available:
            raise ValueError("No ticker data returned by yfinance.")

        prices = prices[available].copy()

        # Handle missing values: forward-fill gaps (e.g. holidays), then
        # back-fill any leading NaNs.
        prices = prices.ffill().bfill()

        # Drop rows where *all* assets are still NaN (very rare)
        prices.dropna(how="all", inplace=True)

        if prices.empty:
            raise ValueError("Price DataFrame is empty after cleaning.")

        return prices

    except Exception as exc:
        # ------------------------------------------------------------------
        # Fallback: generate synthetic price data so the app still runs
        # when yfinance is unavailable (e.g. offline environment).
        # ------------------------------------------------------------------
        print(
            f"[data_loader] yfinance download failed ({exc}). "
            "Generating synthetic data as fallback."
        )
        return _generate_synthetic_prices(tickers)


def compute_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate simple daily percentage returns from price data.

    Formula
    -------
    return_t = (price_t - price_{t-1}) / price_{t-1}

    The first row will be NaN (no prior price) and is dropped.

    Parameters
    ----------
    prices : pd.DataFrame
        Adjusted closing prices.

    Returns
    -------
    pd.DataFrame
        Daily returns as decimals (not percentages).
    """
    returns = prices.pct_change().dropna()
    return returns


def validate_weights(
    tickers: List[str],
    weights: List[float],
) -> Tuple[bool, str]:
    """
    Validate that portfolio weights are non-negative and sum to 1.0.

    Parameters
    ----------
    tickers  : list of str
    weights  : list of float  (values between 0 and 1)

    Returns
    -------
    (is_valid, message) : (bool, str)
    """
    if len(tickers) != len(weights):
        return False, "Number of tickers and weights must match."
    if any(w < 0 for w in weights):
        return False, "All weights must be non-negative."
    total = sum(weights)
    if abs(total - 1.0) > 1e-6:
        return False, f"Weights must sum to 1.0 (currently {total:.4f})."
    return True, "OK"


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _generate_synthetic_prices(
    tickers: List[str],
    n_days: int = 756,          # ~3 years of trading days
    seed: int = 42,
) -> pd.DataFrame:
    """
    Generate synthetic adjusted prices using geometric Brownian motion.

    Uses realistic-looking parameters so that derived returns behave
    similarly to equity returns.

    Parameters
    ----------
    tickers : list of str
    n_days  : int   number of trading days to simulate
    seed    : int   random seed for reproducibility

    Returns
    -------
    pd.DataFrame  — same shape as the yfinance output
    """
    rng = np.random.default_rng(seed)

    # Approximate annual drift and volatility per asset
    annual_mu    = np.array([0.12, 0.15, 0.14, 0.18, 0.10])[:len(tickers)]
    annual_sigma = np.array([0.22, 0.25, 0.27, 0.30, 0.18])[:len(tickers)]

    # Convert to daily
    daily_mu    = annual_mu    / TRADING_DAYS_PER_YEAR
    daily_sigma = annual_sigma / np.sqrt(TRADING_DAYS_PER_YEAR)

    # Simulate correlated log-returns via Cholesky decomposition
    corr = np.array([
        [1.00, 0.75, 0.70, 0.65, 0.50],
        [0.75, 1.00, 0.72, 0.68, 0.52],
        [0.70, 0.72, 1.00, 0.65, 0.48],
        [0.65, 0.68, 0.65, 1.00, 0.45],
        [0.50, 0.52, 0.48, 0.45, 1.00],
    ])[:len(tickers), :len(tickers)]

    # Build covariance matrix for log-returns
    cov = np.outer(daily_sigma, daily_sigma) * corr
    L   = np.linalg.cholesky(cov)

    # Standard normal shocks → correlated shocks
    Z            = rng.standard_normal((n_days, len(tickers)))
    corr_shocks  = Z @ L.T

    # Log-return = drift + shock
    log_returns  = daily_mu + corr_shocks        # shape: (n_days, n_tickers)

    # Starting prices ~$100 for each asset
    start_prices = np.array([150.0, 280.0, 130.0, 120.0, 140.0])[:len(tickers)]
    price_matrix = np.zeros((n_days + 1, len(tickers)))
    price_matrix[0] = start_prices

    for t in range(1, n_days + 1):
        price_matrix[t] = price_matrix[t - 1] * np.exp(log_returns[t - 1])

    # Build date index (business days, ending today)
    dates = pd.bdate_range(end=pd.Timestamp.today().normalize(), periods=n_days + 1)
    df = pd.DataFrame(price_matrix, index=dates, columns=tickers)
    return df
