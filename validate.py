"""
validate.py
===========
Validation script for VaR-RiskLab.

Tests all core modules without starting Streamlit.
Run with: python validate.py
"""
# encoding: utf-8
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))
# Force UTF-8 output on Windows
if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import numpy as np
import pandas as pd

print("=" * 60)
print("VaR-RiskLab - Validation Script")
print("=" * 60)

# ── Test 1: Data loader ──────────────────────────────────────
print("\n[1] Testing data_loader...")
from src.data_loader import load_price_data, compute_returns, validate_weights, DEFAULT_TICKERS

prices = load_price_data(DEFAULT_TICKERS, period="3y")
print(f"    Prices shape : {prices.shape}")
print(f"    Tickers      : {list(prices.columns)}")
print(f"    Date range   : {prices.index[0].date()} to {prices.index[-1].date()}")
assert not prices.empty,             "FAIL: prices is empty"
assert prices.isna().sum().sum() == 0, "FAIL: NaNs remain in prices"
print("    PASS: prices loaded OK")

returns = compute_returns(prices)
print(f"\n    Returns shape: {returns.shape}")
assert not returns.empty, "FAIL: returns is empty"
print("    PASS: returns computed OK")

# Weight validation
ok, msg = validate_weights(DEFAULT_TICKERS, [0.30, 0.25, 0.20, 0.15, 0.10])
assert ok, f"FAIL: {msg}"
bad_ok, bad_msg = validate_weights(DEFAULT_TICKERS, [0.4, 0.4, 0.0, 0.0, 0.0])  # sums to 0.8
assert not bad_ok, "FAIL: should reject weights not summing to 1"
print("    PASS: Weight validation OK")

# ── Test 2: Portfolio ─────────────────────────────────────────
print("\n[2] Testing portfolio...")
from src.portfolio import (
    compute_portfolio_returns, compute_covariance_matrix,
    compute_portfolio_volatility, compute_portfolio_stats
)

weights = {t: w for t, w in zip(DEFAULT_TICKERS, [0.30, 0.25, 0.20, 0.15, 0.10])}
port_returns = compute_portfolio_returns(returns, weights)
assert len(port_returns) == len(returns), "FAIL: length mismatch"
print(f"    Portfolio returns: {len(port_returns)} days, mean={port_returns.mean()*100:.4f}%/day")

cov = compute_covariance_matrix(returns)
assert cov.shape == (len(DEFAULT_TICKERS), len(DEFAULT_TICKERS)), "FAIL: cov matrix wrong shape"
print(f"    Cov matrix shape : {cov.shape}")

vol = compute_portfolio_volatility(weights, cov)
assert vol > 0, "FAIL: vol must be positive"
print(f"    Daily volatility : {vol*100:.4f}%")

stats = compute_portfolio_stats(port_returns, 100_000)
print(f"    Ann. return      : {stats['annualized_return']*100:.2f}%")
print(f"    Ann. volatility  : {stats['annualized_volatility']*100:.2f}%")
print(f"    Sharpe ratio     : {stats['sharpe_ratio']:.3f}")
print(f"    Max drawdown     : {stats['max_drawdown']*100:.2f}%")
print("    PASS: portfolio stats OK")

# ── Test 3: VaR Models ────────────────────────────────────────
print("\n[3] Testing VaR models...")
from src.var_models import historical_es, parametric_es, monte_carlo_var

portfolio_value = 100_000
daily_mean      = float(port_returns.mean())
daily_vol       = vol

for cl, label in [(0.95, "95%"), (0.99, "99%")]:
    hvar, hes         = historical_es(port_returns, cl, portfolio_value)
    pvar, pes         = parametric_es(daily_mean, daily_vol, cl, portfolio_value)
    mvar, mes, sims   = monte_carlo_var(returns, weights, cl, portfolio_value, n_simulations=10_000)

    assert hvar > 0,      f"FAIL: Historical VaR ({label}) must be > 0"
    assert hes  >= hvar,  f"FAIL: ES must be >= VaR ({label})"
    assert pvar > 0,      f"FAIL: Parametric VaR ({label}) must be > 0"
    assert pes  >= pvar,  f"FAIL: Parametric ES must be >= VaR ({label})"
    assert mvar > 0,      f"FAIL: Monte Carlo VaR ({label}) must be > 0"
    assert mes  >= mvar,  f"FAIL: MC ES must be >= VaR ({label})"
    assert len(sims) == 10_000, "FAIL: Wrong number of MC simulations"

    print(f"\n    {label} VaR:")
    print(f"      Historical  VaR=${hvar:,.0f}   ES=${hes:,.0f}")
    print(f"      Parametric  VaR=${pvar:,.0f}   ES=${pes:,.0f}")
    print(f"      Monte Carlo VaR=${mvar:,.0f}   ES=${mes:,.0f}")

# Reproducibility test
_, _, sims1 = monte_carlo_var(returns, weights, 0.99, portfolio_value, n_simulations=1000, seed=42)
_, _, sims2 = monte_carlo_var(returns, weights, 0.99, portfolio_value, n_simulations=1000, seed=42)
assert np.allclose(sims1, sims2), "FAIL: Monte Carlo not reproducible!"
print("\n    PASS: Monte Carlo reproducibility OK")
print("    PASS: VaR models OK")

# ── Test 4: Backtesting ───────────────────────────────────────
print("\n[4] Testing backtesting...")
from src.backtesting import run_backtest, compute_backtest_stats

bt_df = run_backtest(port_returns, 0.99, portfolio_value, window=250)
assert not bt_df.empty, "FAIL: backtest returned empty DataFrame"
assert "exception" in bt_df.columns, "FAIL: missing exception column"
assert bt_df["predicted_var"].min() >= 0, "FAIL: VaR must be non-negative"

bt_stats = compute_backtest_stats(bt_df, 0.99)
n_obs  = bt_stats["n_observations"]
n_exc  = bt_stats["n_exceptions"]
exc_rt = bt_stats["exception_rate"]
print(f"    Observations     : {n_obs}")
print(f"    Exceptions       : {n_exc}")
print(f"    Exception rate   : {exc_rt*100:.2f}% (expected ~1%)")
print(f"    Kupiec p-value   : {bt_stats['kupiec_pvalue']:.4f}")
print("    PASS: backtesting OK")

# ── Test 5: Stress testing ────────────────────────────────────
print("\n[5] Testing stress testing...")
from src.stress_testing import run_stress_scenarios

stress_df = run_stress_scenarios(portfolio_value, custom_shock_pct=-15.0)
assert len(stress_df) == 4, f"FAIL: expected 4 scenarios, got {len(stress_df)}"
assert (stress_df["Loss ($)"] > 0).all(), "FAIL: losses should be positive"
assert (stress_df["Stressed Value ($)"] < portfolio_value).all(), "FAIL: stressed values should be < original"
print(stress_df[["Scenario", "Loss ($)", "Loss (%)"]].to_string(index=False))
print("    PASS: stress testing OK")

# ── Test 6: Missing data handling ─────────────────────────────
print("\n[6] Testing missing data handling...")
prices_with_nans = prices.copy()
prices_with_nans.iloc[5:10, 0] = np.nan
prices_with_nans.iloc[50:55, 2] = np.nan
prices_filled = prices_with_nans.ffill().bfill()
assert prices_filled.isna().sum().sum() == 0, "FAIL: NaNs not handled"
print("    PASS: Missing data handled correctly")

print("\n" + "=" * 60)
print("All validation tests PASSED!")
print("=" * 60)
print("\nTo start the dashboard, run:")
print("    streamlit run app.py")
