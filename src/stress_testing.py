"""
stress_testing.py
=================
Simple scenario-based portfolio stress testing.

Approach
--------
Each stress scenario applies a uniform percentage shock to all asset prices.
This is a simplified "parallel shift" stress test — every asset falls by
the same percentage simultaneously.

In practice, banks use far more granular scenarios (individual factor shocks,
historical crisis scenarios, regulatory scenarios), but this implementation
captures the core concept clearly for educational purposes.

Scenarios implemented
---------------------
- Market Shock    : -10%
- Market Crash    : -20%
- Severe Crash    : -30%
- Custom shock    : user-defined percentage

For each scenario:
    stressed_value = portfolio_value * (1 + shock)
    loss_amount    = portfolio_value - stressed_value
    loss_pct       = shock * 100  (as a percentage)
"""

import pandas as pd
from typing import Dict, List, Optional


# ---------------------------------------------------------------------------
# Pre-defined scenarios
# ---------------------------------------------------------------------------

STANDARD_SCENARIOS: Dict[str, float] = {
    "Market Shock (-10%)"       : -0.10,
    "Market Crash (-20%)"       : -0.20,
    "Severe Crash (-30%)"       : -0.30,
}


def run_stress_scenarios(
    portfolio_value: float,
    custom_shock_pct: Optional[float] = None,
) -> pd.DataFrame:
    """
    Run all standard stress scenarios and an optional custom shock.

    Parameters
    ----------
    portfolio_value  : float
        Current portfolio notional value ($).
    custom_shock_pct : float or None
        User-defined shock as a percentage (e.g. -15 for -15%).
        If None, no custom scenario is added.

    Returns
    -------
    pd.DataFrame with columns:
        Scenario           : str   — scenario name
        Shock (%)          : float — shock as a percentage (e.g. -10.0)
        Stressed Value ($) : float — portfolio value after shock
        Loss ($)           : float — dollar loss (positive = loss)
        Loss (%)           : float — percentage loss (positive = loss)
    """
    scenarios = dict(STANDARD_SCENARIOS)   # copy defaults

    if custom_shock_pct is not None:
        # Convert percentage to decimal; clamp to meaningful range
        shock_decimal = float(custom_shock_pct) / 100.0
        shock_decimal = max(min(shock_decimal, 0.0), -1.0)   # [-100%, 0%]
        label = f"Custom Shock ({custom_shock_pct:+.1f}%)"
        scenarios[label] = shock_decimal

    rows = []
    for name, shock in scenarios.items():
        stressed_value = portfolio_value * (1.0 + shock)
        loss_dollar    = portfolio_value - stressed_value      # positive = loss
        loss_pct       = -shock * 100.0                        # positive = loss %

        rows.append(
            {
                "Scenario"           : name,
                "Shock (%)"          : shock * 100.0,          # negative = adverse
                "Stressed Value ($)" : stressed_value,
                "Loss ($)"           : loss_dollar,
                "Loss (%)"           : loss_pct,
            }
        )

    return pd.DataFrame(rows)


def stress_scenario_single(
    portfolio_value: float,
    shock_pct: float,
    scenario_name: str = "Custom Scenario",
) -> Dict[str, float]:
    """
    Compute the impact of a single stress scenario.

    Parameters
    ----------
    portfolio_value : float
    shock_pct       : float  — shock as a percentage (e.g. -15 for -15%)
    scenario_name   : str

    Returns
    -------
    dict with stressed_value, loss_dollar, loss_pct
    """
    shock_decimal  = shock_pct / 100.0
    stressed_value = portfolio_value * (1.0 + shock_decimal)
    loss_dollar    = portfolio_value - stressed_value
    loss_pct       = -shock_decimal * 100.0

    return {
        "scenario_name"  : scenario_name,
        "shock_pct"      : shock_pct,
        "stressed_value" : stressed_value,
        "loss_dollar"    : loss_dollar,
        "loss_pct"       : loss_pct,
    }
