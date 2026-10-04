"""
app.py
======
VaR-RiskLab — Portfolio Risk Analytics & VaR Backtesting Dashboard
===================================================================

Run with:
    streamlit run app.py

This Streamlit dashboard provides an interactive interface for:
    - Portfolio construction with configurable weights
    - Three VaR methodologies (Historical, Parametric, Monte Carlo)
    - Expected Shortfall calculation
    - Rolling VaR backtesting with exception tracking
    - Scenario-based stress testing
    - Methodology explanations
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st
import sys
import os
import math

# ---------------------------------------------------------------------------
# Path setup — allow imports from the src/ package
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from src.data_loader   import (
    load_price_data, compute_returns, validate_weights, DEFAULT_TICKERS
)
from src.portfolio     import (
    compute_portfolio_returns, compute_covariance_matrix,
    compute_portfolio_volatility, compute_portfolio_stats
)
from src.var_models    import (
    historical_es, parametric_es, monte_carlo_var
)
from src.backtesting   import run_backtest, compute_backtest_stats
from src.stress_testing import run_stress_scenarios


# ---------------------------------------------------------------------------
# Page configuration
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title = "VaR-RiskLab",
    page_icon  = "📊",
    layout     = "wide",
    initial_sidebar_state = "expanded",
)

# ---------------------------------------------------------------------------
# Custom CSS — dark, polished theme
# ---------------------------------------------------------------------------
st.markdown("""
<style>
    /* Base */
    .stApp {
        background-color: #0d1117;
        color: #e6edf3;
        font-family: 'Inter', sans-serif;
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #161b22 0%, #0d1117 100%);
        border-right: 1px solid #21262d;
    }

    /* Metric cards */
    [data-testid="metric-container"] {
        background: linear-gradient(135deg, #161b22 0%, #1c2128 100%);
        border: 1px solid #30363d;
        border-radius: 12px;
        padding: 1rem;
        box-shadow: 0 4px 12px rgba(0,0,0,0.3);
    }

    [data-testid="metric-container"] label {
        color: #8b949e !important;
        font-size: 0.78rem !important;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }

    [data-testid="metric-container"] [data-testid="stMetricValue"] {
        color: #58a6ff !important;
        font-size: 1.5rem !important;
        font-weight: 700;
    }

    /* Section headers */
    .section-header {
        color: #58a6ff;
        font-size: 1.35rem;
        font-weight: 700;
        border-left: 4px solid #58a6ff;
        padding-left: 0.75rem;
        margin: 2rem 0 1rem 0;
    }

    /* Interpretation boxes */
    .info-box {
        background: linear-gradient(135deg, #1f2937 0%, #111827 100%);
        border: 1px solid #374151;
        border-radius: 10px;
        padding: 1rem 1.25rem;
        margin: 0.75rem 0;
        font-size: 0.88rem;
        color: #d1d5db;
        line-height: 1.6;
    }

    .warning-box {
        background: linear-gradient(135deg, #1c1a0f 0%, #14120a 100%);
        border: 1px solid #854d0e;
        border-radius: 10px;
        padding: 1rem 1.25rem;
        margin: 0.75rem 0;
        font-size: 0.88rem;
        color: #fde68a;
    }

    /* Hero title */
    .hero-title {
        font-size: 2.5rem;
        font-weight: 800;
        background: linear-gradient(135deg, #58a6ff 0%, #79c0ff 40%, #a5f3fc 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin: 0;
    }

    .hero-sub {
        color: #8b949e;
        font-size: 1rem;
        margin-top: 0.25rem;
    }

    /* DataFrames */
    .dataframe { font-size: 0.85rem; }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 4px;
        background-color: #161b22;
        border-radius: 10px;
        padding: 4px;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: transparent;
        border-radius: 8px;
        color: #8b949e;
        font-weight: 500;
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #1f6feb 0%, #388bfd 100%) !important;
        color: white !important;
    }

    /* Hide Streamlit footer */
    footer {visibility: hidden;}
    #MainMenu {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Colour palette for Plotly charts (dark theme)
# ---------------------------------------------------------------------------
PLOTLY_DARK = dict(
    paper_bgcolor = "#0d1117",
    plot_bgcolor  = "#161b22",
    font          = dict(color="#e6edf3", family="Inter, sans-serif"),
    xaxis         = dict(gridcolor="#21262d", linecolor="#30363d"),
    yaxis         = dict(gridcolor="#21262d", linecolor="#30363d"),
)

ACCENT_BLUE   = "#58a6ff"
ACCENT_GREEN  = "#3fb950"
ACCENT_ORANGE = "#f0883e"
ACCENT_RED    = "#f85149"
ACCENT_PURPLE = "#bc8cff"


# ===========================================================================
# SIDEBAR — controls
# ===========================================================================
with st.sidebar:
    st.markdown("## ⚙️ Configuration")
    st.divider()

    # Portfolio value
    portfolio_value = st.number_input(
        "Portfolio Value ($)",
        min_value = 10_000,
        max_value = 100_000_000,
        value     = 100_000,
        step      = 10_000,
        help      = "Total notional value of the portfolio in USD.",
    )

    # Confidence level
    confidence_pct = st.selectbox(
        "Confidence Level",
        options  = [95, 99],
        index    = 1,
        help     = "VaR confidence level. 99% means we are modelling the 1% worst-case scenario.",
    )
    confidence_level = confidence_pct / 100.0

    # Backtesting window
    backtest_window = st.slider(
        "Historical Window (days)",
        min_value = 126,
        max_value = 504,
        value     = 250,
        step      = 1,
        help      = "Number of prior trading days used in the rolling VaR backtest.",
    )

    # Monte Carlo simulations
    n_simulations = st.select_slider(
        "Monte Carlo Simulations",
        options = [1_000, 5_000, 10_000, 20_000, 50_000],
        value   = 10_000,
        help    = "Number of simulated scenarios for Monte Carlo VaR.",
    )

    st.divider()
    st.markdown("## 📊 Asset Weights (%)")

    # Weight sliders
    default_weights_pct = {"AAPL": 30, "MSFT": 25, "GOOGL": 20, "AMZN": 15, "JPM": 10}
    raw_weights = {}
    for ticker in DEFAULT_TICKERS:
        raw_weights[ticker] = st.slider(
            ticker,
            min_value = 0,
            max_value = 100,
            value     = default_weights_pct[ticker],
            step      = 1,
        )

    # Normalise to sum to 1.0; warn if all zeros
    total_weight_pct = sum(raw_weights.values())
    if total_weight_pct == 0:
        st.error("⚠️ All weights are zero — using default weights.")
        weights_pct = default_weights_pct
        total_weight_pct = 100

    weights_decimal = {t: w / total_weight_pct for t, w in raw_weights.items()}
    normalised_sum  = sum(weights_decimal.values())

    if abs(total_weight_pct - 100) > 0.01:
        st.warning(
            f"Weights sum to {total_weight_pct}% — normalised automatically to 100%."
        )
    else:
        st.success("✅ Weights sum to 100%")

    st.divider()
    st.caption("Data sourced from Yahoo Finance via yfinance.")


# ===========================================================================
# HERO HEADER
# ===========================================================================
st.markdown("""
<div style="padding: 1rem 0 0.5rem 0;">
    <div class="hero-title">📊 VaR-RiskLab</div>
    <div class="hero-sub">Portfolio Risk Analytics & VaR Backtesting Engine</div>
</div>
""", unsafe_allow_html=True)
st.divider()


# ===========================================================================
# DATA LOADING (with spinner and caching)
# ===========================================================================
@st.cache_data(ttl=3600, show_spinner=False)
def cached_load_prices(tickers, period="3y"):
    """Cache price downloads for 1 hour to avoid repeated API calls."""
    return load_price_data(tickers, period)


with st.spinner("🔄 Loading market data..."):
    try:
        prices = cached_load_prices(DEFAULT_TICKERS)
        returns_all = compute_returns(prices)

        # Keep only assets that have data
        available_tickers = [t for t in DEFAULT_TICKERS if t in returns_all.columns]
        returns_all  = returns_all[available_tickers]
        weights_used = {t: weights_decimal[t] for t in available_tickers}

        # Re-normalise in case some tickers were dropped
        w_sum = sum(weights_used.values())
        if w_sum > 0:
            weights_used = {t: w / w_sum for t, w in weights_used.items()}

        # Guard: if returns are still empty, trigger fallback
        if returns_all.empty or len(returns_all) < 50:
            raise ValueError(
                f"Insufficient return data: {len(returns_all)} rows. "
                "Falling back to synthetic data."
            )

        data_ok = True

    except Exception as e:
        st.warning(f"⚠️ Live data issue ({e}) — using synthetic data as fallback.")
        # Use synthetic prices as fallback
        from src.data_loader import _generate_synthetic_prices
        prices      = _generate_synthetic_prices(DEFAULT_TICKERS)
        returns_all = compute_returns(prices)
        available_tickers = DEFAULT_TICKERS
        weights_used = {t: weights_decimal[t] for t in available_tickers}
        w_sum = sum(weights_used.values())
        if w_sum > 0:
            weights_used = {t: w / w_sum for t, w in weights_used.items()}
        data_ok = True

if not data_ok:
    st.stop()

# Show data info
n_days_data = len(returns_all)
date_start  = returns_all.index[0].strftime('%d %b %Y')
date_end    = returns_all.index[-1].strftime('%d %b %Y')
st.caption(f"Data: {n_days_data} trading days | {date_start} to {date_end} | Tickers: {', '.join(available_tickers)}")


# ===========================================================================
# CORE CALCULATIONS
# ===========================================================================
with st.spinner("⚙️ Computing risk metrics..."):

    # Portfolio returns
    portfolio_returns = compute_portfolio_returns(returns_all, weights_used)

    # Covariance matrix and portfolio volatility
    cov_matrix           = compute_covariance_matrix(returns_all)
    daily_vol            = compute_portfolio_volatility(weights_used, cov_matrix)
    daily_mean           = float(portfolio_returns.mean())

    # Summary statistics
    stats_dict = compute_portfolio_stats(portfolio_returns, portfolio_value)

    # VaR + ES for all three models
    hist_var, hist_es_val     = historical_es(portfolio_returns, confidence_level, portfolio_value)
    param_var, param_es_val   = parametric_es(daily_mean, daily_vol, confidence_level, portfolio_value)
    mc_var, mc_es_val, mc_sim = monte_carlo_var(
        returns_all, weights_used, confidence_level, portfolio_value, n_simulations
    )

    # Backtesting
    backtest_ok = n_days_data > backtest_window
    if backtest_ok:
        backtest_df   = run_backtest(portfolio_returns, confidence_level, portfolio_value, backtest_window)
        backtest_stats = compute_backtest_stats(backtest_df, confidence_level)

    # Stress testing
    stress_df = run_stress_scenarios(portfolio_value)


# ===========================================================================
# TABS
# ===========================================================================
tab_overview, tab_var, tab_es, tab_backtest, tab_stress, tab_method = st.tabs([
    "📈 Overview",
    "🎯 VaR Comparison",
    "📉 Expected Shortfall",
    "🔍 Backtesting",
    "⚡ Stress Testing",
    "📚 Methodology",
])


# ===========================================================================
# TAB 1: PORTFOLIO OVERVIEW
# ===========================================================================
with tab_overview:
    st.markdown('<div class="section-header">Portfolio Overview</div>', unsafe_allow_html=True)

    # Metric cards
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Portfolio Value",      f"${portfolio_value:,.0f}")
    col2.metric("Annualised Return",    f"{stats_dict['annualized_return']*100:.2f}%")
    col3.metric("Annualised Volatility",f"{stats_dict['annualized_volatility']*100:.2f}%")
    col4.metric("Sharpe Ratio",         f"{stats_dict['sharpe_ratio']:.3f}")
    col5.metric("Max Drawdown",         f"{stats_dict['max_drawdown']*100:.2f}%")

    st.divider()

    # Cumulative returns chart
    cum_ret = (1 + portfolio_returns).cumprod() - 1
    fig_cum = go.Figure()
    fig_cum.add_trace(go.Scatter(
        x    = cum_ret.index,
        y    = cum_ret.values * 100,
        mode = "lines",
        name = "Portfolio",
        line = dict(color=ACCENT_BLUE, width=2),
        fill = "tozeroy",
        fillcolor = "rgba(88,166,255,0.08)",
    ))
    fig_cum.update_layout(
        title  = "Cumulative Portfolio Return (%)",
        height = 350,
        xaxis_title = "Date",
        yaxis_title = "Cumulative Return (%)",
        **PLOTLY_DARK,
    )
    st.plotly_chart(fig_cum, use_container_width=True)

    col_left, col_right = st.columns(2)

    with col_left:
        # Return distribution histogram
        fig_hist = go.Figure()
        fig_hist.add_trace(go.Histogram(
            x         = portfolio_returns.values * 100,
            nbinsx    = 60,
            name      = "Daily Returns",
            marker_color = ACCENT_BLUE,
            opacity   = 0.75,
        ))
        # VaR lines
        hist_var_pct = -np.quantile(portfolio_returns, 1 - confidence_level) * 100
        fig_hist.add_vline(
            x          = -hist_var_pct,
            line_dash  = "dash",
            line_color = ACCENT_RED,
            annotation_text = f"{confidence_pct}% VaR",
        )
        fig_hist.update_layout(
            title  = "Daily Return Distribution",
            height = 320,
            xaxis_title = "Daily Return (%)",
            yaxis_title = "Frequency",
            **PLOTLY_DARK,
        )
        st.plotly_chart(fig_hist, use_container_width=True)

    with col_right:
        # Asset weight pie
        fig_pie = go.Figure(go.Pie(
            labels  = list(weights_used.keys()),
            values  = list(weights_used.values()),
            hole    = 0.45,
            marker  = dict(colors=[ACCENT_BLUE, ACCENT_GREEN, ACCENT_ORANGE, ACCENT_PURPLE, ACCENT_RED]),
        ))
        fig_pie.update_layout(
            title  = "Portfolio Allocation",
            height = 320,
            **PLOTLY_DARK,
        )
        st.plotly_chart(fig_pie, use_container_width=True)


# ===========================================================================
# TAB 2: VaR COMPARISON
# ===========================================================================
with tab_var:
    st.markdown('<div class="section-header">Value at Risk — Three Methodologies</div>', unsafe_allow_html=True)

    st.markdown(f"""
    <div class="info-box">
    📌 <strong>Interpretation:</strong> At the {confidence_pct}% confidence level, each VaR estimate represents
    the modelled one-day loss threshold under that method's assumptions. The portfolio would be
    expected to lose more than this amount on approximately {100 - confidence_pct}% of trading days
    (under the model). VaR does <em>not</em> guarantee or bound the worst possible loss.
    </div>
    """, unsafe_allow_html=True)

    # VaR summary table
    var_data = {
        "Model"           : ["Historical Simulation", "Parametric (Var-Covar)", "Monte Carlo"],
        f"VaR ({confidence_pct}%)"  : [f"${hist_var:,.0f}", f"${param_var:,.0f}", f"${mc_var:,.0f}"],
        "% of Portfolio"  : [
            f"{hist_var/portfolio_value*100:.3f}%",
            f"{param_var/portfolio_value*100:.3f}%",
            f"{mc_var/portfolio_value*100:.3f}%",
        ],
    }
    var_df = pd.DataFrame(var_data)
    st.dataframe(var_df, use_container_width=True, hide_index=True)

    # Bar chart comparison
    fig_var = go.Figure()
    model_names = ["Historical", "Parametric", "Monte Carlo"]
    var_values  = [hist_var, param_var, mc_var]
    colors      = [ACCENT_BLUE, ACCENT_GREEN, ACCENT_PURPLE]

    for name, val, col in zip(model_names, var_values, colors):
        fig_var.add_trace(go.Bar(
            x    = [name],
            y    = [val],
            name = name,
            marker_color = col,
            text = [f"${val:,.0f}"],
            textposition = "outside",
        ))

    fig_var.update_layout(
        title       = f"VaR Comparison ({confidence_pct}% confidence, 1-day horizon)",
        height      = 380,
        yaxis_title = "VaR ($)",
        showlegend  = True,
        barmode     = "group",
        **PLOTLY_DARK,
    )
    st.plotly_chart(fig_var, use_container_width=True)

    # Monte Carlo return distribution
    st.markdown('<div class="section-header">Monte Carlo Simulated Return Distribution</div>', unsafe_allow_html=True)
    fig_mc = go.Figure()
    fig_mc.add_trace(go.Histogram(
        x         = mc_sim * 100,
        nbinsx    = 80,
        name      = "Simulated Returns",
        marker_color = ACCENT_PURPLE,
        opacity   = 0.70,
    ))
    mc_var_pct = -np.quantile(mc_sim, 1 - confidence_level) * 100
    fig_mc.add_vline(
        x          = -mc_var_pct,
        line_dash  = "dash",
        line_color = ACCENT_RED,
        annotation_text = f"MC VaR {confidence_pct}%",
    )
    fig_mc.update_layout(
        title  = f"Monte Carlo Distribution ({n_simulations:,} simulations)",
        height = 330,
        xaxis_title = "Simulated Daily Return (%)",
        yaxis_title = "Frequency",
        **PLOTLY_DARK,
    )
    st.plotly_chart(fig_mc, use_container_width=True)

    st.markdown("""
    <div class="info-box">
    🔵 <strong>Historical VaR</strong>: Directly uses the past return distribution — no normality assumption.<br>
    🟢 <strong>Parametric VaR</strong>: Assumes returns are normally distributed; faster but may understate tail risk.<br>
    🟣 <strong>Monte Carlo VaR</strong>: Simulates thousands of scenarios using calibrated distributions; most flexible.
    </div>
    """, unsafe_allow_html=True)


# ===========================================================================
# TAB 3: EXPECTED SHORTFALL
# ===========================================================================
with tab_es:
    st.markdown('<div class="section-header">Expected Shortfall (Conditional VaR)</div>', unsafe_allow_html=True)

    st.markdown(f"""
    <div class="info-box">
    📌 <strong>Expected Shortfall (ES)</strong> — also called Conditional VaR (CVaR) — answers:
    <em>"Given that we exceed VaR, what is the <strong>average</strong> loss?"</em><br><br>
    Unlike VaR, ES captures the severity of losses in the tail, not just the threshold.
    For this reason, the Basel III/IV framework (FRTB) requires banks to use ES instead of VaR
    for internal model regulatory capital calculations.
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Historical Simulation")
        st.metric(f"Historical VaR ({confidence_pct}%)",         f"${hist_var:,.0f}")
        st.metric(f"Historical ES ({confidence_pct}%)",          f"${hist_es_val:,.0f}")
        es_increment_hist = hist_es_val - hist_var
        st.metric("ES Premium over VaR",  f"${es_increment_hist:,.0f}",
                  help="How much worse the average tail loss is compared to VaR.")

    with col2:
        st.subheader("Monte Carlo")
        st.metric(f"Monte Carlo VaR ({confidence_pct}%)",        f"${mc_var:,.0f}")
        st.metric(f"Monte Carlo ES ({confidence_pct}%)",         f"${mc_es_val:,.0f}")
        es_increment_mc = mc_es_val - mc_var
        st.metric("ES Premium over VaR",  f"${es_increment_mc:,.0f}")

    st.divider()

    # Historical tail loss visualisation
    tail_quantile = 1.0 - confidence_level
    var_threshold = float(np.quantile(portfolio_returns, tail_quantile))
    tail_returns  = portfolio_returns[portfolio_returns <= var_threshold]

    fig_es = go.Figure()
    fig_es.add_trace(go.Histogram(
        x    = portfolio_returns.values * 100,
        nbinsx = 60,
        name = "All Returns",
        marker_color = ACCENT_BLUE,
        opacity = 0.5,
    ))
    fig_es.add_trace(go.Histogram(
        x    = tail_returns.values * 100,
        nbinsx = 20,
        name = f"Tail (beyond {confidence_pct}% VaR)",
        marker_color = ACCENT_RED,
        opacity = 0.8,
    ))
    fig_es.add_vline(x=var_threshold * 100, line_dash="dash", line_color=ACCENT_ORANGE,
                     annotation_text=f"VaR ({confidence_pct}%)")
    fig_es.add_vline(x=float(tail_returns.mean()) * 100, line_dash="dot",
                     line_color=ACCENT_RED, annotation_text="ES (avg tail loss)")
    fig_es.update_layout(
        title      = "Return Distribution: VaR Threshold vs Expected Shortfall",
        height     = 380,
        xaxis_title = "Daily Return (%)",
        yaxis_title = "Frequency",
        barmode    = "overlay",
        **PLOTLY_DARK,
    )
    st.plotly_chart(fig_es, use_container_width=True)

    st.markdown("""
    <div class="warning-box">
    ⚠️ <strong>Why ES matters:</strong> VaR tells you the <em>threshold</em> but is silent about
    <em>how bad</em> the losses beyond it can be. Two portfolios can have identical VaR but very
    different ES — the one with higher ES carries more tail risk. Regulators therefore prefer ES
    as the primary risk measure under Basel IV / FRTB.
    </div>
    """, unsafe_allow_html=True)


# ===========================================================================
# TAB 4: BACKTESTING
# ===========================================================================
with tab_backtest:
    st.markdown('<div class="section-header">Rolling VaR Backtesting</div>', unsafe_allow_html=True)

    if not backtest_ok:
        st.warning(
            f"Insufficient data for backtesting. "
            f"Need more than {backtest_window} observations; have {n_days_data}. "
            "Try reducing the historical window in the sidebar."
        )
    else:
        # Summary metrics
        bs = backtest_stats
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Observations",          f"{bs['n_observations']:,}")
        col2.metric("VaR Exceptions",        f"{bs['n_exceptions']}")
        col3.metric("Observed Rate",         f"{bs['exception_rate']*100:.2f}%")
        col4.metric("Expected Rate",         f"{bs['expected_rate']*100:.1f}%")

        st.divider()

        # Kupiec test result
        kupiec_p = bs["kupiec_pvalue"]
        if not math.isnan(kupiec_p):
            if kupiec_p > 0.05:
                st.success(
                    f"✅ **Kupiec Test**: p-value = {kupiec_p:.4f} — We **cannot reject** H₀ at 5% significance. "
                    "The observed exception rate is statistically consistent with the model."
                )
            else:
                st.warning(
                    f"⚠️ **Kupiec Test**: p-value = {kupiec_p:.4f} — We **reject** H₀ at 5% significance. "
                    "The observed exception rate is statistically inconsistent with the model."
                )
        else:
            st.info("ℹ️ Kupiec test could not be computed (0 or N exceptions).")

        # Backtest chart
        exceptions_df = backtest_df[backtest_df["exception"]]

        fig_bt = go.Figure()

        # Predicted VaR (as positive loss)
        fig_bt.add_trace(go.Scatter(
            x    = backtest_df.index,
            y    = backtest_df["predicted_var"],
            mode = "lines",
            name = f"Predicted VaR ({confidence_pct}%)",
            line = dict(color=ACCENT_BLUE, width=1.5, dash="dot"),
        ))

        # Actual losses (negative losses shown as positive, gains as negative)
        fig_bt.add_trace(go.Scatter(
            x    = backtest_df.index,
            y    = backtest_df["actual_loss"],
            mode = "lines",
            name = "Actual Daily P&L (loss = positive)",
            line = dict(color=ACCENT_GREEN, width=1),
            opacity = 0.7,
        ))

        # Exception markers
        if not exceptions_df.empty:
            fig_bt.add_trace(go.Scatter(
                x    = exceptions_df.index,
                y    = exceptions_df["actual_loss"],
                mode = "markers",
                name = "VaR Exception",
                marker = dict(color=ACCENT_RED, size=7, symbol="x"),
            ))

        fig_bt.update_layout(
            title  = f"Rolling {confidence_pct}% VaR vs Actual Losses (window = {backtest_window} days)",
            height = 420,
            xaxis_title = "Date",
            yaxis_title = "Loss ($)",
            legend = dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
            **PLOTLY_DARK,
        )
        st.plotly_chart(fig_bt, use_container_width=True)

        # Exception detail table
        if not exceptions_df.empty:
            st.markdown("#### VaR Exception Days")
            exc_display = exceptions_df[["actual_loss", "predicted_var"]].copy()
            exc_display.index = exc_display.index.strftime("%d %b %Y")
            exc_display.columns = ["Actual Loss ($)", "Predicted VaR ($)"]
            exc_display = exc_display.round(2)
            st.dataframe(exc_display, use_container_width=True)

        st.markdown(f"""
        <div class="info-box">
        📌 <strong>How to read this chart:</strong><br>
        — The dashed blue line is the <strong>rolling {confidence_pct}% VaR prediction</strong> for each day,
        computed using the prior {backtest_window} days.<br>
        — The green line is the <strong>actual daily P&L</strong> (positive = loss, negative = gain).<br>
        — Red ✗ markers indicate <strong>VaR exceptions</strong>: days when the actual loss exceeded the VaR prediction.<br><br>
        A well-calibrated model at {confidence_pct}% confidence should produce approximately
        {(1-confidence_level)*100:.0f}% exceptions (≈ {bs['expected_n_exceptions']:.1f} in this sample).
        Significantly more exceptions suggest the model under-estimates risk.
        </div>
        """, unsafe_allow_html=True)


# ===========================================================================
# TAB 5: STRESS TESTING
# ===========================================================================
with tab_stress:
    st.markdown('<div class="section-header">Scenario-Based Stress Testing</div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="warning-box">
    ⚠️ <strong>Educational stress test:</strong> Each scenario applies a uniform percentage shock
    to the entire portfolio simultaneously. Real bank stress tests use granular factor sensitivities
    and historical crisis scenarios (e.g. 2008 GFC, COVID March 2020). This simplified version
    illustrates the core concept.
    </div>
    """, unsafe_allow_html=True)

    # Custom shock input
    custom_shock = st.slider(
        "Custom Shock (%)",
        min_value = -100,
        max_value = 0,
        value     = -15,
        step      = 1,
        help      = "Enter a custom negative shock to apply to the portfolio.",
    )

    # Run scenarios including custom shock
    stress_df_full = run_stress_scenarios(portfolio_value, custom_shock_pct=float(custom_shock))

    # Display table
    display_cols = ["Scenario", "Shock (%)", "Stressed Value ($)", "Loss ($)", "Loss (%)"]
    formatted = stress_df_full.copy()
    formatted["Stressed Value ($)"] = formatted["Stressed Value ($)"].apply(lambda x: f"${x:,.0f}")
    formatted["Loss ($)"]           = formatted["Loss ($)"].apply(lambda x: f"${x:,.0f}")
    formatted["Loss (%)"]           = formatted["Loss (%)"].apply(lambda x: f"{x:.1f}%")
    formatted["Shock (%)"]          = formatted["Shock (%)"].apply(lambda x: f"{x:.1f}%")
    st.dataframe(formatted[display_cols], use_container_width=True, hide_index=True)

    st.divider()

    # Bar chart — stressed losses
    fig_stress = go.Figure()
    fig_stress.add_trace(go.Bar(
        x    = stress_df_full["Scenario"],
        y    = stress_df_full["Loss ($)"],
        marker_color = [ACCENT_ORANGE, ACCENT_RED, "#7f1d1d", ACCENT_PURPLE],
        text = stress_df_full["Loss ($)"].apply(lambda x: f"${x:,.0f}"),
        textposition = "outside",
    ))
    fig_stress.add_hline(
        y          = hist_var,
        line_dash  = "dash",
        line_color = ACCENT_BLUE,
        annotation_text = f"{confidence_pct}% Historical VaR = ${hist_var:,.0f}",
    )
    fig_stress.update_layout(
        title       = "Portfolio Loss by Stress Scenario (vs Historical VaR)",
        height      = 400,
        xaxis_title = "Scenario",
        yaxis_title = "Portfolio Loss ($)",
        showlegend  = False,
        **PLOTLY_DARK,
    )
    st.plotly_chart(fig_stress, use_container_width=True)

    st.markdown(f"""
    <div class="info-box">
    📌 The dashed blue line shows the {confidence_pct}% Historical VaR (${hist_var:,.0f}).
    Stress scenarios typically produce losses <strong>far larger than VaR</strong>, illustrating
    why VaR alone is an insufficient risk measure for extreme events.
    VaR models are calibrated to normal market conditions; stress tests complement them
    by exploring tail scenarios that VaR cannot fully capture.
    </div>
    """, unsafe_allow_html=True)


# ===========================================================================
# TAB 6: METHODOLOGY
# ===========================================================================
with tab_method:
    st.markdown('<div class="section-header">Methodology Reference</div>', unsafe_allow_html=True)

    with st.expander("📘 What is Value at Risk (VaR)?", expanded=True):
        st.markdown("""
        **Value at Risk (VaR)** is a **quantile-based estimate** of portfolio loss over a specified
        time horizon and confidence level, under normal market conditions.

        **Intuitive Interpretation:**
        A 99%, 1-day VaR of \\$2,000 means that on 99% of trading days, portfolio losses are expected
        to be \\$2,000 or less (equivalently, there is a 1% probability of losing more than \\$2,000 in a single day).

        **Crucial Distinctions & Limitations:**
        - **Not the maximum loss:** VaR is a threshold quantile, not the absolute maximum loss.
        - **Tail severity ignored:** VaR says nothing about the magnitude of losses *beyond* the threshold (which is why Expected Shortfall is used).
        - **Model dependence:** VaR calculations rely heavily on empirical or parametric assumptions.
        """)

    with st.expander("📗 Historical Simulation VaR"):
        st.markdown(r"""
        **Method:** Use the empirical distribution of past portfolio returns.

        **Steps:**
        1. Collect the last $n$ daily portfolio returns.
        2. Sort them from worst to best.
        3. The VaR is the return at the $(1 - \alpha)$ quantile, expressed as a positive loss.

        $$\text{VaR}_{\alpha} = -Q_{1-\alpha}(\text{returns}) \times \text{Portfolio Value}$$

        **Advantages:**
        - No distributional assumptions.
        - Naturally captures fat tails from historical data.
        - Easy to explain and implement.

        **Limitations:**
        - Depends entirely on historical data — past crises may not repeat.
        - Equally weights all past observations.
        - Cannot model events not seen in the historical window.
        """)

    with st.expander("📙 Parametric (Variance-Covariance) VaR"):
        st.markdown(r"""
        **Method:** Assume portfolio returns are normally distributed and compute closed-form analytics.

        **Portfolio Volatility from Covariance Matrix:**
        $$\sigma_p^2 = \mathbf{w}^\top \Sigma \mathbf{w}$$
        $$\sigma_p = \sqrt{\sigma_p^2}$$

        **Parametric VaR:**
        $$\text{VaR}_{\alpha} = (z_\alpha \cdot \sigma_p - \mu_p) \times \text{Portfolio Value}$$

        where $z_\alpha = \Phi^{-1}(1-\alpha)$ is the left-tail normal quantile.

        For 99% VaR: $z_{0.01} \approx -2.326$

        **Advantages:**
        - Computationally fast; closed-form solution.
        - Easy to decompose risk by asset or factor.

        **Limitations:**
        - Normality assumption underestimates fat-tailed returns.
        - Does not capture non-linear option payoffs.
        - Static covariance may not reflect stressed market correlations.
        """)

    with st.expander("📕 Monte Carlo VaR"):
        st.markdown(r"""
        **Method:** Simulate thousands of explicit scenarios from a calibrated distribution.

        **Steps:**
        1. Calibrate $\boldsymbol{\mu}$ (mean return vector) and $\Sigma$ (covariance matrix) from history.
        2. Draw $N$ samples from the multivariate normal: $\mathbf{r} \sim \mathcal{N}(\boldsymbol{\mu}, \Sigma)$.
        3. Compute simulated portfolio returns: $r_p = \mathbf{w}^\top \mathbf{r}$.
        4. VaR = the $(1-\alpha)$ quantile of simulated losses.

        **Key Interview Distinction — Parametric vs. Monte Carlo:**
        - **Parametric VaR** is a closed-form analytical calculation ($z_\alpha \sigma_p - \mu_p$).
        - **Monte Carlo VaR** explicitly generates simulated scenario paths.
        - *Framework Flexibility:* The current implementation uses a multivariate normal model; the Monte Carlo framework can be extended to other distributions (e.g. Student's t, copulas) and nonlinear payoffs (e.g. options full revaluation).

        **Advantages:**
        - Highly flexible — extensible to non-linear payoffs and non-normal distributions.
        - Produces a full simulated loss distribution.

        **Limitations:**
        - Computationally intensive for large portfolios.
        - Results depend on the assumed distribution and calibration period.
        """)

    with st.expander("📓 Expected Shortfall (CVaR)"):
        st.markdown(r"""
        **Expected Shortfall (ES)** — also called Conditional VaR or CVaR — is the average loss
        *conditional on the loss exceeding VaR*.

        $$\text{ES}_\alpha = \mathbb{E}[\text{Loss} \mid \text{Loss} > \text{VaR}_\alpha]$$

        **For a normal distribution, the analytical formula is:**
        $$\text{ES}_\alpha = \mu_p + \sigma_p \cdot \frac{\phi(z_\alpha)}{1-\alpha}$$

        where $\phi(\cdot)$ is the standard normal PDF.

        **Why ES > VaR:** ES averages over the entire tail beyond VaR, making it more
        sensitive to extreme losses. Basel IV / FRTB mandates ES for regulatory capital.
        """)

    with st.expander("📔 VaR Backtesting & Exceptions"):
        st.markdown(r"""
        **Purpose:** Test whether the VaR model is well-calibrated against actual P&L.

        **Rolling backtest process:**
        For each day $t$ after the initial window:
        1. Compute VaR using the prior $W$ days of returns.
        2. Compare to the actual portfolio return on day $t$.
        3. An **exception** occurs if: $\text{Actual Loss}_t > \text{VaR}_t$

        **Expected exception rates:**
        - 99% VaR → ~1% exceptions (≈ 2–3 per year)
        - 95% VaR → ~5% exceptions (≈ 12–13 per year)

        **Kupiec Unconditional Coverage Test (H₀: observed rate = expected rate):**
        $$LR_{uc} = -2 \ln\left[\frac{p_0^x (1-p_0)^{n-x}}{\hat{p}^x (1-\hat{p})^{n-x}}\right] \sim \chi^2(1)$$

        Reject H₀ if $p\text{-value} < 0.05$.
        """)

    with st.expander("⚡ Stress Testing"):
        st.markdown("""
        **Purpose:** Assess portfolio losses under severe, hypothetical market scenarios that
        may not be captured by VaR (which is calibrated to normal market conditions).

        **Approach used here (simplified educational model):**
        Apply a uniform percentage shock to all assets simultaneously.

        $$\text{Stressed Value} = \text{Portfolio Value} \times (1 + \text{shock})$$
        $$\text{Loss} = \text{Portfolio Value} - \text{Stressed Value}$$

        **In practice, banks use:**
        - Historical crisis scenarios (2008 GFC, COVID-19, 9/11)
        - Hypothetical regulatory scenarios (EBA, Fed stress tests)
        - Factor sensitivity shocks (equity, rates, FX, credit spreads)

        This simplified version illustrates the core concept of how stress losses
        dwarf VaR-level losses in tail scenarios.
        """)


# ===========================================================================
# FOOTER
# ===========================================================================
st.divider()
st.markdown("""
<div style="text-align:center; color:#4a5568; font-size:0.8rem; padding: 1rem 0;">
    VaR-RiskLab &nbsp;|&nbsp; Educational Portfolio Risk Analytics &nbsp;|&nbsp;
    Not intended for production risk management &nbsp;|&nbsp;
    Data: Yahoo Finance (yfinance)
</div>
""", unsafe_allow_html=True)
