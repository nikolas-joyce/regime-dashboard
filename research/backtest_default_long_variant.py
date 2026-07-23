"""
Fourth backtest variant (2026-07-22): "default-long, confidence-scaled, market-
anchored" directional exposure, built to directly test Nikolas's proposed fix after
three rounds of diagnostics on the config.yaml structure_matrix (see
backtest_per_ticker.py's docstring history and the regime_distribution_diagnostic /
direction_summary_diagnostic / cell_mix_diagnostic findings):

  1. Even at high market-posterior confidence (61.5% of days), 44% of (name, date)
     trade-entries were still net_delta==0 -- the per-name RS-based tilt cell
     (bull_hi/bull_lo/bear_hi/bear_lo/neut_hi/neut_lo, config.yaml tilt_layer)
     dominates over market confidence as the actual gate.
  2. cell_mix_diagnostic found a positive correlation (~+0.3) between %names
     classified neutral and the trailing 63-day return of the basket -- the
     RS-based per-name signal goes quiet specifically during calm, broadly-bullish
     stretches (low cross-sectional dispersion), and is LEAST neutral during acute
     volatility/crisis (2008-09, 2018, 2020, 2022) -- backwards from what a
     default-long, temper-on-evidence design wants.
  3. Even in cells that DO trade, sizing doesn't scale with confidence tier (three
     of six cells' "_smaller"/moderate variants use identical legs to their
     high_confidence counterpart).

This variant replaces the config.yaml structure_matrix lookup entirely with a
direct, continuous net_delta formula:

    net_delta = clip(MARKET_BIAS[market_cell] * confidence_scalar(posterior)
                      + NAME_TILT[name_cell],  -1.2, 1.2)

  - MARKET_BIAS anchors direction+base size to the MARKET's committed regime (not
    per-name RS), which the dashboard's own step0 validation already showed dwells
    sensibly (min_dwell=3, commit smoothing) rather than flickering. neut_hi/neut_lo
    get a positive (not zero) bias -- "absence of confirmed-bearish evidence stays
    long," per Nikolas's stated prior, generalizing the asymmetric bias config.yaml
    already applies to neut_hi alone.
  - confidence_scalar(p) = 0.5 + 0.5*p is a smooth floor-to-ceiling scalar (0.5 at
    p=0, 1.0 at p=1) -- exposure never drops to zero purely from low confidence
    (that was the low_confidence tier's 100%-flat behavior in the matrix, rejected
    here), but DOES scale up continuously with conviction (which the matrix never
    did within a tier).
  - NAME_TILT is now a small ADDITIVE modulator on top of the market-driven base,
    not an independent gate -- a name's own idiosyncratic RS can nudge exposure up
    or down, but can no longer zero it out on its own.

This is deliberately a fast, simple formula to test the HYPOTHESIS, not a final
design -- MARKET_BIAS/NAME_TILT/confidence_scalar constants below are a reasonable
first pass, not calibrated. Same TRADE_DTE=21/TRADE_STEP=5 cadence and
delta_proxy_return methodology as backtest_per_ticker.py, for apples-to-apples
comparison.

2026-07-22, second pass ("let's advance" -- two follow-ups from the SPY/sub-index
and delta-proxy discussion, added as two SEPARATE, independently-readable columns
rather than folded into one number, so each change's effect stays isolated:

  1. QQQ sub-index anchor (variant2_*). Confirmed in run_nightly.py that the market
     regime HMM is fit on SPY ONLY -- config.yaml lists QQQ/IWM under
     underlyings_index but neither is ever fetched or used. IWM (small caps) isn't
     relevant to this 50-mega-cap universe and isn't added. QQQ is, for the
     tech/growth subset (TECH_SUBSET below) -- a lightweight trend+vol read (price
     vs. 200dma, realized-vol percentile), NOT a full HMM refit, blended 50/50 with
     the existing SPY-anchored market component for those names only. This is
     Nikolas's "look at SPY or SPY+relevant sub-index for the default, then tilt
     with single-name intel" -- variant (v1) already did the SPY-anchor half;
     variant2 adds the sub-index half for the subset where it's actually relevant.
  2. Crude premium/theta accrual on the CURRENT MATRIX only (matrix_premium_*, not
     the variants -- the variants are raw delta targets with no options legs, theta
     doesn't apply to them). Addresses the objection that delta_proxy_return is a
     linear, static approximation that discards time decay and the IV risk premium
     entirely -- for credit-selling structures (short_put, both credit spreads)
     that decay IS the trade, so scoring them on delta alone structurally
     understates them. Adds a flat ANNUAL_PREMIUM_YIELD=0.12 (12%/yr, an
     uncalibrated placeholder -- not derived from any IV data, this repo has none
     validated yet) accrual on |net_delta| for credit structures, subtracted for
     debit structures, zero for no_trade. This does NOT fix the proxy's other
     limitations (naked short_put's understated convex tail risk, no transaction
     costs, Sharpe on a linearized/non-kinked return series) -- it isolates
     specifically the "matrix loses to buy-hold" confound around theta, nothing
     else. Read matrix_premium vs. matrix as "how much of the gap could theta
     plausibly close," not as a validated P&L.

Run locally (needs the same venv/data as backtest_per_ticker.py):
    python research/backtest_default_long_variant.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pipeline.data_pull import pull_prices
from pipeline.matrix import confidence_tier
from pipeline.default_long import (
    MARKET_BIAS, NAME_TILT, NET_DELTA_CAP, confidence_scalar, variant_net_delta,
)
from backtest_per_ticker import (
    REPO_ROOT, TRADE_DTE, TRADE_STEP, BASELINE_LEGS,
    load_config, load_persisted, market_posterior_series,
    net_delta as leg_net_delta, delta_proxy_return, true_daily_sharpe,
)

# 2026-07-22: MARKET_BIAS/NAME_TILT/NET_DELTA_CAP/confidence_scalar/variant_net_delta
# moved to pipeline/default_long.py as the single source of truth, shared with the
# live app (app/app.py's new default-long banner) -- this file now imports them
# instead of defining its own copy. See that module's docstring for the full
# rationale and the backtested numbers that justified shipping it to the app.

# --- QQQ sub-index anchor (tech/growth subset only -- see module docstring) ---
TECH_SUBSET = {
    "AAPL", "MSFT", "NVDA", "GOOGL", "META", "AVGO", "QCOM", "CSCO", "ADBE", "CRM", "AMD", "AMZN",
}
QQQ_SMA_WINDOW = 200
QQQ_VOL_PCT_WINDOW = 252
QQQ_VOL_PCT_THRESH = 0.60
QQQ_CONFIDENCE = 0.75  # fixed scalar -- this is a binary trend flag, not a calibrated probability

# --- crude premium/theta accrual on the current matrix only (see module docstring) ---
ANNUAL_PREMIUM_YIELD = 0.12  # uncalibrated placeholder, not derived from any IV data
CREDIT_STRUCTURES = {"short_put"}  # plus anything with "credit" in the name, matched below
DEBIT_MARKER = "debit"
CREDIT_MARKER = "credit"


def qqq_cell_series(qqq_price: pd.Series) -> pd.Series:
    """Lightweight, non-HMM trend+vol read on QQQ -- deliberately crude (this is the
    'small, immediate' Stage 1 sub-index addition, not a full regime refit). Trend:
    price vs. its own 200dma. Vol: trailing 20d realized-vol percentile within a 252d
    window, same convention as pipeline/tilt.py's name_vol_state. Produces one of
    bull_hi/bull_lo/bear_hi/bear_lo -- no neutral band, since this only feeds a 50/50
    blend with the SPY-anchored component, not a standalone gate."""
    sma = qqq_price.rolling(QQQ_SMA_WINDOW).mean()
    trend = np.where(qqq_price > sma, "bull", "bear")
    ret = np.log(qqq_price).diff()
    rv20 = ret.rolling(20).std() * np.sqrt(252)
    vol_pct = rv20.rolling(QQQ_VOL_PCT_WINDOW).rank(pct=True)
    vol_state = np.where(vol_pct > QQQ_VOL_PCT_THRESH, "hi", "lo")
    cell = pd.Series(
        [f"{t}_{v}" if pd.notna(s) else None for t, v, s in zip(trend, vol_state, sma)],
        index=qqq_price.index,
    )
    return cell


def variant2_net_delta(ticker, market_cell, posterior_p, name_cell, qqq_cell) -> float:
    """variant_net_delta plus a 50/50 blend-in of the QQQ trend read for the
    tech/growth subset. Everything else (all non-tech names) is identical to
    variant_net_delta -- this isolates the sub-index anchor's effect to exactly the
    names it's meant to matter for."""
    spy_component = MARKET_BIAS.get(market_cell, 0.0) * confidence_scalar(posterior_p)
    if ticker in TECH_SUBSET and qqq_cell is not None and pd.notna(qqq_cell):
        qqq_component = MARKET_BIAS.get(qqq_cell, 0.0) * QQQ_CONFIDENCE
        market_component = 0.5 * spy_component + 0.5 * qqq_component
    else:
        market_component = spy_component
    name_component = NAME_TILT.get(name_cell, 0.0) if pd.notna(name_cell) else 0.0
    return float(np.clip(market_component + name_component, -NET_DELTA_CAP, NET_DELTA_CAP))


def qqq_pure_net_delta(qqq_cell) -> float:
    """The QQQ trend/vol read on its OWN, unblended -- no SPY component, no name
    tilt. Used only for the QQQ self-test (see backtest_index_self): isolates
    whether the crude QQQ proxy has any standalone timing value at all, separate
    from the question of whether blending it into other names helps (variant2
    showed blending hurts 48/50 names -- this test tells us whether that's because
    the QQQ signal itself is bad, or just a bad way to combine two signals)."""
    if qqq_cell is None or pd.isna(qqq_cell):
        return 0.0
    return float(np.clip(MARKET_BIAS.get(qqq_cell, 0.0) * QQQ_CONFIDENCE, -NET_DELTA_CAP, NET_DELTA_CAP))


def backtest_index_self(
    ticker: str, price: pd.Series, committed: pd.Series, posterior: pd.Series, qqq_cell: pd.Series,
) -> tuple[dict, list[pd.Timestamp], list[float]]:
    """2026-07-22, third pass (Nikolas: 'add the same view for SPY and QQQ'):
    SPY and QQQ aren't in the 50-name universe, have no per-name RS cell in
    name_cells.parquet (they ARE the market/sub-index, not a name being tilted
    against it), and structure_matrix lookups don't apply to them either. So
    'matrix' is reported as a flat 0 here (labeled, not a real comparison point) --
    the real content is: does the pure market-timing signal (variant_net_delta with
    name_cell=None, i.e. MARKET_BIAS*confidence_scalar only, no RS tilt) beat
    buy-and-hold on the very index it's timed against? This is the cleanest
    possible test of the core hypothesis -- no single-stock idiosyncrasy, and for
    QQQ, also free of the blend-dilution effect seen in variant2 on the tech
    subset. For QQQ specifically, also computes qqq_pure_pnl (the QQQ signal fully
    unblended) to separate 'is the QQQ signal bad' from 'is blending it in bad'."""
    idx = price.index
    n = len(idx)
    trade_dates = []
    variant_pnl = []
    qqq_pure_pnl = []
    buyhold_pnl = []

    for t in range(0, n - TRADE_DTE, TRADE_STEP):
        d0 = idx[t]
        S0, ST = price.iloc[t], price.iloc[t + TRADE_DTE]
        if not (np.isfinite(S0) and np.isfinite(ST)):
            continue
        m_cell = committed.get(d0)
        p = posterior.get(d0)
        q_cell = qqq_cell.get(d0) if qqq_cell is not None else None
        if m_cell is None or pd.isna(m_cell):
            continue

        nd_v = variant_net_delta(m_cell, p, None)  # pure market-anchor, no RS tilt (index has none)
        variant_pnl.append(delta_proxy_return(nd_v, S0, ST))

        nd_qp = qqq_pure_net_delta(q_cell)
        qqq_pure_pnl.append(delta_proxy_return(nd_qp, S0, ST))

        buyhold_pnl.append(delta_proxy_return(1.0, S0, ST))
        trade_dates.append(d0)

    v, qp, h = np.array(variant_pnl), np.array(qqq_pure_pnl), np.array(buyhold_pnl)
    ann_factor = np.sqrt(252 / TRADE_STEP)

    def sharpe(x):
        return float(x.mean() / x.std() * ann_factor) if len(x) > 1 and x.std() > 0 else float("nan")

    summary = {
        "ticker": ticker,
        "n_trades": len(v),
        "matrix_sharpe": float("nan"),  # not applicable -- see docstring
        "matrix_premium_sharpe": float("nan"),
        "variant_sharpe": sharpe(v),
        "qqq_pure_sharpe": sharpe(qp) if ticker == "QQQ" else float("nan"),
        "variant2_sharpe": float("nan"),  # variant2 reduces to variant for a non-TECH_SUBSET ticker anyway
        "buyhold_sharpe_true_daily": true_daily_sharpe(price),
        "variant_vs_buyhold": sharpe(v) - true_daily_sharpe(price),
    }
    zero_matrix_pnl = [0.0] * len(variant_pnl)
    return summary, trade_dates, zero_matrix_pnl, variant_pnl


def is_credit_structure(structure_name: str) -> int:
    """+1 credit (premium received, theta works for us), -1 debit (premium paid,
    theta works against us), 0 no_trade/unrecognized."""
    if not structure_name or structure_name == "no_trade":
        return 0
    if structure_name in CREDIT_STRUCTURES or CREDIT_MARKER in structure_name:
        return 1
    if DEBIT_MARKER in structure_name:
        return -1
    return 0


def theta_accrual(structure_name: str, nd: float, dte: int = TRADE_DTE) -> float:
    """Crude static premium accrual -- see module docstring for what this does and
    doesn't fix. sign(structure) * annual_yield * |net_delta| * (holding days / 252)."""
    sign = is_credit_structure(structure_name)
    if sign == 0:
        return 0.0
    return sign * ANNUAL_PREMIUM_YIELD * abs(nd) * (dte / 252)


def backtest_one_name_compare(
    ticker: str, price: pd.Series, name_cell: pd.Series,
    committed: pd.Series, posterior: pd.Series, tier_series: pd.Series,
    structure_matrix: dict, qqq_cell: pd.Series,
) -> tuple[dict, list[pd.Timestamp], list[float], list[float], dict]:
    """Runs the CURRENT structure_matrix (delta-only and with the crude premium
    accrual), the SPY-anchored variant, and the SPY+QQQ-blended variant2 side by
    side, same entry dates, same price windows -- only the net_delta formula (and,
    for matrix_premium, the theta accrual) differs between columns.

    2026-07-22, fourth pass: the 5th return value (`extra`) exposes the raw
    per-trade nd/pnl arrays that the per-name summary dict collapses into a mean --
    needed by build_aggregate_portfolio() to build an equal-weight portfolio series
    across all names, which a per-name Sharpe average can't tell you anything about
    (diversification effects are invisible in an average-of-Sharpes)."""
    idx = price.index
    n = len(idx)

    trade_dates = []
    matrix_pnl, matrix_nd = [], []
    matrix_premium_pnl = []
    variant_pnl, variant_nd = [], []
    variant2_pnl, variant2_nd = [], []
    baseline_pnl = []
    buyhold_pnl = []

    for t in range(0, n - TRADE_DTE, TRADE_STEP):
        d0 = idx[t]
        S0, ST = price.iloc[t], price.iloc[t + TRADE_DTE]
        if not (np.isfinite(S0) and np.isfinite(ST)):
            continue

        m_cell = committed.get(d0)
        n_cell = name_cell.get(d0)
        p = posterior.get(d0)
        tier = tier_series.get(d0)
        q_cell = qqq_cell.get(d0) if qqq_cell is not None else None
        if m_cell is None or pd.isna(m_cell):
            continue

        # current matrix (needs a name cell + tier to look anything up)
        structure_name = "no_trade"
        legs = []
        if n_cell is not None and tier is not None and pd.notna(n_cell) and pd.notna(tier):
            entry = structure_matrix.get(n_cell, {}).get(tier, {})
            legs = entry.get("legs", [])
            structure_name = entry.get("structure", "no_trade")
        nd_m = leg_net_delta(legs)
        base_matrix_pnl = delta_proxy_return(nd_m, S0, ST)
        matrix_pnl.append(base_matrix_pnl)
        matrix_nd.append(nd_m)
        matrix_premium_pnl.append(base_matrix_pnl + theta_accrual(structure_name, nd_m))

        # default-long variant (SPY-anchored only)
        nd_v = variant_net_delta(m_cell, p, n_cell)
        variant_pnl.append(delta_proxy_return(nd_v, S0, ST))
        variant_nd.append(nd_v)

        # variant2 (SPY+QQQ blended for the tech subset)
        nd_v2 = variant2_net_delta(ticker, m_cell, p, n_cell, q_cell)
        variant2_pnl.append(delta_proxy_return(nd_v2, S0, ST))
        variant2_nd.append(nd_v2)

        nd_b = leg_net_delta(BASELINE_LEGS)
        baseline_pnl.append(delta_proxy_return(nd_b, S0, ST))

        buyhold_pnl.append(delta_proxy_return(1.0, S0, ST))
        trade_dates.append(d0)

    summary = _compare_summarize(
        ticker, price, matrix_pnl, matrix_nd, matrix_premium_pnl,
        variant_pnl, variant_nd, variant2_pnl, variant2_nd, baseline_pnl, buyhold_pnl,
    )
    extra = {"matrix_nd": matrix_nd, "variant_nd": variant_nd, "variant2_pnl": variant2_pnl, "variant2_nd": variant2_nd}
    return summary, trade_dates, matrix_pnl, variant_pnl, extra


def _compare_summarize(
    ticker, price, matrix_pnl, matrix_nd, matrix_premium_pnl,
    variant_pnl, variant_nd, variant2_pnl, variant2_nd, baseline_pnl, buyhold_pnl,
) -> dict:
    m, mp, v, v2, b, h = (
        np.array(matrix_pnl), np.array(matrix_premium_pnl), np.array(variant_pnl),
        np.array(variant2_pnl), np.array(baseline_pnl), np.array(buyhold_pnl),
    )
    ann_factor = np.sqrt(252 / TRADE_STEP)

    def sharpe(x):
        return float(x.mean() / x.std() * ann_factor) if len(x) > 1 and x.std() > 0 else float("nan")

    def max_dd(x):
        if len(x) == 0:
            return float("nan")
        cum = np.cumsum(x)
        peak = np.maximum.accumulate(cum)
        return float((cum - peak).min())

    return {
        "ticker": ticker,
        "n_trades": len(m),
        "matrix_sharpe": sharpe(m),
        "matrix_total_pnl": float(m.sum()) if len(m) else float("nan"),
        "matrix_max_dd": max_dd(m),
        "matrix_avg_abs_nd": float(np.mean(np.abs(matrix_nd))) if matrix_nd else float("nan"),
        "matrix_pct_flat": 100 * float(np.mean(np.array(matrix_nd) == 0)) if matrix_nd else float("nan"),
        "matrix_premium_sharpe": sharpe(mp),
        "matrix_premium_total_pnl": float(mp.sum()) if len(mp) else float("nan"),
        "variant_sharpe": sharpe(v),
        "variant_total_pnl": float(v.sum()) if len(v) else float("nan"),
        "variant_max_dd": max_dd(v),
        "variant_avg_abs_nd": float(np.mean(np.abs(variant_nd))) if variant_nd else float("nan"),
        "variant_pct_flat": 100 * float(np.mean(np.array(variant_nd) == 0)) if variant_nd else float("nan"),
        "variant2_sharpe": sharpe(v2),
        "variant2_total_pnl": float(v2.sum()) if len(v2) else float("nan"),
        "variant2_avg_abs_nd": float(np.mean(np.abs(variant2_nd))) if variant2_nd else float("nan"),
        "baseline_sharpe": sharpe(b),
        "buyhold_sharpe_overlapping": sharpe(h),
        "buyhold_sharpe_true_daily": true_daily_sharpe(price),
        "variant_vs_matrix": sharpe(v) - sharpe(m),
        "variant_vs_buyhold": sharpe(v) - sharpe(h),
        "variant2_vs_variant": sharpe(v2) - sharpe(v),
        "variant2_vs_buyhold": sharpe(v2) - sharpe(h),
        "matrix_premium_vs_matrix": sharpe(mp) - sharpe(m),
        "matrix_premium_vs_buyhold": sharpe(mp) - sharpe(h),
        "matrix_vs_buyhold": sharpe(m) - sharpe(h),
    }


def plot_comparison(
    universe: list[str], prices: pd.DataFrame,
    trade_data: dict[str, tuple[list[pd.Timestamp], list[float], list[float]]],
    out_path: Path,
) -> None:
    """buy-hold (left axis) vs. current matrix (right axis, orange) vs. default-long
    variant (right axis, green) -- one panel per name."""
    tickers = [t for t in universe if t in trade_data]
    ncols = 5
    nrows = -(-len(tickers) // ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 3.4, nrows * 2.4), squeeze=False)

    legend_handles, legend_labels = [], []
    for i, ticker in enumerate(tickers):
        ax = axes[i // ncols][i % ncols]
        dates, matrix_pnl, variant_pnl = trade_data[ticker]
        if not dates:
            ax.set_title(f"{ticker} (no trades)", fontsize=8)
            ax.axis("off")
            continue

        px = prices[ticker].dropna()
        px_window = px.loc[dates[0]:]
        underlying_curve = px_window / px_window.iloc[0] - 1.0
        matrix_curve = pd.Series(np.cumsum(matrix_pnl), index=dates)
        variant_curve = pd.Series(np.cumsum(variant_pnl), index=dates)

        l1, = ax.plot(underlying_curve.index, underlying_curve.values,
                       label="buy-hold (left axis)", linewidth=0.8, color="tab:blue")
        ax.tick_params(axis="y", labelcolor="tab:blue", labelsize=6)
        ax.axhline(0, color="gray", linewidth=0.5, linestyle=":")

        ax2 = ax.twinx()
        l2, = ax2.plot(matrix_curve.index, matrix_curve.values,
                        label="current matrix (right axis)", linewidth=0.8, color="tab:orange")
        l3, = ax2.plot(variant_curve.index, variant_curve.values,
                        label="default-long variant (right axis)", linewidth=1.0, color="tab:green")
        ax2.tick_params(axis="y", labelcolor="black", labelsize=6)
        ax2.axhline(0, color="gray", linewidth=0.4, linestyle=":", alpha=0.5)

        ax.set_title(ticker, fontsize=9)
        ax.tick_params(axis="x", labelsize=6)
        if not legend_handles:
            legend_handles, legend_labels = [l1, l2, l3], [
                "buy-hold (left axis)", "current matrix (right axis)", "default-long variant (right axis)",
            ]

    for j in range(len(tickers), nrows * ncols):
        axes[j // ncols][j % ncols].axis("off")

    if legend_handles:
        fig.legend(legend_handles, legend_labels, loc="upper center", ncol=3, fontsize=9)
    fig.suptitle(
        "Default-long, market-anchored, confidence-scaled variant vs. current structure_matrix vs. buy-hold",
        y=1.0, fontsize=10,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def build_aggregate_portfolio(series_data: dict, spy_price: pd.Series) -> dict:
    """2026-07-22, fourth pass (Nikolas: 'aggregate of long and shorts across all
    tickers compared to SPY'). Everything reported so far is a MEAN OF PER-NAME
    SHARPES -- that says nothing about what a diversified 50-name portfolio running
    this variant would actually have earned. 50 names each with a Sharpe of 1.0
    could aggregate to a much HIGHER portfolio Sharpe (if their idiosyncratic
    return components are weakly correlated) or barely higher at all (if they're
    all just riding the same market beta) -- averaging Sharpes can't distinguish
    those cases. This builds the actual equal-weight aggregate and compares it to
    SPY buy-and-hold directly, plus splits the variant sleeve into its long-only and
    short-only contributions so we can see which side of the book is doing the work.

    Sharpe caveat: unioning 50 names' individually-5-trading-day-staggered entry
    dates produces an aggregate series with close to DAILY spacing -- far denser
    than any single name's own 5-day cadence, and each entry is still a 21-trading-
    day-forward return, so adjacent aggregate observations share most of their
    underlying window (worse overlap than the per-name series already flagged
    elsewhere in this file). Computing Sharpe on month-end changes in cumulative P&L
    instead of the raw per-trade-date series is a standard, easily-defensible way to
    get past most of that overlap without needing to build a full daily
    mark-to-market position tracker (a bigger lift, not done here)."""
    def wide_from(field: str) -> pd.DataFrame:
        cols = {}
        for ticker, d in series_data.items():
            if d["trade_dates"]:
                cols[ticker] = pd.Series(d[field], index=pd.DatetimeIndex(d["trade_dates"]))
        return pd.DataFrame(cols)

    def long_short_wide(pnl_field: str, nd_field: str) -> tuple[pd.DataFrame, pd.DataFrame]:
        long_cols, short_cols = {}, {}
        for ticker, d in series_data.items():
            if not d["trade_dates"]:
                continue
            idx = pd.DatetimeIndex(d["trade_dates"])
            pnl, nd = d[pnl_field], d[nd_field]
            long_cols[ticker] = pd.Series([p if n > 0 else 0.0 for p, n in zip(pnl, nd)], index=idx)
            short_cols[ticker] = pd.Series([p if n < 0 else 0.0 for p, n in zip(pnl, nd)], index=idx)
        return pd.DataFrame(long_cols), pd.DataFrame(short_cols)

    matrix_wide = wide_from("matrix_pnl")
    variant_wide = wide_from("variant_pnl")
    variant2_wide = wide_from("variant2_pnl")
    variant_long_wide, variant_short_wide = long_short_wide("variant_pnl", "variant_nd")

    portfolio_matrix = matrix_wide.mean(axis=1, skipna=True).sort_index()
    portfolio_variant = variant_wide.mean(axis=1, skipna=True).sort_index()
    portfolio_variant2 = variant2_wide.mean(axis=1, skipna=True).sort_index()
    portfolio_variant_long = variant_long_wide.mean(axis=1, skipna=True).sort_index()
    portfolio_variant_short = variant_short_wide.mean(axis=1, skipna=True).sort_index()

    spy_ret = spy_price.pct_change().dropna()
    spy_cum = (1 + spy_ret).cumprod() - 1.0

    def monthly_sharpe_from_cum(cum: pd.Series) -> float:
        # "ME" (month-end), not "M" -- pandas 2.2+ removed the "M" offset alias
        # entirely (raises ValueError, not just a deprecation warning), which is
        # what broke the first run of this on the venv's pandas version.
        m = cum.resample("ME").last().ffill().diff().dropna()
        return float(m.mean() / m.std() * np.sqrt(12)) if len(m) > 1 and m.std() > 0 else float("nan")

    def monthly_sharpe_from_ret(ret: pd.Series) -> float:
        m = (1 + ret).resample("ME").apply(lambda x: x.prod() - 1).dropna()
        return float(m.mean() / m.std() * np.sqrt(12)) if len(m) > 1 and m.std() > 0 else float("nan")

    return {
        "portfolio_matrix_cum": portfolio_matrix.cumsum(),
        "portfolio_variant_cum": portfolio_variant.cumsum(),
        "portfolio_variant2_cum": portfolio_variant2.cumsum(),
        "portfolio_variant_long_cum": portfolio_variant_long.cumsum(),
        "portfolio_variant_short_cum": portfolio_variant_short.cumsum(),
        "spy_cum": spy_cum,
        "portfolio_matrix_monthly_sharpe": monthly_sharpe_from_cum(portfolio_matrix.cumsum()),
        "portfolio_variant_monthly_sharpe": monthly_sharpe_from_cum(portfolio_variant.cumsum()),
        "portfolio_variant2_monthly_sharpe": monthly_sharpe_from_cum(portfolio_variant2.cumsum()),
        "portfolio_variant_long_monthly_sharpe": monthly_sharpe_from_cum(portfolio_variant_long.cumsum()),
        "portfolio_variant_short_monthly_sharpe": monthly_sharpe_from_cum(portfolio_variant_short.cumsum()),
        "spy_monthly_sharpe": monthly_sharpe_from_ret(spy_ret),
        "portfolio_matrix_total": float(portfolio_matrix.sum()),
        "portfolio_variant_total": float(portfolio_variant.sum()),
        "portfolio_variant2_total": float(portfolio_variant2.sum()),
        "portfolio_variant_long_total": float(portfolio_variant_long.sum()),
        "portfolio_variant_short_total": float(portfolio_variant_short.sum()),
        "spy_total": float(spy_cum.iloc[-1]) if len(spy_cum) else float("nan"),
    }


def plot_aggregate_portfolio(agg: dict, out_path: Path) -> None:
    """Single chart, not a grid -- the whole point is one clean comparative picture:
    equal-weight aggregate matrix / variant / variant2 (cumulative P&L, left axis,
    since these run well below full notional) vs. SPY buy-and-hold (cumulative
    return, right axis), plus the variant's long-only and short-only contributions
    so it's visible how much of the aggregate comes from each side of the book."""
    fig, ax1 = plt.subplots(figsize=(12, 6))
    ax1.plot(agg["portfolio_matrix_cum"].index, agg["portfolio_matrix_cum"].values,
              label="matrix (current, aggregate)", color="tab:orange", linewidth=1.0)
    ax1.plot(agg["portfolio_variant_cum"].index, agg["portfolio_variant_cum"].values,
              label="variant (SPY-anchored, aggregate)", color="tab:green", linewidth=1.4)
    ax1.plot(agg["portfolio_variant2_cum"].index, agg["portfolio_variant2_cum"].values,
              label="variant2 (SPY+QQQ, aggregate)", color="tab:purple", linewidth=1.0, linestyle="--")
    ax1.plot(agg["portfolio_variant_long_cum"].index, agg["portfolio_variant_long_cum"].values,
              label="variant long-only sleeve", color="tab:green", linewidth=0.9, linestyle=":")
    ax1.plot(agg["portfolio_variant_short_cum"].index, agg["portfolio_variant_short_cum"].values,
              label="variant short-only sleeve", color="tab:red", linewidth=0.9, linestyle=":")
    ax1.axhline(0, color="gray", linewidth=0.5, linestyle=":")
    ax1.set_ylabel("aggregate cumulative P&L (delta-proxy units, left axis)")
    ax1.tick_params(axis="y")

    ax2 = ax1.twinx()
    ax2.plot(agg["spy_cum"].index, agg["spy_cum"].values,
              label="SPY buy-hold (right axis)", color="tab:blue", linewidth=1.6)
    ax2.set_ylabel("SPY cumulative return (right axis)", color="tab:blue")
    ax2.tick_params(axis="y", labelcolor="tab:blue")

    l1, lab1 = ax1.get_legend_handles_labels()
    l2, lab2 = ax2.get_legend_handles_labels()
    fig.legend(l1 + l2, lab1 + lab2, loc="upper left", ncol=2, fontsize=9)
    fig.suptitle("Equal-weight aggregate across all 50 names vs. SPY buy-and-hold")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main():
    cfg = load_config()
    universe = cfg["data"]["underlyings_names"]
    structure_matrix = cfg["structure_matrix"]
    bands = cfg["confidence_bands"]
    start = cfg["data"]["start_date"]

    # 2026-07-22: single fetch for universe+QQQ together, not two separate pull_prices
    # calls -- yf.download(progress=False) gives no visible feedback during the
    # network round-trip (that flag is shared with the production nightly pipeline,
    # not changed here to avoid noisy CI logs), and a second standalone call for just
    # QQQ was doubling that silent wait for no reason. Elapsed-time prints below are
    # the substitute for a real progress bar -- see the [i/N] counter in the
    # per-ticker loop too.
    fetch_list = universe + ["QQQ", "SPY"]
    print(f"Fetching price history for {len(fetch_list)} tickers ({len(universe)} names + QQQ + SPY) from {start}...")
    print("(yfinance gives no progress feedback mid-download -- this step is silent but not stuck; "
          "typically 15-60s depending on network/rate limits)")
    t0 = time.time()
    prices, source = pull_prices(fetch_list, start)
    print(f"Got prices from {source} in {time.time() - t0:.1f}s: {prices.shape[0]} rows, {prices.shape[1]} tickers")

    qqq_cell = qqq_cell_series(prices["QQQ"].dropna()) if "QQQ" in prices.columns else None
    if qqq_cell is None:
        print("QQQ unavailable in the fetched data -- variant2 will equal variant for all names")

    name_cells, cell_posterior, committed = load_persisted()
    posterior = market_posterior_series(cell_posterior, committed)
    tier_series = posterior.apply(lambda p: confidence_tier(p, bands) if pd.notna(p) else None)

    print(f"\nRunning per-ticker backtest ({len(universe)} names, ~605 entries each -- fast, "
          f"the fetch above was the slow part)...")
    t1 = time.time()
    results = []
    trade_data = {}
    series_data = {}
    for i, ticker in enumerate(universe, start=1):
        if ticker not in prices.columns:
            print(f"  [{i}/{len(universe)}] {ticker}: no price data, skipping")
            continue
        px = prices[ticker].dropna()
        nc = name_cells[name_cells["ticker"] == ticker].set_index("Date")["cell"]
        res, trade_dates, matrix_pnl, variant_pnl, series_extra = backtest_one_name_compare(
            ticker, px, nc, committed, posterior, tier_series, structure_matrix, qqq_cell,
        )
        results.append(res)
        trade_data[ticker] = (trade_dates, matrix_pnl, variant_pnl)
        series_data[ticker] = {
            "trade_dates": trade_dates, "matrix_pnl": matrix_pnl, "matrix_nd": series_extra["matrix_nd"],
            "variant_pnl": variant_pnl, "variant_nd": series_extra["variant_nd"],
            "variant2_pnl": series_extra["variant2_pnl"], "variant2_nd": series_extra["variant2_nd"],
        }
        tech_flag = " [QQQ-blended]" if ticker in TECH_SUBSET else ""
        print(f"  [{i}/{len(universe)}] {ticker}{tech_flag}: n={res['n_trades']}, matrix={res['matrix_sharpe']:.2f} "
              f"(flat {res['matrix_pct_flat']:.0f}%) matrix+premium={res['matrix_premium_sharpe']:.2f} "
              f"-> variant={res['variant_sharpe']:.2f} -> variant2={res['variant2_sharpe']:.2f}, "
              f"true daily buy-hold={res['buyhold_sharpe_true_daily']:.2f}")
    print(f"Backtest loop done in {time.time() - t1:.1f}s")

    print("\nRunning SPY/QQQ self-test (pure market-timing signal against the index itself, "
          "no per-name RS, no matrix -- see backtest_index_self docstring)...")
    index_results = []
    for idx_ticker in ["SPY", "QQQ"]:
        if idx_ticker not in prices.columns:
            print(f"  {idx_ticker}: no price data, skipping")
            continue
        idx_px = prices[idx_ticker].dropna()
        res, trade_dates, zero_matrix_pnl, variant_pnl = backtest_index_self(
            idx_ticker, idx_px, committed, posterior, qqq_cell,
        )
        index_results.append(res)
        trade_data[idx_ticker] = (trade_dates, zero_matrix_pnl, variant_pnl)
        extra = f", qqq_pure_sharpe={res['qqq_pure_sharpe']:.2f}" if idx_ticker == "QQQ" else ""
        print(f"  {idx_ticker}: n={res['n_trades']}, variant_sharpe={res['variant_sharpe']:.2f}{extra}, "
              f"buy-hold true daily={res['buyhold_sharpe_true_daily']:.2f}, "
              f"variant_vs_buyhold={res['variant_vs_buyhold']:+.2f}")

    out = pd.DataFrame(results).sort_values("variant_vs_matrix", ascending=False)
    out_path = REPO_ROOT / "research" / "backtest_default_long_variant_results.csv"
    out.to_csv(out_path, index=False)
    print(f"\nWrote {out_path}")

    if index_results:
        idx_out_path = REPO_ROOT / "research" / "backtest_spy_qqq_self_test.csv"
        pd.DataFrame(index_results).to_csv(idx_out_path, index=False)
        print(f"Wrote {idx_out_path}")

    print("\nBuilding equal-weight aggregate portfolio (all 50 names) vs. SPY...")
    agg = build_aggregate_portfolio(series_data, prices["SPY"].dropna())
    agg_plot_path = REPO_ROOT / "research" / "backtest_aggregate_portfolio_vs_spy.png"
    plot_aggregate_portfolio(agg, agg_plot_path)
    print(f"Wrote {agg_plot_path}")
    print("\n=== Aggregate portfolio (equal-weight, all 50 names) vs. SPY ===")
    print(f"Portfolio matrix        : total P&L {agg['portfolio_matrix_total']:+.3f}, "
          f"monthly Sharpe {agg['portfolio_matrix_monthly_sharpe']:.2f}")
    print(f"Portfolio variant (SPY) : total P&L {agg['portfolio_variant_total']:+.3f}, "
          f"monthly Sharpe {agg['portfolio_variant_monthly_sharpe']:.2f}")
    print(f"Portfolio variant2 (QQQ): total P&L {agg['portfolio_variant2_total']:+.3f}, "
          f"monthly Sharpe {agg['portfolio_variant2_monthly_sharpe']:.2f}")
    print(f"  -- long-only sleeve   : total P&L {agg['portfolio_variant_long_total']:+.3f}, "
          f"monthly Sharpe {agg['portfolio_variant_long_monthly_sharpe']:.2f}")
    print(f"  -- short-only sleeve  : total P&L {agg['portfolio_variant_short_total']:+.3f}, "
          f"monthly Sharpe {agg['portfolio_variant_short_monthly_sharpe']:.2f}")
    print(f"SPY buy-hold            : total return {agg['spy_total']:+.3f}, "
          f"monthly Sharpe {agg['spy_monthly_sharpe']:.2f}")
    print("(monthly-resampled Sharpe -- see build_aggregate_portfolio docstring for why raw "
          "per-trade-date Sharpe would be badly overlap-inflated at the aggregate level)")

    plot_path = REPO_ROOT / "research" / "backtest_default_long_variant_curves.png"
    print(f"Rendering comparison grid to {plot_path}...")
    plot_comparison(universe + ["SPY", "QQQ"], prices, trade_data, plot_path)
    print("Done.")

    print("\n=== Summary: matrix / matrix+premium / variant (SPY) / variant2 (SPY+QQQ) / buy-hold ===")
    print(f"Mean matrix Sharpe (overlapping):         {out['matrix_sharpe'].mean():.2f}")
    print(f"Mean matrix+premium Sharpe (overlapping):  {out['matrix_premium_sharpe'].mean():.2f}  "
          f"<- crude theta accrual added, see docstring for what this does/doesn't fix")
    print(f"Mean variant Sharpe (SPY-anchored):        {out['variant_sharpe'].mean():.2f}")
    print(f"Mean variant2 Sharpe (SPY+QQQ blended):    {out['variant2_sharpe'].mean():.2f}")
    print(f"Mean buy-hold Sharpe (TRUE daily):         {out['buyhold_sharpe_true_daily'].mean():.2f}")
    print(f"Mean matrix %flat:                         {out['matrix_pct_flat'].mean():.1f}%")
    print(f"Mean variant %flat:                        {out['variant_pct_flat'].mean():.1f}%")
    print(f"Mean matrix avg|net_delta|:                {out['matrix_avg_abs_nd'].mean():.3f}")
    print(f"Mean variant avg|net_delta|:                {out['variant_avg_abs_nd'].mean():.3f}")
    print(f"Mean variant2 avg|net_delta|:               {out['variant2_avg_abs_nd'].mean():.3f}")
    print(f"Names where variant beats matrix (Sharpe):        {int((out['variant_vs_matrix'] > 0).sum())}/{len(out)}")
    print(f"Names where variant2 beats variant (Sharpe):      {int((out['variant2_vs_variant'] > 0).sum())}/{len(out)}  "
          f"<- isolates the QQQ sub-index anchor's effect, TECH_SUBSET names only should move")
    print(f"Names where variant beats buy-hold (Sharpe):      {int((out['variant_vs_buyhold'] > 0).sum())}/{len(out)}")
    print(f"Names where variant2 beats buy-hold (Sharpe):     {int((out['variant2_vs_buyhold'] > 0).sum())}/{len(out)}")
    print(f"Names where matrix beats buy-hold (Sharpe):       {int((out['matrix_vs_buyhold'] > 0).sum())}/{len(out)}")
    print(f"Names where matrix+premium beats buy-hold (Sharpe): "
          f"{int((out['matrix_premium_vs_buyhold'] > 0).sum())}/{len(out)}")

    cols = ["ticker", "n_trades", "matrix_sharpe", "matrix_premium_sharpe",
            "variant_sharpe", "variant2_sharpe", "variant2_vs_variant", "buyhold_sharpe_true_daily"]
    print("\nTop 5 by variant2 improvement over variant (QQQ anchor effect):")
    print(out.sort_values("variant2_vs_variant", ascending=False).head(5)[cols].to_string(index=False))
    print("\nBottom 5 by variant2 improvement over variant:")
    print(out.sort_values("variant2_vs_variant", ascending=False).tail(5)[cols].to_string(index=False))

    cols2 = ["ticker", "n_trades", "matrix_sharpe", "matrix_pct_flat",
             "variant_sharpe", "variant_pct_flat", "variant_vs_matrix", "buyhold_sharpe_true_daily"]
    print("\nTop 5 by variant improvement over matrix:")
    print(out.head(5)[cols2].to_string(index=False))
    print("\nBottom 5 by variant improvement over matrix:")
    print(out.tail(5)[cols2].to_string(index=False))


if __name__ == "__main__":
    main()
