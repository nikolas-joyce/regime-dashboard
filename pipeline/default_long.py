"""
Default-long, market-anchored, confidence-scaled directional exposure.

Extracted from research/backtest_default_long_variant.py (2026-07-22) so the
formula has ONE source of truth shared by the research backtest and the live app,
instead of two copies that could drift. Pure functions only, same pattern as
pipeline/matrix.py -- no data-acquisition code here.

Why this exists (see research/backtest_default_long_variant.py's module docstring
for the full history): config.yaml's structure_matrix is a confirmation-gated
design -- a name only gets directional exposure if its own RS-based tilt cell
clears a threshold, and even then sizing is a fixed per-structure delta that
doesn't scale with confidence. Backtested against the 50-name universe (2026-07-22):
mean per-name Sharpe 0.54 (matrix) vs. 1.00 (this variant) vs. 0.58 (buy-and-hold,
true daily). At the equal-weight PORTFOLIO level: monthly Sharpe 0.91 (matrix) vs.
1.23 (variant) vs. 0.77 (SPY buy-hold) -- see backtest_default_long_variant_results.csv
and backtest_aggregate_portfolio_vs_spy.png in research/ for the full run.

Known open issue (not fixed here): the variant's short sleeve was a net drag in
that backtest (aggregate short-only Sharpe -0.48, total P&L negative) -- the
long-only sleeve alone outperformed the combined book. MARKET_BIAS's bear_hi/
bear_lo values below are flagged for a "de-risk toward flat instead of flipping
short" revision once that's tested (see conversation notes, not yet in code).

The QQQ sub-index blend (variant2 in the research script) is deliberately NOT
ported here -- it underperformed the SPY-only variant in 48/50 names when tested,
concentrated in the exact tech-subset names it was meant to help. Not shipped
until that's fixed.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

MARKET_BIAS = {
    "bull_hi": 1.0,
    "bull_lo": 0.6,
    "neut_hi": 0.3,
    "neut_lo": 0.5,
    "bear_lo": -0.3,
    "bear_hi": -0.6,
}
NAME_TILT = {
    "bull_hi": 0.15,
    "bull_lo": 0.10,
    "neut_hi": 0.0,
    "neut_lo": 0.0,
    "bear_lo": -0.10,
    "bear_hi": -0.15,
}
NET_DELTA_CAP = 1.2


def confidence_scalar(p: float) -> float:
    """0.5 at p=0 -> 1.0 at p=1. Exposure floors at half-size, never zero, purely
    from low confidence -- confidence scales conviction, it doesn't gate entry
    (unlike config.yaml's confidence_bands, where low_confidence maps every cell
    to no_trade)."""
    if p is None or (isinstance(p, float) and p != p):  # None or NaN, no pandas import needed for the check
        return 0.5
    return float(0.5 + 0.5 * np.clip(p, 0.0, 1.0))


def variant_net_delta(market_cell: str | None, posterior_p: float | None, name_cell: str | None) -> float:
    """Direction and base size come from the MARKET's committed regime (not the
    name's own RS cell), scaled continuously by confidence. The name's own RS cell
    only nudges the result up/down (NAME_TILT) -- it can no longer zero exposure
    out on its own the way config.yaml's structure_matrix does."""
    market_component = MARKET_BIAS.get(market_cell, 0.0) * confidence_scalar(posterior_p)
    name_ok = name_cell is not None and not (isinstance(name_cell, float) and name_cell != name_cell)
    name_component = NAME_TILT.get(name_cell, 0.0) if name_ok else 0.0
    return float(np.clip(market_component + name_component, -NET_DELTA_CAP, NET_DELTA_CAP))


def direction_label(net_delta: float) -> str:
    """Coarse label for display -- thresholds are display-only, not used in sizing."""
    if net_delta > 0.05:
        return "Long"
    if net_delta < -0.05:
        return "Short"
    return "Flat"


def exposure_summary(per_name: dict, market_cell: str | None, posterior_p: float | None) -> dict:
    """Given state.json's per_name dict (ticker -> {cell, ...}) plus the current
    market cell/posterior, computes the default-long variant's exposure for every
    name and returns aggregate stats -- the live-app equivalent of
    direction_summary_diagnostic() from the research backtest, but for TODAY's
    snapshot instead of full history. No price data needed -- this is a pure
    function over already-loaded state.json fields."""
    rows = []
    for ticker, d in per_name.items():
        nd = variant_net_delta(market_cell, posterior_p, d.get("cell"))
        rows.append({"ticker": ticker, "net_delta": nd, "direction": direction_label(nd)})
    if not rows:
        return {"rows": [], "n": 0, "pct_long": float("nan"), "pct_short": float("nan"),
                "pct_flat": float("nan"), "avg_net_delta": float("nan")}
    df = pd.DataFrame(rows)
    n = len(df)
    return {
        "rows": rows,
        "n": n,
        "pct_long": 100 * (df["net_delta"] > 0.05).mean(),
        "pct_short": 100 * (df["net_delta"] < -0.05).mean(),
        "pct_flat": 100 * (df["net_delta"].abs() <= 0.05).mean(),
        "avg_net_delta": float(df["net_delta"].mean()),
        "avg_abs_net_delta": float(df["net_delta"].abs().mean()),
    }
