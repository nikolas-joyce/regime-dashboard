"""
Phase 3 -- derived analytics computed from already-loaded data (no new pipeline writes).

Pure functions, unit-testable the same way pipeline/ is (see tests/test_app_analytics.py) --
kept separate from app.py so the Streamlit-specific presentation code doesn't tangle with
logic that has actual correctness properties worth locking down.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from pipeline.matrix import bs_price, strike_from_delta, price_structure

CELLS = ["bear_hi", "bear_lo", "neut_hi", "neut_lo", "bull_hi", "bull_lo"]


def empirical_transition_matrix(committed: pd.Series) -> pd.DataFrame:
    """Row-normalized empirical transition matrix from the FULL committed-regime
    history: matrix.loc[from_cell, to_cell] = P(tomorrow == to_cell | today == from_cell).

    This is a live-computed stand-in for the plan's "next-day 6-cell probability bars"
    -- the pipeline doesn't persist a separate forward-looking transition forecast, but
    the committed-regime history already has everything needed to estimate one
    empirically. Cells with zero observed transitions get an all-zero row (caller should
    treat that as "insufficient history", not "impossible").
    """
    s = committed.dropna()
    pairs = pd.DataFrame({"today": s.values[:-1], "tomorrow": s.values[1:]})
    counts = pd.crosstab(pairs["today"], pairs["tomorrow"])
    counts = counts.reindex(index=CELLS, columns=CELLS, fill_value=0)
    row_sums = counts.sum(axis=1)
    matrix = counts.div(row_sums.replace(0, np.nan), axis=0).fillna(0.0)
    return matrix


def next_day_probs(current_cell: str, matrix: pd.DataFrame) -> pd.Series:
    if current_cell not in matrix.index:
        return pd.Series(0.0, index=CELLS)
    return matrix.loc[current_cell]


def exit_probability(current_cell: str, matrix: pd.DataFrame) -> float:
    """P(tomorrow != today's cell | today == current_cell) -- the amber-alert trigger."""
    probs = next_day_probs(current_cell, matrix)
    return float(1.0 - probs.get(current_cell, 0.0))


def regime_runs(committed: pd.Series) -> pd.DataFrame:
    """Collapse a daily committed-regime series into (regime, start, end, duration_days)
    runs -- one row per contiguous stay in a regime. Used for both the duration-
    distribution chart and the days-in-regime counter.
    """
    s = committed.dropna()
    if s.empty:
        return pd.DataFrame(columns=["regime", "start", "end", "duration_days"])
    change = s.ne(s.shift()).cumsum()
    rows = []
    for _, grp in s.groupby(change):
        rows.append({
            "regime": grp.iloc[0],
            "start": grp.index[0],
            "end": grp.index[-1],
            "duration_days": len(grp),
        })
    return pd.DataFrame(rows)


def days_in_current_regime(committed: pd.Series) -> int:
    runs = regime_runs(committed)
    if runs.empty:
        return 0
    return int(runs.iloc[-1]["duration_days"])


def forward_returns(price: pd.Series, horizon: int) -> pd.Series:
    """Log forward return over `horizon` trading days, indexed by the ORIGIN date
    (i.e. value at date t is the return from t to t+horizon) -- matured entries only.
    """
    log_px = np.log(price)
    fwd = log_px.shift(-horizon) - log_px
    return fwd.dropna()


def conditional_vs_unconditional_density(
    price: pd.Series, cell_history: pd.Series, current_cell: str, horizon: int = 5,
) -> dict:
    """Empirical forward-return distribution conditional on `current_cell`, vs. the
    name's full unconditional history -- same KS-test/effect-size methodology as Phase 0's
    TEST 1c (see research/regime-dashboard-plan.md section 7), re-run live per name as an
    ongoing out-of-sample check per the 2026-07-21 empirical-only design decision (no
    parametric baseline).

    Returns a dict with both raw arrays (for the caller to plot) and the summary stats,
    or an 'insufficient_data' flag if there aren't enough matured conditional
    observations to say anything meaningful (< 20, arbitrary but conservative floor).
    """
    fwd = forward_returns(price, horizon)
    aligned_cell = cell_history.reindex(fwd.index).ffill()
    conditional = fwd[aligned_cell == current_cell].dropna()
    unconditional = fwd.dropna()

    if len(conditional) < 20:
        return {
            "insufficient_data": True,
            "n_conditional": len(conditional),
            "n_unconditional": len(unconditional),
        }

    ks_stat, ks_p = stats.ks_2samp(conditional, unconditional)
    pooled_std = np.sqrt(
        ((len(conditional) - 1) * conditional.std() ** 2
         + (len(unconditional) - 1) * unconditional.std() ** 2)
        / (len(conditional) + len(unconditional) - 2)
    )
    effect_size = (conditional.mean() - unconditional.mean()) / pooled_std if pooled_std > 0 else 0.0

    return {
        "insufficient_data": False,
        "conditional": conditional,
        "unconditional": unconditional,
        "n_conditional": len(conditional),
        "n_unconditional": len(unconditional),
        "conditional_mean": float(conditional.mean()),
        "unconditional_mean": float(unconditional.mean()),
        "ks_stat": float(ks_stat),
        "ks_p": float(ks_p),
        "effect_size_sd": float(effect_size),
    }


def structure_terms(legs: list[dict], S0: float, sigma: float, r: float, T0: float) -> dict:
    """Concrete strikes/premium/max-loss/max-gain/breakeven for a recommended structure,
    given a live spot and IV. Max loss/gain/breakeven are derived by evaluating
    price_structure() (the same validated pricing path used in the Phase 0 backtest)
    over a price grid rather than hand-deriving a closed-form per structure type --
    robust across single-leg (short_put) and 2-leg vertical spreads alike, at the cost
    of being a grid approximation (accurate to the grid step) rather than exact.
    no_trade (empty legs) returns an all-zero/None result.
    """
    if not legs:
        return {
            "legs": [], "net_credit_debit": 0.0, "max_gain": 0.0, "max_loss": 0.0,
            "breakevens": [],
        }

    leg_terms = []
    for leg in legs:
        cp = 1 if leg["cp"] == "call" else -1
        is_call = cp == 1
        K = strike_from_delta(S0, T0, sigma, r, leg["delta"], is_call)
        premium = bs_price(S0, K, T0, sigma, r, cp)
        leg_terms.append({
            "cp": leg["cp"], "delta": leg["delta"], "pos": leg["pos"],
            "strike": float(K), "premium": float(premium),
        })
    # net cash at entry: long legs pay (cash out), short legs receive (cash in)
    net_credit_debit = float(sum(-lt["pos"] * lt["premium"] for lt in leg_terms))

    grid = np.linspace(0.4 * S0, 1.8 * S0, 141)
    pnl = np.array([price_structure(legs, S0, ST, T0, sigma, r) * S0 for ST in grid])
    max_gain = float(pnl.max())
    max_loss = float(pnl.min())

    breakevens = []
    sign = np.sign(pnl)
    crossings = np.where(np.diff(sign) != 0)[0]
    for i in crossings:
        x0, x1 = grid[i], grid[i + 1]
        y0, y1 = pnl[i], pnl[i + 1]
        if y1 != y0:
            breakevens.append(float(x0 + (0 - y0) * (x1 - x0) / (y1 - y0)))

    return {
        "legs": leg_terms,
        "net_credit_debit": net_credit_debit,
        "max_gain": max_gain,
        "max_loss": max_loss,
        "breakevens": breakevens,
    }


def call_history_log(price: pd.Series, cell_history: pd.Series, horizon: int = 5) -> pd.DataFrame:
    """Per-date cell assignment with realized forward return once matured (NaN for the
    most recent `horizon` days -- not yet resolved). This is the per-name 'call history'
    the plan's drill-down section asks for: date, committed cell, realized forward return.
    """
    fwd = forward_returns(price, horizon)
    df = cell_history.to_frame("cell")
    df["fwd_return"] = fwd.reindex(df.index)
    return df.dropna(subset=["cell"]).sort_index(ascending=False)
