"""
Phase 3 -- derived analytics computed from already-loaded data (no new pipeline writes).

Pure functions, unit-testable the same way pipeline/ is (see tests/test_app_analytics.py) --
kept separate from app.py so the Streamlit-specific presentation code doesn't tangle with
logic that has actual correctness properties worth locking down.

forward_returns/conditional_vs_unconditional_density live in pipeline/forecast.py, not
here -- run_nightly.py now precomputes forecast density for all 50 names using price data
it already pulls (see run_nightly.py's forecast-density block), and the app's live
per-name drill-down (still on-demand, for a same-day-fresher deep dive) should use the
exact same tested logic rather than a second copy that could drift. Re-exported here so
existing callers/tests importing from app.analytics don't need to change.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from pipeline.matrix import bs_price, strike_from_delta, price_structure
from pipeline.forecast import forward_returns, conditional_vs_unconditional_density  # noqa: F401 -- re-exported

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


def transition_counts(committed: pd.Series) -> pd.Series:
    """Sample size behind each row of empirical_transition_matrix(): total number of
    (today, tomorrow) day-pairs observed starting FROM each cell -- i.e. total days
    historically spent in that cell (every day contributes one pair whether the regime
    stayed or switched the next day). NOT the number of distinct switch events, which
    would be far smaller. empirical_transition_matrix() normalizes these counts away, so
    without this a probability from 400 observed days and one from 12 look identical
    (2026-07-23, added after exactly that question came up for a rare cell)."""
    s = committed.dropna()
    if len(s) < 2:
        return pd.Series(0, index=CELLS)
    today = pd.Series(s.values[:-1])
    return today.value_counts().reindex(CELLS, fill_value=0)


def wilson_interval(k: float, n: float, z: float = 1.96) -> tuple[float, float]:
    """Wilson score confidence interval for a binomial proportion k/n (default z=1.96,
    ~95%). Used per-cell as a standard approximation to the true multinomial confidence
    region (each TO-cell treated as its own binary outcome vs. the same n) -- a
    simplification, not exact joint inference, but fine for a diagnostic display. Chosen
    over a normal/Wald interval because it doesn't produce nonsensical bounds (below 0%
    or above 100%) near p=0 or p=1, which is common here: sticky regimes routinely push
    "stay" probabilities near 1.0 and rare-transition probabilities near 0.0.
    Returns (nan, nan) if n<=0.
    """
    if n <= 0:
        return float("nan"), float("nan")
    p = k / n
    denom = 1 + z ** 2 / n
    center = (p + z ** 2 / (2 * n)) / denom
    half_width = (z * np.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2))) / denom
    return max(0.0, center - half_width), min(1.0, center + half_width)


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
