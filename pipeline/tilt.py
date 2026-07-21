"""
Per-name tilt layer (plan section 3.5, "deliberately simple") -- no per-name HMMs.

Direction tilt: z-scored 21d/63d relative strength (name minus market log-return) vs a
bullish/bearish threshold. Vol state: the name's own realized-vol percentile within its
trailing history (proxy for IV rank until the nightly IV snapshot accumulates -- see
plan section 8). Market regime gates the structure *family*; the name only tilts within
it -- a non-neutral market regime caps the name's direction on the side away from the
market's own sign (e.g. a bearish market caps every name at {bear, neut}, never bull,
regardless of how strong that name's relative-strength tilt is).

This module intentionally has no data-acquisition code -- pure functions over pandas
Series the caller supplies, same pattern as model.py.
"""
from __future__ import annotations

import pandas as pd


def relative_strength_zscore(
    name_ret: pd.Series, market_ret: pd.Series, window: int, z_window: int = 252,
) -> pd.Series:
    """Z-scored cumulative relative strength (name minus market) over `window` trading days."""
    m = market_ret.reindex(name_ret.index)
    rs = name_ret.rolling(window).sum() - m.rolling(window).sum()
    return (rs - rs.rolling(z_window).mean()) / rs.rolling(z_window).std()


def direction_tilt(
    rs_z_short: pd.Series, rs_z_long: pd.Series, bull_thresh: float = 0.5, bear_thresh: float = -0.5,
) -> pd.Series:
    """Combine short/long-horizon RS z-scores into a raw tilt label. Averages the two
    horizons per plan section 3.5 ("z-score of 21d and 63d relative strength") --
    deliberately simple, doesn't require both horizons to agree. Revisit if live results
    show the two horizons disagreeing often enough to matter.
    """
    z = (rs_z_short + rs_z_long) / 2
    tilt = pd.Series("neut", index=z.index)
    tilt[z > bull_thresh] = "bull"
    tilt[z < bear_thresh] = "bear"
    tilt[z.isna()] = None
    return tilt


def name_vol_state(realized_vol: pd.Series, pct_window: int = 252, pct_thresh: float = 0.60) -> pd.Series:
    """Name's own realized-vol percentile within its trailing history. > pct_thresh = high vol.
    Realized-vol proxy for IV rank until the nightly IV snapshot (plan section 5 step 3)
    accumulates enough history -- see plan section 8 risk note ("No historical IV").
    """
    pct = realized_vol.rolling(pct_window).rank(pct=True)
    state = pct.map(lambda p: "hi" if p > pct_thresh else ("lo" if pd.notna(p) else None))
    return state


def gate_direction_by_market(market_direction: pd.Series, name_tilt: pd.Series) -> pd.Series:
    """Market regime gates the structure family; the name only tilts within it. Both
    inputs are direction labels ("bear"/"neut"/"bull") aligned on a common date index.
    A non-neutral market caps the name's direction on the side away from the market's
    own sign -- bearish market: bull tilt -> neut. Bullish market: bear tilt -> neut.
    Neutral market: no cap, the name's own tilt determines direction freely.
    """
    md = market_direction.reindex(name_tilt.index)
    gated = name_tilt.copy()
    gated[(md == "bear") & (name_tilt == "bull")] = "neut"
    gated[(md == "bull") & (name_tilt == "bear")] = "neut"
    return gated


def compute_name_cell(
    market_committed: pd.Series, name_tilt: pd.Series, vol_state: pd.Series,
) -> pd.Series:
    """Final per-name 6-cell label: market-gated direction x name vol state.

    `market_committed` is the full committed market-layer cell (e.g. "bull_lo") --
    only its direction component gates the name; the name's own vol_state is independent
    of the market's vol state (a name can be bull/hi while the market itself is bull/lo).
    """
    market_direction = market_committed.reindex(name_tilt.index).str.split("_").str[0]
    gated = gate_direction_by_market(market_direction, name_tilt)
    cell = gated.astype(str) + "_" + vol_state.astype(str)
    cell[gated.isna() | vol_state.isna()] = None
    return cell
