"""
IV snapshot — pure calculation functions (plan section 5 step 3).

No data-acquisition code here (see data_pull.py's pull_iv_snapshot for the yfinance
chain fetch) -- same separation as model.py/tilt.py/matrix.py. Operates on option-chain
DataFrames the caller supplies (yfinance's option_chain(expiry).calls/.puts shape:
columns include at least "strike" and "impliedVolatility").

Yahoo already computes and publishes impliedVolatility per contract, so this module
does NOT invert Black-Scholes for IV -- only for delta (to locate the 25-delta strikes
for skew), using each strike's own quoted IV, which is standard practice.
"""
from __future__ import annotations

from datetime import date, datetime
from typing import Optional

import numpy as np
from scipy.stats import norm


def bs_delta(S: float, K: float, T: float, sigma: float, r: float, is_call: bool) -> float:
    """Black-Scholes delta. Returns a value in (0, 1) for calls, (-1, 0) for puts."""
    if T <= 0 or sigma <= 0:
        return float("nan")
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    return norm.cdf(d1) if is_call else norm.cdf(d1) - 1


def dte(expiry: str, as_of: Optional[date] = None) -> int:
    """Days to expiry for a yfinance expiry string ('YYYY-MM-DD')."""
    as_of = as_of or date.today()
    exp = datetime.strptime(expiry, "%Y-%m-%d").date()
    return (exp - as_of).days


def select_expiries(
    available: list[str], as_of: Optional[date] = None,
    near_dte_target: int = 17, far_dte_range: tuple[int, int] = (30, 45),
) -> tuple[Optional[str], Optional[str]]:
    """Pick the near-term expiry (closest to near_dte_target) and the target 30-45 DTE
    expiry (closest to the midpoint of far_dte_range) from a list of available expiry
    strings. Returns (None, None) components individually if no expiry qualifies --
    caller should treat a None as "skip this leg", not a hard failure.
    """
    if not available:
        return None, None
    dtes = [(e, dte(e, as_of)) for e in available]
    dtes = [(e, d) for e, d in dtes if d > 0]
    if not dtes:
        return None, None
    near = min(dtes, key=lambda x: abs(x[1] - near_dte_target))[0]
    far_mid = sum(far_dte_range) / 2
    far_candidates = [(e, d) for e, d in dtes if far_dte_range[0] <= d <= far_dte_range[1]]
    if far_candidates:
        far = min(far_candidates, key=lambda x: abs(x[1] - far_mid))[0]
    else:
        far = min(dtes, key=lambda x: abs(x[1] - far_mid))[0]
    return near, far


def atm_iv_from_chain(calls, puts, spot: float) -> Optional[float]:
    """ATM IV: average of the nearest-to-spot call and put implied vols. Returns None
    if either side is empty or has no valid IV near spot.
    """
    if calls is None or puts is None or len(calls) == 0 or len(puts) == 0:
        return None
    c = calls.iloc[(calls["strike"] - spot).abs().argsort()[:1]]
    p = puts.iloc[(puts["strike"] - spot).abs().argsort()[:1]]
    ivs = [v for v in [c["impliedVolatility"].iloc[0] if len(c) else None,
                        p["impliedVolatility"].iloc[0] if len(p) else None] if v is not None and v > 0]
    if not ivs:
        return None
    return float(np.mean(ivs))


def skew_25d_from_chain(
    calls, puts, spot: float, T: float, r: float, target_delta: float = 0.25,
) -> Optional[float]:
    """25-delta put-call skew: put_iv(-25d) - call_iv(+25d), using each strike's own
    quoted IV to compute its delta (standard practice, not a separate BS-IV inversion).
    Positive = puts richer than calls (typical equity skew). None if either side lacks
    a usable delta-IV pairing (e.g. all IVs are zero/stale).
    """
    if calls is None or puts is None or len(calls) == 0 or len(puts) == 0:
        return None

    def nearest_by_delta(df, is_call):
        d = df.copy()
        d["_delta"] = d.apply(
            lambda row: bs_delta(spot, row["strike"], T, row["impliedVolatility"], r, is_call)
            if row["impliedVolatility"] and row["impliedVolatility"] > 0 else np.nan, axis=1,
        )
        d = d.dropna(subset=["_delta"])
        if d.empty:
            return None
        target = target_delta if is_call else -target_delta
        idx = (d["_delta"] - target).abs().idxmin()
        return d.loc[idx]

    call_row = nearest_by_delta(calls, True)
    put_row = nearest_by_delta(puts, False)
    if call_row is None or put_row is None:
        return None
    return float(put_row["impliedVolatility"] - call_row["impliedVolatility"])
