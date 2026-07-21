"""
Regime x structure matrix, and the Black-Scholes pricing helpers used to evaluate it.

Structure choices come from config.yaml, not hardcoded here, so the matrix can be
recalibrated without touching code. See config.yaml's `open_calibration_items` for what
Nikolas still needs to sign off on before this goes live, and regime-dashboard-plan.md
section 7b for what was tested and rejected (direction-conditional vol thresholds,
curve-gating, bear_hi call backspread -- do not re-propose these without new evidence).
"""
from __future__ import annotations

import numpy as np
from scipy.stats import norm


def bs_price(S: float, K: float, T: float, sigma: float, r: float, cp: int) -> float:
    """European option price, Black-Scholes. cp = 1 for call, -1 for put."""
    if T <= 0:
        return max(cp * (S - K), 0.0)
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if cp == 1:
        return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)


def strike_from_delta(S: float, T: float, sigma: float, r: float, delta: float, is_call: bool) -> float:
    """Invert BS delta to a strike. `delta` should be positive for calls, negative for puts
    (matches the sign convention used throughout config.yaml's structure_matrix legs)."""
    if is_call:
        d1 = norm.ppf(delta)
    else:
        d1 = norm.ppf(1 + delta)
    return S * np.exp(-(d1 * sigma * np.sqrt(T) - (r + 0.5 * sigma ** 2) * T))


def confidence_tier(posterior: float, bands: dict) -> str:
    if posterior >= bands["high"]:
        return "high_confidence"
    if posterior >= bands["moderate"]:
        return "moderate_confidence"
    return "low_confidence"


def recommend_structure(cell: str, posterior: float, structure_matrix: dict, bands: dict) -> dict:
    """Look up the recommended structure for a committed cell + confidence tier.

    Returns the config entry as-is: {"structure": str, "legs": [{"cp","delta","pos"}, ...]}.
    """
    tier = confidence_tier(posterior, bands)
    return structure_matrix[cell][tier]


def price_structure(entry_legs: list[dict], S0: float, ST: float, T0: float, sigma: float, r: float) -> float:
    """Entry-to-expiry P&L for a structure's legs, normalized by spot at entry.
    Mirrors the exact pricing logic validated in the step0 backtest (TEST 5/5b/5c)."""
    p = 0.0
    for leg in entry_legs:
        cp = 1 if leg["cp"] == "call" else -1
        is_call = cp == 1
        K = strike_from_delta(S0, T0, sigma, r, leg["delta"], is_call)
        entry = bs_price(S0, K, T0, sigma, r, cp)
        exitv = bs_price(ST, K, 1e-9, sigma, r, cp)
        p += leg["pos"] * (exitv - entry)
    return p / S0
