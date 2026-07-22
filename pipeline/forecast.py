"""
Per-name forecast-density logic (plan section 6 drill-down: empirical forward-return
distribution conditional on a name's current cell, vs. its unconditional full-history
distribution -- same KS-test/effect-size methodology as Phase 0's TEST 1c, empirical only
per the 2026-07-21 design decision).

Lives in pipeline/, not app/, because run_nightly.py now precomputes this for all 50 names
using price data it already pulls for the tilt layer (zero extra API cost) -- see
run_nightly.py's forecast-density block. app/analytics.py imports from here rather than
duplicating the logic, so the nightly precompute and the app's live per-name drill-down
(still computed on demand, for a same-day-fresher deep dive) share one tested
implementation instead of two that could quietly drift apart.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


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
    name's full unconditional history. Returns raw arrays (for plotting) and summary
    stats, or an 'insufficient_data' flag if there aren't enough matured conditional
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


def binned_density(values: pd.Series, bin_edges: np.ndarray) -> list[float]:
    """Histogram bin counts for a return series against a shared set of bin edges --
    used to build directly-comparable conditional/unconditional sparklines (same bins
    for both, so shape differences are real, not a binning artifact).
    """
    counts, _ = np.histogram(values, bins=bin_edges)
    return counts.tolist()


def compute_name_forecast_density(
    price: pd.Series, cell_history: pd.Series, current_cell: str, horizon: int = 5, n_bins: int = 15,
) -> dict:
    """Full precompute payload for one name: summary stats + binned histograms for both
    distributions on shared bin edges, suitable for persisting to parquet and rendering
    as an in-cell sparkline (Streamlit's column_config.BarChartColumn) in the Names table.
    insufficient_data rows still get n_conditional/n_unconditional (diagnostic -- shows
    why a name has no sparkline yet) but null stats/histograms.
    """
    result = conditional_vs_unconditional_density(price, cell_history, current_cell, horizon)
    if result["insufficient_data"]:
        return {
            "insufficient_data": True,
            "n_conditional": result["n_conditional"],
            "n_unconditional": result["n_unconditional"],
            "conditional_mean": None, "unconditional_mean": None,
            "ks_stat": None, "ks_p": None, "effect_size_sd": None,
            "bin_edges": None, "conditional_hist": None, "unconditional_hist": None,
        }

    combined_min = float(min(result["conditional"].min(), result["unconditional"].min()))
    combined_max = float(max(result["conditional"].max(), result["unconditional"].max()))
    if combined_min == combined_max:
        # degenerate (zero-variance) series -- pad so np.histogram doesn't choke on a
        # zero-width range
        combined_min -= 1e-6
        combined_max += 1e-6
    bin_edges = np.linspace(combined_min, combined_max, n_bins + 1)

    return {
        "insufficient_data": False,
        "n_conditional": result["n_conditional"],
        "n_unconditional": result["n_unconditional"],
        "conditional_mean": result["conditional_mean"],
        "unconditional_mean": result["unconditional_mean"],
        "ks_stat": result["ks_stat"],
        "ks_p": result["ks_p"],
        "effect_size_sd": result["effect_size_sd"],
        "bin_edges": bin_edges.tolist(),
        "conditional_hist": binned_density(result["conditional"], bin_edges),
        "unconditional_hist": binned_density(result["unconditional"], bin_edges),
    }
