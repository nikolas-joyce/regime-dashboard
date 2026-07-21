"""
Nightly orchestration: pull data, run the validated market-layer model, write outputs.

STATUS: market layer (SPY direction/vol/drift/smoothing) and the per-name tilt layer
(plan section 3.5) are both wired end-to-end below. NOT YET IMPLEMENTED (Phase 1
remaining work):
  - nightly IV/ATM-vol snapshot (plan section 5 step 3) -- name vol_state below uses
    the realized-vol proxy, not real IV rank
  - expansion from the 20-name validated basket to the full ~50-name universe
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from pipeline.data_pull import pull_prices, pull_vx_curve, build_feature_frame
from pipeline.model import (
    walk_forward_direction, vol_layer, curve_conditioned_drift_posterior, commit_regime,
)
from pipeline.matrix import recommend_structure
from pipeline.tilt import relative_strength_zscore, direction_tilt, name_vol_state, compute_name_cell

HERE = Path(__file__).resolve().parent.parent
DATA_DIR = HERE / "data"
OUTPUT_DIR = HERE / "output"


def load_config() -> dict:
    with open(HERE / "config.yaml") as fh:
        return yaml.safe_load(fh)


def run() -> dict:
    cfg = load_config()
    DATA_DIR.mkdir(exist_ok=True)
    OUTPUT_DIR.mkdir(exist_ok=True)

    names = cfg["data"]["underlyings_names"]
    tickers = ["SPY"] + [t for t in cfg["data"]["vix_tickers"]] + names
    raw_prices, source = pull_prices(tickers, cfg["data"]["start_date"])
    raw_prices = raw_prices.rename(columns={t: t.lstrip("^") for t in cfg["data"]["vix_tickers"]})
    prices = raw_prices[["SPY"] + [t.lstrip("^") for t in cfg["data"]["vix_tickers"]]]
    name_prices = raw_prices[[n for n in names if n in raw_prices.columns]]
    missing_names = [n for n in names if n not in raw_prices.columns]
    if missing_names:
        print(f"[run_nightly] {len(missing_names)} name(s) missing from price pull, skipped: {missing_names}")
    vx = pull_vx_curve(cfg["data"]["start_date"])
    feat = build_feature_frame(prices, vx)

    de = cfg["direction_engine"]
    dirpost, transition_mats, diagnostics = walk_forward_direction(
        feat["ret"], min_train=de["min_train"], refit_every=de["refit_every"], n_states=de["n_states"],
    )

    vl = cfg["vol_layer"]
    vf = feat.reindex(dirpost.index)
    p_high = vol_layer(vf["vix_pct"], vf["backwardation"], vf["rv_pct"], vl["weights"], vl["logistic_scale"])

    dm = cfg["drift_model"]
    fwd = feat["ret"].rolling(dm["forward_horizon_days"]).sum().shift(-dm["forward_horizon_days"])
    fwd = fwd.reindex(dirpost.index)
    slope_z = feat["slope_z"].reindex(dirpost.index)
    p_up, beta_slope = curve_conditioned_drift_posterior(slope_z, fwd)

    cells = ["bear_hi", "bear_lo", "neut_hi", "neut_lo", "bull_hi", "bull_lo"]
    cp = pd.DataFrame(index=dirpost.index, columns=cells, dtype=float)
    for i, d in enumerate(["p_bear", "p_neut", "p_bull"]):
        cp[cells[2 * i]] = dirpost[d] * p_high
        cp[cells[2 * i + 1]] = dirpost[d] * (1 - p_high)

    # dirpost.dropna() (model.py) only guarantees the direction posterior itself is valid --
    # it says nothing about vix_pct/backwardation/rv_pct/slope_z for those same dates. When
    # the VX1-VX3 curve history (vix_utils) doesn't reach as far back as SPY's return history,
    # p_high is NaN for those early dates, which wipes out every column of that cp row (all six
    # cells multiply by the same p_high or 1-p_high) and idxmax(axis=1) raises on an all-NaN
    # row. Drop those rows here rather than in commit_regime -- the validated model.py logic
    # assumes it's only ever handed complete rows; enforcing that is this orchestration layer's
    # job, not model.py's.
    n_before = len(cp)
    cp = cp.dropna(how="any")
    n_dropped = n_before - len(cp)
    if n_dropped:
        print(f"[run_nightly] dropped {n_dropped}/{n_before} cell_posterior rows with "
              f"NaN inputs (likely VX curve history not covering that far back)")
    if cp.empty:
        raise RuntimeError(
            "cell_posterior is empty after dropping NaN rows -- vol-layer inputs "
            "(vix_pct/backwardation/rv_pct) never aligned with the direction posterior. "
            "Check VX curve / VIX data pull coverage."
        )

    sm = cfg["smoothing"]
    committed = commit_regime(cp, sm["commit_p"], sm["commit_days"], sm["min_dwell"])

    latest_cell = committed.iloc[-1]
    latest_posterior = float(cp.iloc[-1].max())
    recommendation = recommend_structure(
        latest_cell, latest_posterior, cfg["structure_matrix"], cfg["confidence_bands"],
    )

    # --- Per-name tilt layer (plan section 3.5) ---------------------------------------
    # Market regime gates the structure family; each name only tilts within it. Confidence
    # tiering for the per-name recommendation reuses the MARKET layer's posterior, not a
    # name-specific one -- the tilt layer is z-score-based, not Bayesian, and the plan
    # doesn't specify a separate name-level confidence measure. Flagged in config.yaml's
    # open_calibration_items; not a validated choice.
    tl = cfg["tilt_layer"]
    name_ret = np.log(name_prices).diff()
    name_rv20 = name_ret.rolling(vl["rv_window"]).std() * np.sqrt(252)
    market_ret = feat["ret"]

    name_cells_hist = {}
    per_name_latest = {}
    for tk in name_ret.columns:
        r = name_ret[tk].dropna()
        if len(r) < max(tl["rs_window_long"], tl["name_vol_pct_window"]) + tl["rs_z_window"]:
            continue  # not enough history for this name yet -- skip rather than emit noise
        rs_z_short = relative_strength_zscore(r, market_ret, tl["rs_window_short"], tl["rs_z_window"])
        rs_z_long = relative_strength_zscore(r, market_ret, tl["rs_window_long"], tl["rs_z_window"])
        tilt = direction_tilt(rs_z_short, rs_z_long, tl["rs_bull_thresh"], tl["rs_bear_thresh"])
        vstate = name_vol_state(name_rv20[tk], tl["name_vol_pct_window"], tl["name_vol_pct_thresh"])
        cell_hist = compute_name_cell(committed, tilt, vstate).dropna()
        if cell_hist.empty:
            continue
        name_cells_hist[tk] = cell_hist
        latest = cell_hist.iloc[-1]
        per_name_latest[tk] = {
            "as_of": str(cell_hist.index[-1].date()),
            "cell": latest,
            "rs_z_short": float(rs_z_short.iloc[-1]) if pd.notna(rs_z_short.iloc[-1]) else None,
            "rs_z_long": float(rs_z_long.iloc[-1]) if pd.notna(rs_z_long.iloc[-1]) else None,
            "recommendation": recommend_structure(
                latest, latest_posterior, cfg["structure_matrix"], cfg["confidence_bands"],
            ),
        }

    state = {
        "as_of": str(committed.index[-1].date()),
        "data_source": source,
        "market_regime": latest_cell,
        "posterior": latest_posterior,
        "hmm_diagnostics": diagnostics,
        "drift_p_up_latest": float(p_up.iloc[-1]) if not np.isnan(p_up.iloc[-1]) else None,
        "curve_beta_latest": float(beta_slope.iloc[-1]) if not np.isnan(beta_slope.iloc[-1]) else None,
        "recommendation": recommendation,
        "per_name": per_name_latest,
        # TODO(Phase 1 remaining): iv_snapshot, 50-name universe expansion
    }

    dirpost.to_parquet(DATA_DIR / "dirpost.parquet")
    cp.to_parquet(DATA_DIR / "cell_posterior.parquet")
    committed.to_frame().to_parquet(DATA_DIR / "committed_regime.parquet")
    if name_cells_hist:
        # Long-format history (date, ticker, cell) -- appendable, and the exact shape the
        # planned Phase 3 per-name call-history log / forecast-density panel needs.
        name_cells_df = pd.concat(
            [s.rename("cell").to_frame().assign(ticker=tk) for tk, s in name_cells_hist.items()]
        ).reset_index().rename(columns={"index": "date"})
        name_cells_df.to_parquet(DATA_DIR / "name_cells.parquet")
    with open(OUTPUT_DIR / "state.json", "w") as fh:
        json.dump(state, fh, indent=2, default=str)
    return state


if __name__ == "__main__":
    result = run()
    print(json.dumps(result, indent=2, default=str))
