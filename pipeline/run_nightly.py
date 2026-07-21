"""
Nightly orchestration: pull data, run the validated market-layer model, write outputs.

STATUS: market layer (SPY direction/vol/drift/smoothing) is wired end-to-end below and
reuses only validated logic from model.py. NOT YET IMPLEMENTED (Phase 1 remaining work):
  - per-name tilt layer (plan section 3.5 -- RS z-score + vol percentile per name)
  - nightly IV/ATM-vol snapshot (plan section 5 step 3)
  - expansion from the 20-name validated basket to the full ~50-name universe
Running this script today produces a correct market regime + forecast for SPY; it does
not yet produce per-name cells or structure recommendations for config.yaml's
underlyings_names list.
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

    tickers = ["SPY"] + [t for t in cfg["data"]["vix_tickers"]]
    prices, source = pull_prices(tickers, cfg["data"]["start_date"])
    prices = prices.rename(columns={t: t.lstrip("^") for t in cfg["data"]["vix_tickers"]})
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

    sm = cfg["smoothing"]
    committed = commit_regime(cp, sm["commit_p"], sm["commit_days"], sm["min_dwell"])

    latest_cell = committed.iloc[-1]
    latest_posterior = float(cp.iloc[-1].max())
    recommendation = recommend_structure(
        latest_cell, latest_posterior, cfg["structure_matrix"], cfg["confidence_bands"],
    )

    state = {
        "as_of": str(committed.index[-1].date()),
        "data_source": source,
        "market_regime": latest_cell,
        "posterior": latest_posterior,
        "hmm_diagnostics": diagnostics,
        "drift_p_up_latest": float(p_up.iloc[-1]) if not np.isnan(p_up.iloc[-1]) else None,
        "curve_beta_latest": float(beta_slope.iloc[-1]) if not np.isnan(beta_slope.iloc[-1]) else None,
        "recommendation": recommendation,
        # TODO(Phase 1 remaining): per_name_cells, per_name_recommendations, iv_snapshot
    }

    dirpost.to_parquet(DATA_DIR / "dirpost.parquet")
    cp.to_parquet(DATA_DIR / "cell_posterior.parquet")
    committed.to_frame().to_parquet(DATA_DIR / "committed_regime.parquet")
    with open(OUTPUT_DIR / "state.json", "w") as fh:
        json.dump(state, fh, indent=2, default=str)
    return state


if __name__ == "__main__":
    result = run()
    print(json.dumps(result, indent=2, default=str))
