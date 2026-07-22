"""
Nightly orchestration: pull data, run the validated market-layer model, write outputs.

STATUS: market layer (SPY direction/vol/drift/smoothing), the per-name tilt layer (plan
section 3.5), and the nightly IV/ATM-vol snapshot (plan section 5 step 3) are all wired
end-to-end below. The per-name vol_state in the tilt layer STILL uses the realized-vol
proxy, not IV rank -- that swap is explicitly Phase 4 ("once >=6 months of snapshot
history exists"), not now. This run just starts accumulating that history.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from pipeline.data_pull import pull_prices, pull_vx_curve, build_feature_frame, pull_iv_snapshot
from pipeline.model import (
    walk_forward_direction, vol_layer, curve_conditioned_drift_posterior, commit_regime,
)
from pipeline.matrix import recommend_structure
from pipeline.tilt import relative_strength_zscore, direction_tilt, name_vol_state, compute_name_cell
from pipeline.forecast import compute_name_forecast_density

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
    # Diagnostic added 2026-07-22: as_of was stuck at 2026-05-22 across multiple nightly
    # runs despite the Action succeeding and committing fresh files each time. No
    # "dropped N/M cell_posterior rows" message ever appeared, which rules out our own
    # dropna logic truncating the tail -- so either the raw pull itself isn't reaching
    # past that date (a Yahoo/yfinance data-availability question, not a bug here), or
    # the VX curve is the thing capping it. This line settles which, next run.
    print(f"[run_nightly] raw_prices covers {raw_prices.index.min().date()} to "
          f"{raw_prices.index.max().date()} (source={source})")
    raw_prices = raw_prices.rename(columns={t: t.lstrip("^") for t in cfg["data"]["vix_tickers"]})
    prices = raw_prices[["SPY"] + [t.lstrip("^") for t in cfg["data"]["vix_tickers"]]]
    name_prices = raw_prices[[n for n in names if n in raw_prices.columns]]
    missing_names = [n for n in names if n not in raw_prices.columns]
    if missing_names:
        print(f"[run_nightly] {len(missing_names)} name(s) missing from price pull, skipped: {missing_names}")
    vx = pull_vx_curve(cfg["data"]["start_date"])
    print(f"[run_nightly] VX curve covers {vx.index.min().date()} to {vx.index.max().date()}")
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
        latest_date = cell_hist.index[-1]
        latest = cell_hist.iloc[-1]
        # Look up rs_z at the SAME date as the reported cell, not rs_z_short/long's own
        # positional tail (.iloc[-1]) -- those series run over r's raw (VX-curve-
        # independent) date range, which can extend past committed's VX-curve-gated
        # coverage. Using .iloc[-1] there reports whatever's at r's own later tail
        # (often NaN, since market_ret has no value past committed's range) against a
        # "cell" that's actually from an earlier, valid date -- a real mismatch caught
        # by every single name coming back null simultaneously in the first live run.
        rzs = rs_z_short.get(latest_date)
        rzl = rs_z_long.get(latest_date)
        per_name_latest[tk] = {
            "as_of": str(latest_date.date()),
            "cell": latest,
            "rs_z_short": float(rzs) if pd.notna(rzs) else None,
            "rs_z_long": float(rzl) if pd.notna(rzl) else None,
            "recommendation": recommend_structure(
                latest, latest_posterior, cfg["structure_matrix"], cfg["confidence_bands"],
            ),
        }

    # --- Per-name forecast density (plan section 6 drill-down) ------------------------
    # Precomputed for all 50 names here rather than left as a live-only app feature --
    # name_prices is already pulled above for the tilt layer's RS z-scores, so this adds
    # zero extra API calls. Same horizon as the drift model for consistency. One row per
    # ticker (a snapshot using ALL history to date), overwritten each run -- not an
    # appending time series like iv_snapshots.parquet, since it's a summary statistic,
    # not itself a date-indexed observation.
    forecast_rows = []
    for tk, cell_hist in name_cells_hist.items():
        current_cell = per_name_latest[tk]["cell"]
        fd = compute_name_forecast_density(
            name_prices[tk].dropna(), cell_hist, current_cell,
            horizon=dm["forward_horizon_days"],
        )
        forecast_rows.append({
            "date": str(committed.index[-1].date()), "ticker": tk, "current_cell": current_cell,
            **fd,
        })
    if forecast_rows:
        forecast_density_df = pd.DataFrame(forecast_rows)
        forecast_density_df.to_parquet(DATA_DIR / "forecast_density.parquet")
        n_ok = int((~forecast_density_df["insufficient_data"]).sum())
        print(f"[run_nightly] forecast density: {n_ok}/{len(forecast_density_df)} names "
              f"had >=20 matured conditional observations")

    # --- Nightly IV/ATM-vol snapshot (plan section 5 step 3) --------------------------
    # Batched, throttled, retry x2 per name (see data_pull.pull_iv_snapshot); a missing
    # name is logged and skipped, not a pipeline error -- unlike the VX curve, per-name
    # options-chain coverage isn't a first-class input to anything live yet. This just
    # accumulates data/iv_snapshots.parquet toward the Phase 4 IV-rank swap.
    iv_cfg = cfg.get("iv_snapshot", {})
    spot_prices = name_prices.iloc[-1].dropna().to_dict()
    iv_today = pull_iv_snapshot(
        list(spot_prices.keys()), spot_prices,
        risk_free=cfg["backtest"]["risk_free"],
        near_dte_target=iv_cfg.get("near_dte_target", 17),
        far_dte_range=tuple(iv_cfg.get("far_dte_range", [30, 45])),
        retries=iv_cfg.get("retries", 2),
        throttle_sec=iv_cfg.get("throttle_sec", 0.35),
    )
    iv_snapshot_path = DATA_DIR / "iv_snapshots.parquet"
    if not iv_today.empty:
        if iv_snapshot_path.exists():
            existing = pd.read_parquet(iv_snapshot_path)
            combined = pd.concat([existing, iv_today], ignore_index=True)
            combined = combined.drop_duplicates(subset=["date", "ticker"], keep="last")
        else:
            combined = iv_today
        combined.to_parquet(iv_snapshot_path)
    print(f"[run_nightly] IV snapshot: {len(iv_today)}/{len(spot_prices)} names covered")

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
        "iv_snapshot_coverage": f"{len(iv_today)}/{len(spot_prices)}",
        # TODO(Phase 4): swap per-name vol_state realized-vol proxy for IV rank once
        # iv_snapshots.parquet has >=6 months of history.
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
