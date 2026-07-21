# Regime Dashboard

Bayesian market regime classifier (direction x volatility, 6 cells) with next-day forecast
and an options structure recommendation per cell. Full design doc and validation history:
`research/regime-dashboard-plan.md` (mirrored from the Second Brain vault).

## Status: Phase 2 CLOSED (2026-07-22)

- **Phase 0 (validation gate): CLOSED.** The market-layer model (HMM direction engine,
  curve-conditioned drift model, vol layer, smoothing, forecast) is live-validated on SPY
  and cross-sectionally confirmed on a 50-name liquid basket. Gate passes on the criteria
  that matter economically: the structure matrix beats an always-on baseline on Sharpe
  (2.93 vs 2.35), reproduced across every live rerun.
- **Phase 1 (this repo): CLOSED.** Every piece below is live-verified against real data,
  not just sandbox-tested -- including the IV snapshot's first live run against actual
  Yahoo options chains (50/50 name coverage, realistic ATM IV/skew/term-slope values,
  graceful null-propagation on the one name with an unusable near-term chain).
  - [x] Repo scaffold, `config.yaml` (validated parameters), `requirements.txt`
  - [x] `pipeline/model.py` -- direction engine, vol layer, drift model, smoothing (ported from validated notebook)
  - [x] `pipeline/data_pull.py` -- yfinance + Stooq failover, VX curve, FRED
  - [x] `pipeline/matrix.py` -- structure matrix + Black-Scholes pricing helpers
  - [x] `pipeline/run_nightly.py` -- orchestration (SPY market layer only, end-to-end)
  - [x] `.github/workflows/nightly.yml` -- live-tested, secret + permissions configured
  - [x] First successful live GitHub Actions run (2026-07-21, run #5, commit `bc078b3`)
  - [x] Per-name tilt layer (`pipeline/tilt.py`, plan section 3.5) -- live-verified run #7,
        commit `6313399`: market-gating confirmed correct on real data (bear-tilted names
        capped to neut under the live bull_lo market regime, bull-tilted names passed
        through uncapped)
  - [x] Nightly IV/ATM-vol snapshot (`pipeline/iv_calc.py` + `data_pull.pull_iv_snapshot`,
        plan section 5 step 3): ATM IV at near-term + 30-45 DTE, IV term slope, 25-delta
        put/call skew. Batched/throttled/retry x2, missing names logged not fatal.
        Accumulates `data/iv_snapshots.parquet`, deduped on (date, ticker). Live-verified
        2026-07-21: 50/50 name coverage against real Yahoo chains, realistic IV/skew
        values (ATM IV 23-60%, skew mostly +/-0.03-0.06, correct expiry selection --
        17/38 DTE across every name). NOT yet wired
        into the per-name vol_state (still realized-vol proxy) -- that's Phase 4, once
        >=6 months of snapshot history exists, per plan section 9.
  - [x] Expansion from the 20-name validated basket to the full 50-name universe
        (`research/regime_dashboard_step0c_expansion.ipynb`, 2026-07-21): 30 new names
        confirmed consistent with the established findings -- bear_lo in 29/30 (vs
        step0b's 18/20), bull_hi-vs-bear_hi reversal negative in 22/29 measurable names
        (76%, median -0.37sd vs SPY's -1.08sd)
- **Phase 2: CLOSED.** 48 unit tests across `tests/test_model.py` (12), `test_tilt.py` (10),
  `test_matrix.py` (11), `test_iv_calc.py` (15) -- all passing together in one `pytest`
  run, committed `866673c`. `config.yaml`'s four open calibration items resolved
  2026-07-22 (Nikolas: "go with your instincts... and proceed") -- see the RESOLVED
  block above `structure_matrix` in `config.yaml` for the rationale on each. No
  `structure_matrix` values changed; this was a documentation/audit-trail resolution,
  not a recalibration.
- **Phase 3:** Streamlit app -- not started

## Validation history (important -- read before changing the model)

Three plausible-looking refinements were tested empirically in Phase 0 and **rejected**:
direction-conditional vol thresholds, curve-based gating on bullish cells, and a call
ratio backspread for `bear_hi`. All three failed on live data despite sound-seeming
economic rationale. See `research/regime-dashboard-plan.md` section 7b for the full
history and why each was rejected -- don't re-propose these without new evidence.

### Phase 1 hardening (2026-07-21, first live Actions run)

The first live run surfaced two real bugs in the Phase 1 port -- neither was a defect
in the validated Phase 0 notebook logic itself, but in how the orchestration layer wired
data into it:

1. `data_pull.build_feature_frame` used `prices.index` + `ffill()` to align the VX1-VX3
   curve with price history, then only dropped NaN rows on `ret`/`rv20`. The notebook's
   equivalent step drops rows missing `slope_z`/`vix_pct`/`rv_pct` too, and its raw data
   comes from an inner join with the VX curve (`px.join(vx, how="inner")`), so it never
   hands the model a leading-NaN-slope window. Fixed at the source in `data_pull.py` to
   match. This was the root cause of both symptoms below.
2. Symptom A: `commit_regime`'s `idxmax(axis=1)` raised on an all-NaN `cp` row (every
   cell in a row shares the same `p_high`, so one NaN `p_high` nulls the whole row).
   Symptom B: `curve_conditioned_drift_posterior` masked NaN in `y` only, not `X`, so a
   leading-NaN `slope_z` row silently contaminated `bn`/`Vn` with NaN for the rest of the
   walk-forward series (`drift_p_up_latest`/`curve_beta_latest` came back `null`).
   Both are now also fixed defensively in `model.py`/`run_nightly.py` independent of #1.

Verified TEST 4 ("curve conditioning adds value") is **not** affected by bug #2 --
checked the notebook's actual feature-construction cell, which never had a NaN-slope
window to contaminate in the first place.

## Setup

```bash
pip install -r requirements.txt        # prod (nightly Action)
pip install -r requirements-dev.txt    # prod + pytest, for running tests/
python -m pipeline.run_nightly         # requires FRED_API_KEY env var
pytest tests/                          # 48 tests, ~20s
```

## Repo layout

```
config.yaml              # all model parameters -- don't hand-tune without re-running research/ notebooks
pipeline/
  data_pull.py           # yfinance/Stooq/vix-utils/FRED acquisition
  model.py               # direction engine, vol layer, drift model, smoothing
  matrix.py              # regime -> structure lookup + BS pricing
  run_nightly.py          # orchestration entrypoint
research/                 # validated notebooks + the full design/validation doc
data/, output/            # written by run_nightly.py, committed by the nightly Action
.github/workflows/        # nightly.yml
```
