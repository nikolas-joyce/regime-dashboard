# Regime Dashboard

Bayesian market regime classifier (direction x volatility, 6 cells) with next-day forecast
and an options structure recommendation per cell. Full design doc and validation history:
`research/regime-dashboard-plan.md` (mirrored from the Second Brain vault).

## Status: Phase 1 in progress

- **Phase 0 (validation gate): CLOSED.** The market-layer model (HMM direction engine,
  curve-conditioned drift model, vol layer, smoothing, forecast) is live-validated on SPY
  and cross-sectionally confirmed on a 20-name liquid basket. Gate passes on the criteria
  that matter economically: the structure matrix beats an always-on baseline on Sharpe
  (2.93 vs 2.35), reproduced across every live rerun.
- **Phase 1 (this repo): in progress.**
  - [x] Repo scaffold, `config.yaml` (validated parameters), `requirements.txt`
  - [x] `pipeline/model.py` -- direction engine, vol layer, drift model, smoothing (ported from validated notebook)
  - [x] `pipeline/data_pull.py` -- yfinance + Stooq failover, VX curve, FRED
  - [x] `pipeline/matrix.py` -- structure matrix + Black-Scholes pricing helpers
  - [x] `pipeline/run_nightly.py` -- orchestration (SPY market layer only, end-to-end)
  - [x] `.github/workflows/nightly.yml` -- live-tested, secret + permissions configured
  - [x] First successful live GitHub Actions run (2026-07-21, run #5, commit `bc078b3`)
  - [ ] Per-name tilt layer (plan section 3.5) -- NOT YET BUILT
  - [ ] Nightly IV/ATM-vol snapshot (plan section 5 step 3) -- NOT YET BUILT
  - [ ] Expansion from the 20-name validated basket to the full ~50-name universe
- **Phase 2:** unit tests on synthetic data, resolve open calibration items (see below)
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

## Open calibration items (Nikolas sign-off needed)

1. `bull_hi` short_put: high-confidence only, or acceptable at moderate with half size?
2. `neut_hi`'s bull-put bias -- intentional, or should it be symmetric?
3. Delta targets vs. putspread v3 conventions

## Setup

```bash
pip install -r requirements.txt
python -m pipeline.run_nightly   # requires FRED_API_KEY env var
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
