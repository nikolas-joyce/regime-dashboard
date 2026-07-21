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
  - [x] `.github/workflows/nightly.yml` -- skeleton, untested against a real remote yet
  - [ ] Per-name tilt layer (plan section 3.5) -- NOT YET BUILT
  - [ ] Nightly IV/ATM-vol snapshot (plan section 5 step 3) -- NOT YET BUILT
  - [ ] Expansion from the 20-name validated basket to the full ~50-name universe
  - [ ] First successful live GitHub Actions run
- **Phase 2:** unit tests on synthetic data, resolve open calibration items (see below)
- **Phase 3:** Streamlit app -- not started

## Validation history (important -- read before changing the model)

Three plausible-looking refinements were tested empirically this session and **rejected**:
direction-conditional vol thresholds, curve-based gating on bullish cells, and a call
ratio backspread for `bear_hi`. All three failed on live data despite sound-seeming
economic rationale. See `research/regime-dashboard-plan.md` section 7b for the full
history and why each was rejected -- don't re-propose these without new evidence.

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
