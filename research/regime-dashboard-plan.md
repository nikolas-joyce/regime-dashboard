# Regime Dashboard — Build Plan v1

**Status:** Phase 0 CLOSED (2026-07-21) — gate PASSED on economically relevant criteria. Proceeding to Phase 1.
**Date:** 2026-07-21
**Tags:** #quant #alpha-research #signal #portfolio

## 1. Objective

Web dashboard (Streamlit Community Cloud, ~$0 cost) that classifies the current market regime into one of six cells — {bullish, neutral, bearish} × {low vol, high vol} — via Bayesian methods, forecasts next-day regime transition probabilities, and maps each of ~50 liquid S&P 500 names into the grid with a recommended options structure.

## 2. Locked decisions

| Decision | Choice |
|---|---|
| Direction engine | 3-state Gaussian HMM (primary) + Bayesian drift posterior (cross-check, shown side by side) |
| Architecture | Two-layer: market regime engine × per-name tilt (no per-name HMMs) |
| Compute split | Nightly GitHub Actions precomputes everything → commits parquet/JSON to repo; Streamlit app is a pure reader |
| Options data | On-demand chain fetch (drill-down only) + nightly ATM IV/slope/skew snapshot for all 50 names |
| Hosting | Streamlit Community Cloud (free) |
| Data | yfinance (prices + VIX complex ^VIX9D/^VIX/^VIX3M/^VVIX/^SKEW + chains), vix-utils (VIX futures term structure), FRED via fredapi (HY OAS `BAMLH0A0HYM2`, 10y–3m curve), CBOE equity put/call ratio, Stooq failover via pandas-datareader |
| Underlyings | SPY + QQQ + IWM (index layer) and top ~50 liquid S&P 500 components (name layer) |
| Collar context | Both modes — standalone entry vs. overlay on standing long — UI toggle |

## 3. Model specification

### 3.1 Market direction layer (3 states)

**Primary — HMM.** `hmmlearn` GaussianHMM, 3 states, fit on SPY daily log returns (10y rolling window, refit weekly in the nightly job). States sorted by state mean → {bearish, neutral, bullish} to prevent label swapping across refits. Output: filtered posterior P(state | data to date) and the 3×3 transition matrix.

**Cross-check — Bayesian drift model (curve-conditioned).** Bayesian linear model μ_t = α + β·slope_t on rolling SPY returns, where **slope = VX futures curve slope, (VX3 − VX1)/VX1 from vix-utils** (positive = contango, negative = backwardation), z-scored — this is Nikolas's researched dataset; spot-index ratios (VIX3M/VIX) are explicitly NOT the conditioning variable. Conjugate normal posterior over (α, β). Roll handling: use vix-utils constant-maturity weighting (or the roll calendar) so VX1 expiry noise doesn't leak into the slope in expiry week. Direction probabilities from the posterior predictive of forward drift given today's curve — per Nikolas's research, term structure is a material conditional-drift predictor, so backwardation directly suppresses P(bullish). Bands on P(μ > 0): > 0.65 bullish, < 0.35 bearish, else neutral. β posterior displayed as a live check on the curve's predictive strength. Shown beside the HMM with an agreement badge; disagreement = regime uncertainty → size down.

**Identifiability guardrail:** the curve enters expected drift and transitions (below) but NOT the HMM emission vector — the nowcast of what the regime *is* stays returns-only; the curve conditions where it's *going*.

### 3.2 Market vol layer (2 states)

Mostly observable — light Bayesian smoothing only. High-vol score from three features:

1. VIX level > its 1y 70th percentile
2. Term structure: VIX > VIX3M (backwardation), with VIX9D/VIX ratio as the fast confirm; vix-utils futures curve as secondary
3. SPY 20d realized vol > its 1y 80th percentile

Logistic combination → P(high vol). Credit spread (HY OAS 20d change) and put/call ratio enter as slow confirmers, not triggers.

### 3.3 Anti-flip-flop (both layers)

Regime switch requires posterior of the new regime > 0.70 for 2 consecutive days, plus min-dwell of 3 trading days after any switch. Posteriors always displayed raw; the *committed* regime is the smoothed one.

### 3.4 Next-day forecast

Not a point label — a 6-cell probability vector. Direction: current filtered HMM state × transition matrix row, with transitions conditioned on the VX futures curve slope (VX1–VX3, per 3.1). Phase 2 implementation: static matrix plus a logistic curve-tilt on the forecast vector (backwardation shifts mass toward bearish exits). Phase 2b (gated on step-0 evidence): full time-varying-transition HMM — custom forward filter where exit hazards are logistic in curve slope, calibrated on labeled history (hmmlearn doesn't support covariate transitions). Vol: empirical 2×2 transition matrix from smoothed vol-regime history. Combined under an independence assumption (documented limitation; check empirically in validation). Alert state when P(exit current cell) > 0.25.

### 3.5 Per-name tilt layer (deliberately simple)

For each of the 50 names:

- **Direction tilt:** z-score of 21d and 63d relative strength vs. SPY. z > +0.5 bullish tilt, z < −0.5 bearish, else inherits market direction.
- **Vol state:** name's 20d realized vol percentile within its own 1y history (proxy for IV rank until snapshot history accumulates). > 60th pct = high vol cell.

Name cell = (market direction adjusted by tilt) × (name vol state). Market regime gates the structure *family*; the name only tilts within it — a bearish market regime caps any name at neutral regardless of relative strength.

## 4. Regime × structure matrix ⚠ NEEDS NIKOLAS REVIEW

Standalone-entry mode. Confidence = smoothed posterior of the name's assigned cell. High-vol cells sell premium; low-vol cells buy it.

| Cell | High confidence (>0.75) | Moderate (0.55–0.75) | Low (<0.55) |
|---|---|---|---|
| Bullish / high vol | Short put (~30Δ) | Bull put credit spread (30/15Δ) | No trade |
| Bullish / low vol | Bull call debit spread (55/30Δ) | Bull call debit spread, smaller | No trade |
| Bearish / high vol | Bear call credit spread (30/15Δ) | Bear call credit spread, smaller | No trade |
| Bearish / low vol | Bear put debit spread (55/30Δ) | Bear put debit spread, smaller | No trade |
| Neutral / high vol | Wide bull put credit spread (20/10Δ), reduced size | No trade | No trade |
| Neutral / low vol | No trade | No trade | No trade |

Overlay mode (standing long, toggle): bullish cells → bullish collar (far call cap, e.g. 15Δ call / 25Δ put); bearish cells → bearish collar (tight cap, 35Δ call / 30Δ put); neutral high vol → standard collar; neutral low vol → no hedge (premium too cheap to sell, too pointless to buy).

Default DTE 30–45 across all structures. All parameters (deltas, DTE, confidence bands) live in one `config.yaml`.

Open items for review: (a) short put only at high confidence, or acceptable at moderate with half size? (b) neutral/high-vol as bull-put rather than symmetric premium sale — intentional long bias? (c) delta targets vs. your putspread v3 conventions.

**bear_hi redesign — tested and REJECTED (2026-07-21).** The original bear_call_spread bets against test1c's finding that bear_hi averages unusually strong forward returns (likely capitulation/washout-bounce dynamics). Nikolas proposed a risk-defined convex alternative — call ratio backspread (1x2, 40Δ short/20Δ long) — tested in isolation against the original structure (TEST 5c, n=12 paired bear_hi trades). Result: the backspread underperformed on EVERY metric — mean/trade (+1.53% vs -1.11%), hit rate (83% vs 58%), worst case (-1.44% vs -7.01%), and even best case (+2.85% vs +1.41%). Diagnosis: the far-OTM (20Δ) long legs need a much larger/faster move than SPY's actual ~2-4% average 21d bounce to overcome their cost — the instrument was calibrated for a bigger tail event than what historically occurs. bear_hi stays as bear_call_spread. If revisited later, a closer-to-the-money bull call debit spread or bull put credit spread (needs "no crash," not "big rally") would fit the actual move magnitude better — but not worth further testing against a 12-trade sample now.

## 5. Data pipeline (GitHub Actions, nightly)

Cron ~22:30 UTC weekdays. Steps:

1. Pull EOD prices: 50 names + SPY/QQQ/IWM + VIX complex (yfinance with curl_cffi session; on failure → Stooq failover; on partial failure → carry forward last good with staleness flag)
2. Pull FRED series (fredapi, key in Actions secrets), CBOE put/call CSV, vix-utils VX futures curve — VX1…VX3 settlements + constant-maturity series; the VX1–VX3 slope is a first-class model input (drift model + transition conditioning), so a failed vix-utils pull is a pipeline error, not a soft skip
3. IV snapshot: per name, fetch chain, extract ATM IV (30–45 DTE), IV term slope, 25Δ put−call skew → append to `data/iv_snapshots.parquet` (batched, throttled, retry ×2; missing names logged not fatal)
4. Fit/update models, apply smoothing, compute forecasts and per-name cells + recommendations
5. Write `data/*.parquet` + `output/state.json`, commit to repo

App reads raw files from the repo. Live chain fetch happens only in drill-down, client-triggered, with Tradier sandbox as documented fallback if Yahoo blocks Streamlit Cloud IPs.

## 6. Dashboard layout (Streamlit, single page + tabs)

**Header — market regime card.** 3×2 grid with current cell highlighted, posterior bars per cell, HMM-vs-drift agreement badge, days-in-regime counter.

**Transition panel.** Next-day 6-cell probability bars; amber alert when exit probability > 0.25.

**Market internals strip.** VIX complex + term structure chart, HY OAS, put/call, SPY with regime ribbon (colored background by historical committed regime).

**Names table (main).** 50 rows: name, tilt z-scores, vol percentile, assigned cell, recommended structure, confidence, ATM IV (latest snapshot). Sortable/filterable by cell and structure. Standalone/overlay toggle switches the recommendation column.

**Drill-down (click a name).** Price chart with name-level regime ribbon; button fetches live chain → concrete strikes per the matrix, mid premium, max loss/gain, breakeven; IV snapshot history sparkline (grows over time). Per-name call history log (date, committed cell, posterior, tilt inputs, realized forward return once matured). Forecast density panel: empirical forward-return distribution conditional on the name's current cell/tilt, overlaid against the name's unconditional full-history empirical distribution (KDE or histogram) — same KS-test/effect-size methodology as Phase 0's TEST 1c, re-run live per name as an ongoing out-of-sample check rather than a one-time gate. Empirical null only (no parametric baseline) per 2026-07-21 decision.

**History tab.** SPY regime ribbon full-history, per-regime forward return/vol stats from validation, regime duration distribution.

**Diagnostics tab.** Model agreement over time, data freshness per source, IV snapshot coverage, last pipeline run status.

## 7. Step 0 — validation gate (before any UI code)

Notebook: label 2005→present with the full market-layer engine (walk-forward, no lookahead: fit on data ≤ t only). Then:

1. Forward 5d and 21d SPY return and realized-vol distributions per cell — require economically meaningful separation (report effect sizes, KS tests)
2. Empirical transition matrices — check the direction×vol independence assumption
3. Approximate matrix backtest: Black-Scholes-priced structures using realized vol as IV proxy (documented limitation), vs. always-on bull-put baseline and buy-hold
4. Flip-flop audit: count regime switches with/without smoothing
5. Curve-conditioning test: next-day regime hit rate and forward-drift R² with vs. without term-structure conditioning (static matrix vs. curve-tilted vs. TVTP) — decides whether Phase 2b is built. Primary slope = VX1–VX3 futures (vix-utils); spot ratios (VIX3M/VIX, VIX9D/VIX) run as robustness comparators only.

**Gate:** if cells don't separate forward distributions, we stop and redesign the classifier — no dashboard gets built on a non-signal.

## 8. Risks and failure modes

- **yfinance breakage / IP blocks** — #1 operational risk. Mitigated: Actions-side pulls, curl_cffi, Stooq failover, staleness flags surfaced in UI.
- **HMM label swapping / refit instability** — states sorted by mean; weekly (not daily) refits; drift posterior as sanity check.
- **No historical IV** — realized-vol percentile proxy now; own snapshot history accumulates; DoltHub `post-no-preference/options` available for later validation of the credit/debit logic.
- **Streamlit Community Cloud sleep** — app sleeps after inactivity; acceptable (personal tool), wakes in ~30s.
- **Independence assumption in forecast** — tested empirically in step 0; replace with joint empirical matrix if violated.
- **Repo-as-database growth** — parquet appends are KBs/day; revisit if repo exceeds ~500MB (years away).

## 7b. Step 0 results (as actually run, 2026-07-21)

Several rounds of live validation surfaced real issues and refined the design beyond what section 7 anticipated:

**HMM stability.** v1's independent per-refit label sorting caused sign-inverted forward-return effects (label swaps across the 63d refit windows). Fixed via nearest-previous-mean continuity matching. An interim z-score separation gate (v2) overcorrected into freezing the model on its first (pre-GFC) fit for the entire 20-year run — worse than the bug it replaced (backtest worst-case drawdown -15% vs -4%). Reverted to diagnostic-only z-tracking (v3); stable at n_refit=64/64 across every subsequent run, median z-separation ~0.90.

**Separation criterion redesigned.** The marginal bull-vs-bear test (collapsing the vol axis) is unreliable — collapsing hi/lo vol buckets together produces misleading sign flips even when cells separate sensibly within a vol bucket. Replaced with vol-conditional pairwise tests + an omnibus Kruskal-Wallis as the primary gate criterion (test1c). Gate: **PASSED** (separation=true, matrix Sharpe 2.93 beats always-on baseline's 2.35, reproduced across every live rerun).

**The bull_hi/bear_hi finding (real, not a bug).** A well-powered, reproducible finding survived every fix: at 21d, `bull_hi` shows lower forward returns than `bear_hi` (~-1.08sd). Cross-sectional check (20 liquid names, step0b) confirmed this generalizes broadly — NOT SPY-specific — with 15/19 names negative, several exceeding SPY's own reading (CVX -1.06sd, MSFT -0.80sd). Reframed on closer inspection: `bull_hi`'s own forward returns are still solidly positive; `bear_hi` is unusually rewarding (plausibly capturing violent oversold bounces), which is what drives the effect. `bear_lo` also generalizes in the opposite direction from the SPY reading — occurs in 18/20 names (SPY's zero occurrences was purely an index artifact; single names can grind down quietly in a way a broad index rarely does).

**Two candidate fixes tested and NOT adopted:**
- *Direction-conditional vol thresholds* (rank vol relative to the current direction state's own history, addressing the equity leverage effect where bear periods are baseline more volatile): tested live, did not shrink the effect (SPY 21d effect unchanged at -1.08sd) and cost backtest performance (Sharpe 2.93→2.64). Reverted.
- *Curve-gating* (skip bullish trades when VX1-VX3 is backwardated, hypothesizing backwardation flags bear-market relief rallies): tested live, found the OPPOSITE sign — backwardated `bull_hi` outperformed contango `bull_hi` (+3.14% vs +1.35% at 21d) — and the gate hurt the backtest (Sharpe 2.71 vs 2.93 ungated). Likely dominated by a handful of autocorrelated crisis-recovery episodes (2009/2020/2022), not a robust pattern. Not implemented, in either direction.

**Decision:** ship the structure matrix ungated (no curve or vol-conditional filter) — it already beats the always-on baseline on the metric that matters. Section 4's open calibration items (short-put confidence threshold, neutral/high-vol long bias, delta targets) remain unresolved and are Phase 2 work, not blockers.

Notebooks: `regime_dashboard_step0_validation.ipynb` (SPY, market layer), `regime_dashboard_step0b_crosssectional.ipynb` (20-name cross-sectional confirmation).

## 9. Execution phases

1. **Phase 0:** validation notebook (gate) — deliver regime-separation evidence
2. **Phase 1:** repo scaffold, `config.yaml`, data pipeline modules + GitHub Actions workflow, first successful nightly run
3. **Phase 2:** model module (HMM + drift + vol layer + smoothing + forecast) with unit tests on synthetic data
4. **Phase 3:** Streamlit app, deploy to Community Cloud
5. **Phase 4 (later):** swap realized-vol proxy → own IV-rank history once ≥6 months of snapshots; optional Tradier integration

Outputs are research signals, not trade instructions; execution stays discretionary.
