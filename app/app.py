"""
Regime Dashboard -- Streamlit app (Phase 3, plan section 6).

Pure reader over the nightly Action's committed outputs (data/*.parquet, output/state.json)
for everything precomputed; live yfinance fetch only inside drill-down, client-triggered,
per the plan's design. See app/data.py's module docstring for the precomputed/live split.

Run locally: streamlit run app/app.py   (from repo root, so pipeline/ and config.yaml resolve)
"""
from __future__ import annotations

import sys
from pathlib import Path

# Streamlit Cloud (and `streamlit run app/app.py` generally) inserts the SCRIPT's own
# directory (app/) into sys.path, not the invocation cwd/repo root -- unlike a plain
# `python3 -c "..."` run from the repo root, which is what made this work in local
# sandbox testing (AppTest via python3 -c has the repo root on sys.path implicitly) and
# silently masked the bug until the first real Streamlit Cloud deploy. Without this, `app`
# is not importable as a package from inside its own directory ("No module named 'app.data';
# 'app' is not a package"), and neither is the sibling `pipeline` package. Insert the repo
# root explicitly, before any app.*/pipeline.* imports below, so this works regardless of
# how the runner sets sys.path[0].
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from datetime import datetime, date

import numpy as np
import pandas as pd
import streamlit as st

from app.data import (
    load_config, load_state, load_cell_posterior, load_committed_regime, load_dirpost,
    load_name_cells, load_iv_snapshots, name_cell_history, name_iv_history,
    fetch_price_history, fetch_option_expiries, fetch_option_chain, CELLS,
)
from app.analytics import (
    empirical_transition_matrix, next_day_probs, exit_probability, regime_runs,
    days_in_current_regime, conditional_vs_unconditional_density, call_history_log,
    structure_terms,
)
from pipeline.matrix import recommend_structure, confidence_tier

st.set_page_config(page_title="Regime Dashboard", layout="wide")

CELL_LABELS = {
    "bull_hi": "Bull / High Vol", "bull_lo": "Bull / Low Vol",
    "neut_hi": "Neutral / High Vol", "neut_lo": "Neutral / Low Vol",
    "bear_hi": "Bear / High Vol", "bear_lo": "Bear / Low Vol",
}
CELL_GRID = [["bear_hi", "neut_hi", "bull_hi"], ["bear_lo", "neut_lo", "bull_lo"]]


def freshness_badge(as_of_str: str) -> str:
    try:
        as_of = datetime.strptime(as_of_str, "%Y-%m-%d").date()
    except (ValueError, TypeError):
        return "unknown as_of"
    days_stale = (date.today() - as_of).days
    if days_stale <= 1:
        return f"fresh (as of {as_of_str})"
    return f"STALE -- {days_stale}d old (as of {as_of_str}, next run may not have landed)"


def render_header(state: dict, committed: pd.Series):
    st.title("Regime Dashboard")
    c1, c2, c3 = st.columns(3)
    c1.metric("Data freshness", freshness_badge(state.get("as_of", "")))
    c1.caption(f"Source: {state.get('data_source', '?')} | IV coverage: {state.get('iv_snapshot_coverage', '?')}")
    c2.metric("Days in current regime", days_in_current_regime(committed))
    hmm_diag = state.get("hmm_diagnostics", {})
    c3.metric("HMM refits", hmm_diag.get("n_refit", "?"),
              help=f"median z-separation: {hmm_diag.get('median_z_separation', float('nan')):.2f}")


def render_regime_card(state: dict, cell_posterior: pd.DataFrame):
    st.subheader("Market regime")
    current = state.get("market_regime", "?")
    latest_post = cell_posterior.iloc[-1] if not cell_posterior.empty else pd.Series(dtype=float)

    for row in CELL_GRID:
        cols = st.columns(3)
        for cell, col in zip(row, cols):
            p = latest_post.get(cell, float("nan"))
            label = CELL_LABELS[cell]
            if cell == current:
                col.success(f"**{label}**\n\n{p:.1%}" if p == p else f"**{label}**")
            else:
                col.info(f"{label}\n\n{p:.1%}" if p == p else label)

    p_bull = None
    dirpost = load_dirpost()
    if not dirpost.empty:
        latest_dir = dirpost.iloc[-1]
        p_bull, p_bear = latest_dir["p_bull"], latest_dir["p_bear"]
        drift_p_up = state.get("drift_p_up_latest")
        if drift_p_up is not None:
            hmm_bullish = p_bull > p_bear
            drift_bullish = drift_p_up > 0.5
            agree = hmm_bullish == drift_bullish
            badge = "AGREE" if agree else "DISAGREE"
            st.caption(
                f"HMM-vs-drift agreement: **{badge}** "
                f"(HMM p_bull={p_bull:.2f} vs p_bear={p_bear:.2f}; drift model p_up={drift_p_up:.2f}, "
                f"curve_beta={state.get('curve_beta_latest', float('nan')):.4f})"
            )
        else:
            st.caption("Drift model p_up not available for this run (null -- see model.py NaN-masking fix).")


def render_transition_panel(committed: pd.Series, current_cell: str):
    st.subheader("Next-day transition (empirical)")
    st.caption(
        "Not a separate model forecast -- the pipeline doesn't persist one. This is an "
        "empirical transition matrix estimated live from the full committed-regime "
        "history, conditioned on today's committed cell."
    )
    matrix = empirical_transition_matrix(committed)
    probs = next_day_probs(current_cell, matrix)
    exit_p = exit_probability(current_cell, matrix)

    if exit_p > 0.25:
        st.warning(f"Exit probability elevated: {exit_p:.1%} chance of leaving {current_cell} tomorrow (empirical).")
    else:
        st.caption(f"Exit probability: {exit_p:.1%}")

    chart_df = probs.rename_axis("cell").rename("probability").reset_index()
    st.bar_chart(chart_df.set_index("cell"))


def render_names_tab(state: dict, cfg: dict):
    st.subheader("Names")
    per_name = state.get("per_name", {})
    if not per_name:
        st.info("No per-name data in the latest state.json.")
        return

    rows = []
    for ticker, d in per_name.items():
        rec = d.get("recommendation", {})
        rows.append({
            "ticker": ticker, "as_of": d.get("as_of"), "cell": d.get("cell"),
            "rs_z_short": d.get("rs_z_short"), "rs_z_long": d.get("rs_z_long"),
            "structure": rec.get("structure"),
        })
    df = pd.DataFrame(rows)

    col1, col2 = st.columns(2)
    cell_filter = col1.multiselect("Filter by cell", CELLS, default=[])
    structure_filter = col2.multiselect("Filter by structure", sorted(df["structure"].dropna().unique()), default=[])

    filtered = df.copy()
    if cell_filter:
        filtered = filtered[filtered["cell"].isin(cell_filter)]
    if structure_filter:
        filtered = filtered[filtered["structure"].isin(structure_filter)]

    st.caption(
        "This table shows the market-GATED cell per name (per-name tilt capped by the "
        "market regime, per pipeline/tilt.py). A 'standalone' ungated view would need the "
        "pre-gating tilt/vol series persisted per date -- not currently written by "
        "run_nightly.py, so that toggle isn't built yet. Flagging rather than faking it."
    )
    st.dataframe(filtered.sort_values("ticker"), width="stretch", hide_index=True)

    st.divider()
    st.subheader("Drill-down")
    ticker = st.selectbox("Select a name", sorted(per_name.keys()))
    if ticker:
        render_drilldown(ticker, per_name[ticker], cfg)


def render_drilldown(ticker: str, name_state: dict, cfg: dict):
    cell_hist = name_cell_history(ticker)
    iv_hist = name_iv_history(ticker)

    tabs = st.tabs(["Regime ribbon + price", "Structure", "Call history", "Forecast density"])

    with tabs[0]:
        if st.button(f"Fetch live price history for {ticker}", key=f"px_{ticker}"):
            px = fetch_price_history(ticker)
            chart_df = px.rename("close").to_frame()
            st.line_chart(chart_df)
            ribbon = cell_hist.reindex(px.index, method="ffill")
            st.caption("Regime ribbon (cell per date, last 20 obs):")
            st.dataframe(ribbon.tail(20).to_frame("cell"), width="stretch")
        else:
            st.caption("Price history is fetched live (not persisted by the nightly pipeline) -- click to load.")
            st.dataframe(cell_hist.tail(10).to_frame("cell"), width="stretch")

        if not iv_hist.empty:
            st.caption("IV snapshot history (ATM IV near/target expiry, term slope, 25d skew):")
            iv_cols = [c for c in ["atm_iv_near", "atm_iv_target"] if c in iv_hist.columns]
            if iv_cols:
                st.line_chart(iv_hist[iv_cols])
            st.caption(f"Latest: term slope {iv_hist['iv_term_slope'].iloc[-1]:.3f}, "
                       f"25d skew {iv_hist['skew_25d'].iloc[-1]:.3f} "
                       f"({iv_hist.index[-1].date()})")

    with tabs[1]:
        rec = name_state.get("recommendation", {})
        st.write(f"Recommended structure: **{rec.get('structure', 'no_trade')}**")
        legs = rec.get("legs", [])
        if not legs:
            st.caption("no_trade -- no legs to price.")
        elif not st.button(f"Fetch live chain + price {ticker}", key=f"chain_{ticker}"):
            st.caption("Concrete strikes need a live option chain + spot fetch -- click to load.")
        else:
            expiries = fetch_option_expiries(ticker)
            if not expiries:
                st.warning("No option expiries returned for this ticker.")
            else:
                target_dte = cfg.get("default_dte", [30, 45])
                today = date.today()
                dtes = [(e, (datetime.strptime(e, "%Y-%m-%d").date() - today).days) for e in expiries]
                dtes = [x for x in dtes if x[1] > 0]
                best = min(dtes, key=lambda x: abs(x[1] - sum(target_dte) / 2)) if dtes else None
                if best:
                    expiry, dte = best
                    st.caption(f"Using expiry {expiry} ({dte} DTE, target {target_dte[0]}-{target_dte[1]})")
                    calls, puts = fetch_option_chain(ticker, expiry)
                    px = fetch_price_history(ticker)
                    S0 = float(px.iloc[-1]) if not px.empty else None
                    atm_side = calls if legs[0]["cp"] == "call" else puts
                    if S0 and not atm_side.empty:
                        sigma = float(atm_side.iloc[(atm_side["strike"] - S0).abs().argsort()[:1]]["impliedVolatility"].iloc[0])
                        terms = structure_terms(legs, S0, sigma, cfg.get("backtest", {}).get("risk_free", 0.03), dte / 365)
                        st.write(f"Spot: {S0:.2f} | IV used: {sigma:.1%}")
                        st.dataframe(pd.DataFrame(terms["legs"]), width="stretch", hide_index=True)
                        st.write(f"Net {'credit' if terms['net_credit_debit'] > 0 else 'debit'}: "
                                 f"${abs(terms['net_credit_debit']):.2f} (per contract, 1x notional)")
                        st.write(f"Max gain: ${terms['max_gain']:.2f} | Max loss: ${terms['max_loss']:.2f}")
                        if terms["breakevens"]:
                            st.write(f"Breakeven(s): {', '.join(f'{b:.2f}' for b in terms['breakevens'])}")

    with tabs[2]:
        st.caption("Realized forward returns need live price history (not persisted) -- click to compute.")
        if st.button(f"Compute call history for {ticker}", key=f"hist_{ticker}"):
            px = fetch_price_history(ticker)
            log_df = call_history_log(px, cell_hist, horizon=cfg.get("drift_model", {}).get("forward_horizon_days", 5))
            st.dataframe(log_df, width="stretch")

    with tabs[3]:
        st.caption(
            "Empirical forward-return density conditional on the name's current cell, vs. "
            "its unconditional full-history distribution -- same KS-test/effect-size "
            "methodology as Phase 0's TEST 1c, re-run live as an ongoing out-of-sample check "
            "(empirical only, no parametric baseline, per the 2026-07-21 design decision)."
        )
        if st.button(f"Compute forecast density for {ticker}", key=f"density_{ticker}"):
            px = fetch_price_history(ticker)
            current_cell = name_state.get("cell")
            result = conditional_vs_unconditional_density(
                px, cell_hist, current_cell,
                horizon=cfg.get("drift_model", {}).get("forward_horizon_days", 5),
            )
            if result.get("insufficient_data"):
                st.warning(
                    f"Only {result['n_conditional']} matured observations in cell "
                    f"'{current_cell}' -- need >=20 to say anything meaningful."
                )
            else:
                st.write(f"n conditional={result['n_conditional']}, n unconditional={result['n_unconditional']}")
                st.write(f"Conditional mean: {result['conditional_mean']:.4f} | "
                         f"Unconditional mean: {result['unconditional_mean']:.4f}")
                st.write(f"KS stat={result['ks_stat']:.3f} (p={result['ks_p']:.3f}), "
                         f"effect size={result['effect_size_sd']:.2f}sd")
                hist_df = pd.DataFrame({
                    "conditional": pd.Series(result["conditional"].values),
                    "unconditional": pd.Series(result["unconditional"].values),
                })
                st.bar_chart(hist_df)


def render_history_tab(committed: pd.Series):
    st.subheader("Regime history")
    runs = regime_runs(committed)
    if runs.empty:
        st.info("No committed-regime history available.")
        return

    st.caption("Regime duration distribution (days per contiguous run):")
    st.bar_chart(runs.groupby("regime")["duration_days"].mean().reindex(CELLS))

    st.caption("Full run log (most recent first):")
    st.dataframe(runs.sort_values("start", ascending=False), width="stretch", hide_index=True)


def render_diagnostics_tab(state: dict, committed: pd.Series, dirpost: pd.DataFrame):
    st.subheader("Diagnostics")
    st.json({
        "as_of": state.get("as_of"),
        "data_source": state.get("data_source"),
        "hmm_diagnostics": state.get("hmm_diagnostics"),
        "drift_p_up_latest": state.get("drift_p_up_latest"),
        "curve_beta_latest": state.get("curve_beta_latest"),
        "iv_snapshot_coverage": state.get("iv_snapshot_coverage"),
    })

    if not dirpost.empty and not committed.empty:
        st.caption("Model agreement over time: HMM bullish lean (p_bull - p_bear) vs. committed regime side.")
        merged = dirpost.copy()
        merged["hmm_lean"] = merged["p_bull"] - merged["p_bear"]
        st.line_chart(merged["hmm_lean"].tail(252))


def main():
    cfg = load_config()
    state = load_state()
    cell_posterior = load_cell_posterior()
    committed = load_committed_regime()
    dirpost = load_dirpost()

    render_header(state, committed)
    tabs = st.tabs(["Market Regime", "Names", "History", "Diagnostics"])

    with tabs[0]:
        render_regime_card(state, cell_posterior)
        st.divider()
        render_transition_panel(committed, state.get("market_regime", ""))

    with tabs[1]:
        render_names_tab(state, cfg)

    with tabs[2]:
        render_history_tab(committed)

    with tabs[3]:
        render_diagnostics_tab(state, committed, dirpost)


if __name__ == "__main__":
    main()
