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

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from app.data import (
    load_config, load_state, load_cell_posterior, load_committed_regime, load_dirpost,
    load_name_cells, load_iv_snapshots, load_forecast_density, name_cell_history, name_iv_history,
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
# Directional hue (bull=green, neut=gray, bear=red) with a darker/more saturated shade
# for the high-vol side of each pair -- used for the drill-down's regime-ribbon overlay.
CELL_COLORS = {
    "bull_hi": "#1B5E20", "bull_lo": "#81C784",
    "neut_hi": "#616161", "neut_lo": "#BDBDBD",
    "bear_hi": "#B71C1C", "bear_lo": "#EF9A9A",
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


def _cell_badge_html(label: str, prob: float, color: str, is_current: bool) -> str:
    prob_str = f"{prob:.1%}" if prob == prob else "—"
    border = "3px solid var(--text-primary, #1a1a1a)" if is_current else "1px solid rgba(128,128,128,0.25)"
    marker = " (current)" if is_current else ""
    # White text on the two darkest/most-saturated fills (bull_hi, bear_hi, both neut
    # shades' gray sits in between and reads fine either way -- checked against both).
    text_color = "#FFFFFF" if color in ("#1B5E20", "#B71C1C") else "#1a1a1a"
    return (
        f'<div style="background:{color};color:{text_color};border:{border};'
        f'border-radius:8px;padding:10px 8px;text-align:center;">'
        f'<div style="font-weight:600;">{label}{marker}</div>'
        f'<div style="font-size:1.15em;">{prob_str}</div>'
        f'</div>'
    )


def render_recommendation_banner(state: dict, cfg: dict):
    rec = state.get("recommendation", {})
    structure = rec.get("structure")
    if not structure:
        st.info("No market-level recommendation in the latest state.json.")
        return

    legs = rec.get("legs", [])
    legs_str = ", ".join(
        f"{'+' if leg['pos'] > 0 else '-'}{leg['cp']} @ {leg['delta']:.2f}delta" for leg in legs
    ) if legs else "no legs (no_trade)"

    posterior = state.get("posterior")
    bands = cfg.get("confidence_bands", {})
    tier = confidence_tier(posterior, bands) if posterior is not None and bands else None
    tier_label = {
        "high_confidence": "High confidence", "moderate_confidence": "Moderate confidence",
        "low_confidence": "Low confidence",
    }.get(tier, tier or "confidence unknown")
    posterior_str = f"{posterior:.1%}" if posterior is not None else "?"

    st.markdown(
        f'<div style="border:1px solid rgba(128,128,128,0.3); border-radius:10px; '
        f'padding:14px 18px; margin-bottom:12px;">'
        f'<div style="font-size:0.85em; opacity:0.7; text-transform:uppercase; letter-spacing:0.03em;">'
        f'Recommended structure</div>'
        f'<div style="font-size:1.4em; font-weight:600; margin:2px 0;">{structure.replace("_", " ").title()}</div>'
        f'<div style="opacity:0.85;">{legs_str}</div>'
        f'<div style="opacity:0.7; font-size:0.9em; margin-top:4px;">{tier_label} (posterior {posterior_str})</div>'
        f'</div>',
        unsafe_allow_html=True,
    )


def render_regime_card(state: dict, cell_posterior: pd.DataFrame, cfg: dict):
    st.subheader("Market regime")
    render_recommendation_banner(state, cfg)

    current = state.get("market_regime", "?")
    latest_post = cell_posterior.iloc[-1] if not cell_posterior.empty else pd.Series(dtype=float)

    # Color now always means the same thing everywhere in the app (bull=green,
    # neut=gray, bear=red; darker=hi-vol -- CELL_COLORS, shared with the drill-down
    # ribbon overlay). Previously this grid used st.success/st.info (green=current,
    # blue=other) regardless of direction, which meant a current bear_hi cell showed
    # green here but red in the ribbon -- same app, opposite color meaning. "Current"
    # is now a bold border + label instead of overloading color for two things at once.
    for row in CELL_GRID:
        cols = st.columns(3)
        for cell, col in zip(row, cols):
            p = latest_post.get(cell, float("nan"))
            html = _cell_badge_html(CELL_LABELS[cell], p, CELL_COLORS[cell], cell == current)
            col.markdown(html, unsafe_allow_html=True)

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
    # st.bar_chart's auto-scale/tick-formatting broke on live data (garbled y-axis
    # labels, e.g. "2e-28"-style ticks) for this exact shape: mostly-near-zero values
    # plus one value near 1 -- the near-guaranteed distribution for any row of a sticky
    # regime's transition matrix. AppTest's exception-only check couldn't catch a bad
    # Vega-Lite spec, only a live screenshot did (2026-07-22). Explicit Altair chart
    # with a forced [0,1] scale removes the auto-formatting ambiguity that caused it.
    chart = (
        alt.Chart(chart_df)
        .mark_bar()
        .encode(
            x=alt.X("cell:N", sort=CELLS, title=None),
            y=alt.Y("probability:Q", scale=alt.Scale(domain=[0, 1]), axis=alt.Axis(format="%")),
        )
    )
    st.altair_chart(chart, width="stretch")


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

    # Forecast density -- precomputed nightly (pipeline/forecast.py), not live-fetched,
    # so this merge is free. Empty until the first post-2026-07-22 nightly run lands it.
    fd = load_forecast_density()
    if not fd.empty:
        fd_cols = fd[["ticker", "effect_size_sd", "ks_p", "n_conditional",
                       "conditional_hist", "unconditional_hist"]]
        df = df.merge(fd_cols, on="ticker", how="left")
        for col in ["conditional_hist", "unconditional_hist"]:
            df[col] = df[col].apply(lambda x: list(x) if isinstance(x, (list, np.ndarray)) else [])

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
    if not fd.empty:
        st.caption(
            "conditional_hist/unconditional_hist sparklines: forward-return distribution "
            "given the name's current cell vs. its full unconditional history, same bin "
            "edges for both (directly comparable shapes). effect_size_sd/ks_p from the "
            "same KS-test/effect-size methodology as Phase 0's TEST 1c. Blank sparkline = "
            "fewer than 20 matured observations in that cell so far."
        )
    column_config = {
        "conditional_hist": st.column_config.BarChartColumn("Density (cond.)", width="small"),
        "unconditional_hist": st.column_config.BarChartColumn("Density (uncond.)", width="small"),
        "effect_size_sd": st.column_config.NumberColumn("Effect size (sd)", format="%.2f"),
        "ks_p": st.column_config.NumberColumn("KS p-value", format="%.3f"),
    } if not fd.empty else None

    # Default sort by |rs_z_short| descending rather than ticker alphabetical -- surfaces
    # the most statistically extreme names first regardless of direction. The table's
    # column headers are still natively clickable to re-sort any other way.
    display_df = filtered.assign(_abs_rs=filtered["rs_z_short"].abs()) \
        .sort_values("_abs_rs", ascending=False).drop(columns="_abs_rs")
    st.dataframe(display_df, width="stretch", hide_index=True, column_config=column_config)

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
            ribbon = cell_hist.reindex(px.index, method="ffill")
            runs = regime_runs(ribbon)
            if runs.empty:
                st.line_chart(px.rename("close").to_frame())
            else:
                price_min, price_max = float(px.min()), float(px.max())
                pad = (price_max - price_min) * 0.02 or 1.0
                band_df = runs.copy()
                band_df["y0"] = price_min - pad
                band_df["y1"] = price_max + pad
                bands = (
                    alt.Chart(band_df)
                    .mark_rect(opacity=0.28)
                    .encode(
                        x=alt.X("start:T", title=None), x2="end:T",
                        y=alt.Y("y0:Q", title="Price"), y2="y1:Q",
                        color=alt.Color(
                            "regime:N",
                            scale=alt.Scale(domain=list(CELL_COLORS.keys()), range=list(CELL_COLORS.values())),
                            legend=alt.Legend(title="Regime"),
                        ),
                        tooltip=["regime:N", "start:T", "end:T", "duration_days:Q"],
                    )
                )
                line = (
                    alt.Chart(px.rename("close").reset_index().rename(columns={"index": "date"}))
                    .mark_line(color="black", strokeWidth=1.5)
                    .encode(x="date:T", y="close:Q")
                )
                st.altair_chart((bands + line).properties(height=400), width="stretch")
                st.caption(
                    "Colored bands are contiguous regime runs (per-name cell, market-gated) "
                    "aligned to the fetched price history -- hover a band for its regime, "
                    "start/end dates, and duration. This is the same regime_runs() logic "
                    "used for the History tab's duration distribution, applied per name."
                )
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

    # Formatted metrics instead of a raw JSON dump (2026-07-22) -- the other three tabs
    # are readable at a glance, this one wasn't. Raw JSON still available below for
    # actual debugging, just collapsed by default.
    hmm_diag = state.get("hmm_diagnostics", {})
    drift_p_up = state.get("drift_p_up_latest")
    curve_beta = state.get("curve_beta_latest")

    c1, c2, c3 = st.columns(3)
    c1.metric("As of", state.get("as_of", "?"))
    c1.metric("Data source", state.get("data_source", "?"))
    c2.metric("Drift model p(up)", f"{drift_p_up:.1%}" if drift_p_up is not None else "n/a")
    c2.metric("Curve beta", f"{curve_beta:.4f}" if curve_beta is not None else "n/a")
    c3.metric(
        "HMM refits", hmm_diag.get("n_refit", "?"),
        help=f"median z-separation: {hmm_diag.get('median_z_separation', float('nan')):.2f}",
    )
    c3.metric("IV snapshot coverage", state.get("iv_snapshot_coverage", "?"))

    with st.expander("Raw state.json (debugging)"):
        st.json({
            "as_of": state.get("as_of"),
            "data_source": state.get("data_source"),
            "hmm_diagnostics": hmm_diag,
            "drift_p_up_latest": drift_p_up,
            "curve_beta_latest": curve_beta,
            "iv_snapshot_coverage": state.get("iv_snapshot_coverage"),
        })

    if not dirpost.empty and not committed.empty:
        st.caption(
            "HMM's bullish lean (p_bull - p_bear, continuous -1..+1) vs. the smoothed "
            "committed regime's own direction side (bull=+1 / neut=0 / bear=-1, "
            "discrete). Divergence between the two flags moments the raw HMM posterior "
            "and the smoothing/min_dwell logic disagree -- e.g. mid-transition, where "
            "the HMM has already moved but commitment hasn't caught up yet."
        )
        # committed's index is a subset of dirpost's (cp.dropna() in run_nightly.py drops
        # early rows where vol-layer inputs lag the direction posterior) -- reindex
        # dirpost onto committed's index rather than the reverse, so this naturally
        # restricts to dates where both series actually exist, no NaN-fill needed.
        aligned_dirpost = dirpost.reindex(committed.index)
        hmm_lean = aligned_dirpost["p_bull"] - aligned_dirpost["p_bear"]
        direction_map = {"bull": 1, "neut": 0, "bear": -1}
        committed_side = committed.map(lambda c: direction_map[c.split("_")[0]])
        agreement_df = pd.DataFrame({"hmm_lean": hmm_lean, "committed_side": committed_side}).tail(252)
        st.line_chart(agreement_df)


def main():
    cfg = load_config()
    state = load_state()
    cell_posterior = load_cell_posterior()
    committed = load_committed_regime()
    dirpost = load_dirpost()

    render_header(state, committed)
    tabs = st.tabs(["Market Regime", "Names", "History", "Diagnostics"])

    with tabs[0]:
        render_regime_card(state, cell_posterior, cfg)
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
