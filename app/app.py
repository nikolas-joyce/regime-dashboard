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
import plotly.express as px
import streamlit as st

from app.data import (
    load_config, load_state, load_cell_posterior, load_committed_regime, load_dirpost,
    load_name_cells, load_iv_snapshots, load_forecast_density, load_name_metadata,
    name_cell_history, name_iv_history,
    fetch_price_history, fetch_option_expiries, fetch_option_chain, CELLS,
)
from app.analytics import (
    empirical_transition_matrix, next_day_probs, exit_probability, regime_runs,
    days_in_current_regime, conditional_vs_unconditional_density, call_history_log,
    structure_terms, transition_counts, wilson_interval,
)
from pipeline.matrix import recommend_structure, confidence_tier
from pipeline.default_long import (
    confidence_scalar, variant_net_delta, direction_label, exposure_summary, NET_DELTA_CAP,
)

st.set_page_config(page_title="Regime Dashboard", layout="wide")

CELL_LABELS = {
    "bull_hi": "Bull / High Vol", "bull_lo": "Bull / Low Vol",
    "neut_hi": "Neutral / High Vol", "neut_lo": "Neutral / Low Vol",
    "bear_hi": "Bear / High Vol", "bear_lo": "Bear / Low Vol",
}
# Hover text for the cell grid badges (native HTML title attribute -- works in any browser,
# no JS needed). Plain-language description of what each of the 6 regime cells means, since
# "Bull / High Vol 34%" on its own doesn't say what "high vol" is measuring.
CELL_DESCRIPTIONS = {
    "bull_hi": "SPY's HMM-estimated direction posterior leans bullish AND the vol layer's "
               "P(high-vol) is elevated -- an up-trending but choppier/more expensive-to-hedge tape.",
    "bull_lo": "Bullish direction posterior with LOW P(high-vol) -- a calmer, grinding uptrend.",
    "neut_hi": "No clear directional lean (bull/bear posteriors roughly balanced) but P(high-vol) "
               "is elevated -- an uncertain, jumpy tape with no dominant trend.",
    "neut_lo": "No clear directional lean and LOW P(high-vol) -- a quiet, range-bound tape.",
    "bear_hi": "Bearish direction posterior AND elevated P(high-vol) -- the classic selloff "
               "signature (down-trending, volatile).",
    "bear_lo": "Bearish direction posterior with LOW P(high-vol) -- a slower, grinding decline.",
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
    st.caption(
        "SPY's market regime, a recommended options structure, and per-name directional "
        "signals -- all precomputed by a nightly pipeline (see freshness below). Hover any "
        "metric, chart, or table column header for a definition; each tab opens with a "
        "one-line summary of what it shows."
    )
    c1, c2, c3 = st.columns(3)
    c1.metric(
        "Data freshness", freshness_badge(state.get("as_of", "")),
        help="Age of the nightly pipeline's last committed output. 'Fresh' = updated within "
             "the last day; 'STALE' means the scheduled run may not have landed -- treat "
             "everything else on this page with that in mind until it clears.",
    )
    c1.caption(f"Source: {state.get('data_source', '?')} | IV coverage: {state.get('iv_snapshot_coverage', '?')}")
    c2.metric(
        "Days in current regime", days_in_current_regime(committed),
        help="Consecutive trading days the smoothed 'committed' regime cell has held without "
             "switching. Commitment is deliberately sticky -- it only flips after the raw "
             "model posterior clears a confidence threshold for several days running (see "
             "the Diagnostics tab for how often that has happened) -- so this number moves "
             "slower than the raw daily probabilities in the Market Regime tab.",
    )
    hmm_diag = state.get("hmm_diagnostics", {})
    c3.metric(
        "HMM refits", hmm_diag.get("n_refit", "?"),
        help=f"Number of times the direction model (a 3-state Hidden Markov Model fit on "
             f"SPY's daily returns) has been refit walk-forward across its full history. "
             f"Median z-separation {hmm_diag.get('median_z_separation', float('nan')):.2f} "
             f"measures how cleanly separated the bull/neutral/bear return distributions "
             f"were at each refit -- a diagnostic only, refits are never rejected for low "
             f"separation (see model.py).",
    )


def _cell_badge_html(label: str, prob: float, color: str, is_current: bool, tooltip: str = "") -> str:
    prob_str = f"{prob:.1%}" if prob == prob else "—"
    border = "3px solid var(--text-primary, #1a1a1a)" if is_current else "1px solid rgba(128,128,128,0.25)"
    marker = " (current)" if is_current else ""
    # White text on the two darkest/most-saturated fills (bull_hi, bear_hi, both neut
    # shades' gray sits in between and reads fine either way -- checked against both).
    text_color = "#FFFFFF" if color in ("#1B5E20", "#B71C1C") else "#1a1a1a"
    title_attr = f' title="{tooltip}"' if tooltip else ""
    return (
        f'<div{title_attr} style="background:{color};color:{text_color};border:{border};'
        f'border-radius:8px;padding:10px 8px;text-align:center;cursor:help;">'
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
    # 2026-07-22: caveat added after the backtest exercise found the delta-hedge-proxy
    # methodology used to score this structure_matrix strips out theta/premium entirely
    # -- for credit-selling structures (short_put, both credit spreads) that decay IS
    # the trade, so a proxy Sharpe understates them. See research/backtest_per_ticker.py
    # and backtest_default_long_variant.py's matrix_premium_* columns for the (still
    # crude, uncalibrated) attempt to bound how much that might matter. Not a claim this
    # structure is bad -- a caveat that its backtested performance hasn't been validated
    # with real options pricing yet.
    st.caption(
        "Backtested performance for this structure matrix uses a delta-only P&L proxy "
        "(no theta/premium, no transaction costs) -- treat any Sharpe you've seen quoted "
        "for it as a directional sanity check, not a validated options P&L."
    )
    st.caption(
        "'Posterior' = the model's current confidence that the committed regime call above "
        "is correct (from the direction HMM's forward-filtered probability). Confidence tier "
        "(High/Moderate/Low) buckets that number per config.yaml's thresholds and determines "
        "structure sizing -- Low confidence can gate a name straight to no_trade."
    )


def render_default_long_banner(state: dict):
    """2026-07-22: added after backtesting a market-anchored, confidence-scaled
    alternative to the structure_matrix above (research/backtest_default_long_variant.py).
    That variant beat the current matrix in 43/50 names (mean Sharpe 1.00 vs. 0.54) and
    beat SPY buy-and-hold at the aggregate portfolio level (monthly Sharpe 1.23 vs. 0.77)
    where the matrix never did (0/50 names, aggregate Sharpe 0.91). Shown here as a
    SEPARATE, clearly-labeled second opinion -- not a replacement for the structure
    recommendation above, since the matrix chooses a concrete options structure and this
    is a raw directional exposure target with no options legs at all. Known open issue,
    not yet fixed: this formula's short sleeve was a net drag in the backtest (aggregate
    short-only Sharpe -0.48) -- treat SHORT signals from this banner with more
    skepticism than LONG ones until that's revisited."""
    market_cell = state.get("market_regime")
    posterior = state.get("posterior")
    if not market_cell:
        return

    nd = variant_net_delta(market_cell, posterior, None)  # market-only, no per-name RS component
    label = direction_label(nd)
    color = {"Long": "#1B5E20", "Short": "#B71C1C", "Flat": "#616161"}[label]
    conf = confidence_scalar(posterior)

    st.markdown(
        f'<div style="border:1px dashed rgba(128,128,128,0.4); border-radius:10px; '
        f'padding:12px 18px; margin-bottom:12px;">'
        f'<div style="font-size:0.85em; opacity:0.7; text-transform:uppercase; letter-spacing:0.03em;">'
        f'Default-long variant (research, market-anchored) — second opinion</div>'
        f'<div style="font-size:1.2em; font-weight:600; margin:2px 0; color:{color};">'
        f'{label} (target exposure {nd:+.2f})</div>'
        f'<div style="opacity:0.7; font-size:0.85em;">'
        f'Backtested mean Sharpe 1.00 vs. matrix 0.54 vs. buy-hold 0.58 (per-name); '
        f'0% flat vs. matrix\'s 66% flat. Short signals are less validated than long ones -- '
        f'see caption below.</div>'
        f'</div>',
        unsafe_allow_html=True,
    )
    # st.progress() doesn't accept a help= kwarg (TypeError, confirmed live 2026-07-23) --
    # unlike st.metric/st.button, ProgressMixin.progress() has no tooltip param at all.
    # Explanation moved to a caption instead.
    st.progress(conf, text=f"Confidence scalar: {conf:.0%} (0.5 floor, scales continuously with posterior)")
    st.caption(
        "This variant sizes exposure by confidence directly (0.5 + 0.5*posterior) -- it "
        "floors at half-size even with zero model confidence and never goes fully flat on "
        "confidence alone, unlike the matrix above which can gate a name all the way to "
        "no_trade at low confidence."
    )


def render_regime_card(state: dict, cell_posterior: pd.DataFrame, cfg: dict):
    st.subheader("Market regime")
    st.caption(
        "What SPY's regime is right now, two independent model opinions on what to do about "
        "it (a concrete options structure, and a simpler directional exposure target below "
        "it), and the full 6-cell probability breakdown the regime call is drawn from."
    )
    render_recommendation_banner(state, cfg)
    render_default_long_banner(state)

    current = state.get("market_regime", "?")
    latest_post = cell_posterior.iloc[-1] if not cell_posterior.empty else pd.Series(dtype=float)

    st.caption(
        "Today's probability of being in each of the 6 regime cells (direction x vol level) "
        "-- hover a cell for what it means. This is the raw daily posterior, not yet smoothed "
        "into a single call; the bold-bordered cell is today's 'committed' regime, and it's "
        "usually but not always the highest-probability one here, since commitment "
        "deliberately lags the raw posterior (see 'Days in current regime' above)."
    )
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
            html = _cell_badge_html(CELL_LABELS[cell], p, CELL_COLORS[cell], cell == current,
                                     tooltip=CELL_DESCRIPTIONS[cell])
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
        "Where the regime is likely to go from here. Not a separate model forecast -- the "
        "pipeline doesn't persist one -- this is an empirical transition matrix built live "
        "from how often each cell has historically switched to each other cell, conditioned "
        "on today's committed cell. 'Exit probability' below is P(tomorrow's committed cell "
        "differs from today's), i.e. 1 minus the chance of staying put."
    )
    matrix = empirical_transition_matrix(committed)
    probs = next_day_probs(current_cell, matrix)
    exit_p = exit_probability(current_cell, matrix)
    counts = transition_counts(committed)
    n_obs = int(counts.get(current_cell, 0))

    # 2026-07-23: added after a direct question about sample size -- the matrix above
    # normalizes counts away, so a probability from 400 observed days and one from 12
    # looked identical without this. n_obs = total historical days that STARTED in
    # current_cell (see transition_counts()), not the number of distinct switch events.
    if n_obs < 30:
        st.warning(
            f"Only {n_obs} historical days observed starting in {current_cell} -- "
            "these probabilities (and their confidence intervals) are noisy. Treat as "
            "directional, not precise."
        )
    else:
        st.caption(
            f"Based on {n_obs} historical days observed starting in {current_cell} "
            f"(out of {int(counts.sum())} total committed-regime days)."
        )

    if exit_p > 0.25:
        st.warning(f"Exit probability elevated: {exit_p:.1%} chance of leaving {current_cell} tomorrow (empirical).")
    else:
        st.caption(f"Exit probability: {exit_p:.1%}")

    chart_df = probs.rename_axis("cell").rename("probability").reset_index()
    chart_df["label"] = chart_df["cell"].map(CELL_LABELS)
    chart_df["n"] = n_obs
    # Per-cell Wilson score CI (95%), treating each TO-cell as its own binary outcome vs.
    # the same n_obs denominator -- a standard per-category approximation to the true
    # multinomial confidence region, not exact joint inference, but sufficient for a
    # diagnostic error bar. k reconstructed from probability*n_obs (exact by construction,
    # since probability WAS count/n_obs before normalization -- round() just guards
    # float noise).
    ci_bounds = [
        wilson_interval(round(p * n_obs), n_obs) if n_obs else (float("nan"), float("nan"))
        for p in chart_df["probability"]
    ]
    chart_df["ci_low"] = [b[0] for b in ci_bounds]
    chart_df["ci_high"] = [b[1] for b in ci_bounds]

    # st.bar_chart's auto-scale/tick-formatting broke on live data (garbled y-axis
    # labels, e.g. "2e-28"-style ticks) for this exact shape: mostly-near-zero values
    # plus one value near 1 -- the near-guaranteed distribution for any row of a sticky
    # regime's transition matrix. AppTest's exception-only check couldn't catch a bad
    # Vega-Lite spec, only a live screenshot did (2026-07-22). Explicit Altair chart
    # with a forced [0,1] scale removes the auto-formatting ambiguity that caused it.
    bars = (
        alt.Chart(chart_df)
        .mark_bar()
        .encode(
            x=alt.X("cell:N", sort=CELLS, title=None),
            y=alt.Y("probability:Q", scale=alt.Scale(domain=[0, 1]), axis=alt.Axis(format="%")),
            tooltip=[
                alt.Tooltip("label:N", title="Cell"),
                alt.Tooltip("probability:Q", title="P(tomorrow)", format=".1%"),
                alt.Tooltip("n:Q", title="n observed"),
            ],
        )
    )
    error_bars = (
        alt.Chart(chart_df)
        .mark_rule(color="black", strokeWidth=2)
        .encode(
            x=alt.X("cell:N", sort=CELLS),
            y="ci_low:Q", y2="ci_high:Q",
            tooltip=[
                alt.Tooltip("ci_low:Q", title="95% CI low", format=".1%"),
                alt.Tooltip("ci_high:Q", title="95% CI high", format=".1%"),
            ],
        )
    )
    st.altair_chart(bars + error_bars, width="stretch")
    st.caption("Black bars are 95% Wilson score confidence intervals -- wide bars mean the point probability isn't well-pinned down yet.")


def render_universe_treemap(state: dict, name_metadata: pd.DataFrame):
    """2026-07-23: main-page treemap, added per Nikolas's explicit request for a Finviz-
    style view -- box SIZE = |net_delta| (conviction magnitude, not market cap: this
    dashboard's whole point is highlighting where the model has a strong call, and that
    needs zero new data, unlike market cap), box COLOR toggles between net_delta
    (continuous, the forward directional call itself) and regime cell (categorical, same
    6-color scheme as the rest of the app). Grouped by GICS sector when reference data
    is available -- see pipeline/fetch_name_metadata.py's docstring for why that's a
    separate, manually-run script rather than wired into the nightly Action (yfinance's
    .info endpoint is slower/more failure-prone than the price pull, and sector/market-
    cap don't need daily freshness the way the regime outputs do)."""
    st.subheader("Universe map")
    st.caption(
        "Every name in the universe in one view. Box size = conviction (|net_delta| -- "
        "bigger box means a stronger long OR short call, not a bigger company). Color "
        "toggles between that same directional call and the discrete regime cell. "
        "Grouped by sector when reference data is available. Hover any box for the "
        "full breakdown."
    )
    per_name = state.get("per_name", {})
    if not per_name:
        st.info("No per-name data in the latest state.json.")
        return

    market_cell = state.get("market_regime")
    posterior = state.get("posterior")
    variant_summary = exposure_summary(per_name, market_cell, posterior)
    df = pd.DataFrame(variant_summary["rows"])  # ticker, net_delta, direction
    df["cell"] = df["ticker"].map(lambda t: per_name.get(t, {}).get("cell"))
    df["cell_label"] = df["cell"].map(CELL_LABELS).fillna("Unknown")
    # Floor, not raw abs -- a true zero net_delta is possible (market/name components can
    # exactly cancel) and a zero-value treemap box either errors or renders invisibly.
    df["abs_net_delta"] = df["net_delta"].abs().clip(lower=0.02)

    has_meta = not name_metadata.empty
    if has_meta:
        df = df.merge(name_metadata[["ticker", "market_cap", "sector", "industry"]], on="ticker", how="left")
        df["sector"] = df["sector"].fillna("Unknown")
        df["industry"] = df["industry"].fillna("Unknown")
        df["market_cap_display"] = df["market_cap"].apply(
            lambda v: f"${v / 1e9:.1f}B" if pd.notna(v) else "n/a"
        )
        path = ["sector", "ticker"]
    else:
        df["market_cap_display"] = "n/a"
        path = ["ticker"]

    color_mode = st.radio(
        "Color by", ["Directional call (net_delta)", "Regime cell"], horizontal=True,
        help="net_delta: continuous red-green scale on the default-long variant's raw "
             "exposure target (-1.2 to +1.2). Regime cell: same 6-color scheme used "
             "everywhere else in the app (bull=green, neut=gray, bear=red; darker=high-vol).",
    )
    hover_data = {"net_delta": ":.2f", "direction": True, "cell_label": True,
                   "market_cap_display": True, "abs_net_delta": False}
    # 2026-07-23: was textinfo="label+percent parent" -- misleading, since that number is
    # each ticker's share of its OWN sector's box area, not its conviction. A 3-name
    # sector (Energy) shows ~33% per name regardless of how strong the call actually is,
    # while an 11-name sector (Technology) tops out near 12% even for its strongest
    # names -- reads like Energy names are more convicted when that's just sector size.
    # Show the actual signed net_delta on the box instead via texttemplate+customdata
    # (built-in "value" field is abs_net_delta -- unsigned, same ambiguity problem, just
    # without the sector-size distortion). Sector header nodes are synthesized by Plotly
    # from the path hierarchy, not rows in df, so customdata isn't defined for them --
    # they'll show just the sector name with a blank second line, which is fine.
    df["net_delta_str"] = df["net_delta"].map(lambda v: f"{v:+.2f}")

    if color_mode == "Directional call (net_delta)":
        fig = px.treemap(
            df, path=path, values="abs_net_delta", color="net_delta",
            color_continuous_scale="RdYlGn", color_continuous_midpoint=0,
            range_color=[-NET_DELTA_CAP, NET_DELTA_CAP], hover_data=hover_data,
            custom_data=["net_delta_str"],
        )
    else:
        discrete_map = {CELL_LABELS[k]: v for k, v in CELL_COLORS.items()}
        discrete_map["Unknown"] = "#9E9E9E"
        fig = px.treemap(
            df, path=path, values="abs_net_delta", color="cell_label",
            color_discrete_map=discrete_map, hover_data=hover_data,
            custom_data=["net_delta_str"],
        )
    fig.update_layout(margin=dict(t=8, l=8, r=8, b=8), height=520)
    fig.update_traces(texttemplate="%{label}<br>%{customdata[0]}")
    st.plotly_chart(fig, width="stretch")

    # 2026-07-23: computed interpretation, not a static caption -- the useful read here
    # (how much of "the book is Long" reflects genuine per-name confirmation vs. pure
    # market-anchoring) isn't visible from either color mode alone; it only showed up by
    # manually toggling between them and cross-referencing, which a new user won't know
    # to do. Do that cross-reference in code instead and state the finding directly.
    bull_cells, bear_cells = {"bull_hi", "bull_lo"}, {"bear_hi", "bear_lo"}
    n_total = len(df)
    n_long, n_short, n_flat = (df["direction"] == "Long").sum(), (df["direction"] == "Short").sum(), (df["direction"] == "Flat").sum()
    long_unconfirmed = ((df["direction"] == "Long") & (~df["cell"].isin(bull_cells))).sum()
    short_unconfirmed = ((df["direction"] == "Short") & (~df["cell"].isin(bear_cells))).sum()
    market_cell_label = CELL_LABELS.get(market_cell, market_cell or "unknown")
    conf_str = f"{posterior:.0%} posterior confidence" if posterior is not None else "confidence unavailable"

    interp = (
        f"**Reading this map**: market regime is **{market_cell_label}** ({conf_str}). "
        f"**{n_long}/{n_total}** names are Long, **{n_short}/{n_total}** Short, **{n_flat}/{n_total}** Flat "
        f"(net_delta thresholds of +/-0.05, same as the Portfolio tab)."
    )
    if n_long:
        interp += (
            f" Of the Long names, **{long_unconfirmed}** have their own regime cell still "
            f"neutral or bearish -- they're Long mainly because the variant anchors to the "
            f"market's {market_cell_label} call, not independent signal in that name. Only "
            f"**{n_long - long_unconfirmed}** have genuine own-cell bullish confirmation "
            f"(toggle 'Color by: Regime cell' above to see which)."
        )
    if n_short:
        interp += (
            f" Similarly, **{short_unconfirmed}/{n_short}** Short names are market-anchored "
            f"rather than independently bearish."
        )
    st.markdown(interp)

    if not has_meta:
        st.caption(
            "No sector/market-cap reference data yet -- boxes are flat (ungrouped) and "
            "sized purely by conviction. Run `python -m pipeline.fetch_name_metadata` "
            "once (locally, not part of the nightly Action) to populate "
            "data/name_metadata.parquet and unlock sector grouping + market cap in the "
            "hover tooltip."
        )
    st.caption(
        "Backtested short-only Sharpe for this variant was negative (-0.48, see the "
        "Portfolio tab) -- treat red boxes with more skepticism than green ones."
    )


def render_names_tab(state: dict, cfg: dict):
    st.subheader("Names")
    st.caption(
        "Scan the whole universe here to decide which names deserve a closer look, then "
        "pick one in 'Drill-down' below for price history, live option pricing, and "
        "interactive forecast diagnostics. Sorted by |RS z (short)| by default -- most "
        "statistically extreme names first -- or use the filters/column headers to sort "
        "your own way. Hover any column header for what it means."
    )
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

        # 2026-07-23: plain-language read of ks_p/n_conditional, added alongside (not
        # instead of) the raw numbers -- nobody can eyeball "ks_p=0.233" and know whether
        # that's meaningful without already knowing what a KS test is. This is that
        # judgment call made explicit and consistent, not a replacement for the raw stats.
        def _support_label(row) -> str:
            n = row.get("n_conditional")
            if pd.isna(n) or n < 20:
                return "Insufficient data"
            return "Yes" if row.get("ks_p", 1.0) < 0.05 else "No"

        df["historically_supported"] = df.apply(_support_label, axis=1)

    col1, col2 = st.columns(2)
    cell_filter = col1.multiselect(
        "Filter by cell", CELLS, default=[],
        help="Show only names currently in these regime cells (direction x vol level -- same "
             "6 cells as the Market Regime tab's grid). Leave empty to show all.",
    )
    structure_filter = col2.multiselect(
        "Filter by structure", sorted(df["structure"].dropna().unique()), default=[],
        help="Show only names whose recommended options structure matches. Leave empty to show all.",
    )

    filtered = df.copy()
    if cell_filter:
        filtered = filtered[filtered["cell"].isin(cell_filter)]
    if structure_filter:
        filtered = filtered[filtered["structure"].isin(structure_filter)]

    st.caption(
        "Three groups of columns, left to right: **classification** (cell + raw RS "
        "z-scores), **recommendation** (the live structure_matrix's actual output), and "
        "**statistical validation** (does this cell historically mean anything for this "
        "specific name). The market-GATED cell shown here is the name's own tilt capped "
        "by the market regime (pipeline/tilt.py) -- an ungated 'standalone' view would "
        "need the pre-gating series persisted per date, not currently written by "
        "run_nightly.py, so that toggle isn't built yet. Flagging rather than faking it."
    )
    if not fd.empty:
        st.caption(
            "Validation columns: **Historically supported** is a plain-language read of "
            "the two numbers next to it (Yes = KS p<0.05 with >=20 matured observations; "
            "No = tested but not significant; Insufficient data = fewer than 20 "
            "observations, don't trust it yet) -- check the raw effect size/p-value "
            "yourself before relying on it. The two sparklines are the forward-return "
            "distribution given this cell vs. this name's full unconditional history, "
            "same bin edges for both (directly comparable shapes) -- same KS-test/effect-"
            "size methodology as Phase 0's TEST 1c. Blank sparkline = insufficient data."
        )
    column_config = {
        "ticker": st.column_config.TextColumn("Ticker"),
        "cell": st.column_config.TextColumn(
            "Cell", help="This name's regime cell -- its own relative-strength tilt, capped by "
                         "the market's committed regime (pipeline/tilt.py). Same 6 cells as "
                         "the Market Regime tab.",
        ),
        "rs_z_short": st.column_config.NumberColumn(
            "RS z (short)", format="%.2f",
            help="Short-window relative-strength z-score: this name's return vs. SPY's, "
                 "z-scored. Positive = outperforming SPY over the short lookback.",
        ),
        "rs_z_long": st.column_config.NumberColumn(
            "RS z (long)", format="%.2f",
            help="Long-window relative-strength z-score -- same idea as RS z (short) but over "
                 "a longer lookback window (see config.yaml tilt_layer.rs_window_long).",
        ),
        "structure": st.column_config.TextColumn(
            "Structure", help="Recommended options structure for this name given its cell and "
                              "confidence tier (pipeline/matrix.py). 'no_trade' = flat.",
        ),
        "conditional_hist": st.column_config.BarChartColumn(
            "Density (cond.)", width="small",
            help="Forward-return distribution for dates historically in this SAME cell.",
        ),
        "unconditional_hist": st.column_config.BarChartColumn(
            "Density (uncond.)", width="small",
            help="Forward-return distribution across this name's FULL history, any cell -- "
                 "compare against 'Density (cond.)' to see if the current cell shifts the "
                 "distribution.",
        ),
        "effect_size_sd": st.column_config.NumberColumn(
            "Effect size (sd)", format="%.2f",
            help="Conditional mean minus unconditional mean, in standard deviations of the "
                 "unconditional distribution -- how many sigma this cell's forward returns "
                 "have historically shifted, one direction or the other.",
        ),
        "ks_p": st.column_config.NumberColumn(
            "KS p-value", format="%.3f",
            help="Kolmogorov-Smirnov test p-value comparing the conditional vs. unconditional "
                 "return distributions. Small (e.g. <0.05) = this cell's forward-return shape "
                 "is statistically distinguishable from the name's baseline; large = not "
                 "distinguishable given the sample size.",
        ),
        "historically_supported": st.column_config.TextColumn(
            "Historically supported?",
            help="Plain-language read of effect size/KS p-value: Yes = statistically "
                 "significant (p<0.05) with >=20 matured observations; No = tested but "
                 "not significant; Insufficient data = fewer than 20 observations so far, "
                 "don't trust it yet. Always check the raw numbers too.",
        ),
    }
    # Explicit order groups the three zones from the caption above (classification |
    # recommendation | validation) and drops as_of -- it duplicates the header's
    # freshness badge and added width without new information in this table specifically.
    column_order = ["ticker", "cell", "rs_z_short", "rs_z_long", "structure"]
    if not fd.empty:
        column_order += ["historically_supported", "effect_size_sd", "ks_p",
                          "conditional_hist", "unconditional_hist"]

    # Default sort by |rs_z_short| descending rather than ticker alphabetical -- surfaces
    # the most statistically extreme names first regardless of direction. The table's
    # column headers are still natively clickable to re-sort any other way.
    display_df = filtered.assign(_abs_rs=filtered["rs_z_short"].abs()) \
        .sort_values("_abs_rs", ascending=False).drop(columns="_abs_rs")
    st.dataframe(display_df, width="stretch", hide_index=True, column_config=column_config,
                 column_order=column_order)

    st.divider()
    st.subheader("Drill-down")
    st.caption(
        "Everything below is per-ticker: price history with the regime overlaid, live "
        "option pricing for the recommended structure, historical outcomes when this cell "
        "occurred before, and the same conditional-vs-unconditional forecast check as the "
        "table above but interactive. Most of this needs a live data fetch (yfinance/option "
        "chain), so nothing loads until you click the button in each sub-tab."
    )
    ticker = st.selectbox("Select a name", sorted(per_name.keys()), help="Choose a ticker to drill into below.")
    if ticker:
        render_drilldown(ticker, per_name[ticker], cfg)


def render_drilldown(ticker: str, name_state: dict, cfg: dict):
    cell_hist = name_cell_history(ticker)
    iv_hist = name_iv_history(ticker)

    tabs = st.tabs(["Regime ribbon + price", "Structure", "Regime call history", "Forecast density"])

    with tabs[0]:
        st.caption(
            f"{ticker}'s price with its own regime-cell history painted as colored bands "
            "underneath -- lets you eyeball whether cell changes lined up with what price "
            "actually did. Colors match the Market Regime tab's grid (bull=green, "
            "neut=gray, bear=red; darker=high-vol)."
        )
        if st.button(f"Fetch live price history for {ticker}", key=f"px_{ticker}",
                     help="Pulls live daily price history via yfinance -- not persisted by "
                          "the nightly pipeline, so this fetches fresh on click."):
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
            st.dataframe(
                cell_hist.tail(10).to_frame("cell"), width="stretch",
                column_config={"cell": st.column_config.TextColumn(
                    "Cell", help="This name's regime cell as of that date (no price fetched yet).")},
            )

        if not iv_hist.empty:
            st.caption("IV snapshot history (ATM IV near/target expiry, term slope, 25d skew):")
            iv_cols = [c for c in ["atm_iv_near", "atm_iv_target"] if c in iv_hist.columns]
            if iv_cols:
                st.line_chart(iv_hist[iv_cols])
            st.caption(f"Latest: term slope {iv_hist['iv_term_slope'].iloc[-1]:.3f}, "
                       f"25d skew {iv_hist['skew_25d'].iloc[-1]:.3f} "
                       f"({iv_hist.index[-1].date()})")

    with tabs[1]:
        st.caption(
            "Same recommended structure as the Names table, expanded here into concrete "
            "strikes and pricing using a live spot price + option chain (target-delta legs "
            "resolved to actual strikes at the nearest available expiry)."
        )
        rec = name_state.get("recommendation", {})
        st.write(f"Recommended structure: **{rec.get('structure', 'no_trade')}**")
        legs = rec.get("legs", [])
        if not legs:
            st.caption("no_trade -- no legs to price.")
        elif not st.button(f"Fetch live chain + price {ticker}", key=f"chain_{ticker}",
                            help="Pulls a live option chain + spot price via yfinance to "
                                 "resolve the recommended delta targets to real strikes."):
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
                        legs_col_config = {
                            "cp": st.column_config.TextColumn("Call/Put"),
                            "delta": st.column_config.NumberColumn(
                                "Target delta", format="%.2f",
                                help="The delta this leg was targeted at before resolving to a "
                                     "real strike -- e.g. -0.20 targets a ~20-delta put.",
                            ),
                            "pos": st.column_config.NumberColumn(
                                "Position", help="+1 = long this leg (pay premium), -1 = short (collect premium).",
                            ),
                            "strike": st.column_config.NumberColumn(
                                "Strike", format="%.2f",
                                help="Actual strike nearest this leg's target delta, from the live option chain.",
                            ),
                            "premium": st.column_config.NumberColumn(
                                "Premium", format="%.2f",
                                help="Black-Scholes theoretical premium at this strike, using the IV noted above.",
                            ),
                        }
                        st.dataframe(pd.DataFrame(terms["legs"]), width="stretch", hide_index=True,
                                     column_config=legs_col_config)
                        st.write(f"Net {'credit' if terms['net_credit_debit'] > 0 else 'debit'}: "
                                 f"${abs(terms['net_credit_debit']):.2f} (per contract, 1x notional)")
                        st.write(f"Max gain: ${terms['max_gain']:.2f} | Max loss: ${terms['max_loss']:.2f}")
                        if terms["breakevens"]:
                            st.write(f"Breakeven(s): {', '.join(f'{b:.2f}' for b in terms['breakevens'])}")

    with tabs[2]:
        st.caption(
            "'Call' here means regime call, not call option -- this is the history of "
            f"{ticker}'s past cell assignments alongside what its price actually did in the "
            "following period, so you can eyeball how each cell has played out historically. "
            "Realized forward returns need live price history (not persisted) -- click to compute."
        )
        if st.button(f"Compute call history for {ticker}", key=f"hist_{ticker}",
                     help="Fetches live price history and joins it against this name's cell "
                          "history to compute realized forward returns per date."):
            px = fetch_price_history(ticker)
            horizon = cfg.get("drift_model", {}).get("forward_horizon_days", 5)
            log_df = call_history_log(px, cell_hist, horizon=horizon)
            hist_col_config = {
                "cell": st.column_config.TextColumn("Cell", help="This name's regime cell as of that date."),
                "fwd_return": st.column_config.NumberColumn(
                    f"Fwd return ({horizon}d)", format="percent",
                    help=f"Realized {horizon}-trading-day forward return from that date. "
                         "Blank for the most recent dates -- not yet matured.",
                ),
            }
            st.dataframe(log_df, width="stretch", column_config=hist_col_config)

    with tabs[3]:
        st.caption(
            "Empirical forward-return density conditional on the name's current cell, vs. "
            "its unconditional full-history distribution -- same KS-test/effect-size "
            "methodology as Phase 0's TEST 1c, re-run live as an ongoing out-of-sample check "
            "(empirical only, no parametric baseline, per the 2026-07-21 design decision)."
        )
        if st.button(f"Compute forecast density for {ticker}", key=f"density_{ticker}",
                     help="Fetches live price history and computes the same conditional-vs-"
                          "unconditional forward-return comparison as the Names table's "
                          "sparklines, for this one ticker, interactively."):
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
                d1, d2 = st.columns(2)
                d1.metric(
                    "n conditional", result["n_conditional"],
                    help=f"Matured observations where {ticker} was in cell '{current_cell}'.",
                )
                d2.metric("n unconditional", result["n_unconditional"],
                          help="Matured observations across this name's full history, any cell.")
                d1.metric(
                    "Conditional mean", f"{result['conditional_mean']:.4f}",
                    help="Mean forward return given the current cell.",
                )
                d2.metric(
                    "Unconditional mean", f"{result['unconditional_mean']:.4f}",
                    help="Mean forward return across full history, any cell.",
                )
                st.metric(
                    "Effect size", f"{result['effect_size_sd']:.2f} sd",
                    help=f"Conditional mean minus unconditional mean, in standard deviations "
                         f"of the unconditional distribution. KS stat={result['ks_stat']:.3f} "
                         f"(p={result['ks_p']:.3f}) -- small p means the two distributions "
                         f"above are statistically distinguishable given the sample size.",
                )
                st.caption(
                    "Chart below: conditional (blue) vs. unconditional (orange) forward-"
                    "return values -- hover a bar for its exact value."
                )
                hist_df = pd.DataFrame({
                    "conditional": pd.Series(result["conditional"].values),
                    "unconditional": pd.Series(result["unconditional"].values),
                })
                st.bar_chart(hist_df)


def render_portfolio_tab(state: dict):
    """2026-07-22: new tab, added after the aggregate-portfolio backtest
    (research/backtest_default_long_variant.py's build_aggregate_portfolio()) showed
    that per-name Sharpe averages hide diversification effects -- the equal-weight
    PORTFOLIO Sharpe was meaningfully different (higher, in both matrix's and the
    variant's case) than the mean of the 50 individual per-name Sharpes. This tab is
    a TODAY'S-SNAPSHOT view of that idea (current %long/%flat/%short across the
    universe, both under the live matrix and the default-long variant), computed
    live from state.json's per_name dict -- no new pipeline data needed. It is NOT
    a live-updating historical equity curve -- that needs the full backtest re-run
    (research/backtest_default_long_variant.py, local, not wired into the nightly
    Action) and isn't reproduced here. See that script's output PNGs/CSVs for the
    historical comparison; this tab only answers "what does the book look like
    right now."""
    st.subheader("Portfolio snapshot")
    st.caption(
        "Today's aggregate exposure across all names, under the live structure_matrix "
        "vs. the default-long variant (research/backtest_default_long_variant.py). "
        "Computed live from state.json -- not a historical backtest; see that script's "
        "own output for the multi-year comparison."
    )

    per_name = state.get("per_name", {})
    if not per_name:
        st.info("No per-name data in the latest state.json.")
        return

    market_cell = state.get("market_regime")
    posterior = state.get("posterior")

    # Matrix side: no_trade (empty legs) counts as flat, any nonzero leg set counts
    # as "trading" -- direction isn't reconstructed here (would need pipeline.matrix's
    # net_delta helper on each name's leg list); flat-vs-trading is the headline number
    # this exercise cared about anyway (matrix: 66% flat at the per-name level, backtest).
    matrix_flat = sum(1 for d in per_name.values() if not d.get("recommendation", {}).get("legs"))
    matrix_trading = len(per_name) - matrix_flat

    variant_summary = exposure_summary(per_name, market_cell, posterior)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Current matrix (structure_matrix)**")
        st.caption("The live production logic (pipeline/matrix.py) -- gates a name to a "
                   "concrete options structure, or no_trade, based on its cell + confidence.")
        st.metric(
            "Names trading (nonzero legs)", f"{matrix_trading}/{len(per_name)}",
            help="Names the matrix has assigned a real options structure to (not no_trade).",
        )
        st.metric(
            "Names flat (no_trade)", f"{matrix_flat}/{len(per_name)}",
            help="Names the matrix currently has no structure for -- either the cell doesn't "
                 "clear the confirmation threshold or confidence is too low.",
        )
    with c2:
        st.markdown("**Default-long variant**")
        st.caption("Research alternative (research/backtest_default_long_variant.py) -- a "
                   "continuous, confidence-scaled directional target with no options legs, "
                   "shown here as a second opinion, not a replacement.")
        st.metric(
            "Long", f"{variant_summary['pct_long']:.0f}%",
            help="Share of names with net_delta > +0.05 under this variant's formula.",
        )
        st.metric(
            "Short", f"{variant_summary['pct_short']:.0f}%",
            help="Share of names with net_delta < -0.05. Backtested short-only Sharpe was "
                 "negative (-0.48) -- treat these with more skepticism than the longs.",
        )
        st.metric(
            "Flat", f"{variant_summary['pct_flat']:.0f}%",
            help="Share of names within +/-0.05 of zero net_delta -- rare by design, since "
                 "this variant floors at half-size exposure rather than gating to zero.",
        )
        st.caption(f"Avg net_delta: {variant_summary['avg_net_delta']:+.2f} "
                   f"(avg |net_delta|: {variant_summary['avg_abs_net_delta']:.2f})")

    st.caption(
        "Backtested (2026-07-22, 50 names, 2005-present): equal-weight portfolio monthly "
        "Sharpe -- matrix 0.91, default-long variant 1.23, SPY buy-hold 0.77. Variant's "
        "short-only sleeve had a NEGATIVE Sharpe (-0.48) in that backtest -- long signals "
        "above are better-validated than short ones."
    )

    rows_df = pd.DataFrame(variant_summary["rows"]).sort_values("net_delta", ascending=False)
    portfolio_col_config = {
        "ticker": st.column_config.TextColumn("Ticker"),
        "net_delta": st.column_config.NumberColumn(
            "Net delta", format="%.2f",
            help="This variant's target exposure for the name, in [-1.2, +1.2]: market-anchored "
                 "direction x confidence, nudged by the name's own RS tilt. Not an options "
                 "delta -- a raw directional-exposure target (see pipeline/default_long.py).",
        ),
        "direction": st.column_config.TextColumn(
            "Direction", help="Long/Short/Flat label derived from net_delta at +/-0.05 thresholds.",
        ),
    }
    st.dataframe(rows_df, width="stretch", hide_index=True, column_config=portfolio_col_config)


def render_history_tab(committed: pd.Series):
    st.subheader("Regime history")
    st.caption(
        "How long the market has historically stayed in each committed regime cell, and the "
        "full log of every contiguous run -- 'committed' means the smoothed regime call (see "
        "'Days in current regime' on the Market Regime tab), not the raw daily posterior."
    )
    runs = regime_runs(committed)
    if runs.empty:
        st.info("No committed-regime history available.")
        return

    st.caption("Average run length per cell (mean days per contiguous run, hover a bar for the exact value):")
    st.bar_chart(runs.groupby("regime")["duration_days"].mean().reindex(CELLS))

    st.caption("Full run log (most recent first) -- hover a column header for what it means:")
    history_col_config = {
        "regime": st.column_config.TextColumn("Regime", help="The committed cell during this run."),
        "start": st.column_config.DateColumn("Start", help="First date this run's cell was committed."),
        "end": st.column_config.DateColumn("End", help="Last date before the committed cell switched away."),
        "duration_days": st.column_config.NumberColumn("Duration (days)", help="Length of this contiguous run, in trading days."),
    }
    st.dataframe(runs.sort_values("start", ascending=False), width="stretch", hide_index=True,
                 column_config=history_col_config)


def render_diagnostics_tab(state: dict, committed: pd.Series, dirpost: pd.DataFrame):
    st.subheader("Diagnostics")
    st.caption(
        "Pipeline health and model-internals detail for debugging -- when it last ran, "
        "how the drift and direction models are currently calibrated, and where the raw "
        "model posterior and the smoothed committed regime currently agree or disagree."
    )

    # Formatted metrics instead of a raw JSON dump (2026-07-22) -- the other three tabs
    # are readable at a glance, this one wasn't. Raw JSON still available below for
    # actual debugging, just collapsed by default.
    hmm_diag = state.get("hmm_diagnostics", {})
    drift_p_up = state.get("drift_p_up_latest")
    curve_beta = state.get("curve_beta_latest")

    c1, c2, c3 = st.columns(3)
    c1.metric("As of", state.get("as_of", "?"), help="Date the nightly pipeline last committed output for.")
    c1.metric("Data source", state.get("data_source", "?"),
              help="Where price data came from for this run (e.g. yfinance) -- see data_pull.py.")
    c2.metric(
        "Drift model p(up)", f"{drift_p_up:.1%}" if drift_p_up is not None else "n/a",
        help="Curve-conditioned Bayesian drift model's P(positive forward return), "
             "conditioned on the VX1-VX3 futures curve slope. 'n/a' means the model's NaN-"
             "masking kicked in for this date (see model.py curve_conditioned_drift_posterior).",
    )
    c2.metric(
        "Curve beta", f"{curve_beta:.4f}" if curve_beta is not None else "n/a",
        help="Regression coefficient on the VX curve slope in the drift model -- how much "
             "the model currently thinks curve slope moves expected forward returns.",
    )
    c3.metric(
        "HMM refits", hmm_diag.get("n_refit", "?"),
        help=f"How many times the direction model has been refit walk-forward. Median "
             f"z-separation {hmm_diag.get('median_z_separation', float('nan')):.2f} is a "
             f"diagnostic of how cleanly separated the bull/neutral/bear state means were "
             f"at each refit -- never gates refit acceptance (see model.py).",
    )
    c3.metric("IV snapshot coverage", state.get("iv_snapshot_coverage", "?"),
              help="Fraction of names with a successful nightly IV/ATM-vol chain snapshot.")

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


def render_sidebar_glossary():
    st.sidebar.header("What each section shows")
    st.sidebar.markdown(
        "- **Universe map** (top of page) -- every name at once, box size = conviction, "
        "color = directional call or regime cell (toggle).\n"
        "- **Market Regime** -- SPY's regime right now, two model opinions on positioning, "
        "and where the regime is likely to go next.\n"
        "- **Names** -- per-ticker cell/signal table, filterable, with a per-name drill-down "
        "(live price+regime chart, option pricing, historical outcomes, forecast check).\n"
        "- **Portfolio** -- today's aggregate long/short/flat exposure across all names, "
        "under two different sizing approaches.\n"
        "- **History** -- how long the market has stayed in each regime historically, and "
        "the full run log.\n"
        "- **Diagnostics** -- pipeline freshness and model-internals detail for debugging."
    )
    with st.sidebar.expander("Glossary"):
        st.markdown(
            "**Committed regime** -- the smoothed, current regime call. The raw model "
            "posterior has to clear a confidence threshold for several consecutive days "
            "before the committed cell switches, so it's deliberately sticky/lagging vs. "
            "the raw daily probabilities.\n\n"
            "**Posterior** -- the direction HMM's current probability estimate for "
            "bull/neutral/bear, forward-filtered (causal, no lookahead).\n\n"
            "**Cell** -- one of 6 combinations of direction (bull/neutral/bear) x vol level "
            "(high/low): e.g. bull_hi = bullish direction with elevated P(high-vol).\n\n"
            "**RS z-score** -- relative-strength z-score: a name's return vs. SPY's, "
            "z-scored over a lookback window. Positive = outperforming SPY.\n\n"
            "**net_delta** -- the default-long variant's raw directional exposure target "
            "(not an options delta) -- market direction x confidence, nudged by the name's "
            "own RS tilt, clipped to [-1.2, +1.2].\n\n"
            "**Confidence tier** -- High/Moderate/Low bucket of the posterior, per "
            "config.yaml's confidence_bands -- determines structure sizing in the matrix.\n\n"
            "**KS test / effect size** -- Kolmogorov-Smirnov test and standard-deviation "
            "shift comparing a name's forward-return distribution conditional on its "
            "current cell vs. its unconditional (full-history) distribution.\n\n"
            "**\"Delta\" means three different things in this app, watch the label**: (1) "
            "*option delta* on the recommended-structure legs (e.g. \"-0.20delta\") -- the "
            "standard options Greek, used only to pick which strike to target; (2) *net_delta* "
            "on the Portfolio tab -- the default-long variant's raw directional exposure "
            "target, NOT an options delta at all; (3) *RS z-score* is unrelated to either "
            "despite also being a signed number -- it's a relative-strength z-score, no "
            "options math involved."
        )


def main():
    cfg = load_config()
    state = load_state()
    cell_posterior = load_cell_posterior()
    committed = load_committed_regime()
    dirpost = load_dirpost()
    name_metadata = load_name_metadata()

    render_sidebar_glossary()
    render_header(state, committed)
    # Main-page placement (2026-07-23, explicit request): the treemap sits above the
    # tabs, not inside one -- it's the "key info" at-a-glance view, visible immediately
    # on load rather than requiring a click into a specific tab.
    render_universe_treemap(state, name_metadata)
    st.divider()
    tabs = st.tabs(["Market Regime", "Names", "Portfolio", "History", "Diagnostics"])

    with tabs[0]:
        render_regime_card(state, cell_posterior, cfg)
        st.divider()
        render_transition_panel(committed, state.get("market_regime", ""))

    with tabs[1]:
        render_names_tab(state, cfg)

    with tabs[2]:
        render_portfolio_tab(state)

    with tabs[3]:
        render_history_tab(committed)

    with tabs[4]:
        render_diagnostics_tab(state, committed, dirpost)


if __name__ == "__main__":
    main()
