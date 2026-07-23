"""
Plot: directional and volatility conviction over time, SPY (market layer) vs. a
second ticker (per-name tilt layer) -- Nikolas's request, 2026-07-22.

Two different signals feed these two "conviction" measures, and they're not
directly comparable in derivation -- only in shape and in the fact that both sides
have a directional line and a vol line:

  - SPY directional conviction: p_bull - p_bear from the market layer's HMM
    (pipeline/model.py walk_forward_direction, persisted in data/dirpost.parquet).
    Bounded [-1, 1] -- same convention as the Diagnostics tab's "HMM lean" chart
    in app/app.py, reused here for consistency.
  - SPY vol conviction: p_high, the market layer's vol_layer() logistic output
    (pipeline/model.py). NOT separately persisted by run_nightly.py -- reconstructed
    exactly from data/cell_posterior.parquet via the same algebraic identity used to
    BUILD cell_posterior there: cp[cell_hi] = dirpost[direction] * p_high for each of
    the three directions (bear/neut/bull), so summing the three "_hi" columns
    recovers p_high * (p_bear+p_neut+p_bull) = p_high exactly. Bounded [0, 1].
  - <TICKER> directional conviction: the per-name tilt layer's relative-strength
    z-score, (rs_z_short + rs_z_long)/2 -- pipeline/tilt.py's direction_tilt() takes
    this same average and THRESHOLDS it into bull/neut/bear; this plots the
    continuous value underneath that threshold. Unbounded, typically roughly -3..3.
  - <TICKER> vol conviction: the per-name tilt layer's realized-vol percentile rank
    -- pipeline/tilt.py's name_vol_state() thresholds this same percentile into
    hi/lo; this plots the continuous percentile underneath. Bounded [0, 1].

Only price history is needed for the per-name side -- pipeline.tilt's RS z-score and
vol percentile are pure functions of name_ret + market_ret (SPY's own log return),
no VX curve or HMM refit required -- so this fetches SPY + TICKER prices live and
computes historically, the same math as run_nightly.py's per-name block but for one
name and without the full market-layer machinery (much lighter/faster).

2026-07-22, second pass (two fixes after the first run):

  1. The chart showed an odd diagonal jump across 2020-2022 in the SPY panel. This
     is the SAME class of bug found and fixed earlier this session in
     backtest_per_ticker.py's cell_mix_diagnostic(): building `directional` and `vol`
     into one DataFrame and calling .dropna() on it, then plotting the RESULT, means
     any date where EITHER series is missing gets dropped from BOTH before plotting --
     if there's a real multi-year gap in one of them (see the gap report this version
     prints at startup -- likely cell_posterior.parquet specifically, since it depends
     on VX-curve-derived features that had known coverage problems earlier this
     session, while dirpost.parquet only depends on SPY's own return and shouldn't
     have the same gap), matplotlib just draws a straight line across the missing
     stretch in the OUTPUT chart, which looks like real data rather than a hole. Fixed
     by reindexing each series onto its own full calendar (not intersecting via
     dropna) before plotting, so genuine gaps now show as a BREAK in the line, not a
     bridge -- and by printing an explicit gap report per source series at startup so
     the actual extent of any real gap is visible in the console, not just inferred
     from the chart.
  2. Added SPY price (and the second ticker's own price) as a third, right-offset axis
     on each panel, per Nikolas's request -- lets you eyeball whether conviction swings
     line up with actual price moves (e.g. does high vol-conviction coincide with a
     selloff, does directional conviction turn before or after price does).

2026-07-22, third pass: "shade the SPY instead of overlaying the oscillators on the
price action." Replaced the two conviction OSCILLATOR LINES (directional, vol) drawn
on top of a muted price line with the reverse: price is now the single bold
foreground line, and conviction is read off a background SHADING behind it --
color hue = directional conviction (red=bearish -> green=bullish, normalized per
panel), shading opacity = vol conviction magnitude (more opaque = higher P(high-vol)
/ vol percentile). Conviction data is aligned onto price's own trading-day index via
a short (5-session) ffill so weekend/holiday gaps in price don't punch transparent
holes; alignment gaps beyond that stay fully transparent rather than guessed at,
same honesty principle as the gap report below. See _conviction_shading().

Run locally:
    python research/plot_conviction_over_time.py [TICKER]
    (TICKER defaults to AAPL if not given)
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib
matplotlib.use("Agg")  # headless -- this script just saves a PNG, no display needed
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
import pandas as pd
import yaml

from pipeline.data_pull import pull_prices
from pipeline.tilt import relative_strength_zscore

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"
CONFIG_PATH = REPO_ROOT / "config.yaml"

DEFAULT_TICKER = "AAPL"
GAP_REPORT_THRESHOLD_DAYS = 10  # calendar days -- flags anything wider than a long weekend


def load_config() -> dict:
    with open(CONFIG_PATH) as fh:
        return yaml.safe_load(fh)


def _report_gaps(s: pd.Series, label: str) -> None:
    """Prints any calendar-day gaps wider than GAP_REPORT_THRESHOLD_DAYS in a
    date-indexed series' own native coverage -- run on RAW series before any
    dropna/merge, so a gap can be attributed to a specific source file rather than
    discovered only as a mysterious bridge in the final chart."""
    if s.empty:
        print(f"  {label}: EMPTY")
        return
    idx = s.dropna().index.sort_values()
    print(f"  {label}: {len(idx)} non-null rows, {idx.min().date()} to {idx.max().date()}")
    if len(idx) < 2:
        return
    diffs = idx.to_series().diff().dt.days
    gaps = diffs[diffs > GAP_REPORT_THRESHOLD_DAYS]
    if gaps.empty:
        print(f"    no gaps > {GAP_REPORT_THRESHOLD_DAYS}d")
    for end_date, gap_days in gaps.items():
        start_date = end_date - pd.Timedelta(days=int(gap_days))
        print(f"    GAP: {int(gap_days)}d, {start_date.date()} to {end_date.date()}")


def _reindex_full_range(df: pd.DataFrame) -> pd.DataFrame:
    """Reindexes onto a continuous calendar-day range spanning the data's own
    min/max date, WITHOUT dropna -- leaves genuine gaps as NaN so matplotlib breaks
    the line there instead of bridging it. (Weekends/holidays show as NaN too, but
    at daily resolution over a 15-20yr chart those are invisible; only a real
    multi-week-or-longer gap is visually distinguishable, which is exactly what we
    want to catch.)"""
    full_idx = pd.date_range(df.index.min(), df.index.max(), freq="D")
    return df.reindex(full_idx)


def spy_conviction_series() -> pd.DataFrame:
    """Reads the market layer's ALREADY-PERSISTED output -- no live fetch, no
    recomputation, just the two series described in the module docstring above.
    SPY's own price (for the third-axis overlay) is handled separately in main()
    via the live price fetch already needed for the ticker side."""
    dirpost = pd.read_parquet(DATA_DIR / "dirpost.parquet")
    dirpost.index = pd.to_datetime(dirpost.index)
    cp = pd.read_parquet(DATA_DIR / "cell_posterior.parquet")
    cp.index = pd.to_datetime(cp.index)

    directional_raw = dirpost["p_bull"] - dirpost["p_bear"]
    vol_raw = cp[["bear_hi", "neut_hi", "bull_hi"]].sum(axis=1)

    print("\nGap report -- SPY market layer (raw, pre-merge):")
    _report_gaps(directional_raw, "dirpost (p_bull - p_bear)")
    _report_gaps(vol_raw, "cell_posterior (p_high, reconstructed)")

    df = pd.DataFrame({"directional": directional_raw, "vol": vol_raw})
    return _reindex_full_range(df)


def name_conviction_series(ticker: str, cfg: dict, prices: pd.DataFrame) -> pd.DataFrame:
    """Recomputes the per-name tilt layer's two CONTINUOUS inputs historically
    (config.yaml's tilt_layer/vol_layer params, same values run_nightly.py uses) --
    these aren't persisted by the nightly pipeline, only the final thresholded cell
    label is (data/name_cells.parquet), so there's no shortcut to reading them.
    `prices` is the already-fetched SPY+ticker frame from main() -- no separate
    fetch here."""
    tl = cfg["tilt_layer"]
    vl = cfg["vol_layer"]

    market_ret = np.log(prices["SPY"]).diff()
    name_ret = np.log(prices[ticker]).diff().dropna()
    name_rv20 = name_ret.rolling(vl["rv_window"]).std() * np.sqrt(252)

    rs_z_short = relative_strength_zscore(name_ret, market_ret, tl["rs_window_short"], tl["rs_z_window"])
    rs_z_long = relative_strength_zscore(name_ret, market_ret, tl["rs_window_long"], tl["rs_z_window"])
    directional_raw = (rs_z_short + rs_z_long) / 2
    vol_raw = name_rv20.rolling(tl["name_vol_pct_window"]).rank(pct=True)

    print(f"\nGap report -- {ticker} tilt layer (raw, pre-merge):")
    _report_gaps(directional_raw, f"{ticker} RS z-score")
    _report_gaps(vol_raw, f"{ticker} vol percentile")

    df = pd.DataFrame({"directional": directional_raw, "vol": vol_raw})
    return _reindex_full_range(df)


def _conviction_shading(ax, price: pd.Series, conviction_df: pd.DataFrame, direction_clip: float,
                         cmap_name: str = "RdYlGn") -> None:
    """Background shading behind the price line -- replaces the earlier two-
    oscillator-lines-over-a-muted-price-line design. Color hue encodes directional
    conviction (red=bearish -> green=bullish, normalized to +/-direction_clip so
    SPY's bounded [-1,1] posterior and the name's unbounded RS z-score both map to
    the same visual scale). Shading OPACITY encodes vol conviction magnitude (more
    opaque = higher P(high-vol) / vol percentile) -- so a calm, low-conviction
    stretch fades toward invisible regardless of direction, and a loud, high-
    conviction stretch reads as a bold color block.

    conviction_df is aligned onto PRICE's own trading-day index (not the other way
    around) via a short ffill (limit=5 sessions) -- long enough to bridge normal
    weekend/holiday spacing, short enough that a real multi-week gap (see the gap
    report printed elsewhere in this file) stays unshaded/transparent rather than
    silently painted over."""
    if price is None or price.empty:
        return
    aligned = conviction_df.reindex(price.index, method="ffill", limit=5)
    direction = (aligned["directional"].clip(-direction_clip, direction_clip) / direction_clip).fillna(0.0)
    vol = np.nan_to_num(aligned["vol"].clip(0, 1).values, nan=0.0)
    missing = aligned["directional"].isna().values

    cmap = plt.get_cmap(cmap_name)
    norm = Normalize(vmin=-1, vmax=1)
    colors = cmap(norm(direction.values))
    # Opacity = floor + vol*scale, not vol*scale alone. Vol conviction spends most of
    # history well under 0.5 (calm markets are the common case, crisis spikes the rare
    # one), so a pure vol*0.55 mapping left most of the chart looking blank -- visually
    # indistinguishable from an actual data gap (alpha=0 below), which defeats the point
    # of reserving alpha=0 specifically for real gaps. The floor guarantees every day with
    # real data reads as at least a faint tint; only true gaps go fully transparent.
    ALPHA_FLOOR, ALPHA_SCALE = 0.14, 0.46
    colors[:, 3] = ALPHA_FLOOR + vol * ALPHA_SCALE
    colors[missing, 3] = 0.0  # real gaps (beyond the 5-session ffill) render fully transparent, not guessed

    y0, y1 = float(price.min()) * 0.97, float(price.max()) * 1.03
    x0, x1 = mdates.date2num(price.index[0]), mdates.date2num(price.index[-1])
    ax.imshow(colors.reshape(1, -1, 4), extent=[x0, x1, y0, y1], aspect="auto",
              origin="lower", zorder=0, interpolation="nearest")
    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)
    ax.xaxis_date()

    ax.plot(price.index, price.values, color="black", linewidth=1.3, zorder=3)
    ax.set_ylabel("Price ($)", fontsize=9)
    ax.tick_params(axis="y", labelsize=8)


def plot_conviction(
    spy_df: pd.DataFrame, name_df: pd.DataFrame, ticker: str,
    spy_price: pd.Series, name_price: pd.Series, out_path: Path,
) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(14, 9), sharex=False)

    # direction_clip normalizes each panel's directional series to +/-1 before the
    # colormap: SPY's p_bull-p_bear is already bounded [-1,1]; the name's RS
    # z-score is unbounded but rarely exceeds +/-3, so that's used as its clip.
    panels = [
        (axes[0], spy_df, "SPY (market layer)", spy_price, 1.0),
        (axes[1], name_df, f"{ticker} (per-name tilt layer)", name_price, 3.0),
    ]
    for ax, df, title, price, direction_clip in panels:
        _conviction_shading(ax, price, df, direction_clip=direction_clip)
        ax.set_title(title, fontsize=11)

        legend_handles = [
            Patch(facecolor="tab:green", alpha=0.55, label="Bullish conviction"),
            Patch(facecolor="tab:red", alpha=0.55, label="Bearish conviction"),
            Line2D([0], [0], color="black", linewidth=1.3, label="Price"),
        ]
        ax.legend(handles=legend_handles, loc="upper left", fontsize=7.5, framealpha=0.85)

    fig.suptitle(
        "Directional + vol conviction as background shading, price as the foreground line\n"
        "(color = direction, shade opacity = vol conviction; SPY vs. name use different "
        "derivations -- see module docstring)",
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Wrote {out_path}")


def main():
    ticker = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_TICKER
    cfg = load_config()
    start = cfg["data"]["start_date"]

    print(f"Fetching SPY + {ticker} price history from {start}...")
    t0 = time.time()
    prices, source = pull_prices(["SPY", ticker], start)
    print(f"Got prices from {source} in {time.time() - t0:.1f}s")

    print("Loading SPY conviction series from persisted market-layer output...")
    spy_df = spy_conviction_series()
    print(f"SPY: {len(spy_df)} rows (calendar-reindexed), {spy_df.index.min().date()} to {spy_df.index.max().date()}")

    name_df = name_conviction_series(ticker, cfg, prices)
    print(f"{ticker}: {len(name_df)} rows (calendar-reindexed), "
          f"{name_df.index.min().date()} to {name_df.index.max().date()}")

    out_path = REPO_ROOT / "research" / f"conviction_over_time_SPY_vs_{ticker}.png"
    plot_conviction(spy_df, name_df, ticker, prices["SPY"].dropna(), prices[ticker].dropna(), out_path)


if __name__ == "__main__":
    main()
