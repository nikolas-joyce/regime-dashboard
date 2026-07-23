"""
Per-ticker backtest of the regime dashboard's structure matrix -- delta-hedge-proxy
version (2026-07-22, second rewrite -- see git history for the Black-Scholes
original and the first delta-proxy pass).

Rewrite #2, after Nikolas flagged that baseline_sharpe was still implausibly high
even after removing BS/IV (rewrite #1). Root cause found: baseline_sharpe and
buyhold_sharpe were printing IDENTICAL to 6 decimal places for every name, which
is a mathematical necessity, not a finding -- the fixed baseline holds a CONSTANT
net_delta=0.15 on every trade, and Sharpe is scale-invariant to a constant positive
multiplier (mean(k*r)/std(k*r) == mean(r)/std(r) for any k>0). So "baseline vs.
buy-and-hold" was structurally incapable of differing on Sharpe; only magnitude
differs. The real open question was whether buy-and-hold ITSELF was showing
inflated Sharpes, and separately whether the overlapping-entry annualization
(sqrt(252/TRADE_STEP) applied to 21-day holds entered every 5 days -- up to ~4
concurrently open) was inflating everything versus a proper daily-return estimate.

Fixes in this version:
  1. Added true_daily_sharpe(): a properly non-overlapping, full-daily-return
     Sharpe per name, computed independently of the trade-entry cadence. This is
     the correct number to compare against "SPY is 0.6-0.8" -- the overlapping-
     entry buyhold_sharpe is kept for context but should not be read as ground
     truth.
  2. Added equity-curve plotting (plot_equity_curves()): cumulative matrix P&L
     (summed at trade entry dates) vs. the underlying's own cumulative daily
     return, one small panel per name, saved to
     research/backtest_equity_curves.png. Visual sanity check per Nikolas's
     request -- makes any remaining implausibility (a curve that jumps, or
     diverges from the underlying in a way the regime logic can't explain)
     immediately visible instead of inferred from summary stats.

Everything else (net_delta = sum(pos*delta), delta_proxy_return, TRADE_DTE=21/
TRADE_STEP=5 cadence, structure_matrix[name cell][market-posterior tier] lookup)
is unchanged from rewrite #1 -- see that version's docstring (git history) for the
full rationale on dropping BS/IV pricing.

Rewrite #3 (2026-07-22, same day): Nikolas looked at the equity-curve grid and
flagged that trades don't look like they're behaving sensibly relative to the
daily probabilities. Two fixes, not a rewrite of the trading logic itself:
  1. plot_equity_curves() now puts matrix and buy-hold on separate y-axes. On the
     old shared axis, the matrix curve was visually crushed to a flat line for
     almost every name -- net_delta tops out around +-0.30 (a defined-risk options
     overlay) vs. buy-hold's full 1.0x exposure compounding to 5x-1000x+ over 15-20
     years, so the matrix's real (much smaller) variation was invisible, not
     evidence it wasn't trading.
  2. Added regime_distribution_diagnostic(): the actual answer to "is sizing
     consistent with the daily posterior" -- reports how often each (cell, tier)
     combination fires, its net_delta, what fraction of all trades are net_delta==0
     (no_trade), and the raw distribution of the market posterior against the
     confidence_bands thresholds. Also documents a real config inconsistency found
     while building this: three cells' "_smaller" (moderate) structure variants --
     bull_lo, bear_hi, bear_lo -- use IDENTICAL legs to their high_confidence
     counterpart in config.yaml, so those three cells don't actually size down
     between tiers (only bull_hi and neut_hi do). Flagged for Nikolas to confirm
     intentional vs. bug -- not changed here.

Run locally (not in a sandbox -- needs yfinance + real historical price data, and
matplotlib for the equity-curve figure, already in requirements.txt):
    python research/backtest_per_ticker.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib
matplotlib.use("Agg")  # headless -- this script just saves a PNG, no display needed
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

from pipeline.data_pull import pull_prices
from pipeline.matrix import confidence_tier

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"
CONFIG_PATH = REPO_ROOT / "config.yaml"

TRADE_DTE = 21    # trading days to expiry -- matches TEST 5
TRADE_STEP = 5     # new entry every N trading days -- matches TEST 5

BASELINE_LEGS = [{"cp": "put", "delta": -0.30, "pos": -1}, {"cp": "put", "delta": -0.15, "pos": 1}]


def load_config() -> dict:
    with open(CONFIG_PATH) as fh:
        return yaml.safe_load(fh)


def load_persisted():
    name_cells = pd.read_parquet(DATA_DIR / "name_cells.parquet")
    name_cells["Date"] = pd.to_datetime(name_cells["Date"])

    cell_posterior = pd.read_parquet(DATA_DIR / "cell_posterior.parquet")
    cell_posterior.index = pd.to_datetime(cell_posterior.index)

    committed_df = pd.read_parquet(DATA_DIR / "committed_regime.parquet")
    committed_df.index = pd.to_datetime(committed_df.index)
    committed = committed_df["regime"]

    return name_cells, cell_posterior, committed


def market_posterior_series(cell_posterior: pd.DataFrame, committed: pd.Series) -> pd.Series:
    """Posterior probability of whatever cell was actually committed, per date --
    same quantity as state.json's top-level "posterior" field, reconstructed for
    every historical date instead of just the latest one."""
    aligned = cell_posterior.reindex(committed.index)
    vals = [aligned.loc[d, c] if c in aligned.columns else np.nan for d, c in committed.items()]
    return pd.Series(vals, index=committed.index)


def net_delta(legs: list[dict]) -> float:
    """Sum of pos * delta across a leg-set -- the structure's net directional
    exposure, in units of the underlying. Empty legs (no_trade) -> 0.0."""
    return sum(leg["pos"] * leg["delta"] for leg in legs)


def delta_proxy_return(nd: float, S0: float, ST: float) -> float:
    """Static-position approximation: nd units of the underlying held from S0 to
    ST. No options pricing, no vol assumption -- see module docstring for why."""
    return nd * (ST / S0 - 1.0)


def true_daily_sharpe(price: pd.Series) -> float:
    """Properly non-overlapping Sharpe from plain daily returns -- independent of
    TRADE_STEP/TRADE_DTE entirely, so not subject to the overlapping-entry
    autocorrelation inflation the rest of this script's Sharpes carry. This is the
    number to compare against "SPY is 0.6-0.8", not buyhold_sharpe."""
    ret = price.pct_change().dropna()
    if len(ret) < 2 or ret.std() == 0:
        return float("nan")
    return float(ret.mean() / ret.std() * np.sqrt(252))


def backtest_one_name(
    ticker: str, price: pd.Series, name_cell: pd.Series, tier_series: pd.Series,
    structure_matrix: dict,
) -> tuple[dict, list[pd.Timestamp], list[float], list[tuple[str, str, float]]]:
    idx = price.index
    n = len(idx)

    trade_dates = []
    matrix_pnl, matrix_delta = [], []
    baseline_pnl, baseline_delta = [], []
    buyhold_pnl = []
    cell_tier_log = []   # (cell, tier, net_delta) per entry -- for the distribution diagnostic
    for t in range(0, n - TRADE_DTE, TRADE_STEP):
        d0 = idx[t]
        S0, ST = price.iloc[t], price.iloc[t + TRADE_DTE]
        if not (np.isfinite(S0) and np.isfinite(ST)):
            continue
        cell = name_cell.get(d0)
        tier = tier_series.get(d0)
        if cell is None or tier is None or pd.isna(cell) or pd.isna(tier):
            continue

        legs = structure_matrix.get(cell, {}).get(tier, {}).get("legs", [])
        nd = net_delta(legs)
        matrix_pnl.append(delta_proxy_return(nd, S0, ST))
        matrix_delta.append(nd)
        cell_tier_log.append((cell, tier, nd))

        nd_b = net_delta(BASELINE_LEGS)
        baseline_pnl.append(delta_proxy_return(nd_b, S0, ST))
        baseline_delta.append(nd_b)

        buyhold_pnl.append(delta_proxy_return(1.0, S0, ST))
        trade_dates.append(d0)

    summary = _summarize(
        ticker, price, matrix_pnl, matrix_delta, baseline_pnl, baseline_delta, buyhold_pnl,
    )
    return summary, trade_dates, matrix_pnl, cell_tier_log


def regime_distribution_diagnostic(
    all_cell_tier_logs: dict[str, list[tuple[str, str, float]]],
    posterior: pd.Series,
    bands: dict,
) -> pd.DataFrame:
    """Answers the actual question Nikolas raised: are trades being sized in a way
    that's consistent with the daily posterior probabilities, or is the matrix
    mostly sitting in no_trade / tiny-delta buckets regardless of what the model
    believes?

    Two things this surfaces that summary Sharpes and a shared-axis equity plot
    both hide:
      1. confidence_bands (high=0.75, moderate=0.55, config.yaml) are thresholds on
         the MARKET-level posterior, not the per-name cell -- so a name can carry a
         strong idiosyncratic bull_hi/bear_lo tilt and still get sized down to
         moderate or zeroed to no_trade purely because the MARKET regime isn't
         confident that day. If the market posterior rarely clears 0.75, high-
         confidence (largest net_delta) structures will fire rarely across the
         entire 50-name universe, independent of any single name's own signal.
      2. Three cells' "_smaller" (moderate) structure variants -- bull_lo,
         bear_hi, bear_lo -- use IDENTICAL legs/deltas to their high_confidence
         counterpart in config.yaml's structure_matrix (only bull_hi and neut_hi
         actually reduce net_delta between tiers). So for those three cells,
         "moderate confidence" and "high confidence" are financially the same
         trade in this backtest -- the tier distinction only matters there insofar
         as it decides no_trade vs. trade at low confidence.
    """
    rows = []
    for ticker, log in all_cell_tier_logs.items():
        for cell, tier, nd in log:
            rows.append({"ticker": ticker, "cell": cell, "tier": tier, "net_delta": nd})
    df = pd.DataFrame(rows)

    print("\n=== Regime/tier distribution across all trades, all names ===")
    dist = (
        df.groupby(["cell", "tier"])
        .agg(n=("net_delta", "size"), avg_net_delta=("net_delta", "mean"))
        .reset_index()
    )
    dist["pct_of_all_trades"] = 100 * dist["n"] / len(df)
    dist = dist.sort_values("n", ascending=False)
    print(dist.to_string(index=False))

    zero_pct = 100 * (df["net_delta"] == 0).mean()
    print(f"\n{zero_pct:.1f}% of all (name, entry-date) trades had net_delta == 0 (no_trade cell/tier).")
    print(f"Mean |net_delta| across all non-zero trades: {df.loc[df['net_delta'] != 0, 'net_delta'].abs().mean():.3f} "
          f"(vs. buy-hold's implicit 1.0 -- this ratio alone compresses the matrix curve on a shared axis).")

    print("\n=== Market posterior distribution (drives the tier, hence a lot of the above) ===")
    p = posterior.dropna()
    high_pct = 100 * (p >= bands["high"]).mean()
    mod_pct = 100 * ((p >= bands["moderate"]) & (p < bands["high"])).mean()
    low_pct = 100 * (p < bands["moderate"]).mean()
    print(f"high_confidence  (posterior >= {bands['high']}): {high_pct:.1f}% of days")
    print(f"moderate_confidence ({bands['moderate']} <= posterior < {bands['high']}): {mod_pct:.1f}% of days")
    print(f"low_confidence   (posterior < {bands['moderate']}): {low_pct:.1f}% of days")
    print("If low_confidence dominates, most cells resolve to no_trade regardless of the per-name signal --")
    print("that's a market-posterior-threshold effect, not a per-name bug.")

    return dist


def direction_summary_diagnostic(all_cell_tier_logs: dict[str, list[tuple[str, str, float]]]) -> None:
    """Nikolas's follow-up question: are we actually leaning into conviction --
    bigger long exposure in bullish high-confidence regimes, bigger short/hedge
    exposure in bearish high-confidence regimes -- or does the matrix spend most of
    its time near flat regardless of confidence? Cross-tabs net_delta sign and
    magnitude against the confidence tier directly, across all names."""
    rows = []
    for ticker, log in all_cell_tier_logs.items():
        for cell, tier, nd in log:
            rows.append({"ticker": ticker, "cell": cell, "tier": tier, "net_delta": nd})
    df = pd.DataFrame(rows)

    long_pct = 100 * (df["net_delta"] > 0).mean()
    short_pct = 100 * (df["net_delta"] < 0).mean()
    flat_pct = 100 * (df["net_delta"] == 0).mean()
    avg_long = df.loc[df["net_delta"] > 0, "net_delta"].mean()
    avg_short = df.loc[df["net_delta"] < 0, "net_delta"].mean()

    print("\n=== Direction/conviction summary (all names, all trade-entries) ===")
    print(f"Long (net_delta>0):  {long_pct:.1f}% of trades, avg net_delta {avg_long:+.3f}")
    print(f"Short (net_delta<0): {short_pct:.1f}% of trades, avg net_delta {avg_short:+.3f}")
    print(f"Flat (net_delta==0): {flat_pct:.1f}% of trades")

    print("\nAvg |net_delta| and %flat, by confidence tier (does size scale with conviction?):")
    by_tier = df.groupby("tier").agg(
        n=("net_delta", "size"),
        avg_abs_delta=("net_delta", lambda x: x.abs().mean()),
        pct_flat=("net_delta", lambda x: 100 * (x == 0).mean()),
    ).reindex(["high_confidence", "moderate_confidence", "low_confidence"])
    print(by_tier.to_string())
    print(
        "\nIf avg_abs_delta doesn't rise from low->high confidence, and/or pct_flat stays high even at "
        "high_confidence, sizing is NOT scaling with conviction -- it's gated on WHICH cell fires, not how "
        "strongly. That's consistent with a 'wait for explicit confirmation, fixed size per structure' "
        "design rather than a 'default-exposed, scale continuously with conviction' design -- worth deciding "
        "which one this should be."
    )


def cell_mix_diagnostic(
    name_cells: pd.DataFrame, prices: pd.DataFrame, universe: list[str], out_path: Path,
) -> tuple[float, pd.DataFrame]:
    """Tests a specific hypothesis about WHY exposure plateaus even during periods
    where names keep rallying in absolute terms (visible in
    backtest_exposure_levels.png as long flat green stretches during clear
    uptrends, e.g. AAPL/XOM 2020-2024): the per-name cell comes from the tilt
    layer's RELATIVE-STRENGTH z-score (config.yaml tilt_layer: rs_z vs a rolling
    window, thresholded at +-0.5sd), not absolute price direction. In a low-
    dispersion market where most/all 50 names are rallying together, relative
    strength across the basket compresses toward zero even though every name is
    individually up a lot -- so names read "neutral" (no_trade) exactly when the
    tape is calm-and-broadly-bullish, which is precisely the environment you'd most
    want default long exposure in, not the environment that zeroes it out.

    But the payoff (delta_proxy_return) is priced against the ABSOLUTE underlying,
    not relative to the basket -- so there's a definitional mismatch between what
    the entry signal measures (relative outperformance) and what the position
    actually earns (absolute return). This is a candidate root cause for "still
    doesn't look correct," distinct from the tier-gating and fixed-notional issues
    already surfaced.

    Checks it directly: cross-sectional %neutral per date vs. an equal-weight,
    unrebalanced basket's rolling 63-trading-day return. A meaningfully positive
    correlation supports the hypothesis.

    2026-07-22 fix (after first run): two issues in the first version of this plot.
    (1) basket_curve used an ARITHMETIC mean of raw price ratios -- with 50 names of
    wildly different total returns over 20 years (NVDA alone is roughly 100x+),
    a simple average of price ratios is dominated by whichever single name has
    flown the furthest, which is why the first chart showed the "basket" pinned
    near zero for a decade and then rocketing -- that's NVDA's shape bleeding
    through an arithmetic mean, not a real basket-level signal. Switched to a
    GEOMETRIC equal-weight index (mean of log price-ratios, then exp) -- the
    standard fix for cross-sectionally averaging returns of assets with very
    different compounding, and much less sensitive to one outlier. (2) the first
    version plotted the dropna()-aligned frame, which bridges any real data gap
    with a straight line -- visible as the odd diagonal jump around 2020 in the
    first chart. Now plots pct_neut and basket_curve on their own native index
    (reindexed to a shared full range) so genuine gaps show as breaks, not
    interpolated jumps; only the correlation itself is computed on the
    dropna()'d frame."""
    pivot = name_cells.pivot_table(index="Date", columns="ticker", values="cell", aggfunc="first")
    is_neut = pivot.isin(["neut_hi", "neut_lo"])
    pct_neut = 100 * is_neut.sum(axis=1) / pivot.notna().sum(axis=1).replace(0, np.nan)

    basket_cols = [t for t in universe if t in prices.columns]
    basket = prices[basket_cols].dropna(how="all")
    log_norm = np.log(basket / basket.bfill().iloc[0])
    basket_curve = np.exp(log_norm.mean(axis=1))  # geometric equal-weight -- robust to one outlier name
    basket_ret_63d = basket_curve.pct_change(63)

    full_idx = pct_neut.index.union(basket_curve.index)
    pct_neut_plot = pct_neut.reindex(full_idx)
    basket_plot = basket_curve.reindex(full_idx)

    aligned = pd.DataFrame({"pct_neut": pct_neut, "basket_ret_63d": basket_ret_63d}).dropna()
    corr = float(aligned["pct_neut"].corr(aligned["basket_ret_63d"]))
    gap_days = len(full_idx) - len(aligned)

    print("\n=== Cell-mix vs. basket-trend diagnostic ===")
    print(f"Correlation(%names classified neutral, trailing 63d geometric-basket return): {corr:+.3f}")
    print(f"({gap_days} of {len(full_idx)} dates dropped for the correlation due to missing data on either side)")
    if corr > 0.15:
        print("Meaningfully POSITIVE: the fraction of names sitting in a no-trade neutral cell rises")
        print("specifically when the basket has been rallying hard -- i.e. the RS-based signal reads")
        print("'neutral' during calm, broadly-bullish stretches (low cross-sectional dispersion), which")
        print("is exactly when backtest_exposure_levels.png shows the long flat plateaus.")
    else:
        print("Not strongly positive -- the neutral-during-rally hypothesis isn't clearly supported by this")
        print("correlation alone; the plateaus are more likely driven by the tier-gating / fixed-notional")
        print("issues already surfaced (see regime_distribution_diagnostic / direction_summary_diagnostic).")

    fig, ax1 = plt.subplots(figsize=(11, 4))
    ax1.plot(pct_neut_plot.index, pct_neut_plot.values, color="tab:purple", linewidth=0.8)
    ax1.set_ylabel("% names in neut_hi/neut_lo", color="tab:purple")
    ax1.tick_params(axis="y", labelcolor="tab:purple")
    ax2 = ax1.twinx()
    ax2.plot(basket_plot.index, basket_plot.values, color="tab:blue", linewidth=0.8)
    ax2.set_ylabel("geometric equal-weight basket (normalized)", color="tab:blue")
    ax2.tick_params(axis="y", labelcolor="tab:blue")
    fig.suptitle(f"% of names classified neutral vs. basket level  (corr with trailing 63d return: {corr:+.2f})")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)

    return corr, aligned


def _summarize(
    ticker: str, price: pd.Series, matrix_pnl: list[float], matrix_delta: list[float],
    baseline_pnl: list[float], baseline_delta: list[float], buyhold_pnl: list[float],
) -> dict:
    m, b, h = np.array(matrix_pnl), np.array(baseline_pnl), np.array(buyhold_pnl)
    ann_factor = np.sqrt(252 / TRADE_STEP)

    def sharpe(x):
        return float(x.mean() / x.std() * ann_factor) if len(x) > 1 and x.std() > 0 else float("nan")

    def max_dd(x):
        if len(x) == 0:
            return float("nan")
        cum = np.cumsum(x)
        peak = np.maximum.accumulate(cum)
        return float((cum - peak).min())

    return {
        "ticker": ticker,
        "n_trades": len(m),
        "matrix_avg_net_delta": float(np.mean(matrix_delta)) if matrix_delta else float("nan"),
        "matrix_sharpe": sharpe(m),
        "matrix_hit_rate": float((m > 0).mean()) if len(m) else float("nan"),
        "matrix_total_pnl": float(m.sum()) if len(m) else float("nan"),
        "matrix_max_dd": max_dd(m),
        "baseline_avg_net_delta": float(np.mean(baseline_delta)) if baseline_delta else float("nan"),
        "baseline_sharpe": sharpe(b),
        "baseline_total_pnl": float(b.sum()) if len(b) else float("nan"),
        "baseline_max_dd": max_dd(b),
        "buyhold_sharpe_overlapping": sharpe(h),
        "buyhold_sharpe_true_daily": true_daily_sharpe(price),
        "buyhold_total_pnl": float(h.sum()) if len(h) else float("nan"),
        "buyhold_max_dd": max_dd(h),
        "edge_vs_baseline": sharpe(m) - sharpe(b),
        "edge_vs_buyhold": sharpe(m) - sharpe(h),
    }


def plot_equity_curves(
    universe: list[str], prices: pd.DataFrame,
    trade_data: dict[str, tuple[list[pd.Timestamp], list[float]]],
    all_cell_tier_logs: dict[str, list[tuple[str, str, float]]],
    out_path: Path,
) -> None:
    """Small-multiples grid: matrix strategy cumulative P&L (summed at trade entry
    dates) vs. the underlying's own cumulative daily return, one panel per name.
    Visual sanity check -- a bug or an implausible result should be visible as a
    curve shape that doesn't make sense, not just a suspicious summary number.

    2026-07-22 fix #1: matrix and buy-hold were originally plotted on a SHARED
    y-axis. Since the matrix's net_delta is capped around +-0.15 to +-0.30 (a
    defined-risk options overlay, never full 1.0 equity exposure) while buy-hold
    compounds to 5x-1000x+ over this history for names like NVDA/AAPL, the matrix
    line was visually crushed to a flat zero line on every panel regardless of what
    it was actually doing -- an axis-scale artifact, not evidence the strategy
    wasn't trading. Now on a separate (twin) y-axis, own color, own scale.

    2026-07-22 fix #2 (same day, Nikolas's follow-up): the single combined matrix
    line conflates long and short periods, so it's impossible to see from the chart
    alone whether the strategy is actually leaning long in bullish stretches and
    short in bearish ones, or just meandering. Now splits the matrix line into three:
    combined (solid), long-only contribution (dashed -- cumulative P&L from periods
    where net_delta>0, flat elsewhere), and short-only contribution (dotted --
    net_delta<0). See regime_distribution_diagnostic() and
    direction_summary_diagnostic() in this file for the numeric version of the same
    question."""
    tickers = [t for t in universe if t in trade_data]
    ncols = 5
    nrows = -(-len(tickers) // ncols)  # ceil division
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 3.4, nrows * 2.4), squeeze=False)

    legend_handles, legend_labels = [], []
    for i, ticker in enumerate(tickers):
        ax = axes[i // ncols][i % ncols]
        dates, pnl = trade_data[ticker]
        if not dates:
            ax.set_title(f"{ticker} (no trades)", fontsize=8)
            ax.axis("off")
            continue

        log = all_cell_tier_logs.get(ticker, [])
        deltas = [nd for _, _, nd in log]
        pnl_long = [p if d > 0 else 0.0 for p, d in zip(pnl, deltas)]
        pnl_short = [p if d < 0 else 0.0 for p, d in zip(pnl, deltas)]

        px = prices[ticker].dropna()
        px_window = px.loc[dates[0]:]
        underlying_curve = px_window / px_window.iloc[0] - 1.0
        strategy_curve = pd.Series(np.cumsum(pnl), index=dates)
        long_curve = pd.Series(np.cumsum(pnl_long), index=dates)
        short_curve = pd.Series(np.cumsum(pnl_short), index=dates)

        l1, = ax.plot(underlying_curve.index, underlying_curve.values,
                       label="buy-hold (left axis)", linewidth=0.8, color="tab:blue")
        ax.tick_params(axis="y", labelcolor="tab:blue", labelsize=6)
        ax.axhline(0, color="gray", linewidth=0.5, linestyle=":")

        ax2 = ax.twinx()
        l2, = ax2.plot(strategy_curve.index, strategy_curve.values,
                        label="matrix combined (right axis)", linewidth=0.9, color="tab:orange")
        l3, = ax2.plot(long_curve.index, long_curve.values,
                        label="long-only contribution", linewidth=0.8, color="tab:green", linestyle="--")
        l4, = ax2.plot(short_curve.index, short_curve.values,
                        label="short-only contribution", linewidth=0.8, color="tab:red", linestyle=":")
        ax2.tick_params(axis="y", labelcolor="tab:orange", labelsize=6)
        ax2.axhline(0, color="tab:orange", linewidth=0.4, linestyle=":", alpha=0.4)

        ax.set_title(ticker, fontsize=9)
        ax.tick_params(axis="x", labelsize=6)
        if not legend_handles:
            legend_handles = [l1, l2, l3, l4]
            legend_labels = ["buy-hold (left axis)", "matrix combined (right axis)",
                              "long-only contribution", "short-only contribution"]

    # Turn off any unused trailing axes (grid doesn't divide evenly into len(tickers))
    for j in range(len(tickers), nrows * ncols):
        axes[j // ncols][j % ncols].axis("off")

    if legend_handles:
        fig.legend(legend_handles, legend_labels, loc="upper center", ncol=4, fontsize=9)
    fig.suptitle(
        "Matrix strategy cumulative P&L (combined + long/short split) vs. buy-and-hold, per name\n"
        "(separate y-axes -- matrix runs a fraction of full 1.0x exposure by design, so scales differ)",
        y=1.0, fontsize=10,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_exposure_levels(
    universe: list[str],
    trade_data: dict[str, tuple[list[pd.Timestamp], list[float]]],
    all_cell_tier_logs: dict[str, list[tuple[str, str, float]]],
    out_path: Path,
) -> None:
    """The chart that most directly answers 'are we largest during highest
    conviction, and are extended flat stretches real': a step plot of the actual
    net_delta POSITION LEVEL (not cumulative P&L) held at each trade entry, per
    name. Positive region shaded green (long), negative shaded red (short), so
    stretches at/near zero -- and how long they last -- are immediately visible."""
    tickers = [t for t in universe if t in trade_data]
    ncols = 5
    nrows = -(-len(tickers) // ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 3.2, nrows * 2.0), squeeze=False)

    for i, ticker in enumerate(tickers):
        ax = axes[i // ncols][i % ncols]
        dates, _ = trade_data[ticker]
        log = all_cell_tier_logs.get(ticker, [])
        if not dates or not log:
            ax.set_title(f"{ticker} (no trades)", fontsize=8)
            ax.axis("off")
            continue

        nd_series = pd.Series([nd for _, _, nd in log], index=dates)
        ax.step(nd_series.index, nd_series.values, where="post", linewidth=0.8, color="black")
        ax.fill_between(nd_series.index, nd_series.values, 0, where=(nd_series.values >= 0),
                         step="post", color="tab:green", alpha=0.3)
        ax.fill_between(nd_series.index, nd_series.values, 0, where=(nd_series.values < 0),
                         step="post", color="tab:red", alpha=0.3)
        ax.axhline(0, color="gray", linewidth=0.5)
        ax.set_title(ticker, fontsize=9)
        ax.tick_params(labelsize=6)

    for j in range(len(tickers), nrows * ncols):
        axes[j // ncols][j % ncols].axis("off")

    fig.suptitle(
        "Net delta position level held at each trade entry, per name\n"
        "(green=long, red=short, white/thin=flat/no_trade -- the actual conviction question)",
        y=1.0, fontsize=10,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main():
    cfg = load_config()
    universe = cfg["data"]["underlyings_names"]
    structure_matrix = cfg["structure_matrix"]
    bands = cfg["confidence_bands"]
    start = cfg["data"]["start_date"]

    print(f"Fetching price history for {len(universe)} names from {start}...")
    print("(yfinance gives no progress feedback mid-download -- silent but not stuck; "
          "typically 15-60s depending on network/rate limits)")
    t0 = time.time()
    prices, source = pull_prices(universe, start)
    print(f"Got prices from {source} in {time.time() - t0:.1f}s: {prices.shape[0]} rows, {prices.shape[1]} names")

    name_cells, cell_posterior, committed = load_persisted()
    posterior = market_posterior_series(cell_posterior, committed)
    tier_series = posterior.apply(lambda p: confidence_tier(p, bands) if pd.notna(p) else None)

    print(f"\nRunning per-ticker backtest ({len(universe)} names -- fast, the fetch above was the slow part)...")
    t1 = time.time()
    results = []
    trade_data = {}
    all_cell_tier_logs = {}
    for i, ticker in enumerate(universe, start=1):
        if ticker not in prices.columns:
            print(f"  [{i}/{len(universe)}] {ticker}: no price data, skipping")
            continue
        px = prices[ticker].dropna()
        nc = name_cells[name_cells["ticker"] == ticker].set_index("Date")["cell"]
        res, trade_dates, matrix_pnl, cell_tier_log = backtest_one_name(
            ticker, px, nc, tier_series, structure_matrix,
        )
        results.append(res)
        trade_data[ticker] = (trade_dates, matrix_pnl)
        all_cell_tier_logs[ticker] = cell_tier_log
        print(f"  [{i}/{len(universe)}] {ticker}: n={res['n_trades']}, matrix_sharpe={res['matrix_sharpe']:.2f} "
              f"(avg delta {res['matrix_avg_net_delta']:+.2f}), "
              f"true daily buy-hold sharpe={res['buyhold_sharpe_true_daily']:.2f}")
    print(f"Backtest loop done in {time.time() - t1:.1f}s")

    out = pd.DataFrame(results).sort_values("edge_vs_buyhold", ascending=False)
    out_path = REPO_ROOT / "research" / "backtest_per_ticker_results.csv"
    out.to_csv(out_path, index=False)
    print(f"\nWrote {out_path}")

    dist = regime_distribution_diagnostic(all_cell_tier_logs, posterior, bands)
    dist_path = REPO_ROOT / "research" / "backtest_regime_distribution.csv"
    dist.to_csv(dist_path, index=False)
    print(f"Wrote {dist_path}")

    direction_summary_diagnostic(all_cell_tier_logs)

    cell_mix_path = REPO_ROOT / "research" / "backtest_cell_mix_vs_basket.png"
    cell_mix_diagnostic(name_cells, prices, universe, cell_mix_path)
    print(f"Wrote {cell_mix_path}")

    plot_path = REPO_ROOT / "research" / "backtest_equity_curves.png"
    print(f"\nRendering equity-curve grid (combined + long/short split) to {plot_path}...")
    plot_equity_curves(universe, prices, trade_data, all_cell_tier_logs, plot_path)

    exposure_path = REPO_ROOT / "research" / "backtest_exposure_levels.png"
    print(f"Rendering exposure-level grid to {exposure_path}...")
    plot_exposure_levels(universe, trade_data, all_cell_tier_logs, exposure_path)
    print("Done.")

    print("\n=== Summary ===")
    print(f"Mean matrix Sharpe (overlapping):        {out['matrix_sharpe'].mean():.2f}")
    print(f"Mean baseline Sharpe (overlapping):       {out['baseline_sharpe'].mean():.2f}  "
          f"(mathematically == buy-hold-overlapping Sharpe -- constant leverage, see docstring)")
    print(f"Mean buy-hold Sharpe (overlapping-entry): {out['buyhold_sharpe_overlapping'].mean():.2f}")
    print(f"Mean buy-hold Sharpe (TRUE daily, no overlap): {out['buyhold_sharpe_true_daily'].mean():.2f}  "
          f"<- compare THIS to your SPY 0.6-0.8 expectation")
    print(f"Names where matrix beats buy-and-hold (overlapping): {int((out['edge_vs_buyhold'] > 0).sum())}/{len(out)}")

    cols = ["ticker", "n_trades", "matrix_sharpe", "baseline_sharpe",
            "buyhold_sharpe_overlapping", "buyhold_sharpe_true_daily", "edge_vs_buyhold"]
    print("\nTop 5 by edge vs buy-and-hold:")
    print(out.head(5)[cols].to_string(index=False))
    print("\nBottom 5 by edge vs buy-and-hold:")
    print(out.tail(5)[cols].to_string(index=False))


if __name__ == "__main__":
    main()
