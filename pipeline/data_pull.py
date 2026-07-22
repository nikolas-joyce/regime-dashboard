"""
Regime dashboard — data acquisition.

yfinance is the #1 operational risk flagged in the plan (research/regime-dashboard-plan.md
section 8) -- Yahoo has historically blocked cloud-provider IPs, which is why this pulls
happen in GitHub Actions (not the Streamlit app itself) and fall back to Stooq on failure.
"""
from __future__ import annotations

import logging
import time
from datetime import date
from typing import Optional

import pandas as pd

from pipeline.iv_calc import select_expiries, atm_iv_from_chain, skew_25d_from_chain, dte

logger = logging.getLogger(__name__)


def pull_prices_yfinance(tickers: list[str], start: str) -> pd.DataFrame:
    import yfinance as yf
    px = yf.download(tickers, start=start, auto_adjust=True, progress=False)["Close"]
    if isinstance(px, pd.Series):
        px = px.to_frame(tickers[0])
    return px


def pull_prices_stooq(tickers: list[str], start: str) -> pd.DataFrame:
    """Failover path when yfinance is unreachable (e.g. blocked cloud IP)."""
    import pandas_datareader.data as web
    frames = {}
    for t in tickers:
        try:
            sym = t.lower().lstrip("^") + ".us" if not t.startswith("^") else t
            df = web.DataReader(sym, "stooq", start=start)
            frames[t] = df["Close"].sort_index()
        except Exception as e:  # noqa: BLE001 -- log and continue, one bad ticker shouldn't kill the run
            logger.warning("Stooq failed for %s: %s", t, e)
    return pd.DataFrame(frames)


def pull_prices(tickers: list[str], start: str) -> tuple[pd.DataFrame, str]:
    """Try yfinance first, fall back to Stooq. Returns (prices, source_used)."""
    try:
        px = pull_prices_yfinance(tickers, start)
        if px.dropna(how="all").empty:
            raise RuntimeError("yfinance returned empty frame")
        return px, "yfinance"
    except Exception as e:  # noqa: BLE001
        logger.warning("yfinance failed (%s) -> falling back to Stooq", e)
        return pull_prices_stooq(tickers, start), "stooq"


def pull_vx_curve(start: Optional[str] = None) -> pd.DataFrame:
    """VX1/VX3 futures curve via vix-utils. This is a first-class model input (drift
    model + forecast conditioning) -- a failed pull here should be treated as a pipeline
    error, not silently skipped (per plan section 5 step 2).
    """
    import vix_utils
    try:
        import nest_asyncio
        nest_asyncio.apply()
    except ImportError:
        pass

    # Diagnostic (2026-07-22): a live Action run showed pull_vx_curve() capping at
    # 2026-05-19 while an identical Colab call to the same two vix_utils functions
    # returned data through 2026-07-21. These prints isolate WHERE the truncation
    # happens (raw fetch vs. pivot step) and whether it's a version difference --
    # remove once the discrepancy is root-caused.
    ver = getattr(vix_utils, "__version__", "unknown")
    print(f"[pull_vx_curve] vix_utils version={ver}")

    ts = vix_utils.load_vix_term_structure()
    print(f"[pull_vx_curve] raw ts: {ts.shape[0]} rows, "
          f"Trade Date {ts['Trade Date'].min()} to {ts['Trade Date'].max()}")

    wide = vix_utils.pivot_futures_on_monthly_tenor(ts)
    vx = pd.DataFrame({"VX1": wide[1]["Close"], "VX3": wide[3]["Close"]})
    vx.index = pd.to_datetime(vx.index)
    print(f"[pull_vx_curve] post-pivot vx: {len(vx)} rows, "
          f"{vx.index.min().date()} to {vx.index.max().date()}")

    if start:
        vx = vx.loc[vx.index >= start]
    return vx


def pull_fred_series(series_ids: dict[str, str], api_key: str, start: Optional[str] = None) -> pd.DataFrame:
    from fredapi import Fred
    fred = Fred(api_key=api_key)
    out = {}
    for name, sid in series_ids.items():
        s = fred.get_series(sid)
        if start:
            s = s.loc[s.index >= start]
        out[name] = s
    return pd.DataFrame(out)


def build_feature_frame(prices: pd.DataFrame, vx: pd.DataFrame) -> pd.DataFrame:
    """Assemble the causal feature frame the model module consumes: returns, realized
    vol, VX1-VX3 curve slope (z-scored), VIX percentile, backwardation flag.
    """
    import numpy as np

    feat = pd.DataFrame(index=prices.index)
    feat["ret"] = np.log(prices["SPY"]).diff()
    feat["rv20"] = feat["ret"].rolling(20).std() * np.sqrt(252)
    feat["rv_pct"] = feat["rv20"].rolling(252).rank(pct=True)
    if "VIX" in prices.columns:
        feat["vix"] = prices["VIX"]
        feat["vix_pct"] = feat["vix"].rolling(252).rank(pct=True)
    vx_aligned = vx.reindex(feat.index).ffill()
    feat["slope"] = (vx_aligned["VX3"] - vx_aligned["VX1"]) / vx_aligned["VX1"]
    feat["slope_z"] = (feat["slope"] - feat["slope"].rolling(252).mean()) / feat["slope"].rolling(252).std()
    feat["backwardation"] = (feat["slope"] < 0).astype(float)
    # Match the validated notebook's guarantee (research/regime_dashboard_step0_validation.ipynb,
    # feature-construction cell): drop any row missing slope_z/vix_pct/rv_pct, not just ret/rv20.
    # The notebook's raw price+VX-curve frame comes from an INNER JOIN (px.join(vx, how="inner")),
    # so it never has a leading-NaN-slope window to begin with; ffill() here can't back-fill dates
    # before VX curve history starts, so without this the model layer gets handed rows where
    # slope_z is NaN. That silently broke two things in the first live Phase 1 run: an all-NaN
    # cp row crashing commit_regime's idxmax, and NaN contaminating the drift model's regression
    # matrix for the entire series (see model.py fixes, same date). Fixing it at the source here
    # is the correct match to validated methodology; the two downstream fixes remain as
    # defensive belt-and-suspenders, not the primary guard.
    subset = [c for c in ["ret", "rv20", "slope_z", "rv_pct", "vix_pct"] if c in feat.columns]
    return feat.dropna(subset=subset)


def _fetch_one_chain_snapshot(
    ticker: str, spot: float, risk_free: float,
    near_dte_target: int, far_dte_range: tuple[int, int], retries: int,
) -> Optional[dict]:
    """One name's IV snapshot: ATM IV at a near-term and a 30-45 DTE expiry, IV term
    slope between them, and 25-delta put-call skew at the 30-45 DTE expiry. Returns
    None (not an exception) on any failure -- missing names are logged, not fatal
    (plan section 5 step 3).
    """
    import yfinance as yf

    last_err = None
    for attempt in range(retries + 1):
        try:
            tk = yf.Ticker(ticker)
            expiries = tk.options
            near_exp, far_exp = select_expiries(list(expiries), near_dte_target=near_dte_target,
                                                 far_dte_range=far_dte_range)
            if far_exp is None:
                return None  # no usable expiry at all -- not worth retrying
            far_chain = tk.option_chain(far_exp)
            far_T = max(dte(far_exp), 1) / 365
            atm_far = atm_iv_from_chain(far_chain.calls, far_chain.puts, spot)
            skew = skew_25d_from_chain(far_chain.calls, far_chain.puts, spot, far_T, risk_free)
            atm_near = None
            if near_exp and near_exp != far_exp:
                near_chain = tk.option_chain(near_exp)
                atm_near = atm_iv_from_chain(near_chain.calls, near_chain.puts, spot)
            term_slope = (atm_far - atm_near) if (atm_far is not None and atm_near is not None) else None
            return {
                "ticker": ticker,
                "spot": spot,
                "near_expiry": near_exp,
                "far_expiry": far_exp,
                "atm_iv_near": atm_near,
                "atm_iv_target": atm_far,
                "iv_term_slope": term_slope,
                "skew_25d": skew,
            }
        except Exception as e:  # noqa: BLE001 -- retry, then give up on this name only
            last_err = e
            if attempt < retries:
                time.sleep(0.5 * (attempt + 1))
    logger.warning("IV snapshot failed for %s after %d attempt(s): %s", ticker, retries + 1, last_err)
    return None


def pull_iv_snapshot(
    names: list[str], spot_prices: dict[str, float], risk_free: float = 0.03,
    near_dte_target: int = 17, far_dte_range: tuple[int, int] = (30, 45),
    retries: int = 2, throttle_sec: float = 0.35,
) -> pd.DataFrame:
    """Nightly ATM IV / term slope / 25-delta skew snapshot for all names (plan section
    5 step 3). Batched, throttled (sleep between names to avoid rate-limiting a 50-name
    loop of live chain fetches), retry x2 per name. A missing/failed name is logged and
    skipped, not a pipeline error -- unlike the VX curve pull, options-chain coverage
    for any single name isn't a first-class model input yet (Phase 4: swap the
    realized-vol proxy for IV rank once >=6 months of this snapshot history exists).
    """
    rows = []
    for i, name in enumerate(names):
        spot = spot_prices.get(name)
        if spot is None or spot != spot:  # NaN check without importing numpy here
            logger.warning("IV snapshot skipped for %s: no spot price available", name)
            continue
        row = _fetch_one_chain_snapshot(name, float(spot), risk_free, near_dte_target, far_dte_range, retries)
        if row is not None:
            rows.append(row)
        if i < len(names) - 1:
            time.sleep(throttle_sec)
    df = pd.DataFrame(rows)
    if not df.empty:
        df.insert(0, "date", str(date.today()))
    return df
