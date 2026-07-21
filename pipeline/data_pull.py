"""
Regime dashboard — data acquisition.

yfinance is the #1 operational risk flagged in the plan (research/regime-dashboard-plan.md
section 8) -- Yahoo has historically blocked cloud-provider IPs, which is why this pulls
happen in GitHub Actions (not the Streamlit app itself) and fall back to Stooq on failure.

NOT YET WIRED: the 50-name universe expansion beyond the 20 cross-sectionally-validated
names in config.yaml, and the nightly IV/ATM-vol snapshot logic (plan section 5 step 3).
Both are Phase 1 remaining work -- this module currently covers prices + VIX complex +
VX1-VX3 curve + FRED, which is everything the validated model actually consumes.
"""
from __future__ import annotations

import logging
from typing import Optional

import pandas as pd

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
    ts = vix_utils.load_vix_term_structure()
    wide = vix_utils.pivot_futures_on_monthly_tenor(ts)
    vx = pd.DataFrame({"VX1": wide[1]["Close"], "VX3": wide[3]["Close"]})
    vx.index = pd.to_datetime(vx.index)
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
    return feat.dropna(subset=["ret", "rv20"])
