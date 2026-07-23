"""
Phase 3 -- Streamlit app data layer.

Split from app.py per the same separation pattern as pipeline/ (data acquisition vs.
logic vs. presentation). Two tiers of data here, matching plan section 5's design:

1. PRECOMPUTED (cheap, cached hard): everything the nightly Action already wrote to
   data/*.parquet and output/state.json. The app is a pure reader for these -- never
   recomputes the model.
2. LIVE (client-triggered, cached soft): price history and option chains fetched
   on-demand only when a user opens a drill-down. The pipeline never persisted raw
   price series (only model outputs), so anything needing actual returns -- the
   drill-down chart, the call-history log's realized returns, the forecast-density
   panel -- has to fetch live. This matches the plan's explicit design ("live chain
   fetch happens only in drill-down, client-triggered"), extended to price history
   since that gap exists too.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st
import yaml

HERE = Path(__file__).resolve().parent.parent
DATA_DIR = HERE / "data"
OUTPUT_DIR = HERE / "output"
CONFIG_PATH = HERE / "config.yaml"

CELLS = ["bear_hi", "bear_lo", "neut_hi", "neut_lo", "bull_hi", "bull_lo"]
DIRECTIONS = ["bear", "neut", "bull"]
VOLS = ["hi", "lo"]


@st.cache_data(ttl=300)
def load_config() -> dict:
    with open(CONFIG_PATH) as fh:
        return yaml.safe_load(fh)


@st.cache_data(ttl=300)
def load_state() -> dict:
    import json
    with open(OUTPUT_DIR / "state.json") as fh:
        return json.load(fh)


@st.cache_data(ttl=300)
def load_cell_posterior() -> pd.DataFrame:
    df = pd.read_parquet(DATA_DIR / "cell_posterior.parquet")
    df.index = pd.to_datetime(df.index)
    return df[CELLS]


@st.cache_data(ttl=300)
def load_committed_regime() -> pd.Series:
    df = pd.read_parquet(DATA_DIR / "committed_regime.parquet")
    df.index = pd.to_datetime(df.index)
    return df["regime"]


@st.cache_data(ttl=300)
def load_dirpost() -> pd.DataFrame:
    df = pd.read_parquet(DATA_DIR / "dirpost.parquet")
    df.index = pd.to_datetime(df.index)
    return df


@st.cache_data(ttl=300)
def load_name_cells() -> pd.DataFrame:
    """Long format: Date, ticker, cell -- full per-name regime history."""
    df = pd.read_parquet(DATA_DIR / "name_cells.parquet")
    df["Date"] = pd.to_datetime(df["Date"])
    return df


@st.cache_data(ttl=300)
def load_iv_snapshots() -> pd.DataFrame:
    path = DATA_DIR / "iv_snapshots.parquet"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_parquet(path)
    df["date"] = pd.to_datetime(df["date"])
    return df


@st.cache_data(ttl=300)
def load_forecast_density() -> pd.DataFrame:
    """Per-name forecast-density summary (one row per ticker, refreshed each nightly
    run -- see run_nightly.py's forecast-density block, pipeline/forecast.py for the
    math). Returns empty if the file doesn't exist yet (first deploy, before the next
    nightly run lands it)."""
    path = DATA_DIR / "forecast_density.parquet"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path)


@st.cache_data(ttl=3600)
def load_name_metadata() -> pd.DataFrame:
    """Market cap + sector/industry per name (pipeline/fetch_name_metadata.py, run
    manually/occasionally -- NOT part of the nightly Action, see that script's docstring
    for why). Longer TTL than the other precomputed loaders (1hr vs. 5min) since this
    data is refreshed on a much slower cadence than the nightly model outputs. Returns
    empty if the file doesn't exist yet (before the first manual run of that script) --
    callers (the universe treemap) degrade to a flat, ungrouped layout in that case."""
    path = DATA_DIR / "name_metadata.parquet"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path)


def name_cell_history(ticker: str) -> pd.Series:
    """Single name's cell history as a Date-indexed Series (subset of load_name_cells)."""
    nc = load_name_cells()
    sub = nc[nc["ticker"] == ticker].sort_values("Date").set_index("Date")["cell"]
    return sub


def name_iv_history(ticker: str) -> pd.DataFrame:
    iv = load_iv_snapshots()
    if iv.empty:
        return iv
    return iv[iv["ticker"] == ticker].sort_values("date").set_index("date")


# ---------------------------------------------------------------------------
# LIVE fetches -- short TTL, client-triggered only (called from drill-down code,
# never on app load for all 50 names at once -- that would hammer yfinance).
# ---------------------------------------------------------------------------

@st.cache_data(ttl=900, show_spinner="Fetching live price history...")
def fetch_price_history(ticker: str, start: str = "2015-01-01") -> pd.Series:
    """Live daily close price history for one ticker. Reuses the pipeline's own
    yfinance-then-Stooq failover so drill-down behaves the same way under a blocked
    Streamlit Cloud IP as the nightly pipeline does.
    """
    from pipeline.data_pull import pull_prices
    px, _source = pull_prices([ticker], start)
    col = ticker if ticker in px.columns else px.columns[0]
    return px[col].dropna()


@st.cache_data(ttl=900, show_spinner="Fetching live option chain...")
def fetch_option_chain(ticker: str, expiry: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Live calls/puts chain for one ticker+expiry. Returns (calls, puts) DataFrames
    in yfinance's native shape (strike, impliedVolatility, bid, ask, ...).
    """
    import yfinance as yf
    tk = yf.Ticker(ticker)
    chain = tk.option_chain(expiry)
    return chain.calls, chain.puts


@st.cache_data(ttl=900, show_spinner="Fetching available expiries...")
def fetch_option_expiries(ticker: str) -> list[str]:
    import yfinance as yf
    tk = yf.Ticker(ticker)
    return list(tk.options)
