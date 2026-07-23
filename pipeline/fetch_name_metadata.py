"""
Reference-data fetch: market cap + GICS sector/industry per name in the universe.

Deliberately NOT wired into run_nightly.py. Two reasons:

1. yfinance's `.info`/`.get_info()` endpoint is a different, slower, and more failure-
   prone call than the price-history pull the nightly Action already relies on -- each
   ticker is roughly a full-page scrape, not a lightweight quote lookup, and this
   session has already hit real fragility on faster endpoints (VX curve gaps, CBOE
   403s, Yahoo truncation). Bolting 50 more of these onto the nightly run risks the
   thing that took real effort to get reliable this session (the self-hosted runner).
2. Sector/industry classification and market cap don't need daily precision for a
   treemap's grouping/sizing -- market cap moves within a name's relative size bucket
   slowly, and sector/industry basically never changes. A stale-by-a-few-weeks value
   is fine here in a way it would NOT be for the regime/direction outputs.

So: run this by hand occasionally (monthly is plenty), not on every nightly Action run.
Writes data/name_metadata.parquet: ticker, market_cap, sector, industry, fetched_at.
The app (app/data.py load_name_metadata()) reads it like any other precomputed file and
degrades gracefully (empty DataFrame) if it doesn't exist yet.

Run locally:
    python -m pipeline.fetch_name_metadata
"""
from __future__ import annotations

import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"
CONFIG_PATH = REPO_ROOT / "config.yaml"

THROTTLE_SEC = 0.35  # same courtesy delay as pipeline's iv_snapshot chain-fetch loop


def load_config() -> dict:
    with open(CONFIG_PATH) as fh:
        return yaml.safe_load(fh)


def fetch_one(ticker: str) -> dict:
    """Best-effort single-ticker fetch -- returns a row with NaN/None fields on failure
    rather than raising, so one bad ticker doesn't kill the whole run (same philosophy
    as pull_prices_stooq's per-ticker try/except in data_pull.py)."""
    import yfinance as yf
    try:
        info = yf.Ticker(ticker).get_info()
        return {
            "ticker": ticker,
            "market_cap": info.get("marketCap"),
            "sector": info.get("sector") or "Unknown",
            "industry": info.get("industry") or "Unknown",
            "short_name": info.get("shortName") or ticker,
            "error": None,
        }
    except Exception as e:  # noqa: BLE001 -- log and continue, see docstring
        return {
            "ticker": ticker, "market_cap": None, "sector": "Unknown",
            "industry": "Unknown", "short_name": ticker, "error": str(e),
        }


def main() -> None:
    cfg = load_config()
    names = cfg["data"]["underlyings_names"]
    DATA_DIR.mkdir(exist_ok=True)

    rows = []
    t0 = time.time()
    for i, ticker in enumerate(names, start=1):
        row = fetch_one(ticker)
        rows.append(row)
        status = "ok" if row["error"] is None else f"FAILED ({row['error']})"
        print(f"[{i}/{len(names)}] {ticker}: {status} in {time.time() - t0:.1f}s total")
        time.sleep(THROTTLE_SEC)

    df = pd.DataFrame(rows)
    n_failed = df["error"].notna().sum()
    if n_failed:
        print(f"\n{n_failed}/{len(df)} tickers failed -- kept as Unknown sector/industry, "
              f"NaN market cap, so the treemap can still render them (small/ungrouped) "
              f"rather than silently dropping names.")
    df["fetched_at"] = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    df = df.drop(columns=["error"])
    out_path = DATA_DIR / "name_metadata.parquet"
    df.to_parquet(out_path)
    print(f"\nWrote {out_path} ({len(df)} rows, {n_failed} with missing data)")


if __name__ == "__main__":
    main()
