"""
data_loader.py

Pulls raw market data used as inputs to regime detection and factor
performance analysis. This module's only job is fetching and caching raw
data — not transforming it (that's feature_engineering.py's job).
"""

from pathlib import Path

import pandas as pd
import yfinance as yf

RAW_DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"

"""
data_loader.py

Pulls raw market data used as inputs to regime detection and factor
performance analysis. This module's only job is fetching and caching raw
data — not transforming it (that's feature_engineering.py's job).
"""

from pathlib import Path

import pandas as pd
import yfinance as yf

RAW_DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"


def fetch_price_history(
    tickers: list[str],
    start: str = "2015-01-01",
    end: str | None = None,
    cache: bool = True,
) -> pd.DataFrame:
    """
    Download adjusted close prices for a list of tickers.

    Parameters
    ----------
    tickers : list[str]
        e.g. ["SPY", "^VIX"]
    start : str
        ISO date string, defaults to ~10 years back.
    end : str | None
        ISO date string; None means "through the most recent available bar".
    cache : bool
        If True, save/read a parquet cache in data/raw/ so we don't
        re-download from Yahoo Finance every time we rerun the notebook.

    Returns
    -------
    pd.DataFrame
        Wide DataFrame, one column per ticker, indexed by date.
    """
    cache_key = "_".join(sorted(t.replace("^", "") for t in tickers))
    cache_path = RAW_DATA_DIR / f"prices_{cache_key}.parquet"

    if cache and cache_path.exists():
        return pd.read_parquet(cache_path)

    raw = yf.download(tickers, start=start, end=end, auto_adjust=True, progress=False)
    prices = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw[["Close"]]
    prices = prices.copy()
    if isinstance(prices, pd.Series):
        prices = prices.to_frame(tickers[0])

    if cache:
        RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
        prices.to_parquet(cache_path)

    return prices

def fetch_vix(start: str = "2015-01-01", end: str | None = None, cache: bool = True) -> pd.Series:
    """Convenience wrapper for the VIX index specifically — returns a Series, not a DataFrame."""
    df = fetch_price_history(["^VIX"], start=start, end=end, cache=cache)
    return df.iloc[:, 0].rename("VIX")