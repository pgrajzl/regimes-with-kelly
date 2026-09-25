"""
fetch_spx.py

Fetches ~10 years of daily price history for:
  1. All current S&P 500 constituents (fetch_spx_constituents_data)
  2. SPY itself, as a separate convenience function (fetch_spy_data)

Design notes:
- The S&P 500 ticker list isn't static — companies get added/removed. We
  pull the CURRENT list from Wikipedia, which means re-running this in a
  year will silently fetch a different set of tickers than today. That's
  expected behavior, not a bug, but worth knowing: this gives you today's
  constituents' full 10yr history, not a survivorship-bias-free historical
  membership list. (Real point-in-time index membership is a much harder,
  usually paid-data problem — flagging this explicitly since it matters
  for anything backtest-like later.)
- yfinance can rate-limit or flake out on individual tickers when you
  request ~500 of them. We batch requests and retry failures individually
  rather than letting one bad ticker kill the whole pull.
- Results are cached to parquet so re-running this doesn't re-hit the
  network every time.
"""

import time
from pathlib import Path

import pandas as pd
import requests
import yfinance as yf

RAW_DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"

WIKI_SPX_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"


def get_spx_tickers() -> list[str]:
    """
    Scrape the current S&P 500 constituent list from Wikipedia.

    Returns
    -------
    list[str]
        Ticker symbols, Yahoo-Finance-formatted (e.g. BRK.B -> BRK-B,
        since Yahoo uses a hyphen where Wikipedia/most sources use a dot).
    """
    # Wikipedia blocks requests with no User-Agent header — set one to
    # avoid a silent 403.
    headers = {"User-Agent": "Mozilla/5.0 (research script; contact: paul)"}
    resp = requests.get(WIKI_SPX_URL, headers=headers, timeout=15)
    resp.raise_for_status()

    tables = pd.read_html(resp.text)
    constituents = tables[0]  # first table on the page is the constituent list

    tickers = constituents["Symbol"].tolist()
    # Yahoo Finance uses '-' where Wikipedia uses '.' for share classes
    # (e.g. Berkshire Hathaway class B: BRK.B on Wikipedia -> BRK-B on Yahoo)
    tickers = [t.replace(".", "-") for t in tickers]

    return sorted(set(tickers))


def _download_batch(
    tickers: list[str],
    start: str,
    end: str | None,
    max_retries: int = 3,
    retry_delay_sec: float = 5.0,
) -> pd.DataFrame:
    """
    Download a batch of tickers with retry logic. yfinance's own internal
    threading handles the batch efficiently, but transient failures
    (network blips, rate limits) happen — retry a few times before giving
    up on the whole batch.
    """
    for attempt in range(1, max_retries + 1):
        try:
            raw = yf.download(
                tickers,
                start=start,
                end=end,
                auto_adjust=True,
                progress=False,
                group_by="ticker",
                threads=True,
            )
            if raw.empty:
                raise ValueError("yfinance returned an empty DataFrame")
            return raw
        except Exception as e:
            if attempt == max_retries:
                print(f"  Batch failed after {max_retries} attempts: {e}")
                return pd.DataFrame()
            print(f"  Attempt {attempt} failed ({e}), retrying in {retry_delay_sec}s...")
            time.sleep(retry_delay_sec)

    return pd.DataFrame()


def fetch_spx_constituents_data(
    start: str = "2015-01-01",
    end: str | None = None,
    batch_size: int = 50,
    cache: bool = True,
    cache_filename: str = "spx_constituents_prices.parquet",
) -> pd.DataFrame:
    """
    Fetch adjusted close prices for every current S&P 500 constituent.

    Batches requests (default 50 tickers per batch) rather than requesting
    all ~500 at once — large single requests to yfinance are more prone to
    silent partial failures and are harder to debug when something goes
    wrong, since you can't tell which ticker(s) caused it.

    Parameters
    ----------
    start : str
        ISO date string.
    end : str | None
        ISO date string; None means through the most recent available bar.
    batch_size : int
        Tickers per yfinance request batch.
    cache : bool
        If True, save/load a parquet cache in data/raw/.
    cache_filename : str
        Filename for the cache (kept as a param so you can version
        different pulls, e.g. if the constituent list changes over time).

    Returns
    -------
    pd.DataFrame
        Wide DataFrame, one column per ticker, indexed by date. Tickers
        that failed to download entirely are simply absent from the
        result (see the printed summary for which ones, if any).
    """
    cache_path = RAW_DATA_DIR / cache_filename

    if cache and cache_path.exists():
        print(f"Loading cached data from {cache_path}")
        return pd.read_parquet(cache_path)

    tickers = get_spx_tickers()
    print(f"Fetching {len(tickers)} S&P 500 constituents in batches of {batch_size}...")

    all_closes = {}
    failed_tickers = []

    for i in range(0, len(tickers), batch_size):
        batch = tickers[i : i + batch_size]
        batch_num = i // batch_size + 1
        total_batches = (len(tickers) - 1) // batch_size + 1
        print(f"  Batch {batch_num}/{total_batches}: {batch[0]}...{batch[-1]}")

        raw = _download_batch(batch, start=start, end=end)
        if raw.empty:
            failed_tickers.extend(batch)
            continue

        # With group_by="ticker", columns are a MultiIndex: (ticker, field).
        # Single-ticker batches (shouldn't happen with our batch_size, but
        # defensive) return flat columns instead — handle both.
        for ticker in batch:
            try:
                if isinstance(raw.columns, pd.MultiIndex):
                    if ticker not in raw.columns.get_level_values(0):
                        failed_tickers.append(ticker)
                        continue
                    close = raw[ticker]["Close"]
                else:
                    close = raw["Close"]

                if close.dropna().empty:
                    failed_tickers.append(ticker)
                    continue

                all_closes[ticker] = close
            except (KeyError, TypeError):
                failed_tickers.append(ticker)

        # Be a reasonably polite citizen between batches
        time.sleep(1)

    prices = pd.DataFrame(all_closes)

    print(f"\nDone: {len(all_closes)}/{len(tickers)} tickers fetched successfully.")
    if failed_tickers:
        print(f"Failed tickers ({len(failed_tickers)}): {sorted(failed_tickers)}")

    if cache and not prices.empty:
        RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
        prices.to_parquet(cache_path)
        print(f"Cached to {cache_path}")

    return prices


def fetch_spy_data(
    start: str = "2015-01-01",
    end: str | None = None,
    cache: bool = True,
) -> pd.Series:
    """
    Fetch SPY itself, separately from the constituent pull above — this is
    the index-level reference series (what feature_engineering.py builds
    regime features from), not one of the 500 individual names.

    Returns
    -------
    pd.Series
        Adjusted close, indexed by date, named 'SPY'.
    """
    cache_path = RAW_DATA_DIR / "spy_prices.parquet"

    if cache and cache_path.exists():
        return pd.read_parquet(cache_path)["SPY"]

    raw = yf.download("SPY", start=start, end=end, auto_adjust=True, progress=False)
    close = raw["Close"]
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]
    close = close.rename("SPY")

    if cache:
        RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
        close.to_frame().to_parquet(cache_path)

    return close


if __name__ == "__main__":
    # Smoke test — SPY first since it's fast, constituents second since
    # it's slow (many minutes for ~500 tickers).
    spy = fetch_spy_data(start="2015-01-01")
    print(f"\nSPY: {spy.index.min()} to {spy.index.max()}, {len(spy)} rows")

    spx = fetch_spx_constituents_data(start="2015-01-01")
    print(f"SPX constituents: {spx.shape[1]} tickers, {spx.shape[0]} rows")

def fetch_ticker_data(
    ticker: str,
    start: str = "2015-01-01",
    end: str | None = None,
    cache: bool = True,
) -> pd.Series:
    """
    Fetch a single arbitrary ticker — for ad hoc lookups (e.g. eyeballing
    AAPL or GOOG) rather than the full SPX universe or SPY specifically.

    Returns
    -------
    pd.Series
        Adjusted close, indexed by date, named after the ticker.
    """
    cache_path = RAW_DATA_DIR / f"ticker_{ticker.replace('-', '_')}.parquet"

    if cache and cache_path.exists():
        return pd.read_parquet(cache_path)[ticker]

    raw = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)
    if raw.empty:
        raise ValueError(f"No data returned for ticker '{ticker}' — check it's a valid, active ticker.")

    close = raw["Close"]
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]
    close = close.rename(ticker)

    if cache:
        RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
        close.to_frame().to_parquet(cache_path)

    return close

def fetch_ticker_hourly_data(
    ticker: str,
    cache: bool = True,
) -> pd.Series:
    """
    Fetch hourly price data for a single ticker, trailing ~1 year.

    Deliberately no start/end params like the daily fetchers — Yahoo caps
    hourly data at roughly 730 days of lookback, and we only need ~1 year
    of it to support 6mo/3mo/1mo hourly views, so this always pulls
    period="1y" rather than letting the caller request an arbitrary range
    that might exceed what Yahoo will actually return.

    Returns
    -------
    pd.Series
        Hourly close prices, datetime-indexed (includes intraday
        timestamps, not just dates), named after the ticker.
    """
    cache_path = RAW_DATA_DIR / f"ticker_{ticker.replace('-', '_')}_hourly.parquet"

    if cache and cache_path.exists():
        return pd.read_parquet(cache_path)[ticker]

    raw = yf.download(ticker, period="1y", interval="1h", auto_adjust=True, progress=False)
    if raw.empty:
        raise ValueError(
            f"No hourly data returned for ticker '{ticker}' — check it's valid and "
            f"actively traded (hourly data isn't available for illiquid/delisted names)."
        )

    close = raw["Close"]
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]
    close = close.rename(ticker)

    if cache:
        RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
        close.to_frame().to_parquet(cache_path)

    return close


def fetch_spy_hourly_data(cache: bool = True) -> pd.Series:
    """Hourly SPY, trailing ~1 year — same pattern as fetch_ticker_hourly_data, just SPY-specific."""
    return fetch_ticker_hourly_data("SPY", cache=cache)