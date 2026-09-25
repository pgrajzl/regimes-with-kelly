"""
miscellaneous_calc.py

Standalone utility calculations — useful for ad hoc analysis, but not part
of the core data/feature/regime pipeline. Grab-bag file by design; don't
worry about tight thematic organization here.
"""

import pandas as pd


def calculate_beta(stock_series: pd.Series, market_series: pd.Series) -> float:
    """
    Calculate beta of a stock relative to a market benchmark (SPY, by
    convention in this repo), over whatever date range the two input
    series actually cover.

    Beta = Cov(stock_returns, market_returns) / Var(market_returns)

    Parameters
    ----------
    stock_series : pd.Series
        Price series for the stock, date-indexed.
    market_series : pd.Series
        Price series for the market benchmark (e.g. SPY), date-indexed.

    Returns
    -------
    float
        Beta over the overlapping date range of the two series. A beta of
        1.0 means the stock moves in line with the market on average; >1
        means more volatile than the market, <1 means less.

    Notes
    -----
    - The two series are aligned on their common dates before computing
      returns — if stock_series and market_series don't fully overlap
      (e.g. different date ranges), only the overlapping portion is used.
    - This is a simple, unconditional (single-regime, single-window) beta
      — no rolling window, no regime-conditioning. That's intentional for
      this file; a rolling/time-varying beta belongs in feature_engineering.py
      or a dedicated module later, not here.
    """
    # Align on common dates — inner join drops any date not present in both
    aligned = pd.concat([stock_series, market_series], axis=1, join="inner")
    aligned.columns = ["stock", "market"]

    stock_returns = aligned["stock"].pct_change().dropna()
    market_returns = aligned["market"].pct_change().dropna()

    # Re-align after pct_change (first row of each is NaN and gets dropped
    # independently above, so re-intersect indices defensively)
    common_idx = stock_returns.index.intersection(market_returns.index)
    stock_returns = stock_returns.loc[common_idx]
    market_returns = market_returns.loc[common_idx]

    if len(common_idx) < 2:
        raise ValueError(
            "Not enough overlapping data points between stock_series and "
            "market_series to compute beta (need at least 2 return "
            "observations)."
        )

    covariance = stock_returns.cov(market_returns)
    market_variance = market_returns.var()

    if market_variance == 0:
        raise ValueError("Market series has zero variance over this window — beta is undefined.")

    return covariance / market_variance

# Approximate trading-day windows for common rolling periods. Calendar
# months don't map to a fixed number of trading days (weekends, holidays),
# so these are standard approximations, not exact conversions.
_ROLLING_WINDOW_DAYS = {
    "3mo": 63,
    "6mo": 126,
    "12mo": 252,
}


def calculate_rolling_beta(
    stock_series: pd.Series,
    market_series: pd.Series,
    window: str = "12mo",
) -> pd.Series:
    """
    Rolling beta of a stock relative to a market benchmark, computed over
    a trailing trading-day window that slides forward one day at a time.

    Unlike calculate_beta (one single number for a whole period), this
    returns a full time series — beta as it evolves day to day. Useful
    for seeing whether a stock's market sensitivity has been stable or
    drifting/regime-dependent over time.

    Parameters
    ----------
    stock_series : pd.Series
        Price series for the stock, date-indexed.
    market_series : pd.Series
        Price series for the market benchmark (e.g. SPY), date-indexed.
    window : str
        One of "3mo", "6mo", "12mo" — the trailing window length.

    Returns
    -------
    pd.Series
        Rolling beta, date-indexed. The first `window` days will be NaN
        (not enough trailing history yet to compute a full window).
    """
    if window not in _ROLLING_WINDOW_DAYS:
        raise ValueError(f"window must be one of {list(_ROLLING_WINDOW_DAYS)}, got '{window}'")

    window_days = _ROLLING_WINDOW_DAYS[window]

    aligned = pd.concat([stock_series, market_series], axis=1, join="inner")
    aligned.columns = ["stock", "market"]

    stock_returns = aligned["stock"].pct_change()
    market_returns = aligned["market"].pct_change()

    # Rolling covariance and rolling variance, both over the same window —
    # beta_t = Cov(stock, market)_t / Var(market)_t, recomputed at each
    # point as the window slides forward.
    rolling_cov = stock_returns.rolling(window_days).cov(market_returns)
    rolling_var = market_returns.rolling(window_days).var()

    rolling_beta = rolling_cov / rolling_var
    rolling_beta.name = f"rolling_beta_{window}"

    return rolling_beta

def calculate_rolling_excess_return(
    stock_series: pd.Series,
    market_series: pd.Series,
    window: str = "12mo",
) -> pd.Series:
    """
    Rolling excess return of a stock vs. a market benchmark: for each day,
    the stock's trailing-window cumulative return minus the benchmark's
    trailing-window cumulative return over that same window.

    This is a simpler, more literal notion of "beating the market" than
    beta — it's not risk-adjusted (says nothing about volatility taken on
    to get there), just a direct return comparison, period return vs.
    period return.

    Parameters
    ----------
    stock_series : pd.Series
        Price series for the stock, date-indexed.
    market_series : pd.Series
        Price series for the market benchmark (e.g. SPY), date-indexed.
    window : str
        One of "3mo", "6mo", "12mo" — the trailing window length.

    Returns
    -------
    pd.Series
        Rolling excess return (as a decimal, e.g. 0.05 = 5 percentage
        points of outperformance), date-indexed. The first `window` days
        will be NaN (not enough trailing history yet).
    """
    if window not in _ROLLING_WINDOW_DAYS:
        raise ValueError(f"window must be one of {list(_ROLLING_WINDOW_DAYS)}, got '{window}'")

    window_days = _ROLLING_WINDOW_DAYS[window]

    aligned = pd.concat([stock_series, market_series], axis=1, join="inner")
    aligned.columns = ["stock", "market"]

    # Trailing-window cumulative return: price today vs. price
    # window_days ago, expressed as a fraction (pct_change(periods=N) does
    # exactly this — not to be confused with rolling().std()-style
    # day-to-day return calcs used elsewhere in this repo).
    stock_period_return = aligned["stock"].pct_change(periods=window_days)
    market_period_return = aligned["market"].pct_change(periods=window_days)

    excess_return = stock_period_return - market_period_return
    excess_return.name = f"rolling_excess_return_{window}"

    return excess_return

def calculate_correlation(series_a: pd.Series, series_b: pd.Series) -> float:
    """
    Pearson correlation coefficient between two series, over their
    overlapping index. Works on any two series — not return-specific, so
    it's fine to pass beta and excess return directly (as opposed to
    calculate_beta, which specifically expects prices and computes
    returns internally).

    Returns
    -------
    float
        Correlation coefficient, -1 to 1. 1 = perfectly move together,
        -1 = perfectly move opposite, 0 = no linear relationship.
    """
    aligned = pd.concat([series_a, series_b], axis=1, join="inner").dropna()

    if len(aligned) < 2:
        raise ValueError("Not enough overlapping, non-NaN data points to compute correlation.")

    return aligned.iloc[:, 0].corr(aligned.iloc[:, 1])