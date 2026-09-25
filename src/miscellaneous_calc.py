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