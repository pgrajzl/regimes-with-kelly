"""
visualize.py

Main plotting module. Interactive price charting with timeframe filtering
and optional SPY overlay for relative comparison.
"""

import matplotlib.pyplot as plt
import pandas as pd
import ipywidgets as widgets
from IPython.display import display
import numpy as np

from src.miscellaneous_calc import (
    calculate_beta,
    calculate_rolling_beta,
    calculate_rolling_excess_return,
    calculate_correlation,
)

# Global plot styling — applied once at import time so every plotting
# function in this module inherits it automatically, rather than needing
# to set it inside each individual plot function.
plt.rcParams["font.family"] = "Times New Roman"

# Ordered timeframe options, each mapped to how far back from the most
# recent date to slice. "max" is handled separately (no offset — full history).
_TIMEFRAME_OFFSETS = {
    "1mo": pd.DateOffset(months=1),
    "3mo": pd.DateOffset(months=3),
    "6mo": pd.DateOffset(months=6),
    "1y": pd.DateOffset(years=1),
    "5y": pd.DateOffset(years=5),
}
_TIMEFRAME_ORDER = ["1mo", "3mo", "6mo", "1y", "5y", "max"]


def _available_timeframes(series: pd.Series) -> list[str]:
    """
    Only offer timeframe options the data can actually support. A ticker
    with 2 years of history shouldn't show a "5y" option that would just
    silently plot the same thing as "max" — better to not offer it at all.
    """
    earliest_date = series.index.min()
    latest_date = series.index.max()

    available = []
    for tf, offset in _TIMEFRAME_OFFSETS.items():
        cutoff = latest_date - offset
        if earliest_date <= cutoff:
            available.append(tf)

    available.append("max")
    # preserve the canonical order (1mo -> max), not dict insertion order
    return [tf for tf in _TIMEFRAME_ORDER if tf in available]


def _filter_by_timeframe(series: pd.Series, timeframe: str) -> pd.Series:
    """Slice a series to the last `timeframe` worth of data, or return it whole for 'max'."""
    if timeframe == "max":
        return series
    cutoff = series.index.max() - _TIMEFRAME_OFFSETS[timeframe]
    return series[series.index >= cutoff]


_HOURLY_TIMEFRAMES = {"1mo", "3mo", "6mo"}


def plot_ticker(
    ticker_name: str,
    ticker_daily: pd.Series,
    spy_daily: pd.Series | None = None,
    ticker_hourly: pd.Series | None = None,
    spy_hourly: pd.Series | None = None,
):
    """
    Interactive price plot for a single ticker, with a timeframe dropdown
    and an optional SPY overlay checkbox.

    For 1mo/3mo/6mo timeframes, uses hourly data (if provided) instead of
    daily — gives genuinely granular intraday shape for short windows,
    rather than a handful of daily dots. Falls back to slicing the daily
    series for these timeframes if no hourly series was passed in.

    Parameters
    ----------
    ticker_name : str
        Display name for the plotted ticker (e.g. "AAPL").
    ticker_daily : pd.Series
        Daily price series for the ticker — used for 1y/5y/max, and as a
        fallback for 1mo/3mo/6mo if ticker_hourly isn't provided.
    spy_daily : pd.Series | None
        Daily SPY series, same role as ticker_daily but for the overlay.
    ticker_hourly : pd.Series | None
        Hourly price series (trailing ~1yr), used for 1mo/3mo/6mo when
        available. Timezone-aware index expected (as returned by
        fetch_ticker_hourly_data).
    spy_hourly : pd.Series | None
        Hourly SPY series, same role as ticker_hourly but for the overlay.
    """
    available_tfs = _available_timeframes(ticker_daily)

    def _select_series(timeframe: str, daily: pd.Series, hourly: pd.Series | None) -> pd.Series:
        """Pick hourly vs. daily source based on timeframe, with a fallback if hourly is missing."""
        if timeframe in _HOURLY_TIMEFRAMES and hourly is not None:
            return _filter_by_timeframe(hourly, timeframe)
        return _filter_by_timeframe(daily, timeframe)

    def _render(timeframe: str, overlay_spy: bool):
        filtered = _select_series(timeframe, ticker_daily, ticker_hourly)
        using_hourly = timeframe in _HOURLY_TIMEFRAMES and ticker_hourly is not None

        fig, ax1 = plt.subplots(figsize=(12, 5))

        if overlay_spy and spy_daily is not None:
            spy_filtered = _select_series(timeframe, spy_daily, spy_hourly)

            # Align both series onto their common trading dates/hours so
            # "position 47" refers to the same actual timestamp on both
            # lines — otherwise the two series could silently drift out
            # of sync on the shared positional x-axis.
            common_index = filtered.index.intersection(spy_filtered.index)
            filtered = filtered.loc[common_index]
            spy_filtered = spy_filtered.loc[common_index]

            x = np.arange(len(common_index))
            ax1.plot(x, filtered.values, color="tab:blue", label=ticker_name)

            ax2 = ax1.twinx()
            ax2.plot(x, spy_filtered.values, color="tab:orange", label="SPY", alpha=0.7)
            ax2.set_ylabel("SPY price", color="tab:orange")
            ax2.tick_params(axis="y", labelcolor="tab:orange")

            _positional_axis_with_dates(ax1, common_index)
        else:
            if overlay_spy and spy_daily is None:
                print("No SPY series was passed to plot_ticker() — nothing to overlay.")

            x = np.arange(len(filtered))
            ax1.plot(x, filtered.values, color="tab:blue", label=ticker_name)
            _positional_axis_with_dates(ax1, filtered.index)

        ax1.set_ylabel(f"{ticker_name} price", color="tab:blue")
        ax1.tick_params(axis="y", labelcolor="tab:blue")

        title = f"{ticker_name} — {timeframe}"
        if using_hourly:
            title += " (hourly)"
        if spy_daily is not None:
            # Beta always computed on daily returns, even when the plot
            # itself is showing hourly data — hourly returns are noisier
            # (microstructure effects) and would give a different, less
            # comparable beta than what you'd expect from daily data.
            ticker_daily_window = _filter_by_timeframe(ticker_daily, timeframe)
            spy_daily_window = _filter_by_timeframe(spy_daily, timeframe)
            try:
                beta = calculate_beta(ticker_daily_window, spy_daily_window)
                title += f" (β = {beta:.2f})"
            except ValueError as e:
                title += " (β unavailable)"
                print(f"Beta calculation skipped: {e}")

        ax1.set_title(title)
        fig.tight_layout()
        plt.show()

    timeframe_dropdown = widgets.Dropdown(
        options=available_tfs, value="max", description="Timeframe:"
    )

    overlay_label = widgets.Label("Overlay SPY")
    overlay_checkbox = widgets.Checkbox(value=False, indent=False)
    overlay_group = widgets.HBox([overlay_label, overlay_checkbox])

    controls_row = widgets.HBox(
        [timeframe_dropdown, overlay_group],
        layout=widgets.Layout(width="100%"),
    )
    overlay_group.layout.margin = "0 0 0 auto"

    controls = controls_row

    out = widgets.interactive_output(
        _render,
        {"timeframe": timeframe_dropdown, "overlay_spy": overlay_checkbox},
    )

    display(controls, out)


def _positional_axis_with_dates(ax, index, n_ticks: int = 8, date_fmt: str = "%m-%d-%Y"):
    """
    Replace a datetime x-axis with a plain integer position axis, labeled
    with a handful of evenly-spaced real dates.

    Why: matplotlib's default datetime axis spaces points by actual
    elapsed time, so non-trading hours/days show up as flat, empty gaps
    (visually "dead" chart space). Plotting against 0..n-1 integer
    positions instead removes every gap — consecutive data points are
    always adjacent on the x-axis, regardless of the real time between
    them (overnight, weekend, holiday, whatever).
    """
    n = len(index)
    if n == 0:
        return
    tick_positions = np.linspace(0, n - 1, min(n_ticks, n)).astype(int)
    tick_labels = [index[i].strftime(date_fmt) for i in tick_positions]
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels, rotation=0, ha="center")


def plot_rolling_beta(ticker_name: str, ticker_daily: pd.Series, spy_daily: pd.Series):
    """
    Interactive rolling beta plot — a dropdown selects the rolling window
    (3mo/6mo/12mo), plotted over the full available daily history for the
    ticker (up to ~10yr, whatever's actually available).

    Always uses daily data — rolling beta on hourly data would be a much
    noisier, less standard measure, and isn't what this function is for.
    """

    def _render(window: str):
        rolling_beta = calculate_rolling_beta(ticker_daily, spy_daily, window=window)
        rolling_beta = rolling_beta.dropna()

        if rolling_beta.empty:
            print(f"Not enough history to compute a {window} rolling beta for {ticker_name}.")
            return

        x = np.arange(len(rolling_beta))

        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(x, rolling_beta.values, color="tab:purple", label=f"{window} rolling beta")
        ax.axhline(1.0, color="gray", linestyle="--", linewidth=1, alpha=0.6, label="Beta = 1")
        ax.set_ylabel("Beta")
        ax.legend(loc="upper left")

        _positional_axis_with_dates(ax, rolling_beta.index)

        ax.set_title(f"{ticker_name} — {window} rolling beta vs. SPY")
        fig.tight_layout()
        plt.show()

    window_dropdown = widgets.Dropdown(
        options=["3mo", "6mo", "12mo"], value="12mo", description="Window:"
    )

    out = widgets.interactive_output(_render, {"window": window_dropdown})
    display(window_dropdown, out)

def plot_rolling_excess_return(ticker_name: str, ticker_daily: pd.Series, spy_daily: pd.Series):
    """
    Interactive rolling excess return plot vs. SPY — a dropdown selects
    the rolling window (3mo/6mo/12mo), plotted over the full available
    daily history for the ticker.

    Always uses daily data, same reasoning as plot_rolling_beta — this is
    a longer-horizon concept, not something intraday granularity adds
    value to.
    """

    def _render(window: str):
        excess_return = calculate_rolling_excess_return(ticker_daily, spy_daily, window=window)
        excess_return = excess_return.dropna()

        if excess_return.empty:
            print(f"Not enough history to compute a {window} rolling excess return for {ticker_name}.")
            return

        x = np.arange(len(excess_return))

        fig, ax = plt.subplots(figsize=(12, 5))
        # Color the fill green where excess return is positive (beating
        # SPY) and red where negative (lagging SPY) — makes the sign
        # visually obvious at a glance rather than requiring the reader
        # to check the y-axis scale each time.
        ax.plot(x, excess_return.values * 100, color="black", linewidth=1)
        ax.fill_between(
            x, excess_return.values * 100, 0,
            where=(excess_return.values >= 0), color="tab:green", alpha=0.3, interpolate=True,
        )
        ax.fill_between(
            x, excess_return.values * 100, 0,
            where=(excess_return.values < 0), color="tab:red", alpha=0.3, interpolate=True,
        )
        ax.axhline(0, color="gray", linestyle="--", linewidth=1, alpha=0.6)
        ax.set_ylabel("Excess return vs. SPY (%)")

        _positional_axis_with_dates(ax, excess_return.index)

        ax.set_title(f"{ticker_name} — {window} rolling excess return vs. SPY")
        fig.tight_layout()
        plt.show()

    window_dropdown = widgets.Dropdown(
        options=["3mo", "6mo", "12mo"], value="12mo", description="Window:"
    )

    out = widgets.interactive_output(_render, {"window": window_dropdown})
    display(window_dropdown, out)

def plot_beta_vs_excess_return(ticker_name: str, ticker_daily: pd.Series, spy_daily: pd.Series):
    """
    Rolling beta and rolling excess return plotted together over time, for
    a given rolling window — same x-axis, beta on the left y-axis and
    excess return on the right, so you can see how the two move relative
    to each other (e.g. does beta spike precede a stretch of
    underperformance, or the opposite?).
    """

    def _render(window: str):
        beta = calculate_rolling_beta(ticker_daily, spy_daily, window=window)
        excess_return = calculate_rolling_excess_return(ticker_daily, spy_daily, window=window)

        combined = pd.concat([beta, excess_return], axis=1, join="inner").dropna()
        combined.columns = ["beta", "excess_return"]

        if combined.empty:
            print(f"Not enough history to compute {window} beta/excess return for {ticker_name}.")
            return

        x = np.arange(len(combined))

        fig, ax1 = plt.subplots(figsize=(12, 5))

        ax1.plot(x, combined["beta"].values, color="tab:purple", label="Rolling beta")
        ax1.axhline(1.0, color="tab:purple", linestyle="--", linewidth=1, alpha=0.4)
        ax1.set_ylabel("Rolling beta", color="tab:purple")
        ax1.tick_params(axis="y", labelcolor="tab:purple")

        ax2 = ax1.twinx()
        ax2.plot(x, combined["excess_return"].values * 100, color="tab:green", label="Rolling excess return", alpha=0.8)
        ax2.axhline(0, color="tab:green", linestyle="--", linewidth=1, alpha=0.4)
        ax2.set_ylabel("Rolling excess return vs. SPY (%)", color="tab:green")
        ax2.tick_params(axis="y", labelcolor="tab:green")

        _positional_axis_with_dates(ax1, combined.index)

        corr = calculate_correlation(combined["beta"], combined["excess_return"])

        ax1.set_title(f"{ticker_name} — {window} rolling beta vs. excess return (corr = {corr:.2f})")
        fig.tight_layout()
        plt.show()

    window_dropdown = widgets.Dropdown(
        options=["3mo", "6mo", "12mo"], value="12mo", description="Window:"
    )

    out = widgets.interactive_output(_render, {"window": window_dropdown})
    display(window_dropdown, out)