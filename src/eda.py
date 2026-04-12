"""
Exploratory Data Analysis: all visualisations produced before modelling.

Every function creates, saves, and closes its own figure so that the module
can be called from a script or a notebook without side effects.

All plots mirror the notebook's EDA sections (cells 25–56).
"""

import calendar
from pathlib import Path

import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.seasonal import STL

from src.utils import get_logger, save_figure, set_plot_style

logger = get_logger(__name__)

def plot_time_series(
    df: pd.DataFrame,
    target_col: str,
    images_dir: Path,
    eu_limit: float = 50,
) -> None:
    """Raw PM10 time series with 30-day rolling average and EU daily limit."""
    set_plot_style()
    fig, ax = plt.subplots(figsize=(12, 7))

    ax.plot(df.index, df[target_col], color="teal", alpha=0.5,
            linewidth=1, label="Daily PM10 Concentration")
    ax.plot(df[target_col].rolling(30).mean(), color="darkorange",
            linewidth=2, label="30-day Rolling Average")
    ax.axhline(eu_limit, color="red", linestyle="--", linewidth=2,
               label=f"EU norm ({eu_limit} µg/m³)")

    ax.set_title("PM10 Levels in Kraków with EU Air Quality Standard",
                 fontsize=16, fontweight="bold")
    ax.set_xlabel("Year", fontsize=12)
    ax.set_ylabel("Concentration [µg/m³]", fontsize=12)
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)

    save_figure(fig, Path(images_dir) / "eda_time_series.png")
    logger.info("Saved eda_time_series.png")


def plot_exceedances_and_distribution(
    df: pd.DataFrame,
    target_col: str,
    images_dir: Path,
    eu_limit: float = 50,
    permitted_days: int = 35,
) -> None:
    """Dual panel: annual EU-limit exceedance bar chart + PM10 histogram."""
    set_plot_style()

    yearly     = df[target_col].resample("YE").agg(["mean", "max", "count"]).copy()
    exceedances = df[df[target_col] > eu_limit][target_col].resample("YE").size()
    yearly["Exceedance_Days"] = exceedances.fillna(0).astype(int)
    yearly.index = yearly.index.year

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 8))
    sns.set_style("white")

    palette = sns.color_palette("Blues_d", len(yearly))
    sns.barplot(x=yearly.index, y=yearly["Exceedance_Days"],
                hue=yearly.index, palette=palette,
                legend=False, ax=ax1)
    ax1.axhline(permitted_days, color="red", linestyle="--", linewidth=2,
                label=f"Permitted {permitted_days} days/year")
    ax1.set_title(f"Days per year with PM10 > {eu_limit} µg/m³",
                  fontsize=14, fontweight="bold")
    ax1.set_ylabel("Number of days", fontsize=12)
    ax1.set_xlabel("Year", fontsize=12)
    ax1.legend(fontsize=11)
    ax1.grid(axis="y", alpha=0.3)

    ax2.hist(df[target_col].dropna(), bins=60, color="steelblue",
             alpha=0.7, edgecolor="white")
    ax2.axvline(eu_limit, color="red", linestyle="--", linewidth=2,
                label=f"EU Limit ({eu_limit} µg/m³)")
    ax2.set_title("PM10 Concentration Distribution", fontsize=14, fontweight="bold")
    ax2.set_xlabel("PM10 [µg/m³]", fontsize=12)
    ax2.set_ylabel("Frequency", fontsize=12)
    ax2.legend(fontsize=11)

    plt.tight_layout()
    save_figure(fig, Path(images_dir) / "eda_time_series.png")
    logger.info("Saved eda_time_series.png (exceedances + distribution panel)")

def plot_seasonal_distribution(
    df: pd.DataFrame,
    target_col: str,
    images_dir: Path,
    eu_limit: float = 50,
) -> None:
    """KDE curves of PM10 concentration by meteorological season.

    Requires a ``season`` column (added by
    :func:`src.feature_engineering.add_calendar_features`).
    """
    set_plot_style()

    if "season" not in df.columns:
        month = df.index.month
        df = df.copy()
        df["season"] = pd.Series(month, index=df.index).map({
            12: "Winter", 1: "Winter",  2: "Winter",
            3:  "Spring", 4: "Spring",  5: "Spring",
            6:  "Summer", 7: "Summer",  8: "Summer",
            9:  "Autumn", 10: "Autumn", 11: "Autumn",
        })

    season_order  = ["Winter", "Spring", "Autumn", "Summer"]
    season_colors = ["#1f77b4", "#2ca02c", "#d62728", "#ff7f0e"]

    fig, ax = plt.subplots(figsize=(12, 7))
    for season, color in zip(season_order, season_colors):
        sns.kdeplot(
            data=df[df["season"] == season][target_col],
            label=season, color=color, fill=True, alpha=0.2,
            linewidth=2, ax=ax,
        )
    ax.axvline(eu_limit, color="red", linestyle="--", alpha=0.5,
               label=f"EU Limit ({eu_limit} µg/m³)")
    ax.set_title("Statistical Distribution of PM10 by Season",
                 fontsize=16, fontweight="bold")
    ax.set_xlabel("PM10 Concentration [µg/m³]", fontsize=12)
    ax.set_ylabel("Density", fontsize=12)
    ax.set_xlim(0, 160)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.2)

    save_figure(fig, Path(images_dir) / "eda_seasonal_distribution.png")
    logger.info("Saved eda_seasonal_distribution.png")


def plot_weekly_cycle(
    df: pd.DataFrame,
    target_col: str,
    images_dir: Path,
    eu_limit: float = 50,
) -> None:
    """Bar chart of mean PM10 by day-of-week with std error bars."""
    set_plot_style()

    _df = df.copy()
    if "week" not in _df.columns:
        _df["week"] = _df.index.dayofweek

    weekly = _df.groupby("week")[target_col].agg(["mean", "std"])
    days   = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]

    norm       = mcolors.Normalize(vmin=weekly["mean"].min(), vmax=weekly["mean"].max())
    cmap       = plt.cm.Blues
    bar_colors = [cmap(norm(v)) for v in weekly["mean"]]

    fig, ax = plt.subplots(figsize=(12, 7))
    bars = ax.bar(days, weekly["mean"], yerr=weekly["std"], capsize=5,
                  color=bar_colors, alpha=0.8, edgecolor="black", width=0.5)
    ax.bar_label(bars, padding=8, fmt="%.1f", fontsize=12)
    ax.axhline(eu_limit, color="#c0392b", linestyle="--", alpha=0.5,
               label=f"EU Daily Limit ({eu_limit} µg/m³)")
    ax.set_title("Weekly PM10 Cycle in Kraków (2019–2024)",
                 fontsize=16, fontweight="bold")
    ax.set_ylabel("PM10 [µg/m³]", fontsize=12)
    ax.set_ylim(0, (weekly["mean"] + weekly["std"]).max() * 1.2)
    ax.grid(axis="y", linestyle=":", alpha=0.6)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False, loc="upper right")

    save_figure(fig, Path(images_dir) / "eda_week_PM10_analysis.png")
    logger.info("Saved eda_week_PM10_analysis.png")


def plot_monthly_boxplots(
    df: pd.DataFrame,
    target_col: str,
    images_dir: Path,
    eu_limit: float = 50,
) -> None:
    """Box-and-whisker plot of PM10 grouped by calendar month."""
    set_plot_style()

    _df = df.copy()
    if "month_name" not in _df.columns:
        _df["month_name"] = _df.index.month_name()

    month_order = list(calendar.month_name)[1:]

    fig, ax = plt.subplots(figsize=(12, 7))
    sns.boxplot(
        data=_df, x="month_name", y=target_col, hue="month_name",
        order=month_order, palette="Blues_d", ax=ax, legend=False,
    )
    ax.axhline(eu_limit, color="red", linestyle="--", linewidth=2,
               label=f"EU Daily Limit ({eu_limit} µg/m³)")
    ax.set_title("Monthly Distribution and Variance of PM10 in Kraków",
                 fontsize=16, fontweight="bold", pad=20)
    ax.set_ylabel("PM10 Concentration [µg/m³]", fontsize=12)
    ax.set_xlabel("Month", fontsize=12)
    ax.tick_params(axis="x", rotation=45)
    ax.legend(loc="upper right")
    ax.grid(axis="y", alpha=0.3)
    sns.despine(ax=ax)

    save_figure(fig, Path(images_dir) / "eda_monthly_boxplots.png")
    logger.info("Saved eda_monthly_boxplots.png")


def plot_heatmap_month_year(
    df: pd.DataFrame,
    target_col: str,
    images_dir: Path,
) -> None:
    """Heatmap of mean monthly PM10 (rows=years, columns=months)."""
    set_plot_style()

    _df = df.copy()
    if "year" not in _df.columns:
        _df["year"] = _df.index.year
    if "month" not in _df.columns:
        _df["month"] = _df.index.month

    pivot = _df.pivot_table(
        values=target_col, index="year", columns="month", aggfunc="mean"
    )
    pivot.columns = [calendar.month_name[m] for m in pivot.columns]

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(pivot, annot=True, fmt=".1f", cmap="coolwarm", ax=ax)
    ax.set_title(
        "Mean Monthly PM10 Concentrations in Kraków (Year × Month)",
        fontsize=16, fontweight="bold", pad=20,
    )
    ax.set_xlabel("Month", fontsize=12)
    ax.set_ylabel("Year", fontsize=12)
    ax.tick_params(axis="x", rotation=45)

    save_figure(fig, Path(images_dir) / "eda_heatmap_month_year.png")
    logger.info("Saved eda_heatmap_month_year.png")

def plot_weather_scatter(
    df: pd.DataFrame,
    target_col: str,
    images_dir: Path,
) -> None:
    """2×3 scatter panel: PM10 vs six meteorological variables with Pearson r."""
    set_plot_style()

    scatter_vars = [
        ("wind_max", "maximum wind speed", "#3498DB"),
        ("temp_avg", "average temperature", "#E74C3C"),
        ("pressure_avg", "average atmospheric pressure","#9B59B6"),
        ("rain_sum",  "precipitation sum", "#1ABC9C"),
        ("humidity_avg", "average relative humidity", "#F39C12"),
        ("wind_mean",  "average wind speed", "#2ECC71"),
    ]
    available = [(col, lbl, clr) for col, lbl, clr in scatter_vars
                 if col in df.columns]

    n_cols = min(3, len(available))
    n_rows = int(np.ceil(len(available) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 6 * n_rows))
    axes = np.array(axes).flatten()

    plot_df = df[[target_col] + [v for v, _, _ in available]].dropna()

    for ax, (col, label, color) in zip(axes, available):
        r = plot_df[[target_col, col]].corr().iloc[0, 1]
        ax.scatter(plot_df[col], plot_df[target_col],
                   alpha=0.3, s=10, color=color)
        m, b = np.polyfit(plot_df[col], plot_df[target_col], 1)
        x_line = np.linspace(plot_df[col].min(), plot_df[col].max(), 100)
        ax.plot(x_line, m * x_line + b, color="red", linewidth=1.5)
        ax.set_title(f"PM10 vs {label}\n(r = {r:.3f})", fontsize=12)
        ax.set_xlabel(label, fontsize=10)
        ax.set_ylabel("PM10 [µg/m³]", fontsize=10)
        ax.grid(True, alpha=0.2)

    for ax in axes[len(available):]:
        ax.set_visible(False)

    fig.suptitle("PM10 vs Exogenous Variables", fontsize=16, fontweight="bold")
    plt.tight_layout()
    save_figure(fig, Path(images_dir) / "weather_scatter_plots.png")
    logger.info("Saved weather_scatter_plots.png")


def plot_dual_axis_timeseries(
    df: pd.DataFrame,
    target_col: str,
    images_dir: Path,
) -> None:
    """Dual-axis line plot: PM10 (left) vs temperature (right), last 12 months."""
    set_plot_style()

    cutoff  = df.index.max() - pd.DateOffset(years=1)
    df_year = df[df.index >= cutoff]

    fig, ax1 = plt.subplots(figsize=(12, 7))

    color1 = "tab:red"
    ax1.set_xlabel("Date", fontsize=12)
    ax1.set_ylabel("PM10 [µg/m³]", color=color1, fontsize=12, fontweight="bold")
    ax1.plot(df_year.index, df_year[target_col],
             color=color1, alpha=0.7, label="PM10")
    ax1.fill_between(df_year.index, df_year[target_col],
                     color=color1, alpha=0.1)
    ax1.tick_params(axis="y", labelcolor=color1)

    ax2 = ax1.twinx()
    color2 = "tab:blue"
    ax2.set_ylabel("Temperature [°C]", color=color2, fontsize=12, fontweight="bold")
    ax2.plot(df_year.index, df_year["temp_avg"],
             color=color2, linestyle="--", label="Temperature")
    ax2.tick_params(axis="y", labelcolor=color2)

    fig.suptitle(
        "Temporal Relationship: PM10 vs Temperature (Last 12 Months)",
        fontsize=16, fontweight="bold",
    )
    plt.tight_layout()
    save_figure(fig, Path(images_dir) / "weather_dual_axis_timeseries.png")
    logger.info("Saved weather_dual_axis_timeseries.png")


def plot_weather_correlation_heatmap(
    df: pd.DataFrame,
    target_col: str,
    images_dir: Path,
) -> None:
    """Pearson correlation heatmap: PM10 vs all meteorological variables."""
    set_plot_style()

    candidate_cols = [
        target_col, "temp_avg", "rain_sum", "wind_mean",
        "pressure_avg", "humidity_avg", "snowfall_sum",
    ]
    cols = [c for c in candidate_cols if c in df.columns]
    corr = df[cols].corr()

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr, annot=True, cmap="RdYlGn", fmt=".2f",
                center=0, linewidths=0.5, ax=ax)
    ax.set_title("Correlation Heatmap: PM10 vs Meteorological Factors",
                 fontsize=16, fontweight="bold", pad=20)
    plt.tight_layout()
    save_figure(fig, Path(images_dir) / "weather_correlation_heatmap.png")
    logger.info("Saved weather_correlation_heatmap.png")

def plot_spatial_correlation(
    df: pd.DataFrame,
    all_stations: list[str],
    images_dir: Path,
) -> None:
    """Heatmap of inter-station Pearson correlations."""
    set_plot_style()

    available = [c for c in all_stations if c in df.columns]
    corr      = df[available].corr()

    fig, ax = plt.subplots(figsize=(8, 7))
    sns.heatmap(corr, annot=True, fmt=".3f", cmap="RdYlGn",
                vmin=0.5, vmax=1.0, linewidths=1, square=True, ax=ax)
    ax.set_title("Correlation Matrix — Kraków PM10 Stations",
                 fontsize=14, fontweight="bold")
    ax.tick_params(axis="x", rotation=30)

    save_figure(fig, Path(images_dir) / "spatial_correlation.png")
    logger.info("Saved spatial_correlation.png")


def plot_top_pollution_events(
    df: pd.DataFrame,
    target_col: str,
    images_dir: Path,
    n: int = 15,
    eu_limit: float = 50,
) -> None:
    """Horizontal bar chart of the n highest single-day PM10 readings."""
    set_plot_style()

    top = df[target_col].nlargest(n).reset_index()
    top.columns = ["Date", target_col]
    top["label"] = top["Date"].dt.strftime("%d %b %Y")

    norm = mcolors.Normalize(vmin=top[target_col].min(), vmax=top[target_col].max())
    cmap_ = plt.cm.Reds
    bar_colors = [cmap_(norm(v)) for v in top[target_col]]

    fig, ax = plt.subplots(figsize=(12, 7))
    bars = ax.barh(
        top["label"][::-1], top[target_col][::-1],
        color=bar_colors[::-1], alpha=0.8, edgecolor="black", height=0.7,
    )
    for bar in bars:
        ax.text(
            bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
            int(bar.get_width()), va="center", fontsize=10,
        )

    ax.axvline(eu_limit, color="black", linestyle="--", linewidth=2,
               label=f"EU norm ({eu_limit} µg/m³)")
    ax.set_title(f"Top {n} Days with Highest PM10 Concentrations (2019–2024)",
                 fontsize=16, fontweight="bold")
    ax.set_xlabel("PM10 [µg/m³]", fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.2, axis="x")

    save_figure(fig, Path(images_dir) / "top15_peak_pollution_events.png")
    logger.info("Saved top15_peak_pollution_events.png")

def plot_stl_decomposition(
    series: pd.Series,
    images_dir: Path,
    period: int = 365,
) -> None:
    """Four-panel STL decomposition (observed / trend / seasonal / residuals)."""
    set_plot_style()
    stl = STL(series, period=period, robust=True).fit()

    fig, axes = plt.subplots(4, 1, figsize=(14, 12), sharex=True)
    panel_data = [
        (series.values, "Observed\n(µg/m³)", "#1e272e", False),
        (stl.trend,  "Secular Trend\n(µg/m³)", "#d35400", False),
        (stl.seasonal, "Seasonal\nVariation", "#009432", True),
        (stl.resid, "Residuals\n(Noise)",  "#7f8c8d", True),
    ]

    for ax, (data, ylabel, color, scatter) in zip(axes, panel_data):
        if scatter:
            ax.scatter(series.index, data, color=color, s=2, alpha=0.5)
            ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
        else:
            ax.plot(series.index, data, color=color,
                    linewidth=2 if ylabel.startswith("Secular") else 0.8)
        ax.set_ylabel(ylabel, fontsize=12, fontweight="bold")
        ax.grid(True, axis="y", linestyle=":", alpha=0.4)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    axes[0].set_title(
        "Longitudinal PM10 Signal Decomposition in Kraków (STL Method)",
        fontsize=18, fontweight="bold", pad=25,
    )
    axes[-1].set_xlabel("Year", fontsize=12, labelpad=15)

    save_figure(fig, Path(images_dir) / "eda_stl_decomposition_analysis.png")
    logger.info("Saved eda_stl_decomposition_analysis.png")


def plot_acf_pacf(
    series: pd.Series,
    images_dir: Path,
    lags: int = 40,
) -> None:
    """ACF and PACF plots for lag-structure identification."""
    set_plot_style()

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    plot_acf(series, lags=lags, ax=ax1, color="#4e8cff",
             vlines_kwargs={"colors": "#4e8cff", "alpha": 0.5},
             title="Autocorrelation (ACF) — Seasonal patterns check")
    plot_pacf(series, lags=lags, ax=ax2, color="#e74c3c",
              vlines_kwargs={"colors": "#e74c3c", "alpha": 0.5},
              title="Partial Autocorrelation (PACF) — Direct lag influence",
              method="ywm")

    for ax in (ax1, ax2):
        ax.grid(True, axis="y", linestyle="--", alpha=0.3)
        ax.set_ylabel("Correlation Strength", fontsize=12, color="#555555")
        sns.despine(ax=ax, left=True, bottom=False)

    ax2.set_xlabel("Lags (time steps back)", fontsize=12, fontweight="bold")
    fig.suptitle("Time Series Diagnostic: ACF & PACF for PM10",
                 fontsize=16, fontweight="bold", y=0.98)

    save_figure(fig, Path(images_dir) / "eda_acf_and_pacf.png")
    logger.info("Saved eda_acf_and_pacf.png")

def run_full_eda(
    df: pd.DataFrame,
    target_col: str,
    images_dir: Path,
    all_stations: list[str] | None = None,
    eu_limit: float = 50,
) -> None:
    """Run every EDA plot in sequence.

    Parameters
    ----------
    df:
        Fully merged, cleaned DataFrame with calendar and weather columns.
    target_col:
        PM10 column name (raw µg/m³).
    images_dir:
        Output directory for all PNG files.
    all_stations:
        All station column names (for spatial correlation plot).
    eu_limit:
        EU daily PM10 limit used for reference lines.
    """
    images_dir = Path(images_dir)
    series = df[target_col].dropna()

    _df = df.copy()
    if "season" not in _df.columns:
        _df["month"] = _df.index.month
        _df["season"] = _df["month"].map({
            12: "Winter", 1: "Winter",  2: "Winter",
            3:  "Spring", 4: "Spring",  5: "Spring",
            6:  "Summer", 7: "Summer",  8: "Summer",
            9:  "Autumn", 10: "Autumn", 11: "Autumn",
        })
    if "week" not in _df.columns:
        _df["week"] = _df.index.dayofweek
    if "year" not in _df.columns:
        _df["year"] = _df.index.year
    if "month_name" not in _df.columns:
        _df["month_name"] = _df.index.month_name()

    plot_time_series(_df, target_col, images_dir, eu_limit)
    plot_exceedances_and_distribution(_df, target_col, images_dir, eu_limit)
    plot_seasonal_distribution(_df, target_col, images_dir, eu_limit)
    plot_weekly_cycle(_df, target_col, images_dir, eu_limit)
    plot_monthly_boxplots(_df, target_col, images_dir, eu_limit)
    plot_heatmap_month_year(_df, target_col, images_dir)
    plot_weather_scatter(_df, target_col, images_dir)
    plot_dual_axis_timeseries(_df, target_col, images_dir)
    plot_weather_correlation_heatmap(_df, target_col, images_dir)

    if all_stations:
        plot_spatial_correlation(_df, all_stations, images_dir)

    plot_top_pollution_events(_df, target_col, images_dir, eu_limit=eu_limit)
    plot_stl_decomposition(series, images_dir)
    plot_acf_pacf(series, images_dir)

    logger.info("Full EDA complete — images saved to %s", images_dir)