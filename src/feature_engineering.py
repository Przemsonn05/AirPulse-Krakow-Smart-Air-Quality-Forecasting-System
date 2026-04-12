"""
Feature engineering: calendar features, Box-Cox transform, lag/rolling features,
domain-specific weather flags, spatial auxiliary features, and interaction terms.

Design rules (all mirroring notebook cells 60–87):
- Box-Cox lambda is estimated on the TRAINING split only to prevent leakage.
- Lag features are computed on ``PM10_transformed`` (Box-Cox scale).
- Rolling statistics are computed on the raw target with ``shift(1)`` so that
  the current day's value is never included in any rolling window.
- All interaction terms that combine a weather flag with a lagged quantity
  consistently apply the same lag to all components.
"""

import holidays
import numpy as np
import pandas as pd
from scipy.stats import boxcox

from src.utils import get_logger

logger = get_logger(__name__)

def apply_boxcox_transform(
    df: pd.DataFrame,
    raw_col: str,
    train_end: str,
    transformed_col: str = "PM10_transformed",
) -> tuple[pd.DataFrame, float]:
    """Fit Box-Cox on the training window and apply to the full series.

    Lambda is estimated exclusively on training data (``index <= train_end``)
    with strictly positive values.  The same lambda is then used to transform
    the entire DataFrame so that val/test rows share the same scale.

    Parameters
    ----------
    df:
        Full merged DataFrame with ``raw_col`` present.
    raw_col:
        Name of the original (raw) PM10 column.
    train_end:
        Last date of the training period, e.g. ``"2022-12-31"``.
    transformed_col:
        Name for the new transformed column (default ``PM10_transformed``).

    Returns
    -------
    df : pd.DataFrame
        DataFrame with ``transformed_col`` appended (copy).
    lambda_bc : float
        Lambda value needed to invert the transform at prediction time.
    """
    df = df.copy()

    train_mask = df.index <= train_end
    pm10_train = df.loc[train_mask, raw_col].dropna()
    pm10_train_pos = pm10_train[pm10_train > 0]

    _, lambda_bc = boxcox(pm10_train_pos)

    mask_pos = df[raw_col] > 0
    df[transformed_col] = np.nan
    df.loc[mask_pos, transformed_col] = boxcox(
        df.loc[mask_pos, raw_col], lambda_bc
    )

    logger.info(
        "Box-Cox transform fitted on training data (lambda=%.4f). "
        "Column '%s' created.",
        lambda_bc, transformed_col,
    )
    return df, lambda_bc

def add_calendar_features(
    df: pd.DataFrame,
    heating_months: list[int] | None = None,
) -> pd.DataFrame:
    """Add calendar columns derived from the DatetimeIndex.

    Columns added
    -------------
    month, week (dayofweek 0–6), year, season, month_name,
    is_heating_season, is_weekend

    Parameters
    ----------
    df:
        Date-indexed DataFrame.
    heating_months:
        Months classified as the heating season. Defaults to
        ``[1, 2, 3, 10, 11, 12]``.
    """
    if heating_months is None:
        heating_months = [1, 2, 3, 10, 11, 12]

    df = df.copy()
    df["month"] = df.index.month
    df["week"] = df.index.dayofweek
    df["year"] = df.index.year
    df["month_name"] = df.index.month_name()

    df["season"] = df["month"].map({
        12: "Winter", 1: "Winter",  2: "Winter",
        3:  "Spring", 4: "Spring",  5: "Spring",
        6:  "Summer", 7: "Summer",  8: "Summer",
        9:  "Autumn", 10: "Autumn", 11: "Autumn",
    })

    df["is_heating_season"] = df["month"].isin(heating_months).astype(int)
    df["is_weekend"] = (df["week"] >= 5).astype(int)

    logger.info("Calendar features added")
    return df

def add_holiday_flag(
    df: pd.DataFrame,
    country: str = "PL",
    years: list[int] | None = None,
) -> pd.DataFrame:
    """Add a binary ``is_holiday`` flag for Polish public holidays.

    Parameters
    ----------
    df:
        Date-indexed DataFrame that already has a ``year`` column.
    country:
        ISO country code accepted by the ``holidays`` library.
    years:
        Years to cover; inferred from the index when ``None``.
    """
    df = df.copy()
    if years is None:
        years = list(df.index.year.unique())

    pl_holidays = holidays.CountryHoliday(country, years=years)
    df["is_holiday"] = df.index.isin(pl_holidays).astype(int)
    logger.info(
        "Holiday flag added (%d holidays across %s)", len(pl_holidays), years
    )
    return df

def add_cyclical_features(df: pd.DataFrame) -> pd.DataFrame:
    """Encode month, day-of-year, and day-of-week as sine / cosine pairs.

    Cyclical encoding eliminates the artificial discontinuity between
    December and January (or Sunday and Monday) that a plain integer
    representation would introduce.

    Columns added
    -------------
    month_sin, month_cos, doy_sin, doy_cos, dow_sin, dow_cos

    Requires: ``month`` and ``week`` columns (from :func:`add_calendar_features`).
    """
    df = df.copy()

    month = df["month"]
    day_of_year = df.index.dayofyear
    weekday = df["week"]

    df["month_sin"] = np.sin(2 * np.pi * month / 12)
    df["month_cos"] = np.cos(2 * np.pi * month / 12)
    df["doy_sin"]   = np.sin(2 * np.pi * day_of_year / 365)
    df["doy_cos"]   = np.cos(2 * np.pi * day_of_year / 365)
    df["dow_sin"]   = np.sin(2 * np.pi * weekday / 7)
    df["dow_cos"]   = np.cos(2 * np.pi * weekday / 7)

    logger.info("Cyclical encoding added (month, day-of-year, day-of-week)")
    return df

def add_lag_features(
    df: pd.DataFrame,
    transformed_col: str = "PM10_transformed",
    lags: list[int] | None = None,
) -> pd.DataFrame:
    """Add lagged versions of the Box-Cox-transformed PM10 column.

    Lags are computed on the *transformed* series so that the autoregressive
    feature scale is consistent with the model's target scale.  All lags are
    ≥ 1 — no same-day leakage.

    Parameters
    ----------
    df:
        DataFrame with ``transformed_col`` present.
    transformed_col:
        Box-Cox-transformed PM10 column (default ``PM10_transformed``).
    lags:
        Lag distances in days. Defaults to ``[1, 2, 7, 14]``.
    """
    if lags is None:
        lags = [1, 2, 7, 14]

    df = df.copy()
    for k in lags:
        df[f"lag_{k}d"] = df[transformed_col].shift(k)
    logger.info("Lag features added (on '%s'): %s", transformed_col, lags)
    return df

def add_rolling_features(
    df: pd.DataFrame,
    raw_col: str,
    windows: list[int] | None = None,
) -> pd.DataFrame:
    """Add rolling mean and std — computed on raw PM10 with ``shift(1)``.

    The base series is always ``shift(1)`` to guarantee that today's
    observation never contaminates any rolling window (no leakage).

    ``min_periods`` is set to ``max(1, window // 2)`` to retain non-NaN
    values at the start of the series rather than producing excessive NaN
    rows at the warm-up boundary.

    Columns added (per window *w*)
    --------------------------------
    rolling_mean_{w}d, rolling_std_{w}d

    Plus momentum features (no window suffix):
    rolling_diff_7d = rolling_mean_7d − rolling_mean_14d

    Parameters
    ----------
    df:
        DataFrame with ``raw_col``.
    raw_col:
        Raw (µg/m³) PM10 column to roll over.
    windows:
        Window sizes in days. Defaults to ``[3, 7, 14, 30]``.
    """
    if windows is None:
        windows = [3, 7, 14, 30]

    df = df.copy()
    base = df[raw_col].shift(1)

    for w in windows:
        mp = max(1, w // 2)
        df[f"rolling_mean_{w}d"] = base.rolling(w, min_periods=mp).mean()
        df[f"rolling_std_{w}d"]  = base.rolling(w, min_periods=mp).std()

    df["rolling_diff_7d"] = df["rolling_mean_7d"] - df["rolling_mean_14d"]

    logger.info("Rolling features added (on '%s'): windows=%s", raw_col, windows)
    return df

def add_weather_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute physically motivated features from raw weather columns.

    Required columns: ``temp_avg``, ``rain_sum``, ``wind_max``.
    Optional columns: ``wind_mean``, ``temp_max``, ``temp_min``,
    ``pressure_avg``.

    All lags and rolling windows use ``shift(1)`` so that weather features
    do not incorporate the target day's unobserved values.

    Features added
    --------------
    is_frost              — 1 if temp_avg ≤ 0 °C (heating threshold)
    is_calm_wind          — 1 if wind_mean ≤ 2 m/s (or wind_max ≤ 3 as fallback)
    wind_inverse          — 1 / (wind_max + 0.1)  dispersion non-linearity
    wind_7d_mean          — 7-day lagged rolling mean wind (shift-1)
    heating_degree_days   — max(0, 15 − temp_avg)
    hdd_7d                — 7-day lagged rolling sum of HDD (shift-1)
    rain_yesterday        — rain_sum shifted by 1 day
    rain_3d_sum           — 3-day rolling sum of rain (shift-1)
    rain_7d_sum           — 7-day rolling sum of rain (shift-1)
    dry_spell_days        — count of zero-rain days in last 14 (shift-1)
    temp_amplitude        — temp_max − temp_min (if both present)
    inversion_proxy       — binary: low temp amplitude + frost + calm wind
    high_pressure_flag    — 1 if pressure > 30-day lagged rolling mean
    """
    df = df.copy()

    df["is_frost"] = (df["temp_avg"] <= 0).astype(int)

    if "wind_mean" in df.columns:
        df["is_calm_wind"] = (df["wind_mean"] <= 2).astype(int)
    else:
        df["is_calm_wind"] = (df["wind_max"] <= 3).astype(int)

    df["wind_inverse"] = 1 / (df["wind_max"] + 0.1)
    df["wind_7d_mean"] = df["wind_max"].shift(1).rolling(7).mean()

    df["heating_degree_days"] = (15 - df["temp_avg"]).clip(lower=0)
    df["hdd_7d"] = df["heating_degree_days"].shift(1).rolling(7).sum()

    df["rain_yesterday"] = df["rain_sum"].shift(1)
    df["rain_3d_sum"]    = df["rain_sum"].shift(1).rolling(3, min_periods=1).sum()
    df["rain_7d_sum"]    = df["rain_sum"].shift(1).rolling(7, min_periods=1).sum()
    df["dry_spell_days"] = (
        df["rain_sum"].shift(1)
        .rolling(14)
        .apply(lambda x: (x == 0).sum(), raw=True)
    )

    if "temp_max" in df.columns and "temp_min" in df.columns:
        df["temp_amplitude"] = df["temp_max"] - df["temp_min"]
        low_amplitude = (df["temp_amplitude"] < 4).astype(int)
    else:
        low_amplitude = (df["temp_avg"].rolling(3).std() < 1.5).astype(int)

    df["inversion_proxy"] = (
        low_amplitude
        * (df["temp_avg"] < 5).astype(int)
        * df["is_calm_wind"]
    )

    if "pressure_avg" in df.columns:
        df["pressure_trend_3d"] = df["pressure_avg"].diff(3)
        rolling_pressure = df["pressure_avg"].shift(1).rolling(30, min_periods=15).mean()
        df["high_pressure_flag"] = (
            df["pressure_avg"] > rolling_pressure
        ).astype(int)

    logger.info("Weather-derived features added")
    return df

def add_aux_station_features(
    df: pd.DataFrame,
    aux_stations: list[str],
) -> pd.DataFrame:
    """Create lag-1 features from auxiliary stations and spatial aggregates.

    The EDA confirmed inter-station Pearson correlations > 0.90, meaning
    yesterday's readings at nearby stations are strong predictors of today's
    target.  Spatial aggregates (mean, max, spread) provide a low-dimensional
    regional summary without introducing redundant columns.

    All values are lagged by 1 day (``shift(1)``).

    Columns added
    -------------
    {aux_col}_lag1     — per-station lag-1
    aux_mean_lag1      — mean across auxiliary station lags
    aux_std_lag1       — std  across auxiliary station lags
    aux_max_lag1       — max  across auxiliary station lags
    aux_spread_lag1    — max − min across auxiliary station lags

    Parameters
    ----------
    df:
        DataFrame containing the raw auxiliary station columns.
    aux_stations:
        Column names of auxiliary stations present in ``df``.
    """
    df = df.copy()
    available = [c for c in aux_stations if c in df.columns]

    if not available:
        logger.warning("No auxiliary station columns found — skipping spatial features")
        return df

    lag1_cols = []
    for col in available:
        lag_col = f"{col}_lag1"
        df[lag_col] = df[col].shift(1)
        lag1_cols.append(lag_col)

    df["aux_mean_lag1"] = df[lag1_cols].mean(axis=1)
    df["aux_std_lag1"] = df[lag1_cols].std(axis=1)
    df["aux_max_lag1"] = df[lag1_cols].max(axis=1)
    df["aux_spread_lag1"] = (
        df[lag1_cols].max(axis=1) - df[lag1_cols].min(axis=1)
    )

    logger.info("Auxiliary station spatial features added: %s", available)
    return df

def add_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create multiplicative interaction terms between domain flags.

    Requires: ``is_frost``, ``is_calm_wind``, ``is_heating_season``,
    ``heating_degree_days``, ``rain_sum``, ``temp_avg``.

    Columns added
    -------------
    is_frost_calm           — frost × calm wind (double stagnation)
    is_heating_season_calm  — heating season × calm (sustained smog risk)
    hdd_calm                — HDD × calm (heating demand + no dispersion)
    cold_dry_calm           — sub-zero × no-rain-yesterday × calm
    """
    df = df.copy()

    df["is_frost_calm"]          = df["is_frost"]           * df["is_calm_wind"]
    df["is_heating_season_calm"] = df["is_heating_season"]  * df["is_calm_wind"]
    df["hdd_calm"]               = df["heating_degree_days"] * df["is_calm_wind"]

    df["cold_dry_calm"] = (
        (df["temp_avg"] < 0).astype(int)
        * (df["rain_sum"].shift(1) == 0).astype(int)
        * df["is_calm_wind"]
    )

    logger.info("Interaction features added")
    return df

def build_features(
    df: pd.DataFrame,
    target_col: str,
    train_end: str,
    aux_stations: list[str] | None = None,
    heating_months: list[int] | None = None,
    lag_days: list[int] | None = None,
    rolling_windows: list[int] | None = None,
) -> tuple[pd.DataFrame, float]:
    """Run the complete feature-engineering pipeline in the correct order.

    Call sequence matters for correctness:
    1. Calendar features (``month``, ``season``, etc.).
    2. Holiday flag.
    3. Cyclical encoding (requires ``month``, ``week``).
    4. Box-Cox transform (fitted on training split only → no leakage).
    5. Lag features (on ``PM10_transformed``).
    6. Rolling features (on raw target, shift-1 safe).
    7. Weather-derived features.
    8. Auxiliary station spatial features.
    9. Interaction terms.
    10. Drop NaN rows from rolling/lag warm-up.

    Parameters
    ----------
    df:
        Merged PM10 + weather DataFrame from the preprocessing stage.
    target_col:
        Raw (µg/m³) PM10 column used as the modelling target.
    train_end:
        Last date (inclusive) of the training period, e.g. ``"2022-12-31"``.
        Used to fit the Box-Cox lambda without data leakage.
    aux_stations:
        Auxiliary station column names for spatial features.  ``None`` skips
        spatial feature creation.
    heating_months:
        Forwarded to :func:`add_calendar_features`.
    lag_days:
        Forwarded to :func:`add_lag_features`.
    rolling_windows:
        Forwarded to :func:`add_rolling_features`.

    Returns
    -------
    df_features : pd.DataFrame
        Feature-complete DataFrame.  NaN rows from the rolling / lag warm-up
        have been dropped.
    lambda_bc : float
        Box-Cox lambda — required by :func:`src.utils.safe_inv_boxcox` when
        back-transforming predictions at evaluation time.
    """
    df = add_calendar_features(df, heating_months)
    df = add_holiday_flag(df)
    df = add_cyclical_features(df)

    df, lambda_bc = apply_boxcox_transform(df, target_col, train_end)

    df = add_lag_features(df, transformed_col="PM10_transformed", lags=lag_days)
    df = add_rolling_features(df, raw_col=target_col, windows=rolling_windows)
    df = add_weather_features(df)

    if aux_stations:
        df = add_aux_station_features(df, aux_stations)

    df = add_interaction_features(df)

    before = len(df)
    df = df.dropna(subset=["PM10_transformed", "lag_1d"])
    logger.info(
        "Dropped %d NaN rows (rolling/lag warm-up). Final shape: %s",
        before - len(df), df.shape,
    )
    return df, lambda_bc