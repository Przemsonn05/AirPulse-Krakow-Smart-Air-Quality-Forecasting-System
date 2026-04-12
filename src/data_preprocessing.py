"""
Data preprocessing: gap imputation, long-gap flagging, and weather join.

The notebook distinguishes short gaps (≤ 3 days — interpolated linearly) from
long gaps (> 3 days — flagged with a binary column and left as NaN to avoid
artificially introducing data far from true sensor readings).  A final
forward- / back-fill pass closes any remaining boundary NaNs.
"""

import pandas as pd

from src.utils import get_logger

logger = get_logger(__name__)

def impute_gaps(
    df: pd.DataFrame,
    stations: list[str],
    limit: int = 3,
) -> pd.DataFrame:
    """Impute short gaps and flag long gaps for each station column.

    Logic (per station, mirrors notebook cells 15–16):
    1. Identify runs of consecutive NaN values.
    2. Gaps of ``limit`` days or fewer → time-based linear interpolation.
    3. Gaps longer than ``limit`` → binary ``{col}_long_gap`` flag (1 = long
       gap), values remain NaN.
    4. A final ``ffill`` / ``bfill`` pass closes any isolated boundary NaNs.

    Parameters
    ----------
    df:
        Date-indexed DataFrame with one column per station (daily frequency).
    stations:
        List of station column names to process.
    limit:
        Maximum gap length (in days) eligible for interpolation.

    Returns
    -------
    pd.DataFrame
        DataFrame with imputed values and ``*_long_gap`` indicator columns
        appended (copy).
    """
    df = df.copy()

    for col in stations:
        is_null   = df[col].isnull()
        gap_id    = is_null.ne(is_null.shift()).cumsum()
        gap_sizes = is_null.groupby(gap_id).transform("sum")

        df[f"{col}_long_gap"] = ((is_null) & (gap_sizes > limit)).astype(int)

        df[col] = df[col].interpolate(method="time", limit=limit)

        df[col] = df[col].ffill().bfill()

        n_long = df[f"{col}_long_gap"].sum()
        logger.info(
            "Column '%s': %d long-gap days flagged (gap > %d days)",
            col, n_long, limit,
        )

    return df

def merge_weather(
    pm10_df: pd.DataFrame,
    weather_df: pd.DataFrame,
) -> pd.DataFrame:
    """Left-join PM10 station data with weather data on the date index.

    Parameters
    ----------
    pm10_df:
        Date-indexed PM10 DataFrame (output of :func:`impute_gaps`).
    weather_df:
        Date-indexed weather DataFrame (output of
        :func:`src.data_loading.fetch_weather`).

    Returns
    -------
    pd.DataFrame
        Merged DataFrame (same row count as ``pm10_df``).
    """
    merged  = pm10_df.join(weather_df, how="left")
    missing = merged[weather_df.columns].isna().sum().sum()
    if missing:
        logger.warning(
            "Weather join left %d missing values — check date alignment",
            missing,
        )
    logger.info("Merged DataFrame shape: %s", merged.shape)
    return merged