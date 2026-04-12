"""
Data loading: PM10 Excel files and weather data from the Open-Meteo archive API.
"""

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import requests

from src.utils import get_logger

logger = get_logger(__name__)

def load_pm10_raw(data_dir: Path, years: Iterable[int]) -> pd.DataFrame:
    """Read and concatenate yearly PM10 Excel files.

    Each file follows the Polish GIOŚ format where the header row containing
    'Kod stacji' is located dynamically (its position varies across years).
    The function reads once without a header to find the row, then re-reads
    with the correct ``header`` argument — exactly as done in the notebook.

    Parameters
    ----------
    data_dir:
        Directory that contains ``{year}_PM10_24g.xlsx`` files.
    years:
        Iterable of integer years to load (e.g. ``range(2019, 2025)``).

    Returns
    -------
    pd.DataFrame
        Raw concatenated data before any cleaning.
    """
    frames = []
    for year in years:
        path = Path(data_dir) / f"{year}_PM10_24g.xlsx"
        logger.info("Loading %s", path)

        df_raw = pd.read_excel(path, header=None)
        header_row = None
        for i, row in df_raw.iterrows():
            if row.astype(str).str.contains("Kod stacji", na=False).any():
                header_row = i
                break
        if header_row is None:
            raise ValueError(f"Could not find 'Kod stacji' header in {path}")

        df = pd.read_excel(path, header=header_row)
        df = df.rename(columns={"Kod stacji": "Date"})
        frames.append(df)

    combined = pd.concat(frames, ignore_index=True)
    logger.info("Loaded %d rows across %d files", len(combined), len(frames))
    return combined


def parse_pm10_stations(
    raw: pd.DataFrame,
    stations: list[str],
) -> pd.DataFrame:
    """Extract station columns, cast to float, and enforce a daily index.

    Decimal separators in the GIOŚ export use commas.  Each station column is
    cast via ``pd.to_numeric`` with ``errors='coerce'`` so that non-numeric
    entries become NaN rather than raising.  Missing dates are inserted as NaN
    rows by ``asfreq('D')``.

    Note: gap imputation (interpolation + long-gap flags) is deferred to
    :mod:`src.data_preprocessing` so that the two concerns stay separate.

    Parameters
    ----------
    raw:
        Output of :func:`load_pm10_raw`.
    stations:
        Station codes to keep (e.g. ``['MpKrakWadow', ...]``).

    Returns
    -------
    pd.DataFrame
        Float columns per station, ``DatetimeIndex`` named ``Date``,
        strict daily frequency, NaN where data is missing.
    """
    df = raw[["Date"] + stations].copy()

    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"]).set_index("Date")

    for col in stations:
        df[col] = (
            df[col]
            .astype(str)
            .str.replace(",", ".", regex=False)
            .pipe(pd.to_numeric, errors="coerce")
        )

    df = df.asfreq("D")

    logger.info(
        "Parsed PM10 data: %d days, stations: %s", len(df), stations
    )
    return df

def fetch_weather(api_url: str, params: dict, col_rename: dict) -> pd.DataFrame:
    """Download daily weather data from the Open-Meteo archive API.

    After downloading, dominant wind direction (reported as a compass string
    or degrees) is encoded as cyclical sin / cos components to preserve the
    circular topology of the variable.

    Parameters
    ----------
    api_url:
        Base URL of the Open-Meteo archive endpoint.
    params:
        Query parameters accepted by the API (latitude, longitude, dates, …).
    col_rename:
        Mapping from raw API column names to project-standard names.

    Returns
    -------
    pd.DataFrame
        Weather DataFrame with a ``DatetimeIndex`` named ``Date``,
        including ``wind_dir_sin`` and ``wind_dir_cos`` columns.

    Raises
    ------
    requests.HTTPError
        If the API returns a non-2xx status code.
    """
    logger.info("Fetching weather data from Open-Meteo …")
    response = requests.get(api_url, params=params, timeout=60)
    response.raise_for_status()

    data = response.json()
    weather = pd.DataFrame(data["daily"])
    weather["time"] = pd.to_datetime(weather["time"])
    weather = weather.rename(columns=col_rename)
    weather = weather.set_index("Date")

    DIR_MAP = {
        "N": 0, "NNE": 22.5, "NE": 45,  "ENE": 67.5,
        "E": 90, "ESE": 112.5, "SE": 135, "SSE": 157.5,
        "S": 180, "SSW": 202.5, "SW": 225, "WSW": 247.5,
        "W": 270, "WNW": 292.5, "NW": 315, "NNW": 337.5,
    }
    col = "wind_dir_dominant"
    if col in weather.columns:
        if weather[col].dtype == object:
            weather["wind_dir_deg"] = weather[col].map(DIR_MAP).fillna(0)
        else:
            weather["wind_dir_deg"] = weather[col].fillna(0)

        rad = np.deg2rad(weather["wind_dir_deg"])
        weather["wind_dir_sin"] = np.sin(rad)
        weather["wind_dir_cos"] = np.cos(rad)
        weather = weather.drop(columns=[col, "wind_dir_deg"])

    logger.info("Downloaded %d rows of weather data", len(weather))
    return weather