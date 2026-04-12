"""
Central configuration shared by backend and frontend.
Import from here instead of hard-coding values anywhere.

Environment variables (override via .env or container env):
  API_HOST      – URL of the FastAPI backend  (default: http://localhost:8000)
  BACKEND_PORT  – backend listen port         (default: 8000)
  FRONTEND_PORT – frontend listen port        (default: 8501)
"""

import os
from pathlib import Path

ROOT_DIR   = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT_DIR / "models"
DATA_DIR   = ROOT_DIR / "data"

API_HOST      = os.getenv("API_HOST",      "http://localhost:8000")
BACKEND_PORT  = int(os.getenv("BACKEND_PORT",  "8000"))
FRONTEND_PORT = int(os.getenv("FRONTEND_PORT", "8501"))

TARGET_STATION = "MpKrakWadow"

_ALL_STATION_META: dict[str, dict] = {
    "MpKrakWadow":  {"lat": 50.0577, "lon": 19.9265, "name": "Wadowicka"},
    "MpKrakAlKras": {"lat": 50.0621, "lon": 19.9355, "name": "Al. Krasińskiego"},
    "MpKrakBujaka": {"lat": 50.0080, "lon": 19.9020, "name": "Bujaka"},
    "MpKrakBulwar": {"lat": 50.0573, "lon": 19.9346, "name": "Bulwarowa"},
    "MpKrakOsPias": {"lat": 50.0749, "lon": 20.0234, "name": "Os. Piastów"},
    "MpKrakSwoszo": {"lat": 50.0200, "lon": 19.8960, "name": "Swoszowice"},
    "MpKrakZloRog": {"lat": 50.0201, "lon": 20.0207, "name": "Złoty Róg"},
}

def _build_stations_meta() -> dict[str, dict]:
    """Return STATIONS_META restricted to stations actually present in data/.

    Reads only the header rows of each Excel workbook (fast).  Falls back to
    the four original stations if the data directory is missing or unreadable.
    """
    try:
        import sys
        sys.path.insert(0, str(ROOT_DIR))
        from src.data_loading import detect_krakow_stations
        detected = detect_krakow_stations(DATA_DIR)
    except Exception:
        detected = []

    if not detected:
        detected = ["MpKrakWadow", "MpKrakBujaka", "MpKrakBulwar", "MpKrakSwoszo"]

    return {
        sid: _ALL_STATION_META[sid]
        for sid in detected
        if sid in _ALL_STATION_META
    }

STATIONS_META: dict[str, dict] = _build_stations_meta()

PM10_GOOD = 25   
PM10_MODERATE = 50  
PM10_HIGH = 100  

EU_DAILY_LIMIT = 50

MODEL_NAMES = ["LightGBM", "SARIMAX", "ARIMA"]

HEATING_MONTHS  = [1, 2, 3, 10, 11, 12]
SARIMAX_EXOG    = [
    "temp_avg", "wind_max", "is_heating_season",
    "is_calm_wind", "hdd_calm", "rain_3d_sum", "inversion_proxy",
]

REGIME_LABELS = {0: "Clean", 1: "Moderate", 2: "Polluted"}
REGIME_COLORS = {"Clean": "#2ecc71", "Moderate": "#f39c12", "Polluted": "#e74c3c"}

COLOR_LGBM    = "#4B9965"
COLOR_SARIMAX = "#64748B"
COLOR_ARIMA   = "#6396E7"
COLOR_ACTUAL  = "#1e3a5f"
COLOR_NAIVE   = "#aaaaaa"