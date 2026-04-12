"""
Central configuration for the PM10 air-quality forecasting project.
Import this module everywhere instead of hard-coding paths or parameters.
"""

from pathlib import Path

ROOT_DIR   = Path(__file__).resolve().parent.parent
DATA_DIR   = ROOT_DIR / "data"
IMAGES_DIR = ROOT_DIR / "images"

STATIONS = ["MpKrakWadow", "MpKrakSwoszo", "MpKrakBujaka", "MpKrakBulwar"]
TARGET          = "MpKrakWadow"          
AUX_STATIONS    = ["MpKrakBujaka", "MpKrakBulwar", "MpKrakSwoszo"]
PM10_BC_COL     = "PM10_transformed"     

YEARS = range(2019, 2025)

TRAIN_END = "2022-12-31"  
VAL_END   = "2023-12-31"   

WEATHER_API_URL = "https://archive-api.open-meteo.com/v1/archive"

WEATHER_PARAMS = {
    "latitude":   50.0577717,
    "longitude":  19.9265492,
    "start_date": "2019-01-01",
    "end_date":   "2024-12-31",
    "daily": [
        "temperature_2m_mean",
        "temperature_2m_min",
        "temperature_2m_max",
        "precipitation_sum",
        "wind_speed_10m_max",
        "wind_speed_10m_mean",
        "surface_pressure_mean",
        "wind_direction_10m_dominant",
        "relative_humidity_2m_mean",
        "snowfall_sum",
    ],
    "timezone": "Europe/Berlin",
}

WEATHER_COL_RENAME = {
    "time": "Date",
    "temperature_2m_mean": "temp_avg",
    "temperature_2m_min": "temp_min",
    "temperature_2m_max": "temp_max",
    "precipitation_sum": "rain_sum",
    "wind_speed_10m_max": "wind_max",
    "wind_speed_10m_mean": "wind_mean",
    "surface_pressure_mean": "pressure_avg",
    "wind_direction_10m_dominant": "wind_dir_dominant",
    "relative_humidity_2m_mean": "humidity_avg",
    "snowfall_sum": "snowfall_sum",
}

EU_PM10_DAILY_LIMIT = 50    
EU_PM10_ANNUAL_DAYS = 35 

HEATING_MONTHS   = [1, 2, 3, 10, 11, 12]
LAG_DAYS         = [1, 2, 7, 14]           
ROLLING_WINDOWS  = [3, 7, 14, 30]         
GAP_INTERP_LIMIT = 3                       

SARIMAX_EXOG = [
    "temp_avg",
    "wind_max",
    "is_heating_season",
    "is_calm_wind",
    "hdd_calm",
    "rain_3d_sum",
    "inversion_proxy",
]

PROPHET_REGRESSORS = [
    "temp_avg",
    "wind_max",
    "is_heating_season",
    "is_calm_wind",
    "hdd_calm",
    "rain_3d_sum",
    "inversion_proxy",
]

LGBM_FEATURES = [
    "month_sin", "month_cos",
    "dow_sin", "dow_cos",
    "doy_sin", "doy_cos",
    "is_heating_season", "is_weekend", "is_holiday",
    "temp_avg", "wind_max", "wind_mean",
    "pressure_avg", "humidity_avg", "snowfall_sum",
    "heating_degree_days", "hdd_7d",
    "wind_inverse", "wind_7d_mean",
    "rain_yesterday", "rain_3d_sum", "rain_7d_sum", "dry_spell_days",
    "is_frost", "is_calm_wind",
    "is_frost_calm", "is_heating_season_calm",
    "hdd_calm", "cold_dry_calm", "inversion_proxy",
    "lag_1d", "lag_2d", "lag_7d", "lag_14d",
    "rolling_mean_3d", "rolling_mean_7d", "rolling_mean_14d", "rolling_mean_30d",
    "rolling_std_7d",  "rolling_std_14d",
    "rolling_diff_7d",
    "aux_mean_lag1", "aux_max_lag1", "aux_spread_lag1",
]

REFIT_EVERY = 7  

LGBM_PARAMS = dict(
    objective="regression_l1", 
    metric="mae",
    n_estimators=3000,
    learning_rate=0.02,
    num_leaves=63,
    min_child_samples=20,
    feature_fraction=0.8,
    bagging_fraction=0.8,
    bagging_freq=1,
    reg_alpha=0.1,
    reg_lambda=1.0,
    verbosity=-1,
    random_state=42,
    n_jobs=-1,
)

LGBM_EARLY_STOPPING_ROUNDS = 100
LGBM_ES_FRACTION = 0.15   