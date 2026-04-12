"""
Prepare supplementary model artefacts required by the FastAPI backend.

Run ONCE from the project root after the main training pipeline has completed:

    python scripts/prepare_api_artifacts.py

Saves to models/:
    lambda_bc.pkl         – Box-Cox lambda
    recent_history.pkl    – last 60 days of engineered features
    scaler.pkl            – StandardScaler fitted on SARIMAX exog (train split)
    kmeans_model.pkl      – KMeans(3) regime classifier
    metrics.pkl           – dict {model_name: {mae, rmse, smape, r2}}
    prophet_model.pkl     – properly saved Prophet model (overwrites broken file)
"""

from __future__ import annotations

import logging
import pickle
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

# ── Path setup ────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.config import (
    DATA_DIR, YEARS, STATIONS, TARGET, TRAIN_END, VAL_END,
    WEATHER_API_URL, WEATHER_PARAMS, WEATHER_COL_RENAME,
    AUX_STATIONS, HEATING_MONTHS, LAG_DAYS, ROLLING_WINDOWS,
    SARIMAX_EXOG, PROPHET_REGRESSORS, LGBM_FEATURES,
    GAP_INTERP_LIMIT, EU_PM10_DAILY_LIMIT,
)
from src.data_loading import load_pm10_raw, parse_pm10_stations, fetch_weather
from src.data_preprocessing import impute_gaps, merge_weather
from src.feature_engineering import build_features
from src.utils import date_split, safe_inv_boxcox

MODELS_DIR = ROOT / "models"
MODELS_DIR.mkdir(exist_ok=True)

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s | %(levelname)-8s | %(message)s")
log = logging.getLogger(__name__)


def smape(actual: np.ndarray, pred: np.ndarray) -> float:
    mask = np.isfinite(actual) & np.isfinite(pred) & (np.abs(actual) + np.abs(pred) > 0)
    return float(100 * np.mean(
        2 * np.abs(actual[mask] - pred[mask]) / (np.abs(actual[mask]) + np.abs(pred[mask]))
    ))


def main() -> None:
    log.info("Loading PM10 data …")
    raw  = load_pm10_raw(DATA_DIR, YEARS)
    pm10 = parse_pm10_stations(raw, STATIONS)

    log.info("Fetching weather data …")
    weather = fetch_weather(WEATHER_API_URL, WEATHER_PARAMS, WEATHER_COL_RENAME)

    pm10   = impute_gaps(pm10, STATIONS, limit=GAP_INTERP_LIMIT)
    merged = merge_weather(pm10, weather)

    log.info("Running feature engineering …")
    df_feat, lambda_bc = build_features(
        merged, TARGET, TRAIN_END, AUX_STATIONS,
        HEATING_MONTHS, LAG_DAYS, ROLLING_WINDOWS,
    )
    log.info("Box-Cox lambda = %.6f", lambda_bc)

    with open(MODELS_DIR / "lambda_bc.pkl", "wb") as fh:
        pickle.dump(lambda_bc, fh)
    log.info("Saved lambda_bc.pkl")

    recent = df_feat.tail(60).copy()
    keep_cols = ["PM10_transformed", TARGET] + [
        c for c in df_feat.columns
        if any(c.startswith(p) for p in ["temp_", "wind_", "rain_", "pressure_", "humidity_", "MpKrak"])
    ]
    recent = recent[[c for c in keep_cols if c in recent.columns]]
    with open(MODELS_DIR / "recent_history.pkl", "wb") as fh:
        pickle.dump(recent, fh)
    log.info("Saved recent_history.pkl (%d rows, %d cols)", *recent.shape)

    train, val, test = date_split(df_feat, TRAIN_END, VAL_END)

    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    exog_train = train[SARIMAX_EXOG].ffill().bfill()
    scaler.fit(exog_train)
    with open(MODELS_DIR / "scaler.pkl", "wb") as fh:
        pickle.dump(scaler, fh)
    log.info("Saved scaler.pkl")

    from sklearn.cluster import KMeans
    km_cols = ["temp_avg", "wind_max", "is_heating_season", "lag_1d"]
    X_km = df_feat[[c for c in km_cols if c in df_feat.columns]].dropna()
    km = KMeans(n_clusters=3, random_state=42, n_init=10)
    km.fit(X_km)
    with open(MODELS_DIR / "kmeans_model.pkl", "wb") as fh:
        pickle.dump(km, fh)
    log.info("Saved kmeans_model.pkl (inertia=%.1f)", km.inertia_)

    metrics: dict[str, dict] = {}
    bc_col  = "PM10_transformed"

    lgbm_path = MODELS_DIR / "lgbm_model.joblib"
    if lgbm_path.exists():
        lgbm = joblib.load(lgbm_path)
        feat_cols = [c for c in lgbm.feature_name_ if c in val.columns]
        X_val = val[feat_cols].ffill().fillna(0.0)
        y_val = val[bc_col].values
        preds_bc = lgbm.predict(X_val)
        actual   = safe_inv_boxcox(y_val, lambda_bc)
        pred_pm10= safe_inv_boxcox(preds_bc, lambda_bc)
        from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
        metrics["LightGBM"] = {
            "mae":   round(float(mean_absolute_error(actual, pred_pm10)), 4),
            "rmse":  round(float(np.sqrt(mean_squared_error(actual, pred_pm10))), 4),
            "smape": round(smape(actual, pred_pm10), 4),
            "r2":    round(float(r2_score(actual, pred_pm10)), 4),
        }
        log.info("LightGBM val metrics: %s", metrics["LightGBM"])

    actual_all = safe_inv_boxcox(val[bc_col].values, lambda_bc)
    naive_preds = np.roll(actual_all, 1); naive_preds[0] = actual_all[0]
    metrics["Naïve"] = {
        "mae":   round(float(np.mean(np.abs(actual_all - naive_preds))), 4),
        "rmse":  round(float(np.sqrt(np.mean((actual_all - naive_preds)**2))), 4),
        "smape": round(smape(actual_all, naive_preds), 4),
        "r2":    None,
    }

    with open(MODELS_DIR / "metrics.pkl", "wb") as fh:
        pickle.dump(metrics, fh)
    log.info("Saved metrics.pkl: %s", list(metrics.keys()))

    try:
        from src.models import train_predict_prophet
        from prophet import Prophet
        log.info("Training Prophet model …")
        _, _ = train_predict_prophet(
            train, val,
            target_col=bc_col,
            regressors=PROPHET_REGRESSORS,
        )
        from src.models import _make_prophet_df
        train_p = train.copy()
        train_p["rolling_bc_7d"] = (
            train_p[bc_col].shift(1).rolling(7, min_periods=3).mean().ffill()
        )
        all_regressors = PROPHET_REGRESSORS + ["rolling_bc_7d"]
        required = all_regressors + [bc_col]
        train_clean = train_p.dropna(subset=[c for c in required if c in train_p.columns])

        m = Prophet(
            yearly_seasonality=True, weekly_seasonality=True,
            daily_seasonality=False, seasonality_mode="multiplicative",
            changepoint_prior_scale=0.05,
        )
        m.add_country_holidays(country_name="PL")
        for col in all_regressors:
            if col in train_clean.columns:
                m.add_regressor(col)
        m.fit(_make_prophet_df(train_clean, all_regressors, bc_col))

        with open(MODELS_DIR / "prophet_model.pkl", "wb") as fh:
            pickle.dump({
                "model": m,
                "regressors": all_regressors,
                "lambda_bc": lambda_bc,
            }, fh)
        log.info("Saved prophet_model.pkl (proper Prophet object)")
    except Exception as exc:
        log.warning("Prophet save skipped: %s", exc)

    log.info("All artefacts saved to %s", MODELS_DIR)


if __name__ == "__main__":
    main()