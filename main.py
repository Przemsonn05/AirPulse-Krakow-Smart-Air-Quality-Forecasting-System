"""
main.py — end-to-end PM10 air-quality forecasting pipeline.

Run from the project root:
    python main.py                        # all models, with EDA
    python main.py --skip-eda            # skip plot generation (faster)
    python main.py --models lgbm         # train LightGBM only
    python main.py --models lgbm --optuna  # LightGBM with Optuna tuning

Each stage is a thin wrapper around the corresponding src/ module.
"""

import argparse
import sys

import pandas as pd

from src import config
from src.data_loading import fetch_weather, load_pm10_raw, parse_pm10_stations
from src.data_preprocessing import impute_gaps, merge_weather
from src.eda import run_full_eda
from src.evaluation import (
    build_metrics_table,
    compute_metrics,
    plot_feature_importance,
    plot_forecast_comparison,
    plot_metrics_comparison,
)
from src.feature_engineering import build_features
from src.models import (
    train_predict_arima,
    train_predict_lgbm,
    train_predict_prophet,
    train_predict_sarimax,
)
from src.utils import date_split, get_logger, safe_inv_boxcox

logger = get_logger(__name__)

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="PM10 air-quality forecasting pipeline")
    parser.add_argument(
        "--skip-eda", action="store_true",
        help="Skip EDA plot generation (faster during repeated runs)",
    )
    parser.add_argument(
        "--models", nargs="+",
        choices=["arima", "sarimax", "prophet", "lgbm"],
        default=["arima", "sarimax", "prophet", "lgbm"],
        help="Which models to train (default: all four)",
    )
    parser.add_argument(
        "--optuna", action="store_true",
        help="Run Optuna hyperparameter search for LightGBM (slow)",
    )
    return parser.parse_args()

def run_sanity_checks(
    raw: pd.DataFrame,
    merged: pd.DataFrame,
    df_features: pd.DataFrame,
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
) -> None:
    """Lightweight assertions to catch data pipeline issues early."""
    logger.info("--- Sanity checks ---")

    assert len(raw) > 0, "Raw PM10 data is empty"

    for col in config.STATIONS:
        assert col in merged.columns, f"Missing station column: {col}"
    for col in ["temp_avg", "wind_max", "rain_sum"]:
        assert col in merged.columns, f"Missing weather column: {col}"

    assert merged.index.is_unique, "Duplicate dates in merged DataFrame"
    assert merged.index.freq == "D", "Merged index is not daily frequency"

    nan_cols = [c for c in df_features.columns if df_features[c].isna().all()]
    assert not nan_cols, f"Fully-NaN feature columns: {nan_cols}"

    assert len(train) > 0 and len(val) > 0 and len(test) > 0, \
        "One or more splits is empty"
    assert train.index.max() < val.index.min(), "Train/val overlap detected"
    assert val.index.max() < test.index.min(), "Val/test overlap detected"

    for split_name, split_df in [("train", train), ("val", val), ("test", test)]:
        assert config.PM10_BC_COL in split_df.columns, \
            f"'{config.PM10_BC_COL}' missing from {split_name}"
        null_frac = split_df[config.PM10_BC_COL].isna().mean()
        assert null_frac < 0.05, \
            f">{null_frac*100:.0f}% NaN in {config.PM10_BC_COL} for {split_name}"

    assert "lag_1d" in df_features.columns, "lag_1d feature missing"
    sample = df_features[[config.TARGET, config.PM10_BC_COL, "lag_1d"]].dropna().head(200)
    expected = df_features[config.PM10_BC_COL].shift(1).dropna()
    actual   = df_features["lag_1d"].dropna()
    diff = (expected - actual).abs().max()
    assert diff < 1e-6, f"lag_1d does not equal shift(1) of PM10_transformed (max diff={diff})"

    logger.info("All sanity checks passed.")

def stage_load() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load PM10 Excel files and weather data from Open-Meteo."""
    logger.info("=== STAGE 1: Data Loading ===")

    raw_pm10 = load_pm10_raw(config.DATA_DIR, config.YEARS)
    pm10     = parse_pm10_stations(raw_pm10, config.STATIONS)

    weather = fetch_weather(
        config.WEATHER_API_URL,
        config.WEATHER_PARAMS,
        config.WEATHER_COL_RENAME,
    )
    return raw_pm10, pm10, weather


def stage_preprocess(
    pm10: pd.DataFrame,
    weather: pd.DataFrame,
) -> pd.DataFrame:
    """Impute gaps, flag long gaps, and merge PM10 with weather."""
    logger.info("=== STAGE 2: Preprocessing ===")

    pm10   = impute_gaps(pm10, config.STATIONS, limit=config.GAP_INTERP_LIMIT)
    merged = merge_weather(pm10, weather)
    return merged


def stage_eda(df: pd.DataFrame) -> None:
    """Generate and save all EDA visualisations."""
    logger.info("=== STAGE 3: EDA ===")
    run_full_eda(
        df,
        target_col=config.TARGET,
        images_dir=config.IMAGES_DIR,
        all_stations=config.STATIONS,
        eu_limit=config.EU_PM10_DAILY_LIMIT,
    )


def stage_features(df: pd.DataFrame) -> tuple[pd.DataFrame, float]:
    """Build all features; Box-Cox lambda fitted on training split only."""
    logger.info("=== STAGE 4: Feature Engineering ===")
    return build_features(
        df,
        target_col=config.TARGET,
        train_end=config.TRAIN_END,
        aux_stations=config.AUX_STATIONS,
        heating_months=config.HEATING_MONTHS,
        lag_days=config.LAG_DAYS,
        rolling_windows=config.ROLLING_WINDOWS,
    )


def stage_split(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Chronological date-based train / val / test split."""
    logger.info("=== STAGE 5: Train/Val/Test Split ===")
    return date_split(df, config.TRAIN_END, config.VAL_END)


def stage_train(
    train: pd.DataFrame,
    val: pd.DataFrame,
    lambda_bc: float,
    models_to_run: list[str],
    use_optuna: bool = False,
) -> tuple[dict, dict, object]:
    """Train each requested model; return predictions, metrics, and LGBM model."""
    logger.info("=== STAGE 6: Model Training ===")

    predictions: dict = {}
    results: dict = {}
    lgbm_model = None

    if "arima" in models_to_run:
        logger.info("Training ARIMA …")
        preds, _, _ = train_predict_arima(
            train, val,
            target_col=config.PM10_BC_COL,
            refit_every=config.REFIT_EVERY,
        )
        predictions["ARIMA"] = preds
        results["ARIMA"]     = compute_metrics(
            val[config.PM10_BC_COL].values, preds, lambda_bc, "ARIMA",
            eu_limit=config.EU_PM10_DAILY_LIMIT,
        )

    if "sarimax" in models_to_run:
        logger.info("Training SARIMAX …")
        preds = train_predict_sarimax(
            train, val,
            target_col=config.PM10_BC_COL,
            exog_cols=config.SARIMAX_EXOG,
            refit_every=config.REFIT_EVERY,
        )
        predictions["SARIMAX"] = preds
        results["SARIMAX"]     = compute_metrics(
            val[config.PM10_BC_COL].values, preds, lambda_bc, "SARIMAX",
            eu_limit=config.EU_PM10_DAILY_LIMIT,
        )

    if "prophet" in models_to_run:
        logger.info("Training Prophet …")
        preds = train_predict_prophet(
            train, val,
            target_col=config.PM10_BC_COL,
            regressors=config.PROPHET_REGRESSORS,
        )
        predictions["Prophet"] = preds
        results["Prophet"]     = compute_metrics(
            val[config.PM10_BC_COL].values, preds, lambda_bc, "Prophet",
            eu_limit=config.EU_PM10_DAILY_LIMIT,
        )

    if "lgbm" in models_to_run:
        logger.info("Training LightGBM (optuna=%s) …", use_optuna)
        preds, lgbm_model = train_predict_lgbm(
            train, val,
            target_col=config.PM10_BC_COL,
            feature_cols=config.LGBM_FEATURES,
            params=config.LGBM_PARAMS,
            early_stopping_rounds=config.LGBM_EARLY_STOPPING_ROUNDS,
            es_fraction=config.LGBM_ES_FRACTION,
            use_optuna=use_optuna,
        )
        predictions["LightGBM"] = preds
        results["LightGBM"]     = compute_metrics(
            val[config.PM10_BC_COL].values, preds, lambda_bc, "LightGBM",
            eu_limit=config.EU_PM10_DAILY_LIMIT,
        )

    return predictions, results, lgbm_model


def stage_evaluate(
    val: pd.DataFrame,
    predictions: dict,
    results: dict,
    lambda_bc: float,
    lgbm_model,
) -> None:
    """Print the metric table and save all comparison visualisations."""
    logger.info("=== STAGE 7: Evaluation ===")

    table = build_metrics_table(results)
    print("\n" + "=" * 60)
    print("  Validation Metrics (2023)")
    print("=" * 60)
    print(table.to_string())
    print("=" * 60 + "\n")

    plot_metrics_comparison(table, config.IMAGES_DIR)
    plot_forecast_comparison(
        val, predictions,
        target_col=config.PM10_BC_COL,
        lambda_bc=lambda_bc,
        images_dir=config.IMAGES_DIR,
    )

    if lgbm_model is not None:
        plot_feature_importance(lgbm_model, config.IMAGES_DIR)

    logger.info("Evaluation complete — plots saved to %s", config.IMAGES_DIR)

def main() -> None:
    args = parse_args()

    raw_pm10, pm10, weather = stage_load()

    df_merged = stage_preprocess(pm10, weather)

    if not args.skip_eda:
        stage_eda(df_merged)
    else:
        logger.info("EDA skipped (--skip-eda)")

    df_features, lambda_bc = stage_features(df_merged)

    train, val, test = stage_split(df_features)

    run_sanity_checks(raw_pm10, df_merged, df_features, train, val, test)

    predictions, results, lgbm_model = stage_train(
        train, val, lambda_bc,
        models_to_run=args.models,
        use_optuna=args.optuna,
    )

    stage_evaluate(val, predictions, results, lambda_bc, lgbm_model)

    if results:
        logger.info("=== STAGE 8: Test Set Sanity (2024) ===")
        first_model = next(iter(predictions))
        first_preds = predictions[first_model]
        test_actual  = safe_inv_boxcox(test[config.PM10_BC_COL].dropna().values, lambda_bc)
        logger.info(
            "Test set has %d rows.  First model (%s) val MAE=%.2f  "
            "— retrain on train+val before reporting test metrics.",
            len(test), first_model, results[first_model]["MAE"],
        )

    logger.info("Pipeline complete.")

if __name__ == "__main__":
    main()