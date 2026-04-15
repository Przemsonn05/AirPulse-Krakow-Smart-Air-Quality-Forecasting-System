"""
Model evaluation: metric computation, exceedance classification,
result comparison tables, and comparison plots.

All regression metrics are computed in the original µg/m³ scale after
inverting the Box-Cox transformation.  SMAPE (Symmetric Mean Absolute
Percentage Error) is used instead of MAPE for robustness: it is bounded
[0, 200%] and does not blow up when actual values are near zero.
"""

from pathlib import Path

import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    precision_score,
    recall_score,
    f1_score,
)

from src.utils import get_logger, safe_inv_boxcox, save_figure, set_plot_style

logger = get_logger(__name__)

def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    lambda_bc: float,
    label: str = "",
    eu_limit: float = 50.0,
) -> dict[str, float]:
    """Compute regression + exceedance-classification metrics.

    All metrics are computed after inverting the Box-Cox transform so that
    numbers are directly interpretable in µg/m³.  SMAPE (Symmetric Mean
    Absolute Percentage Error) is used as the percentage metric for
    consistency with the API serving layer.

    Parameters
    ----------
    y_true:
        True Box-Cox-transformed values.
    y_pred:
        Predicted Box-Cox-transformed values.
    lambda_bc:
        Lambda used during feature engineering (for back-transformation).
    label:
        Human-readable model name (only used for logging).
    eu_limit:
        Threshold in µg/m³ used for exceedance classification metrics
        (default 50 µg/m³ — EU daily limit).

    Returns
    -------
    dict
        Keys: ``R2``, ``MAE``, ``RMSE``, ``SMAPE``,
              ``exc_precision``, ``exc_recall``, ``exc_f1``.
    """
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    actual    = safe_inv_boxcox(y_true[mask], lambda_bc)
    predicted = safe_inv_boxcox(y_pred[mask], lambda_bc)

    r2   = r2_score(actual, predicted)
    mae  = mean_absolute_error(actual, predicted)
    rmse = np.sqrt(mean_squared_error(actual, predicted))

    denom = np.abs(actual) + np.abs(predicted)
    valid = denom > 0
    smape = (
        float(np.mean(2 * np.abs(actual[valid] - predicted[valid]) / denom[valid]) * 100)
        if valid.sum() > 0 else np.nan
    )

    y_exc_true = (actual    >= eu_limit).astype(int)
    y_exc_pred = (predicted >= eu_limit).astype(int)

    if y_exc_true.sum() > 0:
        exc_precision = precision_score(y_exc_true, y_exc_pred, zero_division=0)
        exc_recall  = recall_score(y_exc_true, y_exc_pred, zero_division=0)
        exc_f1 = f1_score(y_exc_true, y_exc_pred, zero_division=0)
    else:
        exc_precision = exc_recall = exc_f1 = np.nan

    metrics = {
        "R2": round(r2, 4),
        "MAE": round(mae, 4),
        "RMSE": round(rmse, 4),
        "SMAPE": round(float(smape), 4),
        "exc_precision": round(exc_precision, 4),
        "exc_recall": round(exc_recall, 4),
        "exc_f1": round(exc_f1, 4),
    }

    logger.info(
        "%-30s  R²=%.4f  MAE=%.2f  RMSE=%.2f  SMAPE=%.1f%%  "
        "exc_recall=%.2f  exc_precision=%.2f",
        label, r2, mae, rmse, float(smape),
        exc_recall, exc_precision,
    )
    return metrics

def build_metrics_table(
    results: dict[str, dict[str, float]],
) -> pd.DataFrame:
    """Assemble a tidy DataFrame from a dict of per-model metric dicts.

    Parameters
    ----------
    results:
        ``{model_name: metrics_dict}`` — output of :func:`compute_metrics`.

    Returns
    -------
    pd.DataFrame
        One row per model, sorted by RMSE ascending, with columns
        ``R2 / MAE / RMSE / SMAPE / exc_precision / exc_recall / exc_f1``.
    """
    table = pd.DataFrame(results).T
    cols  = [c for c in ["R2", "MAE", "RMSE", "SMAPE",
                          "exc_precision", "exc_recall", "exc_f1"]
             if c in table.columns]
    return table[cols].sort_values("RMSE")

def plot_metrics_comparison(
    metrics_table: pd.DataFrame,
    images_dir: Path,
    filename: str = "model_metrics_comparison.png",
) -> None:
    """Three-panel bar chart: MAE, RMSE, SMAPE across all models.

    Parameters
    ----------
    metrics_table:
        Output of :func:`build_metrics_table`.
    images_dir:
        Directory where the PNG is saved.
    filename:
        Output filename (default ``model_metrics_comparison.png``).
    """
    set_plot_style()

    COLORS = {
        "ARIMA":  "#6396E7",
        "SARIMAX": "#64748B",
        "Prophet": "#66D1B6",
        "LightGBM": "#4B9965",
    }
    model_names = list(metrics_table.index)
    bar_colors  = [COLORS.get(m, "#999999") for m in model_names]

    fig, axes = plt.subplots(1, 3, figsize=(15, 6))

    for ax, metric in zip(axes, ["MAE", "RMSE", "SMAPE"]):
        vals     = metrics_table[metric].values.astype(float)
        best_idx = int(np.nanargmin(vals))

        bars = ax.bar(model_names, vals, color=bar_colors, alpha=0.9, width=0.5)

        bars[best_idx].set_edgecolor("black")
        bars[best_idx].set_linewidth(2)

        for bar, v in zip(bars, vals):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + vals.max() * 0.01,
                f"{v:.2f}",
                ha="center", va="bottom", fontsize=10,
            )

        ax.set_title(metric, fontweight="bold")
        ax.set_xlabel("")
        ax.set_ylabel(metric)
        ax.tick_params(axis="x", rotation=20)
        ax.grid(axis="y", alpha=0.3)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.suptitle("Model Comparison — Metrics (lower is better)",
                 fontweight="bold", y=1.02)
    plt.tight_layout()

    save_figure(fig, Path(images_dir) / filename)
    logger.info("Saved %s", filename)

def plot_forecast_comparison(
    val: pd.DataFrame,
    predictions: dict[str, np.ndarray],
    target_col: str,
    lambda_bc: float,
    images_dir: Path,
    eu_limit: float = 50,
    filename: str = "model_comparison_forecast.png",
) -> None:
    """Four-panel stacked line chart: actual vs each model's predictions.

    Each panel shows one model with the forecast/actual gap filled in the
    model's colour to make error magnitude visually immediate.

    Parameters
    ----------
    val:
        Validation / test DataFrame with a ``DatetimeIndex``.
    predictions:
        ``{model_name: np.ndarray}`` of Box-Cox-scale predictions.
    target_col:
        Box-Cox-transformed target column in ``val``.
    lambda_bc:
        Back-transformation lambda.
    images_dir:
        Directory where the PNG is saved.
    eu_limit:
        Horizontal reference line.
    filename:
        Output filename.
    """
    set_plot_style()

    COLORS = {
        "ARIMA":    "tomato",
        "SARIMAX":  "darkorange",
        "Prophet":  "seagreen",
        "LightGBM": "mediumpurple",
    }

    actual = safe_inv_boxcox(val[target_col].values, lambda_bc)
    dates  = val.index

    n_models = len(predictions)
    fig, axes = plt.subplots(n_models, 1, figsize=(14, 4 * n_models), sharex=False)
    if n_models == 1:
        axes = [axes]

    fig.suptitle(
        "Model Comparison — PM10 Forecasts vs Actual",
        fontsize=15, fontweight="bold", y=1.01,
    )

    for ax, (name, preds) in zip(axes, predictions.items()):
        color = COLORS.get(name, "steelblue")
        back  = safe_inv_boxcox(preds, lambda_bc)

        ax.plot(dates, actual, color="steelblue", lw=1.2,
                label="Actual PM10", alpha=0.85)
        ax.plot(dates, back,   color=color, lw=1.2,
                label=f"{name} Forecast", alpha=0.85)
        ax.fill_between(dates, actual, back, alpha=0.15, color=color)
        ax.axhline(eu_limit, color="gray", ls="--", lw=0.8,
                   label=f"EU Limit ({eu_limit} µg/m³)")
        ax.set_title(name, fontweight="bold")
        ax.set_ylabel("PM10 [µg/m³]")
        ax.legend(loc="upper right", fontsize=9)
        ax.grid(True, alpha=0.2)

    axes[-1].set_xlabel("Date")
    plt.tight_layout()

    save_figure(fig, Path(images_dir) / filename)
    logger.info("Saved %s", filename)

def plot_feature_importance(
    model: lgb.LGBMRegressor,
    images_dir: Path,
    top_n: int = 15,
    filename: str = "lgbm_feature_importance.png",
) -> None:
    """Side-by-side bar charts for gain and split feature importances.

    Parameters
    ----------
    model:
        A fitted ``lgb.LGBMRegressor``.
    images_dir:
        Directory where the PNG is saved.
    top_n:
        Number of top features to display.
    filename:
        Output filename.
    """
    set_plot_style()

    booster = model.booster_
    gain_vals = booster.feature_importance(importance_type="gain")
    split_vals = booster.feature_importance(importance_type="split")
    names  = model.feature_name_

    gain_df = (pd.DataFrame({"feature": names, "importance": gain_vals})
                .sort_values("importance", ascending=False)
                .head(top_n)
                .sort_values("importance"))
    split_df = (pd.DataFrame({"feature": names, "importance": split_vals})
                .sort_values("importance", ascending=False)
                .head(top_n)
                .sort_values("importance"))

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    for ax, df_imp, title in zip(
        axes,
        [gain_df, split_df],
        [f"Top {top_n} Features — Gain", f"Top {top_n} Features — Split Count"],
    ):
        ax.barh(df_imp["feature"], df_imp["importance"],
                color="#2563EB", alpha=0.85)
        ax.set_title(title, fontweight="bold")
        ax.set_xlabel("Importance")
        ax.grid(axis="x", alpha=0.3)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.suptitle("LightGBM Feature Importance", fontweight="bold", y=1.02)
    plt.tight_layout()

    save_figure(fig, Path(images_dir) / filename)
    logger.info("Saved %s", filename)