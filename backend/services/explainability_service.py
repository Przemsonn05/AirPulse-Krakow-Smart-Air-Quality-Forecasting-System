"""
ExplainabilityService — SHAP-based feature attribution for LightGBM predictions.

Requires the shap package (pip install shap>=0.44).
Falls back to native LightGBM gain-importance when SHAP is unavailable.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class ExplainabilityService:
    """Computes per-prediction SHAP values for a fitted LightGBM model."""

    def __init__(self, lgbm_model, top_n: int = 10) -> None:
        self.model = lgbm_model
        self.top_n = top_n
        self._explainer = None
        self._shap_available = False
        self._init_explainer()

    def _init_explainer(self) -> None:
        if self.model is None:
            return
        try:
            import shap 
            import shap as _shap
            self._explainer = _shap.TreeExplainer(self.model)
            self._shap_available = True
            logger.info("SHAP TreeExplainer initialised")
        except ImportError:
            logger.warning("shap not installed — falling back to gain importance")
        except Exception as exc:
            logger.warning("SHAP initialisation failed: %s", exc)


    def explain(
        self, X: pd.DataFrame
    ) -> tuple[list[dict], Optional[float]]:
        """
        Compute feature contributions for a single-row feature DataFrame.

        Returns
        -------
        contributions : list of {feature, value, contribution} dicts
            Sorted by |contribution| descending, top_n only.
        base_value : float or None
            SHAP expected output value (Box-Cox scale).
        """
        if self.model is None:
            return self._fallback_contributions(X), None

        if self._shap_available and self._explainer is not None:
            return self._shap_contributions(X)
        else:
            return self._gain_contributions(X), None


    def _shap_contributions(
        self, X: pd.DataFrame
    ) -> tuple[list[dict], float]:
        import shap as _shap

        sv   = self._explainer.shap_values(X)
        base = float(self._explainer.expected_value)

        contribs = pd.Series(sv[0], index=X.columns)
        top = (
            contribs.abs()
            .sort_values(ascending=False)
            .head(self.top_n)
        )

        result = [
            {
                "feature":      feat,
                "value":        float(X[feat].iloc[0]),
                "contribution": float(contribs[feat]),
            }
            for feat in top.index
        ]
        return result, base

    def _gain_contributions(self, X: pd.DataFrame) -> list[dict]:
        """Use LightGBM gain importance as a proxy when SHAP is unavailable."""
        booster   = self.model.booster_
        gain_vals = booster.feature_importance(importance_type="gain")
        names     = self.model.feature_name_

        importance = pd.Series(gain_vals, index=names)
        top = importance.sort_values(ascending=False).head(self.top_n)

        scale = top.sum() or 1.0
        result = [
            {
                "feature":      feat,
                "value":        float(X[feat].iloc[0]) if feat in X.columns else 0.0,
                "contribution": float(importance[feat] / scale),
            }
            for feat in top.index
        ]
        return result

    @staticmethod
    def _fallback_contributions(X: pd.DataFrame) -> list[dict]:
        return [
            {"feature": col, "value": float(X[col].iloc[0]), "contribution": 0.0}
            for col in X.columns[:10]
        ]