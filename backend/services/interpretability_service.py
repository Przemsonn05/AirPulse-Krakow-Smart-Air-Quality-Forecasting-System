"""
InterpretabilityService — rule-based NLG engine that produces human-readable
air-quality forecasts.

Simulates an LLM response using deterministic templates keyed on PM10 level,
dominant weather drivers, and current season.  Replace _generate_llm_response()
with a real Anthropic / OpenAI call when an API key is available.
"""

from __future__ import annotations

import math
from datetime import date



_SUMMARY_TEMPLATES = {
    "Good": (
        "Air quality in Kraków is expected to be {level} on {date}, with PM10 at "
        "{pm10:.1f} µg/m³ — well below the EU daily limit of 50 µg/m³. "
        "Favourable conditions ({drivers}) are keeping pollution low. "
        "No restrictions are required."
    ),
    "Moderate": (
        "Moderate air quality is forecast for Kraków on {date} "
        "(PM10 ≈ {pm10:.1f} µg/m³, EU limit: 50 µg/m³). "
        "Key contributing factors include {drivers}. "
        "Sensitive groups should consider reducing prolonged outdoor activity."
    ),
    "High": (
        "Elevated PM10 levels ({pm10:.1f} µg/m³) are predicted for Kraków on {date}, "
        "exceeding the EU daily limit of 50 µg/m³. "
        "The main drivers are {drivers}. "
        "People with respiratory or cardiovascular conditions should limit outdoor exposure."
    ),
    "Very High": (
        "A significant air-quality episode is forecast for Kraków on {date}: "
        "PM10 is expected to reach {pm10:.1f} µg/m³ — {times:.1f}× the EU daily limit. "
        "Primary causes: {drivers}. "
        "Health authorities recommend that all residents avoid strenuous outdoor activity."
    ),
}

_RECOMMENDATIONS = {
    "Good":     "Enjoy outdoor activities freely.",
    "Moderate": "Sensitive groups (asthmatics, elderly, children) may wish to reduce prolonged outdoor exertion.",
    "High":     "All residents should limit outdoor exercise. Consider wearing a face mask (FFP2/N95) if outdoors.",
    "Very High": "Stay indoors with windows closed. Use air purifiers if available. Follow official health advisories.",
}

_DRIVER_LABELS = {
    "lag_1d":               "high PM10 carry-over from yesterday",
    "lag_7d":               "persistent elevated PM10 this week",
    "is_heating_season":    "active heating season (coal/biomass combustion)",
    "is_calm_wind":         "very low wind speed (poor dispersion)",
    "hdd_calm":             "cold + calm conditions (heating + stagnation)",
    "inversion_proxy":      "likely temperature inversion trapping pollutants",
    "temp_avg":             "low temperature driving heating demand",
    "wind_max":             "weak maximum winds limiting dispersion",
    "rain_3d_sum":          "little recent rainfall (no wet deposition)",
    "dry_spell_days":       "extended dry period (dust accumulation)",
    "rolling_mean_7d":      "elevated 7-day PM10 background",
    "humidity_avg":         "high humidity enhancing aerosol formation",
    "pressure_avg":         "high atmospheric pressure suppressing vertical mixing",
}


class InterpretabilityService:
    """Generates structured natural language forecasts from model outputs."""

    def interpret(
        self,
        pm10: float,
        pm10_level: str,
        regime: str,
        feature_contributions: list[dict],
        forecast_date: date,
        weather: dict,
    ) -> dict:
        """
        Parameters
        ----------
        pm10              : point forecast in µg/m³
        pm10_level        : "Good" | "Moderate" | "High" | "Very High"
        regime            : "Clean" | "Moderate" | "Polluted"
        feature_contributions : list of {feature, value, contribution} dicts
        forecast_date     : date being forecast
        weather           : raw weather dict

        Returns
        -------
        dict with keys: summary, risk_level, key_drivers, recommendation
        """
        drivers     = self._top_driver_labels(feature_contributions, weather, pm10_level)
        drivers_str = self._drivers_sentence(drivers)
        times       = pm10 / 50.0

        template = _SUMMARY_TEMPLATES.get(pm10_level, _SUMMARY_TEMPLATES["Moderate"])
        summary  = template.format(
            level=pm10_level.lower(),
            date=forecast_date.strftime("%A, %d %B %Y"),
            pm10=pm10,
            drivers=drivers_str,
            times=times,
        )

        return {
            "summary":       summary,
            "risk_level":    pm10_level,
            "key_drivers":   drivers,
            "recommendation": _RECOMMENDATIONS.get(pm10_level, "Monitor official advisories."),
        }


    @staticmethod
    def _top_driver_labels(
        contributions: list[dict],
        weather: dict,
        pm10_level: str,
    ) -> list[str]:
        """Map top contributing features to human-readable labels."""
        labels = []

        for item in contributions[:6]:
            feat = item["feature"]
            if feat in _DRIVER_LABELS and item["contribution"] > 0:
                labels.append(_DRIVER_LABELS[feat])

        if weather.get("wind_mean", 5) <= 2 and "very low wind speed" not in " ".join(labels):
            labels.append("very low wind speed (poor dispersion)")
        if weather.get("temp_avg", 10) < 2 and "low temperature" not in " ".join(labels):
            labels.append("low temperature driving heating demand")
        if weather.get("rain_sum", 1) == 0 and "little recent rainfall" not in " ".join(labels):
            labels.append("no recent rainfall")

        return list(dict.fromkeys(labels))[:4] 

    @staticmethod
    def _drivers_sentence(drivers: list[str]) -> str:
        if not drivers:
            return "seasonal and meteorological conditions"
        if len(drivers) == 1:
            return drivers[0]
        return ", ".join(drivers[:-1]) + " and " + drivers[-1]