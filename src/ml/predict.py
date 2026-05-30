"""Load trained models and predict risk scores."""
from pathlib import Path
from dataclasses import dataclass

import joblib
import numpy as np

from src.ml.features import FEATURE_NAMES, calculate_features

MODELS_DIR = Path(__file__).parent.parent.parent / "models"


@dataclass
class RiskModelResult:
    anomaly_score: float
    risk_probability: float
    top_risk_factors: list[tuple[str, float]]
    model_versions: dict


_iso_forest = None
_xgb_model = None


def _load_models():
    global _iso_forest, _xgb_model
    if _iso_forest is None:
        _iso_forest = joblib.load(MODELS_DIR / "isolation_forest.joblib")
    if _xgb_model is None:
        _xgb_model = joblib.load(MODELS_DIR / "xgboost_risk.joblib")
    return _iso_forest, _xgb_model


def predict_risk(merchant_id: str) -> RiskModelResult:
    iso_forest, xgb_model = _load_models()
    features = calculate_features(merchant_id)
    feature_array = np.array([[features[f] for f in FEATURE_NAMES]])

    if_raw = iso_forest.decision_function(feature_array)[0]
    anomaly_score = float(1.0 / (1.0 + np.exp(if_raw)))

    risk_prob = float(xgb_model.predict_proba(feature_array)[0][1])

    importances = dict(zip(FEATURE_NAMES, xgb_model.feature_importances_))
    weighted_factors = {
        feat: imp * abs(features[feat])
        for feat, imp in importances.items()
    }
    top_factors = sorted(weighted_factors.items(), key=lambda x: x[1], reverse=True)[:5]

    return RiskModelResult(
        anomaly_score=anomaly_score,
        risk_probability=risk_prob,
        top_risk_factors=top_factors,
        model_versions={"isolation_forest": "1.0", "xgboost": "1.0"},
    )


def predict_all_merchants(merchant_ids: list[str]) -> dict[str, float]:
    iso_forest, xgb_model = _load_models()
    scores = {}
    for mid in merchant_ids:
        features = calculate_features(mid)
        feature_array = np.array([[features[f] for f in FEATURE_NAMES]])
        risk_prob = float(xgb_model.predict_proba(feature_array)[0][1])
        scores[mid] = risk_prob
    return scores
