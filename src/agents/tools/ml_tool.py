"""ML model wrapper as a typed tool for the agent graph."""
from src.ml.predict import predict_risk, predict_all_merchants
from src.ml.features import calculate_features


def run_risk_prediction(merchant_id: str) -> dict:
    result = predict_risk(merchant_id)
    features = calculate_features(merchant_id)

    return {
        "anomaly_score": result.anomaly_score,
        "risk_probability": result.risk_probability,
        "top_risk_factors": [
            {"feature": f, "importance": float(v)}
            for f, v in result.top_risk_factors
        ],
        "model_versions": result.model_versions,
        "raw_features": {
            "peer_volume_zscore": features.get("peer_volume_zscore", 0.0),
            "peer_amount_zscore": features.get("peer_amount_zscore", 0.0),
            "volume_growth_ratio": features.get("volume_growth_ratio", 0.0),
            "txn_count_24h": features.get("txn_count_24h", 0),
            "unique_customers_24h": features.get("unique_customers_24h", 0),
            "customer_txn_ratio": features.get("customer_txn_ratio", 0.0),
            "pct_card_present": features.get("pct_card_present", 1.0),
            "pct_international": features.get("pct_international", 0.0),
            "avg_ticket_24h": features.get("avg_ticket_24h", 0.0),
            "hour_entropy": features.get("hour_entropy", 0.0),
            "business_age_days": features.get("business_age_days", 0),
        },
    }


def run_batch_prediction(merchant_ids: list[str]) -> dict[str, float]:
    return predict_all_merchants(merchant_ids)
