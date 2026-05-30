"""Node 2: Runs ML models for risk scoring. Pure Python — no LLM."""
from src.agents.state import AgentState
from src.agents.tools.ml_tool import run_risk_prediction
from src.agents.observability import traced


@traced(name="anomaly_detector")
def anomaly_detector(state: AgentState) -> dict:
    merchant_id = state["merchant_id"]
    result = run_risk_prediction(merchant_id)

    risk_score = result["risk_probability"]
    anomaly_score = result["anomaly_score"]
    peer_zscore = result["raw_features"]["peer_volume_zscore"]
    features = result["raw_features"]

    trace_parts = [
        f"[AnomalyDetector] Risk probability: {risk_score:.3f}",
        f"  Anomaly score: {anomaly_score:.3f}",
        f"  Peer volume z-score: {peer_zscore:.1f}σ",
        f"  Volume growth ratio: {features['volume_growth_ratio']:.1f}x",
        f"  Unique customers 24h: {int(features['unique_customers_24h'])}",
        f"  Card present: {features['pct_card_present']:.0%}",
        f"  International: {features['pct_international']:.0%}",
        f"  Top factors: {', '.join(f['feature'] for f in result['top_risk_factors'][:3])}",
    ]

    return {
        "risk_score": risk_score,
        "anomaly_score": anomaly_score,
        "feature_importances": result["top_risk_factors"],
        "peer_zscore": peer_zscore,
        "model_versions": result["model_versions"],
        "reasoning_trace": ["\n".join(trace_parts)],
    }
