"""Agent state definition for the Risk Copilot LangGraph pipeline."""
import operator
from typing import Annotated, TypedDict


class AgentState(TypedDict):
    merchant_id: str
    merchant_info: dict
    transactions_summary: dict
    peer_stats: dict

    risk_score: float
    anomaly_score: float
    feature_importances: list
    peer_zscore: float

    context_findings: list
    kyb_status: dict

    decision: str
    confidence: float
    reason_codes: list
    explanation: str
    narrative_report: str
    counterfactual: str

    policy_context: str

    reasoning_trace: Annotated[list[str], operator.add]
    model_versions: dict


REASON_CODES = [
    "VOLUME_ANOMALY",
    "PEER_GROUP_OUTLIER",
    "TICKET_SIZE_ANOMALY",
    "CNP_RATIO_SPIKE",
    "FOREIGN_CARD_SPIKE",
    "LOW_CUSTOMER_DIVERSITY",
    "NEW_MERCHANT_HIGH_VOLUME",
    "MCC_MISMATCH",
    "LEGITIMATE_EVENT",
    "SEASONAL_PATTERN",
    "WITHIN_NORMAL_RANGE",
    "KYB_FLAG",
    "BUSINESS_AGE_CONCERN",
]
