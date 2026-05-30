"""LangGraph agent graph for the Risk Copilot pipeline."""
from langgraph.graph import StateGraph, END

from src.agents.state import AgentState
from src.agents.nodes.data_collector import data_collector
from src.agents.nodes.anomaly_detector import anomaly_detector
from src.agents.nodes.context_researcher import context_researcher
from src.agents.nodes.kyb_verifier import kyb_verifier
from src.agents.nodes.policy_retriever import policy_retriever
from src.agents.nodes.decision_agent import decision_agent
from src.agents.nodes.narrative_generator import narrative_generator


def auto_approve(state: AgentState) -> dict:
    return {
        "decision": "APPROVE",
        "confidence": 0.95,
        "reason_codes": ["WITHIN_NORMAL_RANGE"],
        "explanation": (
            f"Merchant {state.get('merchant_info', {}).get('name', 'Unknown')} shows transaction patterns "
            f"within expected range for its MCC peer group. Risk score {state.get('risk_score', 0):.3f} "
            f"is below investigation threshold. Auto-approved with continued monitoring."
        ),
        "counterfactual": "If the risk score exceeded 0.5 or peer z-score exceeded 3.0σ, a full investigation would be triggered.",
        "narrative_report": (
            f"**Auto-Approved** — Merchant activity within normal parameters. "
            f"Risk: {state.get('risk_score', 0):.3f}, Peer z-score: {state.get('peer_zscore', 0):.1f}σ. "
            f"No investigation required."
        ),
        "context_findings": [],
        "kyb_status": {"overall": "SKIPPED", "reason": "Low risk — KYB not required"},
        "reasoning_trace": [
            f"[AutoApprove] Risk {state.get('risk_score', 0):.3f} < 0.5 and peer z-score "
            f"{state.get('peer_zscore', 0):.1f} < 3.0. Auto-approved — LLM investigation skipped."
        ],
    }


def route_after_anomaly(state: AgentState) -> str:
    risk = state.get("risk_score", 0)
    peer_z = state.get("peer_zscore", 0)
    if risk < 0.5 and abs(peer_z) < 3.0:
        return "auto_approve"
    return "investigate"


builder = StateGraph(AgentState)

builder.add_node("data_collector", data_collector)
builder.add_node("anomaly_detector", anomaly_detector)
builder.add_node("auto_approve", auto_approve)
builder.add_node("context_researcher", context_researcher)
builder.add_node("kyb_verifier", kyb_verifier)
builder.add_node("policy_retriever", policy_retriever)
builder.add_node("decision_agent", decision_agent)
builder.add_node("narrative_generator", narrative_generator)

builder.set_entry_point("data_collector")
builder.add_edge("data_collector", "anomaly_detector")

builder.add_conditional_edges(
    "anomaly_detector",
    route_after_anomaly,
    {"auto_approve": "auto_approve", "investigate": "context_researcher"},
)

builder.add_edge("auto_approve", END)
builder.add_edge("context_researcher", "kyb_verifier")
builder.add_edge("kyb_verifier", "policy_retriever")
builder.add_edge("policy_retriever", "decision_agent")
builder.add_edge("decision_agent", "narrative_generator")
builder.add_edge("narrative_generator", END)

graph = builder.compile()
