"""Node 5: Synthesizes all evidence and produces a typed APPROVE/REVIEW/BLOCK decision. Uses LLM."""
import os
import json

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

from src.agents.state import AgentState, REASON_CODES
from src.agents.observability import traced

load_dotenv()

GEMINI_MODEL = "gemini-2.5-flash-lite"

RISK_POLICY = """RISK POLICY — Decision Criteria:

BLOCK (high confidence of fraud/laundering/violation):
- Risk score > 0.8 AND no legitimate explanation found
- Evidence of transaction laundering (few customers, many txns, CNP spike)
- KYB flags: sanctions hit, PEP match, business < 30 days with extreme volume
- Foreign card percentage > 40% with no business justification

REVIEW (anomalous but inconclusive — needs human analyst):
- Risk score 0.5-0.8 without clear explanation
- Mixed signals (some legitimate factors, some suspicious)
- New merchant (< 180 days) with unusual but not extreme patterns
- KYB flags present but not critical

APPROVE (anomaly explained — auto-clear with monitoring):
- Legitimate external event explains the spike (festival, holiday, local event)
- Customer diversity is proportional to volume increase
- All card-present, low/zero international, normal ticket sizes
- Business has established history (> 180 days)
- Peer comparison within 2σ OR explained by known event

AVAILABLE REASON CODES (use ONLY from this list):
""" + "\n".join(f"- {code}" for code in REASON_CODES)


def _rule_based_decision(state: AgentState) -> dict:
    risk = state.get("risk_score", 0)
    context = state.get("context_findings", [])
    kyb = state.get("kyb_status", {})

    has_explanation = any(
        f.get("classification") == "EXPLAINS_ANOMALY" for f in context
    )

    if risk > 0.8 and not has_explanation and kyb.get("overall") != "PASS":
        decision, confidence = "BLOCK", 0.85
        codes = ["VOLUME_ANOMALY", "PEER_GROUP_OUTLIER"]
    elif risk > 0.8 and has_explanation:
        decision, confidence = "APPROVE", 0.80
        codes = ["LEGITIMATE_EVENT", "VOLUME_ANOMALY"]
    elif risk > 0.5:
        decision, confidence = "REVIEW", 0.70
        codes = ["VOLUME_ANOMALY"]
    else:
        decision, confidence = "APPROVE", 0.95
        codes = ["WITHIN_NORMAL_RANGE"]

    explanation = (
        f"Risk score {risk:.2f}. "
        f"{'Legitimate event found explaining anomaly. ' if has_explanation else 'No external explanation found. '}"
        f"KYB status: {kyb.get('overall', 'N/A')}."
    )
    counterfactual = (
        "If no legitimate event were found and customer diversity were low, this would be escalated to BLOCK."
        if has_explanation
        else "If a legitimate local event were found explaining the volume spike, this could be auto-approved."
    )

    return {
        "decision": decision,
        "confidence": confidence,
        "reason_codes": codes,
        "explanation": explanation,
        "counterfactual": counterfactual,
        "reasoning_trace": [f"[DecisionAgent] Rule-based fallback: {decision} (confidence {confidence:.0%}). Codes: {codes}"],
    }


@traced(name="decision_agent")
def decision_agent(state: AgentState) -> dict:
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        return _rule_based_decision(state)

    merchant = state.get("merchant_info", {})
    context_findings = state.get("context_findings", [])
    kyb = state.get("kyb_status", {})
    txn_summary = state.get("transactions_summary", {})

    context_text = "\n".join(
        f"  - [{f.get('classification', 'UNKNOWN')}] {f.get('finding', 'N/A')} (source: {f.get('source', 'N/A')})"
        for f in context_findings
    ) or "  No context findings available."

    feature_text = "\n".join(
        f"  - {f['feature']}: importance {f['importance']:.4f}"
        for f in state.get("feature_importances", [])[:5]
    )

    policy_context = state.get("policy_context", "")
    policy_section = (
        f"\n\nRETRIEVED POLICY CONTEXT (from internal policy documents):\n{policy_context}"
        if policy_context
        else ""
    )

    prompt = f"""{RISK_POLICY}{policy_section}

INVESTIGATION EVIDENCE:

MERCHANT: {merchant.get('name', 'Unknown')} | MCC: {merchant.get('mcc_code', 'N/A')} ({merchant.get('mcc_description', '')}) | City: {merchant.get('city', 'N/A')}, {merchant.get('state', '')}
Registration: {merchant.get('registered_date', 'N/A')} | Risk Tier: {merchant.get('risk_tier', 'N/A')}

TRANSACTION SUMMARY (last 24h):
  - Transactions: {txn_summary.get('txn_count_24h', 0)}
  - Avg amount: ${txn_summary.get('avg_amount_24h', 0):.2f}
  - Total amount: ${txn_summary.get('total_amount_24h', 0):.2f}
  - Unique customers: {txn_summary.get('unique_customers_24h', 0)}
  - Card present: {txn_summary.get('pct_card_present', 0):.0%}
  - International: {txn_summary.get('pct_international', 0):.0%}

ML RISK ASSESSMENT:
  - Risk probability: {state.get('risk_score', 0):.3f}
  - Anomaly score: {state.get('anomaly_score', 0):.3f}
  - Peer volume z-score: {state.get('peer_zscore', 0):.1f}σ
  - Top risk factors:
{feature_text}

CONTEXTUAL INVESTIGATION:
{context_text}

KYB VERIFICATION:
  - Business age: {kyb.get('business_age_days', 'N/A')} days
  - Sanctions: {kyb.get('sanctions_clear', 'N/A')}
  - PEP: {kyb.get('pep_clear', 'N/A')}
  - Overall: {kyb.get('overall', 'N/A')}
  - Flags: {kyb.get('flags', [])}

TASK: Based on ALL evidence above and the risk policy, provide your decision.

Respond in JSON format ONLY:
{{
    "decision": "APPROVE" or "REVIEW" or "BLOCK",
    "confidence": 0.0 to 1.0,
    "reason_codes": ["CODE1", "CODE2"],
    "explanation": "2-3 sentence explanation of your decision",
    "counterfactual": "What specific change in evidence would flip your decision?"
}}"""

    try:
        llm = ChatGoogleGenerativeAI(model=GEMINI_MODEL, google_api_key=api_key, temperature=0.1)
        response = llm.invoke(prompt)
        content = response.content.strip()

        if content.startswith("```"):
            content = content.split("\n", 1)[1] if "\n" in content else content[3:]
            if content.endswith("```"):
                content = content[:-3]
            content = content.strip()

        parsed = json.loads(content)

        decision = parsed.get("decision", "REVIEW")
        if decision not in ("APPROVE", "REVIEW", "BLOCK"):
            decision = "REVIEW"

        valid_codes = [c for c in parsed.get("reason_codes", []) if c in REASON_CODES]
        if not valid_codes:
            valid_codes = ["VOLUME_ANOMALY"]

        return {
            "decision": decision,
            "confidence": min(max(float(parsed.get("confidence", 0.5)), 0.0), 1.0),
            "reason_codes": valid_codes,
            "explanation": parsed.get("explanation", "Decision made based on available evidence."),
            "counterfactual": parsed.get("counterfactual", "N/A"),
            "reasoning_trace": [
                f"[DecisionAgent] LLM decision: {decision} "
                f"(confidence: {parsed.get('confidence', 0):.0%}). "
                f"Codes: {valid_codes}"
            ],
        }

    except Exception as e:
        result = _rule_based_decision(state)
        result["reasoning_trace"] = [
            f"[DecisionAgent] LLM failed ({type(e).__name__}: {str(e)[:60]}), fell back to rules: {result['decision']}"
        ]
        return result
