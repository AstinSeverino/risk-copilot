"""Node 4: KYB (Know Your Business) verification stub. Pure Python — no LLM, no external API."""
from datetime import datetime

from src.agents.state import AgentState
from src.agents.observability import traced


@traced(name="kyb_verifier")
def kyb_verifier(state: AgentState) -> dict:
    merchant = state.get("merchant_info", {})
    reg_date_str = merchant.get("registered_date", "2020-01-01")
    reg_date = datetime.strptime(reg_date_str, "%Y-%m-%d")
    business_age_days = (datetime.now() - reg_date).days

    business_age_ok = business_age_days > 90
    risk_tier = merchant.get("risk_tier", "UNKNOWN")
    mcc = merchant.get("mcc_code", "")
    high_risk_mccs = {"7995", "7994", "5967", "5966"}

    mcc_risk_flag = mcc in high_risk_mccs

    overall = "PASS"
    flags = []
    if not business_age_ok:
        flags.append("BUSINESS_TOO_NEW")
        overall = "FLAG"
    if mcc_risk_flag:
        flags.append("HIGH_RISK_MCC")
        overall = "FLAG"
    if risk_tier == "HIGH":
        flags.append("HIGH_RISK_TIER")
        overall = "FLAG"

    kyb_status = {
        "business_age_days": business_age_days,
        "business_age_ok": business_age_ok,
        "mcc_risk_flag": mcc_risk_flag,
        "risk_tier": risk_tier,
        "sanctions_clear": True,
        "pep_clear": True,
        "adverse_media": "NONE_FOUND",
        "flags": flags,
        "overall": overall,
    }

    return {
        "kyb_status": kyb_status,
        "reasoning_trace": [
            f"[KYBVerifier] Business age: {business_age_days}d ({'OK' if business_age_ok else 'WARNING: <90d'}). "
            f"MCC risk: {'HIGH' if mcc_risk_flag else 'NORMAL'}. "
            f"Sanctions: CLEAR. PEP: CLEAR. "
            f"Overall: {overall}{' — Flags: ' + ', '.join(flags) if flags else ''}."
        ],
    }
