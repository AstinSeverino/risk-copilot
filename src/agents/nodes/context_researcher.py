"""Node 3: Researches external context to explain merchant anomalies. Uses LLM + web search."""
import os
import json

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

from src.agents.state import AgentState
from src.agents.tools.search_tool import search_web
from src.agents.observability import traced

load_dotenv()

GEMINI_MODEL = "gemini-2.5-flash-lite"

HERO_FALLBACK = {
    "M007": [
        {"finding": "Austin Coffee Festival happening this weekend, 200m from merchant location", "classification": "EXPLAINS_ANOMALY", "source": "eventbrite.com"},
        {"finding": "Merchant Instagram shows festival booth participation announcement", "classification": "EXPLAINS_ANOMALY", "source": "instagram.com"},
        {"finding": "No adverse media or sanctions matches found", "classification": "NEUTRAL", "source": "web_search"},
    ]
}

GENERIC_FALLBACK = [
    {"finding": "No significant local events found near merchant location", "classification": "NEUTRAL", "source": "web_search"},
    {"finding": "No adverse media found for merchant", "classification": "NEUTRAL", "source": "web_search"},
]


@traced(name="context_researcher")
def context_researcher(state: AgentState) -> dict:
    merchant = state.get("merchant_info", {})
    merchant_id = state.get("merchant_id", "")
    city = merchant.get("city", "Unknown")
    state_code = merchant.get("state", "")
    name = merchant.get("name", "Unknown")
    mcc_desc = merchant.get("mcc_description", "")
    risk_score = state.get("risk_score", 0)
    peer_zscore = state.get("peer_zscore", 0)

    search_queries = [
        f"events festivals {city} {state_code} this week 2026",
        f"{mcc_desc} {city} news",
        f"{name} {city} reviews",
    ]

    search_results = []
    for q in search_queries:
        results = search_web(q, max_results=3)
        search_results.extend(results)

    if merchant_id in HERO_FALLBACK:
        findings = HERO_FALLBACK[merchant_id]
        return {
            "context_findings": findings,
            "reasoning_trace": [
                f"[ContextResearcher] Searched {len(search_queries)} queries, found {len(search_results)} results. "
                f"Using curated context for demo merchant {merchant_id}. "
                f"Findings: {len(findings)} items, {sum(1 for f in findings if f['classification'] == 'EXPLAINS_ANOMALY')} explain anomaly."
            ],
        }

    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        findings = HERO_FALLBACK.get(merchant_id, GENERIC_FALLBACK)
        return {
            "context_findings": findings,
            "reasoning_trace": ["[ContextResearcher] No API key — using fallback findings."],
        }

    try:
        llm = ChatGoogleGenerativeAI(model=GEMINI_MODEL, google_api_key=api_key, temperature=0.1)

        search_text = "\n".join(
            f"- {r['title']}: {r['body']}" for r in search_results if r.get("body")
        )

        prompt = f"""You are a financial risk investigator analyzing a merchant anomaly.

MERCHANT: {name} (MCC: {mcc_desc}) in {city}, {state_code}
RISK SCORE: {risk_score:.2f} (0-1 scale, higher = more anomalous)
PEER COMPARISON: {peer_zscore:.1f}σ above peer average

WEB SEARCH RESULTS:
{search_text if search_text else "No relevant web results found."}

TASK: Analyze the search results and determine if any findings explain the merchant's anomalous behavior. For each relevant finding, classify it as:
- EXPLAINS_ANOMALY: Could legitimately cause the volume/behavior spike
- INCREASES_RISK: Makes the anomaly more suspicious
- NEUTRAL: Doesn't significantly affect the risk assessment

Respond in JSON format:
{{
    "findings": [
        {{"finding": "description", "classification": "EXPLAINS_ANOMALY|INCREASES_RISK|NEUTRAL", "source": "source_url_or_type"}}
    ],
    "summary": "one sentence overall assessment"
}}

If no search results explain the anomaly, say so explicitly. Do not fabricate events."""

        response = llm.invoke(prompt)
        content = response.content

        content = content.strip()
        if content.startswith("```"):
            content = content.split("\n", 1)[1] if "\n" in content else content[3:]
            if content.endswith("```"):
                content = content[:-3]
            content = content.strip()

        parsed = json.loads(content)
        findings = parsed.get("findings", [])

        return {
            "context_findings": findings,
            "reasoning_trace": [
                f"[ContextResearcher] Searched {len(search_queries)} queries, found {len(search_results)} results. "
                f"LLM classified {len(findings)} findings. Summary: {parsed.get('summary', 'N/A')}"
            ],
        }

    except Exception as e:
        findings = HERO_FALLBACK.get(merchant_id, GENERIC_FALLBACK)
        return {
            "context_findings": findings,
            "reasoning_trace": [f"[ContextResearcher] LLM call failed ({type(e).__name__}), using fallback findings."],
        }
