# Architecture — Risk Copilot

## System Overview

Risk Copilot is a multi-agent decision engine that combines ML scoring with LLM reasoning to investigate merchant transaction anomalies. The core principle: **ML handles numbers, LLMs handle judgment**.

```
┌──────────────────────────────────────────────────────────────┐
│                    STREAMLIT UI (app.py)                      │
│  ┌──────────┐  ┌──────────────────┐  ┌──────────────────┐   │
│  │ Dashboard │  │ Alert            │  │ Merchant         │   │
│  │           │  │ Investigation    │  │ Detail           │   │
│  └──────────┘  └──────────────────┘  └──────────────────┘   │
└───────────────────────┬──────────────────────────────────────┘
                        │
┌───────────────────────▼──────────────────────────────────────┐
│                 LANGGRAPH AGENT PIPELINE                      │
│                                                               │
│  ┌──────────────┐    ┌─────────────────┐                     │
│  │ data_collector│───▶│ anomaly_detector │                    │
│  │ (Python)      │    │ (ML Models)      │                    │
│  └──────────────┘    └────────┬─────────┘                    │
│                           ┌───┴────┐                          │
│                     risk<0.5│      │risk≥0.5                  │
│                           ▼        ▼                          │
│                  ┌────────────┐  ┌──────────────────┐        │
│                  │auto_approve│  │context_researcher │        │
│                  │            │  │(LLM + Web Search) │        │
│                  └─────┬──────┘  └────────┬──────────┘        │
│                        │                  ▼                   │
│                        │         ┌──────────────┐            │
│                        │         │ kyb_verifier  │            │
│                        │         │ (Stub)        │            │
│                        │         └──────┬───────┘            │
│                        │                ▼                     │
│                        │         ┌──────────────────┐        │
│                        │         │policy_retriever   │        │
│                        │         │(ChromaDB RAG)     │        │
│                        │         └──────┬───────────┘        │
│                        │                ▼                     │
│                        │         ┌──────────────┐            │
│                        │         │decision_agent │            │
│                        │         │(LLM + Policy) │            │
│                        │         └──────┬───────┘            │
│                        │                ▼                     │
│                        │         ┌──────────────────┐        │
│                        │         │narrative_generator│        │
│                        │         │(LLM)             │        │
│                        │         └──────┬───────────┘        │
│                        ▼                ▼                     │
│                       END              END                    │
└──────────────────────────────────────────────────────────────┘
                        │
┌───────────────────────▼──────────────────────────────────────┐
│                      DATA LAYER                               │
│  ┌──────────┐  ┌──────────────┐  ┌────────────────────┐     │
│  │ SQLite   │  │ IsolationFor.│  │ XGBoost            │     │
│  │ 1.1M txns│  │ .joblib      │  │ .joblib            │     │
│  └──────────┘  └──────────────┘  └────────────────────┘     │
└──────────────────────────────────────────────────────────────┘
```

## Design Decisions

### Why ML + LLM (not pure LLM)?

| Task | ML | LLM |
|------|-----|-----|
| Numeric risk scoring | ✅ Fast (<5ms), calibrated | ❌ Unreliable with numbers |
| Peer comparison z-scores | ✅ Exact calculation | ❌ Can't access data directly |
| Web search + context | ❌ No reasoning ability | ✅ Interprets search results |
| Policy interpretation | ❌ Rigid rules only | ✅ Nuanced judgment |
| Narrative generation | ❌ Template only | ✅ Natural, audit-ready |
| Feature importance | ✅ TreeSHAP/native | ❌ Would hallucinate |

### Why LangGraph (not CrewAI)?

- **Explicit graph** = auditable control flow that maps to policy documents
- **Conditional edges** = cost control (skip LLM for low-risk cases)
- **TypedDict state** with append-only `reasoning_trace` = regulatory replay
- **`graph.stream()`** = real-time UI updates per node

### Why Conditional Routing?

The auto-approve gate (risk < 0.5 AND peer_zscore < 3.0σ) serves two purposes:
1. **Cost control**: ~95% of merchants are normal — no LLM tokens spent on them
2. **Speed**: Auto-approve in ~1s vs full investigation in ~100s

### Why Fallback Mode?

Every LLM node has a rule-based/template fallback:
- **Context researcher**: hardcoded findings for hero case
- **Decision agent**: threshold-based rules (risk > 0.8 = BLOCK)
- **Narrative generator**: f-string template report

This ensures the demo works even if the Gemini API is unavailable.

## Data Flow

1. **Input**: `merchant_id` (e.g., "M007")
2. **Data Collector**: Queries SQLite for merchant info, 90-day transaction history, peer cohort stats
3. **Anomaly Detector**: Runs IsolationForest + XGBoost, computes 15 features including peer z-scores
4. **Routing**: If risk < 0.5 → auto-approve. Otherwise → full investigation.
5. **Context Researcher**: Searches DuckDuckGo for local events/news, LLM classifies findings
6. **KYB Verifier**: Checks business age, MCC risk, simulated sanctions/PEP screening
7. **Policy Retriever**: Queries ChromaDB vector store for relevant risk policies (RAG)
8. **Decision Agent**: LLM synthesizes ALL evidence + retrieved policy context → typed decision
8. **Narrative Generator**: LLM produces audit-ready investigation report
9. **Output**: Complete `AgentState` with decision, reason codes, explanation, counterfactual, narrative, and full reasoning trace

## Feature Engineering Pipeline

15 features computed per merchant across 5 families:

- **Volume/Velocity**: txn_count_24h, txn_count_7d, txn_count_30d, volume_growth_ratio
- **Amounts**: avg_ticket_24h, max_ticket_24h, total_amount_24h
- **Behavior**: unique_customers_24h, customer_txn_ratio, pct_card_present, pct_international
- **Peer Comparison**: peer_volume_zscore (most important), peer_amount_zscore
- **Temporal/Business**: hour_entropy, business_age_days

## Reason Code Taxonomy (13 codes)

| Code | Description |
|------|-------------|
| VOLUME_ANOMALY | Transaction volume significantly above baseline |
| PEER_GROUP_OUTLIER | Volume/amount outside peer MCC norms |
| TICKET_SIZE_ANOMALY | Average ticket significantly different |
| CNP_RATIO_SPIKE | Card-not-present percentage increase |
| FOREIGN_CARD_SPIKE | International card percentage increase |
| LOW_CUSTOMER_DIVERSITY | Few unique customers, many transactions |
| NEW_MERCHANT_HIGH_VOLUME | Recently registered with unusual volume |
| MCC_MISMATCH | Declared MCC doesn't match observed behavior |
| LEGITIMATE_EVENT | External event explains anomaly |
| SEASONAL_PATTERN | Known seasonal pattern for this MCC |
| WITHIN_NORMAL_RANGE | All metrics within expected parameters |
| KYB_FLAG | KYB verification raised concerns |
| BUSINESS_AGE_CONCERN | Business too new for observed volume |
