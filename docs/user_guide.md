# User Guide — Risk Copilot

## Prerequisites

- Python 3.11+
- Google Gemini API key (free tier: https://aistudio.google.com)

## Setup

```bash
# Clone and create virtual environment
git clone <repo-url> && cd risk-copilot
python -m venv venv && source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Configure API key
cp .env.example .env
# Edit .env and add your GOOGLE_API_KEY
```

## Generate Data & Train Models

```bash
# Generate synthetic merchant and transaction data
python -m src.data.generator
# Output: data/transactions.db (100 merchants, 1.1M transactions)

# Train ML risk models
python -m src.ml.train
# Output: models/isolation_forest.joblib, models/xgboost_risk.joblib
```

## Launch the UI

```bash
streamlit run app.py
# Opens at http://localhost:8501
```

## Using the Dashboard

### Tab 1: Dashboard
- Overview of all 100 merchants with risk scores
- KPI tiles: total merchants, active alerts, avg risk score, high-risk count
- Bar chart showing flagged merchants by MCC category
- Top 10 flagged merchants table

### Tab 2: Alert Investigation
1. Select a flagged merchant from the dropdown
2. Review the agent pipeline diagram
3. Click **"Run Investigation"** to trigger the full agent graph
4. Watch the streaming execution — each node reports its findings in real-time
5. Review the verdict card (green/yellow/red)
6. Examine feature contribution chart and decision explanation
7. Expand the full narrative report and reasoning trace

### Tab 3: Merchant Detail
- Select any merchant to view detailed information
- Transaction history chart (90 days)
- Peer comparison (z-score vs same-MCC merchants)
- KYB verification status

## Running the Agent Graph from CLI

```bash
# Investigate a specific merchant
python -c "
from src.agents.graph import graph
result = graph.invoke({'merchant_id': 'M007', 'reasoning_trace': []})
print(f'Decision: {result[\"decision\"]}')
print(f'Explanation: {result[\"explanation\"]}')
"
```

## Running the Evaluation Suite

```bash
python -m src.evaluation.eval_suite
# Runs 10 labeled test cases and reports precision/recall
```

## Fallback Mode (No API Key)

The system works without a Gemini API key:
- Nodes 1-2 (data_collector, anomaly_detector) are pure Python — always work
- Node 3 (context_researcher) uses hardcoded fallback findings for the hero case
- Node 5 (policy_retriever) always works — uses ChromaDB in-memory with local embeddings
- Node 6 (decision_agent) falls back to rule-based logic: risk > 0.8 = BLOCK, 0.5-0.8 = REVIEW, < 0.5 = APPROVE
- Node 7 (narrative_generator) uses template-based reports

## Project Structure

```
risk-copilot/
├── app.py                          # Streamlit UI (3 tabs)
├── requirements.txt                # Python dependencies
├── .env.example                    # API key template
│
├── src/
│   ├── data/
│   │   ├── generator.py            # Synthetic data generation
│   │   └── database.py             # SQLite query helpers
│   ├── ml/
│   │   ├── features.py             # 15-feature engineering pipeline
│   │   ├── train.py                # IsolationForest + XGBoost training
│   │   └── predict.py              # Risk prediction interface
│   ├── agents/
│   │   ├── state.py                # AgentState TypedDict definition
│   │   ├── graph.py                # LangGraph graph construction
│   │   ├── observability.py        # Langfuse integration (optional)
│   │   ├── nodes/
│   │   │   ├── data_collector.py   # Node 1: merchant data pull
│   │   │   ├── anomaly_detector.py # Node 2: ML risk scoring
│   │   │   ├── context_researcher.py # Node 3: web search + LLM
│   │   │   ├── kyb_verifier.py     # Node 4: KYB stub
│   │   │   ├── policy_retriever.py # Node 5: ChromaDB policy RAG
│   │   │   ├── decision_agent.py   # Node 6: final decision (LLM + RAG)
│   │   │   └── narrative_generator.py # Node 7: report generation
│   │   └── tools/
│   │       ├── ml_tool.py          # ML model wrapper
│   │       └── search_tool.py      # DuckDuckGo search wrapper
│   └── evaluation/
│       └── eval_suite.py           # 10-case evaluation suite
│
├── models/                         # Trained ML model artifacts
├── data/
│   ├── transactions.db             # SQLite database
│   └── policies/                   # Policy documents for RAG
│       ├── risk_scoring_thresholds.txt
│       ├── reason_codes.txt
│       ├── aml_policy.txt
│       ├── transaction_laundering.txt
│       ├── kyb_requirements.txt
│       └── mcc_risk_categories.txt
└── docs/                           # Project documentation
```

## Key Merchants for Demo

| ID | Name | MCC | Behavior | Expected |
|----|------|-----|----------|----------|
| M007 | Café Aurora | 5812 | 10x volume spike | APPROVE (hero case) |
| M001 | Normal restaurant | 5812 | Normal activity | APPROVE |
| M089 | New merchant | 5999 | Sudden high volume | REVIEW/BLOCK |
| M067 | Gas station | 5541 | CNP + foreign surge | BLOCK |
