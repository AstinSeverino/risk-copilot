# Changelog — Risk Copilot

## 2026-05-30 — Open-sourced

Published the project as a public repository: **github.com/AstinSeverino/risk-copilot**. Added per-tab screenshots (Dashboard, Investigation, Merchant Detail) and a Screenshots section in the README. Hardened `.gitignore` so secrets, trained models, the database, and local tooling stay out of version control.

## Session 2026-04-24 — Feature Sprint

### 1. Policy RAG System (ChromaDB)

**What**: Added a Retrieval-Augmented Generation (RAG) pipeline that feeds relevant risk policies into the Decision Agent before it makes its verdict.

**Why**: RAG keeps the system extensible — moving policies from hardcoded prompts to a vector store means new risk policies can be added as TXT files without modifying code.

**Implementation**:
- Created 6 policy documents in `data/policies/`:
  - `risk_scoring_thresholds.txt` — Auto-approve, review, and block thresholds
  - `reason_codes.txt` — FCRA-compliant reason code taxonomy (13 codes)
  - `aml_policy.txt` — Anti-money laundering policy (BSA, AMLD6, FATF)
  - `transaction_laundering.txt` — Laundering detection indicators and scoring
  - `kyb_requirements.txt` — Know Your Business verification requirements
  - `mcc_risk_categories.txt` — MCC-specific risk classifications
- Created `src/agents/nodes/policy_retriever.py`:
  - ChromaDB in-memory vector store with all-MiniLM-L6-v2 embeddings (local, no API needed)
  - Splits policy documents into 500-char chunks with 50-char overlap → 47 total chunks
  - Dynamically builds semantic queries based on merchant risk profile
  - Returns top 5 most relevant policy chunks
- Added `policy_context: str` field to `AgentState`
- Updated graph: `kyb_verifier → policy_retriever → decision_agent`
- Decision Agent now includes retrieved policy context in its LLM prompt

**Result**: Policy retriever successfully pulls relevant chunks (e.g., for a high-risk restaurant, it retrieves from reason_codes.txt, mcc_risk_categories.txt, and transaction_laundering.txt). Evaluation results unchanged — 5/10 exact match, 7/10 directionally correct.

### 2. Langfuse Observability

**What**: Integrated Langfuse for LLM call tracing and observability across all agent nodes.

**Why**: Production ML systems need observability. Langfuse provides prompt versioning, latency tracking, and cost monitoring — all visible in a web dashboard.

**Implementation**:
- Created `src/agents/observability.py` — centralized Langfuse module with graceful degradation
- Integrated `CallbackHandler` into all 3 LLM nodes:
  - `context_researcher.py`
  - `decision_agent.py`
  - `narrative_generator.py`
- Configuration via environment variables (optional):
  - `LANGFUSE_PUBLIC_KEY`
  - `LANGFUSE_SECRET_KEY`
  - `LANGFUSE_HOST` (defaults to cloud.langfuse.com)
- Updated `.env.example` with Langfuse config template

**Result**: System works identically with or without Langfuse keys. When configured, all LLM calls are traced with full prompt/response capture.

### 3. UI Rework — Dark Modern Theme

**What**: Completely redesigned the Streamlit UI with a dark, modern aesthetic inspired by canirun.ai.

**Why**: The original UI used default Streamlit styling which looks generic. A polished dark theme demonstrates frontend awareness and makes a stronger impression.

**Design system**:
- Background: `#0a0a0f` (near black)
- Cards: `#111118` with `#1e1e2a` borders
- Accent: `#6366f1` (indigo-500)
- Success: `#22c55e`, Warning: `#f59e0b`, Danger: `#ef4444`
- Typography: Inter font, uppercase labels, -0.02em letter spacing on headings
- Components: gradient verdict banners, styled tab bar, badge system, monospace reason code chips

**Implementation**:
- Created `.streamlit/config.toml` with dark theme
- Rewrote `app.py` with comprehensive CSS (~180 lines)
- Updated Plotly charts: transparent backgrounds, dark grid lines, indigo-to-red gradient scales
- Styled all 3 tabs: Dashboard, Investigation, Merchant Detail
- Area-fill line charts, heat-mapped risk tables

### 4. Matplotlib Bug Fix

**What**: Fixed `ModuleNotFoundError: No module named 'matplotlib'` that crashed the Dashboard tab.

**Root cause**: The `pd.DataFrame.style.background_gradient()` call in the flagged merchants table requires matplotlib for colormap rendering, but matplotlib was not in `requirements.txt`.

**Fix**: Installed matplotlib and added it to `requirements.txt`.

### 5. Documentation Updates

**What**: Updated all project documentation to reflect new features.

**Files updated**:
- `README.md` — Updated architecture diagram, agent table, tech stack (added ChromaDB, Langfuse)
- `docs/architecture.md` — Added policy_retriever to data flow and ASCII diagram
- `docs/user_guide.md` — Updated project structure, node numbering, fallback mode description

### Pipeline (8 Nodes)

The full agent pipeline is now:

```
data_collector → anomaly_detector → [conditional]
  ├── risk < 0.5 → auto_approve → END
  └── risk ≥ 0.5 → context_researcher → kyb_verifier → policy_retriever → decision_agent → narrative_generator → END
```

### Files Created
- `data/policies/risk_scoring_thresholds.txt`
- `data/policies/reason_codes.txt`
- `data/policies/aml_policy.txt`
- `data/policies/transaction_laundering.txt`
- `data/policies/kyb_requirements.txt`
- `data/policies/mcc_risk_categories.txt`
- `src/agents/nodes/policy_retriever.py`
- `src/agents/observability.py`
- `.streamlit/config.toml`
- `docs/changelog.md`

### Files Modified
- `src/agents/state.py` — Added `policy_context` field
- `src/agents/graph.py` — Added policy_retriever node and edge
- `src/agents/nodes/decision_agent.py` — Integrated policy context + Langfuse callback
- `src/agents/nodes/context_researcher.py` — Langfuse callback
- `src/agents/nodes/narrative_generator.py` — Langfuse callback
- `app.py` — Complete UI rewrite
- `requirements.txt` — Added chromadb, langfuse, matplotlib
- `.env.example` — Added Langfuse config
- `README.md` — Updated architecture + tech stack
- `docs/architecture.md` — Added policy_retriever to diagrams
- `docs/user_guide.md` — Updated structure + node numbering

### Evaluation Results (Post-Changes)
| Metric | Value |
|--------|-------|
| Exact match | 5/10 (50%) |
| Directionally correct | 7/10 (70%) |
| Normal merchants approved | 5/5 (100% specificity) |
| Suspicious merchants flagged | 3/5 (60% recall) |
| Avg investigation time | 54.8s |
