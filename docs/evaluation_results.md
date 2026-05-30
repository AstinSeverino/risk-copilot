# Evaluation Results — Risk Copilot Agent Pipeline

## Test Run: 2026-04-24

### Summary Metrics

| Metric | Value |
|--------|-------|
| **Exact match** | 5/10 (50%) |
| **Directionally correct** | 7/10 (70%) |
| **Normal merchants approved** | 5/5 (100% specificity) |
| **Suspicious merchants flagged** | 3/5 (60% recall) |
| **Avg investigation time** | 56.9s |
| **ML Model PR-AUC** | 0.833 |

### Per-Case Results

| Merchant | Scenario | Risk Score | Expected | Actual | Status |
|----------|----------|------------|----------|--------|--------|
| M007 | Café 10x volume spike (hero) | 0.988 | APPROVE | APPROVE | ✅ Exact |
| M001 | Normal restaurant | 0.047 | APPROVE | APPROVE | ✅ Exact |
| M031 | Normal grocery store | 0.010 | APPROVE | APPROVE | ✅ Exact |
| M051 | Normal gas station | 0.013 | APPROVE | APPROVE | ✅ Exact |
| M071 | Normal misc retail | 0.008 | APPROVE | APPROVE | ✅ Exact |
| M089 | New merchant sudden spike | 0.988 | REVIEW | BLOCK | 🔶 More severe |
| M067 | CNP + foreign card surge | 0.973 | BLOCK | REVIEW | ❌ Less severe |
| M078 | New gambling merchant | 0.047 | REVIEW | APPROVE | ❌ Not flagged by ML |
| M091 | Few customers many txns | 0.072 | REVIEW | APPROVE | ❌ Not flagged by ML |
| M096 | Gambling high risk tier | 0.974 | REVIEW | BLOCK | 🔶 More severe |

### Analysis of Mismatches

**M089 (REVIEW → BLOCK)**: The LLM was more conservative than expected. With risk=0.988 and no legitimate explanation found, BLOCK is arguably the safer decision. This is acceptable behavior — erring on the side of caution.

**M067 (BLOCK → REVIEW)**: The LLM found some contextual factors that introduced ambiguity, leading to REVIEW instead of BLOCK. In production, this would be caught by the human-in-the-loop gate on REVIEW cases.

**M078 and M091 (REVIEW → APPROVE)**: These merchants have low ML risk scores (0.047 and 0.072) because the XGBoost model with weak labels doesn't capture their suspicious patterns (gambling MCC newness, low customer diversity). The auto-approve path triggers before the LLM ever sees them. **Root cause**: weak-labeled XGBoost doesn't learn these patterns well. **Production fix**: train on confirmed fraud labels + add rule-based feature checks before auto-approve.

**M096 (REVIEW → BLOCK)**: Similar to M089 — the LLM is more conservative with high-risk merchants. With risk=0.974 and gambling MCC, BLOCK is a defensible decision.

### Key Findings

1. **The hero case works perfectly**: M007 (Café Aurora) with risk=0.988 is correctly APPROVED after the LLM finds contextual explanations. This is the core demo story.

2. **100% specificity on normal merchants**: Zero false positives on legitimate businesses. This is critical for merchant experience.

3. **The auto-approve gate is effective**: Merchants with risk < 0.5 skip expensive LLM calls entirely, saving both time (~1s vs ~100s) and API costs.

4. **LLM tends to be more conservative than expected**: 2 out of 5 suspicious cases got escalated to BLOCK when REVIEW was expected. This is acceptable in fintech — false negatives (missing fraud) are worse than false positives (extra review).

5. **Weak labels limit ML recall**: M078 and M091 show the cold-start limitation. In production, confirmed fraud labels would fix this.

### Honest Caveats

- "10 cases is not statistically significant. In production, you'd have 1000+ analyst-labeled cases running in CI/CD."
- "PR-AUC 0.833 is solid for weak labels but would improve with confirmed chargebacks."
- "The auto-approve threshold (0.5) could be tuned — lower catches more edge cases but costs more LLM tokens."
- "Investigation time of ~100s for LLM cases is acceptable for async batch processing but too slow for real-time decisions. Production would use faster models + caching."
