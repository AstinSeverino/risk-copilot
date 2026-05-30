"""Evaluation suite for the Risk Copilot agent pipeline."""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.agents.graph import graph
from src.ml.predict import predict_risk

EVAL_CASES = [
    {"merchant_id": "M007", "scenario": "Café with 10x volume spike (hero case)", "expected_decision": "APPROVE", "category": "legitimate_surge"},
    {"merchant_id": "M001", "scenario": "Normal restaurant, no anomaly", "expected_decision": "APPROVE", "category": "normal"},
    {"merchant_id": "M031", "scenario": "Normal grocery store", "expected_decision": "APPROVE", "category": "normal"},
    {"merchant_id": "M051", "scenario": "Normal gas station", "expected_decision": "APPROVE", "category": "normal"},
    {"merchant_id": "M071", "scenario": "Normal misc retail", "expected_decision": "APPROVE", "category": "normal"},

    {"merchant_id": "M089", "scenario": "New merchant, sudden high volume", "expected_decision": "REVIEW", "category": "suspicious"},
    {"merchant_id": "M067", "scenario": "CNP + foreign card surge", "expected_decision": "BLOCK", "category": "suspicious"},
    {"merchant_id": "M078", "scenario": "New gambling merchant, irregular", "expected_decision": "REVIEW", "category": "suspicious"},
    {"merchant_id": "M091", "scenario": "Few customers, many transactions", "expected_decision": "REVIEW", "category": "suspicious"},
    {"merchant_id": "M096", "scenario": "Gambling MCC, high risk tier", "expected_decision": "REVIEW", "category": "suspicious"},
]


def run_evaluation(verbose: bool = True):
    results = []
    decision_order = {"APPROVE": 0, "REVIEW": 1, "BLOCK": 2}

    for i, case in enumerate(EVAL_CASES):
        if verbose:
            print(f"\n[{i+1}/{len(EVAL_CASES)}] {case['merchant_id']} — {case['scenario']}")

        start = time.time()
        try:
            risk_result = predict_risk(case["merchant_id"])
            risk_score = risk_result.risk_probability

            output = graph.invoke({"merchant_id": case["merchant_id"], "reasoning_trace": []})
            actual = output.get("decision", "ERROR")
            elapsed = time.time() - start

            expected_level = decision_order.get(case["expected_decision"], 1)
            actual_level = decision_order.get(actual, 1)
            directionally_correct = actual_level >= expected_level if case["category"] == "suspicious" else actual_level <= expected_level

            result = {
                "merchant_id": case["merchant_id"],
                "scenario": case["scenario"],
                "category": case["category"],
                "risk_score": risk_score,
                "expected": case["expected_decision"],
                "actual": actual,
                "exact_match": actual == case["expected_decision"],
                "directionally_correct": directionally_correct,
                "confidence": output.get("confidence", 0),
                "reason_codes": output.get("reason_codes", []),
                "elapsed_s": elapsed,
            }

        except Exception as e:
            result = {
                "merchant_id": case["merchant_id"],
                "scenario": case["scenario"],
                "category": case["category"],
                "risk_score": 0,
                "expected": case["expected_decision"],
                "actual": "ERROR",
                "exact_match": False,
                "directionally_correct": False,
                "confidence": 0,
                "reason_codes": [],
                "elapsed_s": time.time() - start,
                "error": str(e),
            }

        results.append(result)
        if verbose:
            status = "✅" if result["exact_match"] else ("🔶" if result["directionally_correct"] else "❌")
            print(f"  Risk: {result['risk_score']:.3f} | Expected: {result['expected']} | Got: {result['actual']} {status} ({result['elapsed_s']:.1f}s)")

    if verbose:
        print("\n" + "=" * 60)
        print("EVALUATION SUMMARY")
        print("=" * 60)

        exact = sum(1 for r in results if r["exact_match"])
        directional = sum(1 for r in results if r["directionally_correct"])
        total = len(results)

        print(f"Exact match:   {exact}/{total} ({exact/total:.0%})")
        print(f"Directionally correct: {directional}/{total} ({directional/total:.0%})")

        suspicious = [r for r in results if r["category"] == "suspicious"]
        normal = [r for r in results if r["category"] != "suspicious"]

        if suspicious:
            sus_detected = sum(1 for r in suspicious if r["actual"] in ("REVIEW", "BLOCK"))
            print(f"\nSuspicious merchants flagged: {sus_detected}/{len(suspicious)} (recall)")

        if normal:
            normal_approved = sum(1 for r in normal if r["actual"] == "APPROVE")
            print(f"Normal merchants approved: {normal_approved}/{len(normal)} (specificity)")

        avg_time = sum(r["elapsed_s"] for r in results) / len(results)
        print(f"\nAvg investigation time: {avg_time:.1f}s")

        print("\nMismatches:")
        for r in results:
            if not r["exact_match"]:
                print(f"  {r['merchant_id']}: expected {r['expected']}, got {r['actual']} "
                      f"(risk={r['risk_score']:.3f}, codes={r['reason_codes']})")

    return results


if __name__ == "__main__":
    run_evaluation()
