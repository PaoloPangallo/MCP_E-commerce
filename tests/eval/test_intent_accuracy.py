import asyncio
import json
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(__file__), '../../.env'))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from app.services.parser import parse_query_service


# ---------------------------------------------------------------------------
# Normalisation helpers
# ---------------------------------------------------------------------------

def _norm(value: Any) -> str:
    """Lowercase + strip for scalar comparison."""
    if value is None:
        return ""
    return str(value).lower().strip()


def _set_precision_recall(predicted_set: set, expected_set: set) -> Tuple[float, float]:
    if not expected_set and not predicted_set:
        return 1.0, 1.0
    tp = len(predicted_set & expected_set)
    precision = tp / len(predicted_set) if predicted_set else (1.0 if not expected_set else 0.0)
    recall = tp / len(expected_set) if expected_set else 1.0
    return precision, recall


def _f1(precision: float, recall: float) -> float:
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


# ---------------------------------------------------------------------------
# Constraint matching
# ---------------------------------------------------------------------------

def _constraint_match(expected_c: dict, predicted_constraints: List[dict]) -> bool:
    """
    Check if the expected constraint is satisfied by any predicted constraint of
    the same type. Operator comparison is lenient: we only check the value.
    """
    ctype = expected_c.get("type")
    matches = [c for c in predicted_constraints if c.get("type") == ctype]
    if not matches:
        return False

    if ctype == "price":
        exp_val = expected_c.get("value")
        exp_op = expected_c.get("operator", "<=")
        for m in matches:
            pred_val = m.get("value")
            pred_op = m.get("operator", "<=")
            # Accept if value is within 5% tolerance and operator class matches
            # (e.g., "<=" and "<" are both "upper bound")
            try:
                upper_ops = {"<=", "<"}
                lower_ops = {">=", ">"}
                if exp_op in upper_ops and pred_op in upper_ops:
                    if isinstance(pred_val, (int, float)) and isinstance(exp_val, (int, float)):
                        if abs(pred_val - exp_val) / max(exp_val, 1) <= 0.05:
                            return True
                elif exp_op in lower_ops and pred_op in lower_ops:
                    if isinstance(pred_val, (int, float)) and isinstance(exp_val, (int, float)):
                        if abs(pred_val - exp_val) / max(exp_val, 1) <= 0.05:
                            return True
                elif exp_op == "between":
                    if pred_op == "between" and isinstance(pred_val, list) and isinstance(exp_val, list):
                        if len(pred_val) == 2 and len(exp_val) == 2:
                            lo_ok = abs(pred_val[0] - exp_val[0]) / max(exp_val[0], 1) <= 0.05
                            hi_ok = abs(pred_val[1] - exp_val[1]) / max(exp_val[1], 1) <= 0.05
                            if lo_ok and hi_ok:
                                return True
            except (TypeError, ZeroDivisionError):
                pass
        return False

    if ctype == "condition":
        exp_val = _norm(expected_c.get("value"))
        for m in matches:
            if _norm(m.get("value")) == exp_val:
                return True
        return False

    if ctype == "aspect":
        exp_name = _norm(expected_c.get("name"))
        exp_val = _norm(expected_c.get("value"))
        for m in matches:
            if _norm(m.get("name")) == exp_name and _norm(m.get("value")) == exp_val:
                return True
        return False

    # Generic fallback
    exp_val = _norm(expected_c.get("value"))
    for m in matches:
        if _norm(m.get("value")) == exp_val:
            return True
    return False


# ---------------------------------------------------------------------------
# Multi-level comparison
# ---------------------------------------------------------------------------

def compare_parser_output(expected: dict, predicted: dict) -> dict:
    """
    Three-level structured comparison:
      1. Scalars   — product, target_seller, product_type
      2. Lists     — brands, exclusions  (set intersection precision/recall)
      3. Constraints — grouped by type, with normalised value matching
    Returns a dict with per-field results and aggregate F1 scores.
    """
    results: dict = {
        "scalar": {},
        "lists": {},
        "constraints": {},
        "aggregate": {}
    }

    # 1. Scalars
    scalar_scores = []
    for key in ["product", "target_seller", "product_type"]:
        if key not in expected:
            continue
        match = _norm(expected[key]) == _norm(predicted.get(key))
        results["scalar"][key] = match
        scalar_scores.append(1.0 if match else 0.0)

    # 2. Lists (brands, exclusions)
    list_f1s = []
    for key in ["brands", "exclusions"]:
        if key not in expected:
            continue
        exp_set = {_norm(v) for v in (expected.get(key) or [])}
        pred_set = {_norm(v) for v in (predicted.get(key) or [])}
        p, r = _set_precision_recall(pred_set, exp_set)
        f = _f1(p, r)
        results["lists"][key] = {"precision": round(p, 3), "recall": round(r, 3), "f1": round(f, 3)}
        list_f1s.append(f)

    # 3. Constraints
    constraint_scores = []
    expected_constraints = expected.get("constraints") or []
    predicted_constraints = predicted.get("constraints") or []

    for exp_c in expected_constraints:
        ctype = exp_c.get("type", "unknown")
        match = _constraint_match(exp_c, predicted_constraints)
        key = f"{ctype}:{exp_c.get('operator', exp_c.get('name', ''))}" if ctype in ("price", "aspect") else ctype
        results["constraints"][key] = match
        constraint_scores.append(1.0 if match else 0.0)

    # Aggregate F1
    all_scores = scalar_scores + list_f1s + constraint_scores
    macro_f1 = sum(all_scores) / len(all_scores) if all_scores else 1.0
    results["aggregate"]["macro_f1"] = round(macro_f1, 3)
    results["aggregate"]["scalar_acc"] = round(sum(scalar_scores) / len(scalar_scores), 3) if scalar_scores else None
    results["aggregate"]["list_f1"] = round(sum(list_f1s) / len(list_f1s), 3) if list_f1s else None
    results["aggregate"]["constraint_acc"] = round(sum(constraint_scores) / len(constraint_scores), 3) if constraint_scores else None

    return results


# ---------------------------------------------------------------------------
# Evaluation runner
# ---------------------------------------------------------------------------

async def run_evaluation():
    setup_file = os.path.join(os.path.dirname(__file__), "intent_dataset.json")
    with open(setup_file, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    total_macro_f1 = 0.0
    exact_matches = 0

    print("=" * 60)
    print("  NLU PARSING EVALUATION STARTED")
    print("=" * 60)

    for item in dataset:
        query = item["query"]
        expected = item["expected"]

        print(f"\n[ID {item['id']}] Query: '{query}'")
        predicted = await parse_query_service(query)
        report = compare_parser_output(expected, predicted)

        agg = report["aggregate"]
        macro_f1 = agg["macro_f1"]
        total_macro_f1 += macro_f1

        exact = macro_f1 == 1.0
        if exact:
            exact_matches += 1

        print(f"  SCALAR     : {report['scalar']}")
        print(f"  LISTS      : {report['lists']}")
        print(f"  CONSTRAINTS: {report['constraints']}")
        print(f"  AGGREGATE  : scalar_acc={agg['scalar_acc']}  list_f1={agg['list_f1']}  constraint_acc={agg['constraint_acc']}  macro_f1={macro_f1}  EM={exact}")

    n = len(dataset)
    print("\n" + "=" * 60)
    print(f"  RISULTATI FINALI SU {n} QUERY:")
    print(f"  Mean Macro F1   : {total_macro_f1 / n:.3f}")
    print(f"  Exact Matches   : {exact_matches}/{n} ({exact_matches / n * 100:.1f}%)")
    print("=" * 60)


if __name__ == "__main__":
    if sys.platform == 'win32':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(run_evaluation())
