"""Advisory agent assessment of the expert-review DAG checklist (mig 097).

Grades the six reviewer-checklist questions from machine evidence ONLY: the
persisted DAG snapshot (``dag_structure_json``) and the refutation rows the
review links to (``related_validation_ids`` -> ``causal_validations``). The
output is ADVISORY — cached in ``agent_assessment_json``, displayed beside the
human checklist, never a substitute for it.

Honesty discipline (matches the src/insights pattern):
- The LM path is grounded in the provided evidence; each answer must start with
  a valid verdict token, and any digit in its rationale must appear in the
  grounding text (vouched) — otherwise THAT item falls back to the
  deterministic, evidence-derived one.
- No LM (CI / no key) -> the deterministic fallback grades what the evidence
  actually shows; questions with no machine evidence say ``no_evidence``
  plainly (SUTVA is never machine-assessable).
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, List, Optional

from src.insights.common import run_signature

logger = logging.getLogger(__name__)

# Single source of truth for the six questions; ids MUST stay in sync with the
# review UI checkboxes (frontend/src/pages/ExpertReviews.tsx CHECKLIST_ITEMS)
# so verdict chips join onto the right rows.
CHECKLIST_QUESTIONS: List[Dict[str, str]] = [
    {"id": "conf_complete", "question": "Are all known confounders included?"},
    {"id": "edge_plausible", "question": "Do causal arrows reflect domain knowledge?"},
    {"id": "no_forbidden", "question": "Are there no forbidden edges (future→past)?"},
    {
        "id": "mediators_correct",
        "question": "Are intermediate variables correctly positioned?",
    },
    {"id": "sutva_plausible", "question": "Is the no-interference assumption reasonable?"},
    {"id": "positivity", "question": "Is there sufficient overlap in treatment groups?"},
]

VALID_VERDICTS = ("supports", "concern", "unclear", "no_evidence")

# Which refutation tests speak to which question (see 010 refutation_test_type):
# random_common_cause / sensitivity_e_value probe confounding; data_subset /
# bootstrap probe estimate stability across subsamples (an overlap proxy).
_CONFOUNDING_TESTS = ("random_common_cause", "sensitivity_e_value")
_STABILITY_TESTS = ("data_subset", "bootstrap")

try:
    import dspy

    class DagReviewAssessmentSignature(dspy.Signature):
        """Assess a causal DAG awaiting expert review against six checklist
        questions, STRICTLY grounded in the provided structure, structural
        facts, and refutation evidence. For EACH question answer with a verdict
        token — one of supports / concern / unclear / no_evidence — then " — "
        and ONE sentence of rationale citing ONLY the given evidence. NEVER
        invent nodes, edges, tests, or numbers; if the evidence does not speak
        to a question, answer no_evidence. You are ADVISORY: the human reviewer
        decides. Prefer unclear over supports when domain judgment is required
        (edge plausibility, mediator placement)."""

        context: str = dspy.InputField(desc="Brand, treatment, outcome, gate context")
        dag_summary: str = dspy.InputField(desc="Nodes and directed edges of the DAG")
        structural_facts: str = dspy.InputField(
            desc="Deterministic graph facts: acyclicity, paths, mediators, forbidden edges"
        )
        refutation_evidence: str = dspy.InputField(
            desc="Refutation test results (test, status, effect shift)"
        )

        conf_complete: str = dspy.OutputField(desc="Are all known confounders included?")
        edge_plausible: str = dspy.OutputField(desc="Do causal arrows reflect domain knowledge?")
        no_forbidden: str = dspy.OutputField(desc="Are there no forbidden edges (future→past)?")
        mediators_correct: str = dspy.OutputField(
            desc="Are intermediate variables correctly positioned?"
        )
        sutva_plausible: str = dspy.OutputField(
            desc="Is the no-interference assumption reasonable?"
        )
        positivity: str = dspy.OutputField(desc="Is there sufficient overlap in treatment groups?")

    DSPY_AVAILABLE = True
except ImportError:
    DSPY_AVAILABLE = False
    DagReviewAssessmentSignature = None  # type: ignore[assignment,misc]


# ---------------------------------------------------------------------------
# Deterministic graph analysis
# ---------------------------------------------------------------------------


def compute_structural_facts(structure: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Pure-Python graph facts for the persisted DAG snapshot.

    No networkx dependency: the snapshots are small (tens of nodes) and the
    checks are simple DFS/BFS, so plain adjacency sets keep this importable
    everywhere (including the slim API image).
    """
    if not structure or not structure.get("nodes"):
        return {"has_structure": False}

    nodes = [str(n) for n in structure.get("nodes", [])]
    edges = [(str(e[0]), str(e[1])) for e in structure.get("edges", [])]
    treatment = next(iter(structure.get("treatment_nodes") or []), None)
    outcome = next(iter(structure.get("outcome_nodes") or []), None)

    adjacency: Dict[str, set] = {n: set() for n in nodes}
    for src, tgt in edges:
        adjacency.setdefault(src, set()).add(tgt)
        adjacency.setdefault(tgt, set())

    def _reachable(start: str) -> set:
        seen: set = set()
        stack = [start]
        while stack:
            current = stack.pop()
            for nxt in adjacency.get(current, ()):
                if nxt not in seen:
                    seen.add(nxt)
                    stack.append(nxt)
        return seen

    # Cycle detection: iterative DFS with colors.
    WHITE, GRAY, BLACK = 0, 1, 2
    color = dict.fromkeys(adjacency, WHITE)
    is_acyclic = True
    for root in adjacency:
        if color[root] != WHITE:
            continue
        stack: List[tuple] = [(root, iter(adjacency[root]))]
        color[root] = GRAY
        while stack:
            node, it = stack[-1]
            advanced = False
            for nxt in it:
                if color[nxt] == GRAY:
                    is_acyclic = False
                elif color[nxt] == WHITE:
                    color[nxt] = GRAY
                    stack.append((nxt, iter(adjacency[nxt])))
                    advanced = True
                    break
            if not advanced:
                color[node] = BLACK
                stack.pop()

    has_path = False
    mediators: List[str] = []
    if treatment and outcome:
        from_treatment = _reachable(treatment)
        has_path = outcome in from_treatment
        if has_path:
            # A mediator sits on a directed treatment->outcome path: reachable
            # FROM the treatment and able to reach the outcome.
            mediators = sorted(
                n
                for n in from_treatment
                if n not in (treatment, outcome) and outcome in _reachable(n)
            )

    outcome_to_treatment_edge = bool(treatment and outcome and (outcome, treatment) in set(edges))

    return {
        "has_structure": True,
        "n_nodes": len(nodes),
        "n_edges": len(edges),
        "treatment": treatment,
        "outcome": outcome,
        "is_acyclic": is_acyclic,
        "has_treatment_outcome_path": has_path,
        "mediators": mediators,
        "outcome_to_treatment_edge": outcome_to_treatment_edge,
        "augmented_edges": [[str(e[0]), str(e[1])] for e in structure.get("augmented_edges", [])],
        "discovery_gate_decision": structure.get("discovery_gate_decision"),
        "adjustment_sets": structure.get("adjustment_sets") or [],
    }


# ---------------------------------------------------------------------------
# Grounding
# ---------------------------------------------------------------------------


def _coerce_structure(raw: Any) -> Optional[Dict[str, Any]]:
    """dag_structure_json arrives as dict (PostgREST JSONB) or a JSON string."""
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str) and raw.strip():
        try:
            parsed = json.loads(raw)
            return parsed if isinstance(parsed, dict) else None
        except (ValueError, TypeError):
            return None
    return None


def build_grounding(review: Dict[str, Any], validations: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Assemble the evidence strings the assessment is grounded in."""
    structure = _coerce_structure(review.get("dag_structure_json"))
    facts = compute_structural_facts(structure)

    if facts["has_structure"]:
        edge_lines = [f"{src} -> {tgt}" for src, tgt in (structure or {}).get("edges", [])][:60]
        dag_summary = (
            f"treatment={facts['treatment']}, outcome={facts['outcome']}, "
            f"{facts['n_nodes']} nodes, {facts['n_edges']} edges:\n" + "\n".join(edge_lines)
        )
        fact_lines = [
            f"acyclic={facts['is_acyclic']}",
            f"directed treatment->outcome path exists={facts['has_treatment_outcome_path']}",
            f"mediators on treatment->outcome paths={facts['mediators'] or 'none'}",
            f"outcome->treatment edge present={facts['outcome_to_treatment_edge']}",
            f"adjustment sets={facts['adjustment_sets'] or 'none'}",
        ]
        if facts["augmented_edges"]:
            fact_lines.append(
                f"discovery-augmented edges={facts['augmented_edges']} "
                f"(gate={facts['discovery_gate_decision']})"
            )
        structural_facts_text = "\n".join(fact_lines)
    else:
        dag_summary = "DAG structure not captured for this review (pre-097 row)."
        structural_facts_text = "No structural facts available."

    evidence_lines = []
    for row in validations:
        line = f"{row.get('test_type')}: {row.get('status')}"
        if row.get("original_effect") is not None and row.get("refuted_effect") is not None:
            line += (
                f" (effect {row['original_effect']} -> {row['refuted_effect']}"
                + (
                    f", delta {row['delta_percent']}%"
                    if row.get("delta_percent") is not None
                    else ""
                )
                + ")"
            )
        if row.get("p_value") is not None:
            line += f" p={row['p_value']}"
        evidence_lines.append(line)
    refutation_evidence = "\n".join(evidence_lines) or "No refutation rows linked."

    context = ", ".join(
        f"{k}={review[k]}"
        for k in ("brand", "treatment_variable", "outcome_variable", "analysis_context")
        if review.get(k)
    )

    return {
        "context": context or "no context recorded",
        "dag_summary": dag_summary,
        "structural_facts": structural_facts_text,
        "refutation_evidence": refutation_evidence,
        "facts": facts,
        "validations": validations,
        "validations_used": len(validations),
        "has_dag_structure": facts["has_structure"],
    }


# ---------------------------------------------------------------------------
# Deterministic fallback verdicts
# ---------------------------------------------------------------------------


def _test_bucket_verdict(validations: List[Dict[str, Any]], test_types: tuple) -> tuple:
    relevant = [v for v in validations if v.get("test_type") in test_types]
    ran = [v for v in relevant if v.get("status") in ("passed", "failed", "warning")]
    if not ran:
        return "no_evidence", None
    failed = [v for v in ran if v.get("status") == "failed"]
    if failed:
        return "concern", sorted({str(v.get("test_type")) for v in failed})
    warned = [v for v in ran if v.get("status") == "warning"]
    if warned:
        return "unclear", sorted({str(v.get("test_type")) for v in warned})
    return "supports", sorted({str(v.get("test_type")) for v in ran})


def _fallback_items(g: Dict[str, Any]) -> List[Dict[str, str]]:
    facts = g["facts"]
    validations = g.get("validations", [])
    has_structure = facts.get("has_structure", False)

    verdict, tests = _test_bucket_verdict(validations, _CONFOUNDING_TESTS)
    if verdict == "supports":
        conf = (
            verdict,
            f"Confounder-sensitivity refuters passed ({', '.join(tests)}) — "
            "consistent with, but not proof of, adequate confounder coverage.",
        )
    elif verdict == "concern":
        conf = (verdict, f"Confounder-sensitivity refuter(s) FAILED: {', '.join(tests)}.")
    elif verdict == "unclear":
        conf = (verdict, f"Borderline confounder-sensitivity result(s): {', '.join(tests)}.")
    else:
        conf = (verdict, "No confounder-sensitivity refutation evidence linked.")

    if facts.get("augmented_edges"):
        edge = (
            "unclear",
            f"{len(facts['augmented_edges'])} edge(s) were discovery-augmented "
            f"(gate={facts.get('discovery_gate_decision')}); verify each against "
            "domain knowledge.",
        )
    elif has_structure:
        edge = (
            "unclear",
            "Edge directions require domain judgment; the graph is shown for review.",
        )
    else:
        edge = ("no_evidence", "DAG structure not captured for this review.")

    if not has_structure:
        forbidden = ("no_evidence", "DAG structure not captured for this review.")
    elif facts.get("outcome_to_treatment_edge") or not facts.get("is_acyclic"):
        reasons = []
        if facts.get("outcome_to_treatment_edge"):
            reasons.append("an outcome->treatment edge is present")
        if not facts.get("is_acyclic"):
            reasons.append("the graph contains a cycle")
        forbidden = ("concern", f"Structural check failed: {' and '.join(reasons)}.")
    else:
        forbidden = (
            "supports",
            "Graph is acyclic with no outcome->treatment edge (structural check "
            "only; variable timing is not encoded in the DAG).",
        )

    if not has_structure:
        mediators = ("no_evidence", "DAG structure not captured for this review.")
    elif not facts.get("has_treatment_outcome_path"):
        mediators = (
            "concern",
            "No directed path from treatment to outcome — the DAG cannot carry "
            "the estimated effect.",
        )
    elif facts.get("mediators"):
        mediators = (
            "unclear",
            f"Mediator(s) on the treatment->outcome path: {', '.join(facts['mediators'])}. "
            "Verify their placement reflects the real mechanism.",
        )
    else:
        mediators = (
            "unclear",
            "No intermediate variables on the treatment->outcome path (direct "
            "edge only); confirm no mediator is missing.",
        )

    sutva = (
        "no_evidence",
        "Interference/spillover cannot be assessed from the DAG or refutation "
        "output; requires domain judgment (e.g. HCP-level spillover across patients).",
    )

    verdict, tests = _test_bucket_verdict(validations, _STABILITY_TESTS)
    if verdict == "supports":
        positivity = (
            verdict,
            f"Estimate stable across subsamples ({', '.join(tests)}) — consistent "
            "with adequate overlap, though overlap is not directly measured.",
        )
    elif verdict == "concern":
        positivity = (
            verdict,
            f"Stability refuter(s) FAILED: {', '.join(tests)} — the estimate moves "
            "materially across subsamples, which can indicate poor overlap.",
        )
    elif verdict == "unclear":
        positivity = (verdict, f"Borderline stability result(s): {', '.join(tests)}.")
    else:
        positivity = (verdict, "No subset/bootstrap stability evidence linked.")

    by_id = {
        "conf_complete": conf,
        "edge_plausible": edge,
        "no_forbidden": forbidden,
        "mediators_correct": mediators,
        "sutva_plausible": sutva,
        "positivity": positivity,
    }
    return [
        {
            "id": q["id"],
            "question": q["question"],
            "verdict": by_id[q["id"]][0],
            "rationale": by_id[q["id"]][1],
        }
        for q in CHECKLIST_QUESTIONS
    ]


# ---------------------------------------------------------------------------
# LM parse + digit vouching
# ---------------------------------------------------------------------------

_DIGIT_RE = re.compile(r"\d+(?:\.\d+)?")
_VERDICT_RE = re.compile(r"^\s*(supports|concern|unclear|no_evidence)\b[\s:—–-]*", re.IGNORECASE)


def _digits_vouched(text: str, grounding_text: str) -> bool:
    grounded = set(_DIGIT_RE.findall(grounding_text))
    return all(d in grounded for d in _DIGIT_RE.findall(text))


def _parse_item(raw: Any, fallback: Dict[str, str], grounding_text: str) -> Dict[str, str]:
    text = str(raw or "").strip()
    match = _VERDICT_RE.match(text)
    if not match:
        return fallback
    rationale = text[match.end() :].strip()
    if not rationale or not _digits_vouched(rationale, grounding_text):
        return fallback
    return {
        "id": fallback["id"],
        "question": fallback["question"],
        "verdict": match.group(1).lower(),
        "rationale": rationale,
    }


def generate_assessment(g: Dict[str, Any]) -> Dict[str, Any]:
    """Grade the six checklist questions; LM when available, else deterministic.

    Per-item honesty: an LM answer without a valid leading verdict token, with
    an empty rationale, or with an unvouched digit is replaced by that item's
    deterministic fallback — a single bad answer never poisons the set.
    """
    fallback = _fallback_items(g)
    pred = run_signature(
        DagReviewAssessmentSignature,
        context=g["context"],
        dag_summary=g["dag_summary"],
        structural_facts=g["structural_facts"],
        refutation_evidence=g["refutation_evidence"],
    )
    evidence = {
        "refutation_tests": g.get("validations_used", 0),
        "has_dag_structure": g.get("has_dag_structure", False),
    }
    if pred is None:
        return {"items": fallback, "is_fallback": True, "evidence": evidence}

    grounding_text = "\n".join(
        (g["context"], g["dag_summary"], g["structural_facts"], g["refutation_evidence"])
    )
    items = [_parse_item(getattr(pred, fb["id"], None), fb, grounding_text) for fb in fallback]
    return {"items": items, "is_fallback": False, "evidence": evidence}
