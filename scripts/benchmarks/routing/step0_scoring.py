"""Step 0 candidate scoring — pure functions (#1337).

Everything here is deterministic and env-free so it can be unit-tested
red-first. LLM/network/node access lives in step0_candidates.py; the CLI
lives in score_candidates.py.

Vocabulary:
- ``gold_pattern`` / ``pred_pattern``: one of the four RoutingPattern values
  (SINGLE_AGENT, PARALLEL_DELEGATION, TOOL_COMPOSER, CLARIFICATION_NEEDED).
- ``agents``: contract agent names; compared as sets (order/dupes ignored).
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from typing import Any, Dict, FrozenSet, Iterable, List, Optional, Tuple

VALID_PATTERNS = (
    "SINGLE_AGENT",
    "PARALLEL_DELEGATION",
    "TOOL_COMPOSER",
    "CLARIFICATION_NEEDED",
)

# =============================================================================
# Row scoring
# =============================================================================


@dataclass(frozen=True)
class RowScore:
    pattern_correct: bool
    agents_exact: bool
    agents_jaccard: float


def _agent_set(agents: Optional[Iterable[str]]) -> FrozenSet[str]:
    return frozenset(a.strip() for a in (agents or []) if a and a.strip())


def score_row(
    gold_pattern: str,
    gold_agents: Optional[Iterable[str]],
    pred_pattern: str,
    pred_agents: Optional[Iterable[str]],
) -> RowScore:
    """Score one prediction against gold."""
    g, p = _agent_set(gold_agents), _agent_set(pred_agents)
    if not g and not p:
        jaccard = 1.0
    elif not g or not p:
        jaccard = 0.0
    else:
        jaccard = len(g & p) / len(g | p)
    return RowScore(
        pattern_correct=(gold_pattern == pred_pattern),
        agents_exact=(g == p),
        agents_jaccard=jaccard,
    )


# =============================================================================
# Wilson 95% CI
# =============================================================================


def wilson_ci(k: int, n: int, z: float = 1.959964) -> Tuple[float, float]:
    """Wilson score interval for a binomial proportion; (0, 1) when n == 0."""
    if n == 0:
        return (0.0, 1.0)
    phat = k / n
    denom = 1 + z * z / n
    center = (phat + z * z / (2 * n)) / denom
    half = (z / denom) * math.sqrt(phat * (1 - phat) / n + z * z / (4 * n * n))
    lo, hi = center - half, center + half
    if k == 0:
        lo = 0.0
    if k == n:
        hi = 1.0
    return (max(0.0, lo), min(1.0, hi))


# =============================================================================
# Aggregation
# =============================================================================


def confusion_matrix(rows: List[Dict[str, Any]]) -> Dict[Tuple[str, str], int]:
    """(gold_pattern, pred_pattern) -> count."""
    cm: Dict[Tuple[str, str], int] = {}
    for r in rows:
        key = (r["gold_pattern"], r["pred_pattern"])
        cm[key] = cm.get(key, 0) + 1
    return cm


def _per_pattern_stats(rows: List[Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
    out: Dict[str, Dict[str, float]] = {}
    for pattern in VALID_PATTERNS:
        gold_n = sum(1 for r in rows if r["gold_pattern"] == pattern)
        pred_n = sum(1 for r in rows if r["pred_pattern"] == pattern)
        hits = sum(1 for r in rows if r["gold_pattern"] == pattern and r["pred_pattern"] == pattern)
        recall = hits / gold_n if gold_n else 0.0
        precision = hits / pred_n if pred_n else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        out[pattern] = {
            "gold_n": float(gold_n),
            "pred_n": float(pred_n),
            "recall": recall,
            "precision": precision,
            "f1": f1,
        }
    return out


def aggregate(rows: List[Dict[str, Any]], slice_key: Optional[str] = None) -> Dict[str, Any]:
    """Aggregate scored rows into the summary block for one candidate.

    Each row needs: gold_pattern, pred_pattern, score (RowScore), plus any
    slice keys referenced by ``slice_key``.
    """
    n = len(rows)
    hits = sum(1 for r in rows if r["score"].pattern_correct)
    agents_exact = sum(1 for r in rows if r["score"].agents_exact)
    jaccard_sum = sum(r["score"].agents_jaccard for r in rows)
    agg: Dict[str, Any] = {
        "n": n,
        "pattern_accuracy": hits / n if n else 0.0,
        "pattern_accuracy_ci95": wilson_ci(hits, n),
        "agents_exact_rate": agents_exact / n if n else 0.0,
        "agents_jaccard_mean": jaccard_sum / n if n else 0.0,
        "per_pattern": _per_pattern_stats(rows),
        "confusion": {f"{g}->{p}": c for (g, p), c in sorted(confusion_matrix(rows).items())},
    }
    if slice_key:
        slices: Dict[str, Any] = {}
        values = sorted({str(r.get(slice_key)) for r in rows})
        for v in values:
            sub = [r for r in rows if str(r.get(slice_key)) == v]
            k = sum(1 for r in sub if r["score"].pattern_correct)
            slices[v] = {
                "n": len(sub),
                "pattern_accuracy": k / len(sub) if sub else 0.0,
                "pattern_accuracy_ci95": wilson_ci(k, len(sub)),
            }
        agg["slices"] = slices
    return agg


# =============================================================================
# Legacy pattern derivation
# =============================================================================


def derive_legacy_pattern(primary_intent: str, agent_names: List[str]) -> str:
    """Map the incumbent's (intent, dispatch agents) onto the 4-pattern space.

    The legacy path has no abstention concept, so CLARIFICATION_NEEDED is
    never produced — scoring it as a class is exactly the point (#1337).
    """
    if primary_intent == "multi_faceted" or "tool_composer" in agent_names:
        return "TOOL_COMPOSER"
    if len(set(agent_names)) > 1:
        return "PARALLEL_DELEGATION"
    return "SINGLE_AGENT"


# =============================================================================
# LLM-candidate output parsing
# =============================================================================

_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL)


def parse_candidate_json(text: str, known_agents: FrozenSet[str]) -> Optional[Dict[str, Any]]:
    """Parse an LLM candidate reply into {routing_pattern, target_agents, confidence}.

    Returns None when the reply is unusable (no JSON / invalid pattern) so the
    runner can count parse failures explicitly rather than mis-scoring them.
    Unknown agent names are dropped; confidence is clamped to [0, 1].
    """
    raw = text.strip()
    m = _FENCE_RE.search(raw)
    if m:
        raw = m.group(1).strip()
    else:
        # Bare object: take the first {...} span if there is leading prose.
        start, end = raw.find("{"), raw.rfind("}")
        if start == -1 or end <= start:
            return None
        raw = raw[start : end + 1]
    try:
        data = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return None
    if not isinstance(data, dict):
        return None
    pattern = data.get("routing_pattern")
    if pattern not in VALID_PATTERNS:
        return None
    agents_raw = data.get("target_agents") or []
    if not isinstance(agents_raw, list):
        return None
    agents = [a for a in agents_raw if isinstance(a, str) and a in known_agents]
    try:
        confidence = float(data.get("confidence", 0.0))
    except (TypeError, ValueError):
        confidence = 0.0
    confidence = min(1.0, max(0.0, confidence))
    return {
        "routing_pattern": pattern,
        "target_agents": agents,
        "confidence": confidence,
    }


# =============================================================================
# Disagreement worksheet
# =============================================================================


def disagreement_rows(rows_by_candidate: Dict[str, List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
    """Rows where at least one candidate's pattern misses gold.

    Output is keyed by query_id with a per-candidate {pred, correct} map —
    the protocol requires human review of candidate disagreements.
    """
    by_qid: Dict[str, Dict[str, Any]] = {}
    for cand, rows in rows_by_candidate.items():
        for r in rows:
            entry = by_qid.setdefault(
                r["query_id"],
                {
                    "query_id": r["query_id"],
                    "gold_pattern": r["gold_pattern"],
                    "text": r.get("text", ""),
                    "candidates": {},
                },
            )
            entry["candidates"][cand] = {
                "pred_pattern": r["pred_pattern"],
                "correct": r["score"].pattern_correct,
            }
    return [
        e
        for _, e in sorted(by_qid.items())
        if any(not c["correct"] for c in e["candidates"].values())
    ]


# =============================================================================
# Contract cards (shared knowledge block for the LLM candidates)
# =============================================================================


def contract_cards_from_registry(registry: Dict[str, Any]) -> str:
    """Render the contract registry into one compact line per agent.

    Both LLM candidates receive the same card block so the comparison
    isolates architecture (staged rules + LLM vs single call), not knowledge.
    """
    lines: List[str] = []
    for name, spec in sorted((registry.get("agents") or {}).items()):
        covers = ", ".join(spec.get("covers") or [])
        not_covers = ", ".join(spec.get("does_not_cover") or [])
        line = f"- {name}: {covers}"
        if not_covers:
            line += f" | NOT: {not_covers}"
        lines.append(line)
    return "\n".join(lines)
