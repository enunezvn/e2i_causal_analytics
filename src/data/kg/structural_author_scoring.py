"""Scoring of authored structure against the literature golden set.

Lane B of the real-data causal estimation program (spec §3 Lane B item 3):
author once on the 91 blind briefs, score once against ``ground_truth_role``;
report per-role precision / recall and the missed-leak rate. **Gate: zero
missed leaks.**

The briefs are the LABEL-FREE projection of the golden set
(``feature_name``, ``derivation_pseudocode``, ``dataset_context``; the
committed CSU blind briefs ``tests/fixtures/causal_role_csu_blind_briefs.json``
are exactly that projection for the CSU cohort, pinned by a test). Roles enter
only from the golden set at scoring time (author-once / score-once).

A *missed leak* is a feature whose golden role is a LEAK role (mediator /
collider / descendant) that the author's derived role puts in the ACCEPT
bucket (ancestor / confounder / instrument) — the safety-critical error the
guide's §4 names. A feature routed to review (no derived role) is neither a
miss nor a hit: it is counted as ``review`` and listed. The bucket constants
are the voter's own (``LEAK_ROLES`` / ``ACCEPT_ROLES``), so this scorer and the
leak decision cannot drift apart.
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

from src.data.kg.ensemble_voter import ACCEPT_ROLES, LEAK_ROLES

__all__ = [
    "BRIEF_FIELDS",
    "ScoreReport",
    "golden_briefs",
    "load_golden_entries",
    "score_roles",
]

#: The label-free fields an author sees (and nothing else).
BRIEF_FIELDS: tuple[str, ...] = ("feature_name", "derivation_pseudocode", "dataset_context")
ROLES: tuple[str, ...] = tuple(sorted(LEAK_ROLES | ACCEPT_ROLES))


def load_golden_entries(path: Path | str) -> list[dict[str, Any]]:
    """The golden set's ``entries`` (91 literature-derived features)."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    entries = payload.get("entries")
    if not isinstance(entries, list) or not entries:
        raise ValueError(f"{path}: no entries")
    for e in entries:
        for key in (*BRIEF_FIELDS, "ground_truth_role", "cohort"):
            if key not in e:
                raise ValueError(f"{path}: entry missing {key!r}: {e.get('feature_name')!r}")
        if e["ground_truth_role"] not in ROLES:
            raise ValueError(f"{path}: unknown role {e['ground_truth_role']!r}")
    return entries


def golden_briefs(
    entries: Sequence[Mapping[str, Any]], *, cohort: Optional[str] = None
) -> list[dict[str, str]]:
    """The label-free projection of the golden entries (what the author sees).

    Keys are exactly :data:`BRIEF_FIELDS` plus ``cohort`` (needed to route the
    brief, carries no label). Order follows the golden set.
    """
    out: list[dict[str, str]] = []
    for e in entries:
        if cohort is not None and e["cohort"] != cohort:
            continue
        brief = {k: str(e[k]) for k in BRIEF_FIELDS}
        brief["cohort"] = str(e["cohort"])
        out.append(brief)
    return out


@dataclass
class RoleMetrics:
    role: str
    support: int
    predicted: int
    tp: int
    precision: Optional[float]
    recall: Optional[float]
    f1: Optional[float]


@dataclass
class ScoreReport:
    n: int
    n_scored: int
    n_review: int
    exact_role_agreement: int
    exact_role_accuracy: Optional[float]
    leak_decision_agreement: int
    leak_decision_accuracy: Optional[float]
    missed_leaks: list[dict[str, str]]
    #: missed leaks over the golden LEAK-role features that were scored (the
    #: false-negative rate of the leak decision), not over every feature.
    missed_leak_rate: Optional[float]
    n_leak_truth: int
    n_leak_scored: int
    conservative_errors: list[dict[str, str]]
    review: list[dict[str, str]]
    per_role: list[RoleMetrics]
    per_cohort: dict[str, dict[str, Any]]
    confusion: dict[str, dict[str, int]]
    gate_passed: bool
    gate: str = "missed_leaks == 0"
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def summary_lines(self) -> list[str]:
        acc = "n/a" if self.exact_role_accuracy is None else f"{self.exact_role_accuracy:.3f}"
        leak = (
            "n/a" if self.leak_decision_accuracy is None else f"{self.leak_decision_accuracy:.3f}"
        )
        mlr = "n/a" if self.missed_leak_rate is None else f"{self.missed_leak_rate:.3f}"
        verdict = "PASS" if self.gate_passed else "FAIL"
        lines = [
            f"{verdict}: gate {self.gate} — missed leaks {len(self.missed_leaks)} "
            f"(rate {mlr} over {self.n_leak_scored} scored golden-leak features; "
            f"{self.n_scored} scored, {self.n_review} routed to review, n={self.n})",
            f"exact role agreement {self.exact_role_agreement}/{self.n_scored} ({acc}); "
            f"leak-decision agreement {self.leak_decision_agreement}/{self.n_scored} ({leak}); "
            f"conservative errors {len(self.conservative_errors)}",
        ]
        for m in self.per_role:
            p = "n/a" if m.precision is None else f"{m.precision:.2f}"
            r = "n/a" if m.recall is None else f"{m.recall:.2f}"
            lines.append(
                f"  {m.role:<11} support={m.support:<3} predicted={m.predicted:<3} "
                f"tp={m.tp:<3} precision={p} recall={r}"
            )
        for cohort, row in sorted(self.per_cohort.items()):
            lines.append(
                f"  cohort {cohort}: n={row['n']} scored={row['n_scored']} "
                f"exact={row['exact_role_agreement']} missed_leaks={row['missed_leaks']} "
                f"review={row['n_review']}"
            )
        return lines


def _ratio(num: int, den: int) -> Optional[float]:
    return None if den == 0 else num / den


def _bucket(role: Optional[str]) -> Optional[str]:
    if role is None:
        return None
    if role in LEAK_ROLES:
        return "leak"
    if role in ACCEPT_ROLES:
        return "accept"
    raise ValueError(f"unknown role {role!r}")


def score_roles(
    predicted: Mapping[str, Optional[str]],
    entries: Sequence[Mapping[str, Any]],
) -> ScoreReport:
    """Score derived roles against the golden labels.

    ``predicted`` maps ``(cohort, feature_name)`` joined as ``"<cohort>/<feature>"``
    — or, when unambiguous, the bare ``feature_name`` — to the DERIVED role
    (``None`` = routed to review). Every golden entry must have a key; a
    missing key is an error (an author that silently skipped a brief must not
    score as if it had reviewed it).
    """
    n = len(entries)
    exact = 0
    leak_agree = 0
    missed: list[dict[str, str]] = []
    conservative: list[dict[str, str]] = []
    review: list[dict[str, str]] = []
    confusion: dict[str, dict[str, int]] = {r: dict.fromkeys(ROLES, 0) for r in ROLES}
    support: Counter[str] = Counter()
    pred_count: Counter[str] = Counter()
    tp: Counter[str] = Counter()
    per_cohort: dict[str, dict[str, Any]] = {}

    for e in entries:
        cohort = str(e["cohort"])
        feat = str(e["feature_name"])
        key = f"{cohort}/{feat}"
        if key in predicted:
            pred = predicted[key]
        elif feat in predicted:
            pred = predicted[feat]
        else:
            raise KeyError(f"no prediction for {key!r} (author skipped a brief?)")
        truth = str(e["ground_truth_role"])
        row = per_cohort.setdefault(
            cohort,
            {"n": 0, "n_scored": 0, "n_review": 0, "exact_role_agreement": 0, "missed_leaks": 0},
        )
        row["n"] += 1
        support[truth] += 1
        if pred is None:
            review.append({"cohort": cohort, "feature_name": feat, "ground_truth_role": truth})
            row["n_review"] += 1
            continue
        if pred not in ROLES:
            raise ValueError(f"{key}: predicted role {pred!r} is not a role")
        row["n_scored"] += 1
        pred_count[pred] += 1
        confusion[truth][pred] += 1
        if pred == truth:
            exact += 1
            tp[pred] += 1
            row["exact_role_agreement"] += 1
        tb, pb = _bucket(truth), _bucket(pred)
        if tb == pb:
            leak_agree += 1
        elif tb == "leak":
            missed.append(
                {
                    "cohort": cohort,
                    "feature_name": feat,
                    "ground_truth_role": truth,
                    "derived_role": pred,
                }
            )
            row["missed_leaks"] += 1
        else:
            conservative.append(
                {
                    "cohort": cohort,
                    "feature_name": feat,
                    "ground_truth_role": truth,
                    "derived_role": pred,
                }
            )

    n_review = len(review)
    n_scored = n - n_review
    n_leak_truth = sum(1 for e in entries if _bucket(str(e["ground_truth_role"])) == "leak")
    n_leak_scored = n_leak_truth - sum(
        1 for x in review if _bucket(x["ground_truth_role"]) == "leak"
    )
    per_role = []
    for role in ROLES:
        p = _ratio(tp[role], pred_count[role])
        r = _ratio(
            tp[role], support[role] - sum(1 for x in review if x["ground_truth_role"] == role)
        )
        f1 = None if p is None or r is None or (p + r) == 0 else 2 * p * r / (p + r)
        per_role.append(
            RoleMetrics(
                role=role,
                support=support[role],
                predicted=pred_count[role],
                tp=tp[role],
                precision=p,
                recall=r,
                f1=f1,
            )
        )
    notes: list[str] = []
    if n_review:
        notes.append(
            f"{n_review} feature(s) routed to review are excluded from precision/recall "
            "(neither a hit nor a miss); they are listed under 'review'."
        )
    return ScoreReport(
        n=n,
        n_scored=n_scored,
        n_review=n_review,
        exact_role_agreement=exact,
        exact_role_accuracy=_ratio(exact, n_scored),
        leak_decision_agreement=leak_agree,
        leak_decision_accuracy=_ratio(leak_agree, n_scored),
        missed_leaks=missed,
        missed_leak_rate=_ratio(len(missed), n_leak_scored),
        n_leak_truth=n_leak_truth,
        n_leak_scored=n_leak_scored,
        conservative_errors=conservative,
        review=review,
        per_role=per_role,
        per_cohort=per_cohort,
        confusion=confusion,
        gate_passed=len(missed) == 0,
        notes=notes,
    )
