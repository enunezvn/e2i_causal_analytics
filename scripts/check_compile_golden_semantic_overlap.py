"""Plan-239 §4.3 — semantic-overlap check between compile-set and golden-set.

Detects same-role same-signature derivation collisions between
``build_compile_set()`` entries and the literature golden set. Signature is
``(source, frozenset(derivation_inputs), aggregation, window_days)`` parsed
from the structured derivation_pseudocode shape used by the golden set and by
the 17 plan-239 compile-set additions.

Compile-set entries whose ``derivation_pseudocode`` does not match the
structured regex (e.g., the legacy 33 free-form entries authored pre-#239)
are skipped — the gate only enforces no-near-duplicate on entries that
adopt the structured shape, which is by-construction the plan-239 §3.0
addition set.

Usage::

    python scripts/check_compile_golden_semantic_overlap.py
    # Exit 0 if no unauthorized collisions; 1 otherwise.

Importable function ``find_unauthorized_collisions(...)`` is used by
``tests/unit/test_data/test_causal_role_classifier.py``.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

DEFAULT_GOLDEN_SET = _REPO_ROOT / "tests" / "fixtures" / "causal_role_golden_set.json"
DEFAULT_ALLOWLIST = _REPO_ROOT / "tests" / "fixtures" / "compile_golden_semantic_allowlist.json"

# Match the structured derivation_pseudocode used by the golden set + plan-239
# additions. Mirrors src/ml/causal_role_dgp/golden_set.py
# DERIVATION_PSEUDOCODE_REGEX but exposes named capture groups so we can
# extract (source, derivation_inputs, aggregation, window_days).
_SIGNATURE_REGEX = re.compile(
    r"^source=(?P<source>[A-Za-z_][A-Za-z0-9_]*); "
    r"derivation_inputs=\[(?P<inputs>(?:'[^']*'(?:, '[^']*')*)?)\]; "
    r"aggregation=(?P<aggregation>[A-Za-z_]+|None); "
    r"window_days=(?P<window_days>\d+|None); "
    r"knowable_at=[a-z_]+(?:[+-]\d+d)?$"
)

Signature = tuple[str, frozenset[str], str, str]


def parse_signature(derivation_pseudocode: str) -> Signature | None:
    """Parse a structured derivation_pseudocode into a comparable signature.

    Returns None if the string does not match the structured shape (legacy
    compile-set free-form entries fall into this bucket and are skipped).
    """
    m = _SIGNATURE_REGEX.match(derivation_pseudocode.strip())
    if m is None:
        return None
    inputs_str = m.group("inputs")
    inputs = (
        frozenset(s.strip().strip("'") for s in inputs_str.split(", "))
        if inputs_str
        else frozenset()
    )
    return (m.group("source"), inputs, m.group("aggregation"), m.group("window_days"))


def load_golden_signatures(
    path: Path = DEFAULT_GOLDEN_SET,
) -> dict[Signature, list[dict[str, str]]]:
    """Map signature → list of {feature_name, ground_truth_role} from the golden set."""
    raw = json.loads(path.read_text())
    out: dict[Signature, list[dict[str, str]]] = {}
    for entry in raw["entries"]:
        sig = parse_signature(entry["derivation_pseudocode"])
        if sig is None:
            continue
        out.setdefault(sig, []).append(
            {
                "feature_name": entry["feature_name"],
                "ground_truth_role": entry["ground_truth_role"],
                "cohort": entry.get("cohort", "?"),
            }
        )
    return out


def load_allowlist(path: Path = DEFAULT_ALLOWLIST) -> list[dict[str, Any]]:
    """Return the list of allowlisted (compile_feature, golden_feature) pairs."""
    raw = json.loads(path.read_text())
    allowlist_raw = raw.get("allowlist", [])
    allowlist: list[dict[str, Any]] = list(allowlist_raw)
    # Validate that each allowlist row carries a non-empty justification.
    for row in allowlist:
        if not row.get("justification", "").strip():
            raise ValueError(
                f"Allowlist row {row!r} missing non-empty `justification` (plan-239 §4.3 policy)."
            )
    return allowlist


def _is_allowlisted(
    allowlist: list[dict[str, Any]],
    compile_feature: str,
    golden_feature: str,
) -> bool:
    for row in allowlist:
        if (
            row.get("compile_feature_name") == compile_feature
            and row.get("golden_feature_name") == golden_feature
        ):
            return True
    return False


def find_unauthorized_collisions(
    compile_examples: list[Any],
    *,
    golden_path: Path = DEFAULT_GOLDEN_SET,
    allowlist_path: Path = DEFAULT_ALLOWLIST,
) -> list[dict[str, str]]:
    """Return per-pair collision records (compile vs golden) not in the allowlist.

    Each record: {compile_feature, golden_feature, role, signature_repr}.
    Compares only when compile_entry.causal_role == golden_entry["ground_truth_role"]
    (the field-name discipline from plan-239 §4.1 iter-2 fix).
    """
    golden_sigs = load_golden_signatures(golden_path)
    allowlist = load_allowlist(allowlist_path)

    collisions: list[dict[str, str]] = []
    for ex in compile_examples:
        sig = parse_signature(getattr(ex, "derivation_pseudocode", ""))
        if sig is None:
            continue
        compile_role = getattr(ex, "causal_role", None)
        if compile_role is None:
            continue
        for golden_entry in golden_sigs.get(sig, []):
            if golden_entry["ground_truth_role"] != compile_role:
                continue
            compile_feature = getattr(ex, "feature_name", "?")
            golden_feature = golden_entry["feature_name"]
            if _is_allowlisted(allowlist, compile_feature, golden_feature):
                continue
            collisions.append(
                {
                    "compile_feature": compile_feature,
                    "golden_feature": golden_feature,
                    "role": compile_role,
                    "cohort": golden_entry["cohort"],
                    "signature_repr": (
                        f"source={sig[0]}; inputs={sorted(sig[1])}; "
                        f"aggregation={sig[2]}; window_days={sig[3]}"
                    ),
                }
            )
    return collisions


def main() -> int:
    from src.data.causal_role_classifier import build_compile_set

    collisions = find_unauthorized_collisions(build_compile_set())
    if not collisions:
        print(
            "OK: no unauthorized same-role same-signature collisions between compile-set and golden-set."
        )
        return 0
    print("FAIL: unauthorized collisions detected (plan-239 §4.3):")
    for c in collisions:
        print(
            f"  - compile.{c['compile_feature']} <-> golden.{c['golden_feature']} "
            f"(cohort={c['cohort']}, role={c['role']}); "
            f"signature: {c['signature_repr']}"
        )
    return 1


if __name__ == "__main__":
    sys.exit(main())
