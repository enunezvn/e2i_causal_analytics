"""Plan v4 Gate G2 — pre-spec dataset-hash verifier.

Closes G2's threshold-shopping defense for cohort drift between
``S_prespec`` (the commit that introduces the pre-spec memo
``docs/specs/tier1b_b2_prespec_20260510.md``) and the experiment commit.
The memo pins sha256 values for the cohort artifacts referenced by the
G2 experiment harness; this script re-computes the live hashes and
compares them to the pinned values. Mismatch is a hard failure.

Usage
-----

Verify (CI mode):
    python scripts/verify_g2_prespec_dataset_hashes.py
    # exits non-zero on hash mismatch OR on a present-artifact-vs-
    # placeholder mismatch (the operator forgot to run --update before
    # tagging the experiment).

Update (operator-only, used at first green run when cohort first
lands on disk):
    python scripts/verify_g2_prespec_dataset_hashes.py --update
    # writes live sha256 values into the memo (in-place edit) and
    # exits zero. The operator MUST commit the change as a separate
    # PR diff that the threshold-shopping audit can review (no
    # threshold edits in the same commit).

The script is intentionally dependency-free (stdlib only) and uses
plain string substitution to minimize the chance of an audit-breaking
edit. Mirrors ``scripts/verify_g5_prespec_hashes.py`` exactly so the
audit reviewer only needs to verify ONE pattern.

Behaviour on missing artifact
-----------------------------

If a pinned artifact is absent:
  * Verify mode: treats as a soft skip and prints a warning; the
    experiment harness's M2 fixture is the load-bearing CI gate for
    "is this cohort artifact required to be present?"
  * Update mode: prints a warning and leaves the placeholder in the
    memo unchanged.

This script is a defense-in-depth secondary check; the experiment
harness's CI=true cohort-presence guard is the primary CI gate.
"""

from __future__ import annotations

import argparse
import hashlib
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
MEMO_PATH = REPO_ROOT / "docs" / "specs" / "tier1b_b2_prespec_20260510.md"
PLACEHOLDER = "TODO_PIN_AT_FIRST_GREEN_RUN"

# Map: short label → (relative path within REPO_ROOT, memo key).
# The memo key is the YAML key used in the pinned-hashes block.
ARTIFACTS: Dict[str, Tuple[str, str]] = {
    "optum_initiation_patient_journeys_parquet": (
        "data/rwd/optum/initiation/e2i_ml_v3_patient_journeys.parquet",
        "optum_initiation_patient_journeys_parquet",
    ),
    "optum_initiation_treatment_events_parquet": (
        "data/rwd/optum/initiation/e2i_ml_v3_treatment_events.parquet",
        "optum_initiation_treatment_events_parquet",
    ),
}


def _sha256_of(path: Path) -> str:
    """sha256-hexdigest a file in 64KB chunks (memory-efficient on
    large parquet artifacts)."""
    hasher = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(64 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _parse_pinned_hashes(memo_text: str) -> Dict[str, Optional[str]]:
    """Parse the YAML-ish pinned-hashes block from the memo. Returns
    {memo_key: pinned_sha256_or_None_if_placeholder}.

    Tolerates the dual indentation patterns that markdown-rendered YAML
    produces.
    """
    pinned: Dict[str, Optional[str]] = {}
    for memo_key in ARTIFACTS:
        # Match: ``  <memo_key>:\n    path: "..."\n    sha256: "..."\n``
        # Anchored on the memo_key prefix; non-greedy to avoid bleed.
        pattern = re.compile(
            rf"^\s*{re.escape(memo_key)}:\s*\n\s*path:.*\n\s*sha256:\s*\"([^\"]*)\"",
            re.MULTILINE,
        )
        m = pattern.search(memo_text)
        if m is None:
            pinned[memo_key] = None
            continue
        value = m.group(1).strip()
        pinned[memo_key] = None if value == PLACEHOLDER else value
    return pinned


def _replace_pinned_hash(memo_text: str, memo_key: str, new_hash: str) -> str:
    """Replace the placeholder for ``memo_key`` with ``new_hash``. If
    a value is already present (non-placeholder), leave it alone (the
    operator must explicitly delete + re-pin to satisfy the
    threshold-shopping audit)."""
    pattern = re.compile(
        rf"(^\s*{re.escape(memo_key)}:\s*\n\s*path:.*\n\s*sha256:\s*\")"
        rf"({re.escape(PLACEHOLDER)})(\")",
        re.MULTILINE,
    )

    def _sub(match: re.Match[str]) -> str:
        return f"{match.group(1)}{new_hash}{match.group(3)}"

    return pattern.sub(_sub, memo_text, count=1)


def _verify(pinned: Dict[str, Optional[str]]) -> int:
    """Verify mode: re-compute live hashes and compare to pinned. Exit
    non-zero if any mismatch."""
    failures: List[str] = []
    skipped: List[str] = []
    verified: List[str] = []

    for memo_key, (relpath, _) in ARTIFACTS.items():
        path = REPO_ROOT / relpath
        if not path.exists():
            skipped.append(f"{memo_key}: artifact missing at {relpath}")
            continue

        live = _sha256_of(path)
        pinned_value = pinned.get(memo_key)
        if pinned_value is None:
            failures.append(
                f"{memo_key}: pinned value is {PLACEHOLDER!r} but artifact "
                f"is present (live sha256={live}). Run with --update to pin."
            )
            continue
        if live != pinned_value:
            failures.append(
                f"{memo_key}: HASH MISMATCH\n"
                f"  pinned: {pinned_value}\n"
                f"  live:   {live}\n"
                f"  artifact: {relpath}\n"
                f"  Resolution: cohort drifted between memo-lock and run; "
                "either revert the cohort or write a new "
                "tier1b_b2_prespec_<date>.md memo at a fresh date."
            )
            continue
        verified.append(f"{memo_key}: OK (sha256={live[:16]}…)")

    print("=" * 70)
    print("G2 pre-spec dataset-hash verification")
    print("=" * 70)
    for line in verified:
        print(f"  [OK]   {line}")
    for line in skipped:
        print(f"  [SKIP] {line}")
    for line in failures:
        print(f"  [FAIL] {line}")
    print()
    if failures:
        print(f"FAILED: {len(failures)} hash mismatch(es)")
        return 1
    if skipped and not verified:
        print(
            "WARNING: every artifact is missing on disk; verification "
            "vacuously passed. The experiment harness's CI-presence "
            "guard is the primary gate."
        )
    print("OK: all present artifacts match pinned hashes")
    return 0


def _update(pinned: Dict[str, Optional[str]]) -> int:
    """Update mode: write live hashes into the memo's TODO placeholders.

    Refuses to overwrite non-placeholder values (the operator must
    delete the old hash + re-pin in a separate diff so the
    threshold-shopping audit can see the action).
    """
    if not MEMO_PATH.exists():
        print(f"FATAL: spec memo not found at {MEMO_PATH}", file=sys.stderr)
        return 2

    memo_text = MEMO_PATH.read_text(encoding="utf-8")
    updated = 0
    skipped: List[str] = []

    for memo_key, (relpath, _) in ARTIFACTS.items():
        path = REPO_ROOT / relpath
        if not path.exists():
            skipped.append(f"{memo_key}: artifact missing at {relpath}")
            continue

        existing = pinned.get(memo_key)
        if existing is not None:
            skipped.append(
                f"{memo_key}: already pinned ({existing[:16]}…); "
                "delete the existing value to re-pin via a fresh memo."
            )
            continue

        live = _sha256_of(path)
        memo_text_new = _replace_pinned_hash(memo_text, memo_key, live)
        if memo_text_new == memo_text:
            skipped.append(
                f"{memo_key}: regex did not match the memo's pinned-hashes "
                "block; check memo formatting"
            )
            continue
        memo_text = memo_text_new
        print(f"  [PIN]  {memo_key}: sha256={live}")
        updated += 1

    if skipped:
        print()
        for line in skipped:
            print(f"  [SKIP] {line}")

    if updated:
        MEMO_PATH.write_text(memo_text, encoding="utf-8")
        print(f"\nUpdated {updated} hash(es) in {MEMO_PATH}")
    else:
        print("\nNo updates applied.")
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--update",
        action="store_true",
        help="Write live sha256 values into the memo's TODO placeholders.",
    )
    args = parser.parse_args(argv)

    if not MEMO_PATH.exists():
        print(f"FATAL: spec memo not found at {MEMO_PATH}", file=sys.stderr)
        return 2

    memo_text = MEMO_PATH.read_text(encoding="utf-8")
    pinned = _parse_pinned_hashes(memo_text)

    if args.update:
        return _update(pinned)
    return _verify(pinned)


if __name__ == "__main__":
    sys.exit(main())
