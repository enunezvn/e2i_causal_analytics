"""Plan v4 Gate G2 — pre-spec dataset-hash verifier.

Closes G2's threshold-shopping defense for cohort drift between
``S_prespec`` (the commit that introduces the pre-spec memo
``docs/specs/tier1b_b2_prespec_20260510.md``) and the experiment commit.
The memo pins sha256 values for the cohort artifacts referenced by the
G2 experiment harness; this script re-computes the live hashes and
compares them to the pinned values. Mismatch is a hard failure.

Usage
-----

Verify (CI mode — STRICT):
    CI=true python scripts/verify_g2_prespec_dataset_hashes.py
    # In CI, MISSING artifacts are a HARD FAILURE: an absent cohort
    # parquet means the verifier cannot compute the live hash, which
    # means the pinned hash cannot be verified, which means the
    # threshold-shopping defense's data-content half is unobservable.
    # This is the codex pass-1 HIGH-2 fix: vacuous-pass on missing
    # artifacts is no longer permitted in CI.

Verify (local diagnostic — LENIENT):
    python scripts/verify_g2_prespec_dataset_hashes.py --allow-missing
    # When CI=false (or unset) AND --allow-missing is passed, missing
    # artifacts are reported as [SKIP] and the verifier exits 0 if no
    # explicit failures occurred. This path exists so an operator on
    # a fresh checkout can run the verifier without the cohort-on-disk
    # for a quick sanity check; the load-bearing CI gate enforces
    # presence.

Verify (explicit strict mode — overrides CI=false):
    python scripts/verify_g2_prespec_dataset_hashes.py --strict
    # Forces present-artifact mode regardless of the CI env var.

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

Strict-mode resolution
----------------------

The verifier's strict mode is the OR of:
  * ``CI=true`` (or ``CI=1`` / ``CI=yes``) — the CI-controlled first
    execution gate.
  * ``--strict`` flag — explicit operator opt-in.

In strict mode, ANY missing artifact is reported as a HARD FAILURE
(exit 1). ``--allow-missing`` is REJECTED in strict mode so an
operator cannot accidentally pass it in CI.

Threshold-shopping defense (HIGH-3)
-----------------------------------

When invoked with ``--prespec-sha <SHA>``, the verifier loads pinned
hashes from the memo content AT THAT SHA via ``git show
<SHA>:docs/specs/tier1b_b2_prespec_20260510.md`` instead of the working
copy. This defeats a threshold-shopper who edits the memo's pinned
hashes in a child commit; the immutable S_prespec content is the source
of truth.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def _resolve_repo_root() -> Path:
    """Resolve the actual git worktree root.

    NEW HIGH-2 (iter-3) fix: when the workflow copies this script into
    ``governance_checkout/scripts/`` (HIGH-6 protected verifier
    staging), ``Path(__file__).resolve().parents[1]`` resolves to
    ``governance_checkout``, NOT the actual worktree the workflow
    checked out. Subsequent ``REPO_ROOT / "docs/..."``,
    ``REPO_ROOT / "data/..."`` paths then point at empty
    ``governance_checkout/docs/...`` and ``governance_checkout/data/...``
    (which never exist), and the verifier silently miscategorizes
    every artifact as MISSING / fails to find the memo.

    Resolution order:
      1. ``E2I_GOVERNANCE_REPO_ROOT`` env var — explicit override.
      2. ``git rev-parse --show-toplevel`` from CWD — preferred.
      3. ``Path(__file__).resolve().parents[1]`` — legacy fallback.

    The legacy path is kept so unit tests that ``monkeypatch.setattr(V,
    "REPO_ROOT", tmp_path)`` continue to work; production CI invocations
    rely on (1) or (2).
    """
    env_root = os.environ.get("E2I_GOVERNANCE_REPO_ROOT", "").strip()
    if env_root:
        return Path(env_root).resolve()
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            check=True,
            capture_output=True,
            text=True,
        )
        return Path(result.stdout.strip()).resolve()
    except (FileNotFoundError, subprocess.CalledProcessError):
        return Path(__file__).resolve().parents[1]


REPO_ROOT = _resolve_repo_root()
MEMO_PATH = REPO_ROOT / "docs" / "specs" / "tier1b_b2_prespec_20260510.md"
MEMO_RELPATH = "docs/specs/tier1b_b2_prespec_20260510.md"
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


def _git_show(sha: str, relpath: str, *, cwd: Optional[Path] = None) -> Optional[str]:
    """Return the contents of ``relpath`` at ``sha`` via ``git show``.

    Returns None on git error. The HIGH-3 fix uses this to load the
    memo content from the immutable ``S_prespec`` SHA instead of the
    mutable working tree, so a child commit cannot edit the pinned
    hash block and bypass verification.
    """
    try:
        result = subprocess.run(
            ["git", "show", f"{sha}:{relpath}"],
            cwd=str(cwd or REPO_ROOT),
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def _discover_s_prespec_sha(*, cwd: Optional[Path] = None) -> Optional[str]:
    """Return the SHA that introduced the pre-spec memo.

    Mirrors ``check_g2_commit_graph._discover_introducing_commit`` but
    inlined here so this verifier remains stdlib-only and dependency-
    free of the sibling script.
    """
    try:
        result = subprocess.run(
            [
                "git",
                "log",
                "--diff-filter=A",
                "--follow",
                "--format=%H",
                "--",
                MEMO_RELPATH,
            ],
            cwd=str(cwd or REPO_ROOT),
            check=True,
            capture_output=True,
            text=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None
    lines = [ln.strip() for ln in result.stdout.splitlines() if ln.strip()]
    if not lines:
        return None
    return lines[-1]


# Load-bearing memo sections that, if edited after S_prespec, MUST
# invalidate the run. Per-key substring matching kept as a
# defense-in-depth backstop so a token rename is also surfaced.
_MEMO_LOAD_BEARING_KEYS: Tuple[str, ...] = (
    "G2_DELTA_AUC_MIN",
    "G2_ECE_RATIO_MAX",
    "G2_CV_STABILITY_RATIO_MAX",
    "G2_SEEDS",
    "g2_dataset_hashes",
    "optum_initiation_default",
    "optum_initiation_relaxed",
    "treatment_initiated",
)


# Regex extractors for the canonical load-bearing values. Each yields
# a normalized form so that whitespace, ordering, or prose edits do
# NOT register as drift, but a value change DOES.
#
# HIGH-3 (iter-3) — codex pass-2 raised that token-line comparison
# misses multi-line load-bearing edits. The structured extractors
# below produce a JSON fingerprint of the load-bearing values; the
# fingerprint comparison is the primary signal, and the per-token
# line check from pass-1 is kept as a backstop for rename detection.
# Match either ``G2_DELTA_AUC_MIN: float = 0.03`` or
# ``G2_DELTA_AUC_MIN = 0.03`` or any whitespace-tolerant variant.
_RE_T1 = re.compile(r"G2_DELTA_AUC_MIN[^=\n]*=\s*([0-9.]+)")
_RE_T2 = re.compile(r"G2_ECE_RATIO_MAX[^=\n]*=\s*([0-9.]+)")
_RE_T3 = re.compile(r"G2_CV_STABILITY_RATIO_MAX[^=\n]*=\s*([0-9.]+)")
_RE_SEEDS = re.compile(r"G2_SEEDS[^=\n]*=\s*[\[(]([0-9,\s]+)[\])]")
_RE_DELTA_INEQ = re.compile(r"`?\s*Δ?_?AUC\s*≥\s*([0-9.]+)\s*`?", re.IGNORECASE)
_RE_ECE_INEQ = re.compile(r"ECE_post\s*≤\s*([0-9.]+)\s*[×x*]\s*ECE_pre", re.IGNORECASE)
_RE_CV_INEQ = re.compile(
    r"\(?std/mean\)?_post\s*≤\s*([0-9.]+)\s*[×x*]\s*\(?std/mean\)?_pre",
    re.IGNORECASE,
)
_RE_COHORT_LABEL = re.compile(
    r"Cohort label:\*?\*?\s*`([a-z_0-9]+)`",
    re.IGNORECASE,
)
_RE_PATIENT_COUNT = re.compile(
    r"Patient[- ]journey count:\*?\*?\s*([0-9]+)",
    re.IGNORECASE,
)
_RE_TARGET_COL = re.compile(
    r"Target column:\*?\*?\s*`([a-z_0-9]+)`",
    re.IGNORECASE,
)


def _extract_load_bearing_fingerprint(memo_text: str) -> Dict[str, Any]:
    """Extract a canonical fingerprint of the memo's load-bearing values.

    HIGH-3 (iter-3) fix: replaces pass-1's per-line token comparison
    (which only captures the LINE containing a token, missing
    multi-line load-bearing edits) with a structured extraction. The
    fingerprint covers:

      * Pre-spec threshold constants (T1, T2, T3) — both code-block
        form (``G2_DELTA_AUC_MIN: float = 0.03``) and prose-inequality
        form (``Δ_AUC ≥ 0.03``).
      * Seeds list.
      * Pinned cohort sha256s — sourced from
        ``_parse_pinned_hashes`` (already canonicalized).
      * Cohort labels + patient-journey counts + target column.

    Two memos with identical fingerprints are equivalent for the
    threshold-shopping defense; differing fingerprints are drift.
    """
    fp: Dict[str, Any] = {}
    # Threshold constants — code block.
    m = _RE_T1.search(memo_text)
    fp["t1_code"] = float(m.group(1)) if m else None
    m = _RE_T2.search(memo_text)
    fp["t2_code"] = float(m.group(1)) if m else None
    m = _RE_T3.search(memo_text)
    fp["t3_code"] = float(m.group(1)) if m else None
    # Threshold inequalities — prose form (catches edits that change
    # the inequality without touching the code-block constant).
    fp["t1_prose"] = sorted({m.group(1) for m in _RE_DELTA_INEQ.finditer(memo_text)})
    fp["t2_prose"] = sorted({m.group(1) for m in _RE_ECE_INEQ.finditer(memo_text)})
    fp["t3_prose"] = sorted({m.group(1) for m in _RE_CV_INEQ.finditer(memo_text)})
    # Seeds list — normalized (sorted ints).
    m = _RE_SEEDS.search(memo_text)
    if m:
        seeds_raw = m.group(1)
        seeds_parts = [s.strip() for s in seeds_raw.split(",") if s.strip()]
        try:
            fp["seeds"] = sorted(int(s) for s in seeds_parts)
        except ValueError:
            fp["seeds"] = seeds_parts
    else:
        fp["seeds"] = None
    # Cohort identifiers (sorted to be order-insensitive).
    fp["cohort_labels"] = sorted(set(_RE_COHORT_LABEL.findall(memo_text)))
    fp["cohort_patient_counts"] = sorted(set(_RE_PATIENT_COUNT.findall(memo_text)))
    fp["target_columns"] = sorted(set(_RE_TARGET_COL.findall(memo_text)))
    # Pinned hashes — re-use the existing parser. Map keys are sorted
    # by virtue of dict-of-string-keys; values are placeholder ("None")
    # or sha256.
    pinned = _parse_pinned_hashes(memo_text)
    fp["pinned_hashes"] = {k: pinned.get(k) for k in sorted(pinned.keys())}
    return fp


def _check_memo_unchanged_since_prespec(
    *,
    s_prespec_sha: str,
    cwd: Optional[Path] = None,
) -> Tuple[bool, List[str]]:
    """HIGH-3 secondary check: verify that load-bearing memo content
    has not changed between ``S_prespec`` and the working tree.

    Returns ``(is_unchanged, [diagnostic_messages])``.

    HIGH-3 (iter-3) implementation: structured comparison of the
    load-bearing FINGERPRINT extracted from each memo, plus the
    per-token line backstop from pass-1 for rename detection.
    Whitespace, prose, ordering, or section-reorganization edits do
    NOT register as drift; threshold-value, seed, hash, cohort-count,
    or target-column edits DO.
    """
    snapshot = _git_show(s_prespec_sha, MEMO_RELPATH, cwd=cwd)
    if snapshot is None:
        return False, [
            f"could not load memo content at S_prespec={s_prespec_sha} via git show; "
            "the memo content unchanged check cannot run",
        ]
    if not MEMO_PATH.exists():
        return False, [f"working-copy memo missing at {MEMO_PATH}"]
    current = MEMO_PATH.read_text(encoding="utf-8")
    diagnostics: List[str] = []

    # Primary: structured fingerprint comparison.
    snap_fp = _extract_load_bearing_fingerprint(snapshot)
    curr_fp = _extract_load_bearing_fingerprint(current)
    if snap_fp != curr_fp:
        diff_keys = sorted(
            k for k in (set(snap_fp) | set(curr_fp)) if snap_fp.get(k) != curr_fp.get(k)
        )
        for key in diff_keys:
            diagnostics.append(
                f"load-bearing fingerprint key {key!r} differs between "
                f"S_prespec and working tree:\n"
                f"  S_prespec value: {snap_fp.get(key)!r}\n"
                f"  current value:   {curr_fp.get(key)!r}"
            )

    # Backstop: per-token presence-of-line comparison. Catches the
    # case where a token is REMOVED entirely (so the regex extractor
    # returns None on both sides and the fingerprint matches by
    # mutual absence). The per-token check distinguishes "removed
    # everywhere" from "present in S_prespec but removed in current".
    for key in _MEMO_LOAD_BEARING_KEYS:
        snap_count = sum(1 for ln in snapshot.splitlines() if key in ln)
        curr_count = sum(1 for ln in current.splitlines() if key in ln)
        if snap_count != curr_count:
            diagnostics.append(
                f"load-bearing token {key!r} occurrence count differs: "
                f"S_prespec={snap_count}, current={curr_count}"
            )

    return (not diagnostics), diagnostics


def _is_ci_env() -> bool:
    """Return True iff the CI env var is one of the truthy values.

    The verifier uses this to default to strict mode in CI. Mirrors
    the harness's ``run_experiment`` CI detection.
    """
    return os.environ.get("CI", "").lower() in ("true", "1", "yes")


def _verify(
    pinned: Dict[str, Optional[str]],
    *,
    strict: bool,
) -> int:
    """Verify mode: re-compute live hashes and compare to pinned.

    HIGH-2 fix: when ``strict=True`` (CI env var or --strict flag),
    missing artifacts are a HARD FAILURE. When ``strict=False`` (local
    diagnostic with --allow-missing), missing artifacts are reported
    as [SKIP] and do NOT cause exit non-zero.

    Returns:
      * 0 — all pinned artifacts verified (or skipped under
        ``strict=False``).
      * 1 — at least one hash mismatch OR (under strict) at least one
        missing artifact.
    """
    failures: List[str] = []
    missing_failures: List[str] = []
    skipped: List[str] = []
    verified: List[str] = []

    for memo_key, (relpath, _) in ARTIFACTS.items():
        path = REPO_ROOT / relpath
        if not path.exists():
            msg = f"{memo_key}: artifact missing at {relpath}"
            if strict:
                missing_failures.append(
                    f"{msg} (strict mode: missing artifact is a hard failure; "
                    "the threshold-shopping defense's data-content half "
                    "cannot be verified without the cohort on disk)"
                )
            else:
                skipped.append(msg)
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
    print(f"G2 pre-spec dataset-hash verification (strict={strict})")
    print("=" * 70)
    for line in verified:
        print(f"  [OK]   {line}")
    for line in skipped:
        print(f"  [SKIP] {line}")
    for line in missing_failures:
        print(f"  [FAIL] {line}")
    for line in failures:
        print(f"  [FAIL] {line}")
    print()
    n_failures = len(failures) + len(missing_failures)
    if n_failures:
        kinds: List[str] = []
        if failures:
            kinds.append(f"{len(failures)} hash mismatch(es)")
        if missing_failures:
            kinds.append(f"{len(missing_failures)} missing artifact(s)")
        print("FAILED: " + " + ".join(kinds))
        return 1
    if skipped and not verified:
        print(
            "WARNING: every artifact is missing on disk; verification "
            "vacuously passed (lenient mode). The experiment harness's "
            "CI-presence guard is the primary gate. To force missing-as-"
            "failure locally, pass --strict."
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
    parser.add_argument(
        "--strict",
        action="store_true",
        help=(
            "Force strict mode: missing artifacts are a hard failure. "
            "Strict is the implicit default in CI (CI=true env var). "
            "Pass --strict explicitly to force it locally."
        ),
    )
    parser.add_argument(
        "--allow-missing",
        action="store_true",
        help=(
            "Local diagnostic flag: report missing artifacts as [SKIP] "
            "instead of [FAIL]. REJECTED in strict mode (CI or --strict)."
        ),
    )
    parser.add_argument(
        "--prespec-sha",
        default=None,
        help=(
            "HIGH-3 fix: load pinned hashes from the memo content at "
            "this SHA via `git show <SHA>:<MEMO>` instead of the working "
            "copy. Pass the immutable S_prespec SHA so a child commit "
            "cannot bypass verification by editing the memo. If omitted, "
            "auto-discovers S_prespec by `git log --diff-filter=A --follow`. "
            "Pass `--prespec-sha working` to use the working-copy memo "
            "(legacy / unsafe)."
        ),
    )
    parser.add_argument(
        "--repo-root",
        default=None,
        help=(
            "NEW HIGH-2 (iter-3) fix: explicit override for the worktree "
            "root. Required when this script is invoked from a staged "
            "governance checkout (e.g. governance_checkout/scripts/) "
            "where Path(__file__).parents[1] resolves to the staging "
            "directory, NOT the actual worktree. Workflows should pass "
            '--repo-root "$GITHUB_WORKSPACE" or set '
            "E2I_GOVERNANCE_REPO_ROOT in env."
        ),
    )
    args = parser.parse_args(argv)

    # NEW HIGH-2 (iter-3): allow CLI override of REPO_ROOT — the
    # scripts get copied into governance_checkout/scripts/ and need to
    # know the actual worktree root.
    if args.repo_root is not None:
        global REPO_ROOT, MEMO_PATH
        REPO_ROOT = Path(args.repo_root).resolve()
        MEMO_PATH = REPO_ROOT / "docs" / "specs" / "tier1b_b2_prespec_20260510.md"

    is_ci = _is_ci_env()
    strict = args.strict or is_ci
    if args.allow_missing and strict:
        ci_reason = "CI env var" if is_ci else "--strict flag"
        print(
            f"FATAL: --allow-missing is rejected in strict mode ({ci_reason}). "
            "Strict mode requires all pinned artifacts to be present.",
            file=sys.stderr,
        )
        return 2

    if not MEMO_PATH.exists():
        print(f"FATAL: spec memo not found at {MEMO_PATH}", file=sys.stderr)
        return 2

    # Resolve S_prespec for hash loading + memo-content-unchanged check.
    if args.prespec_sha == "working":
        s_prespec_sha = None
        memo_text_for_hashes = MEMO_PATH.read_text(encoding="utf-8")
        memo_source_label = "working tree (legacy / unsafe)"
    elif args.prespec_sha:
        s_prespec_sha = args.prespec_sha
        snapshot = _git_show(s_prespec_sha, MEMO_RELPATH)
        if snapshot is None:
            print(
                f"FATAL: could not load memo at --prespec-sha={s_prespec_sha} "
                "via `git show`. Check the SHA is reachable in the local git "
                "graph.",
                file=sys.stderr,
            )
            return 2
        memo_text_for_hashes = snapshot
        memo_source_label = f"git show {s_prespec_sha[:12]}:{MEMO_RELPATH}"
    else:
        s_prespec_sha = _discover_s_prespec_sha()
        if s_prespec_sha is not None:
            snapshot = _git_show(s_prespec_sha, MEMO_RELPATH)
            if snapshot is None:
                # Auto-discovered SHA but `git show` failed; fall back
                # to the working-copy memo (with a loud warning).
                memo_text_for_hashes = MEMO_PATH.read_text(encoding="utf-8")
                memo_source_label = (
                    f"working tree (auto-discovered S_prespec={s_prespec_sha[:12]} "
                    "but git show failed; falling back)"
                )
                s_prespec_sha = None
            else:
                memo_text_for_hashes = snapshot
                memo_source_label = (
                    f"git show {s_prespec_sha[:12]}:{MEMO_RELPATH} (auto-discovered)"
                )
        else:
            memo_text_for_hashes = MEMO_PATH.read_text(encoding="utf-8")
            memo_source_label = "working tree (no git history available)"

    pinned = _parse_pinned_hashes(memo_text_for_hashes)
    print(f"  loading pinned hashes from: {memo_source_label}")

    if args.update:
        return _update(pinned)

    rc_verify = _verify(pinned, strict=strict)

    # HIGH-3 secondary check: memo-content-unchanged across S_prespec
    # → working tree. Runs only when we have a real S_prespec SHA AND
    # we're in strict mode (the unchanged check is the strict-only
    # gate; lenient mode reports it informationally).
    if s_prespec_sha is not None:
        unchanged, diagnostics = _check_memo_unchanged_since_prespec(s_prespec_sha=s_prespec_sha)
        if unchanged:
            print(f"[OK] memo load-bearing content unchanged since S_prespec={s_prespec_sha[:12]}")
        else:
            print(f"[FAIL] memo load-bearing content CHANGED since S_prespec={s_prespec_sha[:12]}:")
            for diag in diagnostics:
                print(f"  {diag}")
            if strict:
                print(
                    "FAILED: memo load-bearing content drifted (strict mode); "
                    "the threshold-shopping defense requires a fresh "
                    "tier1b_b2_prespec_<date>.md memo."
                )
                return 1
            print(
                "WARNING: lenient mode — memo drift reported but not failed; "
                "pass --strict to enforce."
            )
    return rc_verify


if __name__ == "__main__":
    sys.exit(main())
