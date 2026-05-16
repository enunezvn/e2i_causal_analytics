#!/usr/bin/env python3
"""Count accepted compile-set candidates and surface backlog vs threshold.

Phase 4.5 (issue #236): auto-trigger surface for
``scripts/compile_causal_role_classifier.py``.

A "compile-set candidate" is a row in the JSON manifests produced by
``scripts/curate_compile_set_candidates.py`` (one per ``make
curate-candidates`` run). The curator hand-edits these manifests:
filling in ``expected_causal_role``, ``expected_remediation``,
``derivation_pseudocode``, and ``dataset_context``. A candidate is
"accepted" iff every one of those four fields is non-null. (Partial
fill-ins are deliberately treated as not-accepted — a half-filled row
is malformed and the markdown reviewer checklist exists exactly to
prevent it from sneaking in.)

The backlog is the count of accepted candidates from manifests whose
**file mtime is newer than the compiled artifact's mtime**. Manifests
older than the artifact contain candidates that were already folded
into ``build_compile_set()`` and recompiled — re-counting them would
produce a permanent positive backlog and defeat the purpose.

Usage::

    # Informational (default): always exit 0, print backlog & ready signal
    python scripts/check_compile_set_candidate_backlog.py \\
        --candidates-dir ./candidates \\
        --artifact artifacts/dspy/causal_role_classifier.json

    # Strict (compile pre-flight): exit non-zero when backlog == 0
    python scripts/check_compile_set_candidate_backlog.py \\
        --candidates-dir ./candidates \\
        --artifact artifacts/dspy/causal_role_classifier.json \\
        --strict

Exit codes:
  0  — backlog computed; if ``--strict`` then backlog >= 1
  1  — ``--strict`` mode and backlog == 0 (refuse to recompile)
  2  — argparse / unrecoverable input error
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path

# Single source of truth for the four nullable fields a curator MUST
# complete before a manifest row counts as accepted. Mirrors
# ``src/data/audit_candidate_formatter._REQUIRED_FILL_INS``. A unit
# test (``test_required_fill_ins_matches_formatter``) trips if either
# side adds a 5th required key without updating the other.
REQUIRED_FILL_INS: tuple[str, ...] = (
    "expected_causal_role",
    "expected_remediation",
    "derivation_pseudocode",
    "dataset_context",
)

# Issue #236 acceptance: "notifies … when backlog crosses threshold"
# default 5. Tunable via ``--threshold``.
DEFAULT_THRESHOLD: int = 5


@dataclass
class BacklogResult:
    """Outcome of a backlog scan.

    Attributes:
        count: Number of accepted-candidate rows across all manifests
            newer than the compiled artifact.
        accepted_features: Feature names from those accepted rows. Same
            feature appearing in two manifests counts twice — at this
            layer we don't de-dup (the curator's the dedup boundary
            via ``make curate-candidates`` window selection).
        scanned_manifests: Paths of JSON manifests parsed successfully.
        malformed_paths: Paths that exist but couldn't be parsed as the
            expected manifest shape. Surfaced for operator visibility,
            never fatal.
    """

    count: int
    accepted_features: list[str] = field(default_factory=list)
    scanned_manifests: list[Path] = field(default_factory=list)
    malformed_paths: list[Path] = field(default_factory=list)


def _is_accepted(candidate: dict) -> bool:
    """A candidate row is accepted iff every required fill-in is non-null.

    Empty string also counts as "not filled" — an operator who hits
    Enter on an empty prompt hasn't actually filled the field. Lists
    and booleans are passed through (we don't currently use those
    types but defensive coding).
    """
    for key in REQUIRED_FILL_INS:
        value = candidate.get(key)
        if value is None:
            return False
        if isinstance(value, str) and not value.strip():
            return False
    return True


def _artifact_mtime(path: Path) -> float | None:
    """Return the artifact's mtime in seconds, or ``None`` if missing.

    A missing artifact means cold-start: no prior compile, so every
    accepted candidate counts as backlog.
    """
    try:
        return path.stat().st_mtime
    except (FileNotFoundError, NotADirectoryError):
        return None


def count_accepted_candidates(
    *,
    candidates_dir: Path,
    compiled_artifact_path: Path,
    logger: logging.Logger | None = None,
) -> BacklogResult:
    """Walk ``candidates_dir`` for manifest JSON files newer than the
    compiled artifact, parse each, and count accepted-candidate rows.

    Non-JSON files (e.g. markdown reports written alongside the manifest)
    are skipped silently. JSON files that fail to parse, or that don't
    match the expected ``{"candidates": [...]}`` shape, are recorded in
    ``result.malformed_paths`` and surfaced to the caller — but never
    fatal (the candidates dir is the operator's audit trail, not a
    control channel).
    """
    log = logger or logging.getLogger(__name__)
    result = BacklogResult(count=0)

    if not candidates_dir.exists():
        log.info("candidates_dir %s does not exist; backlog=0", candidates_dir)
        return result

    artifact_mtime = _artifact_mtime(compiled_artifact_path)
    if artifact_mtime is None:
        log.info(
            "compiled artifact %s missing — counting every accepted candidate",
            compiled_artifact_path,
        )

    for manifest_path in sorted(candidates_dir.iterdir()):
        if not manifest_path.is_file() or manifest_path.suffix != ".json":
            continue

        try:
            mtime = manifest_path.stat().st_mtime
        except OSError as exc:  # broken symlink, racey delete, etc.
            log.warning("stat failed on %s: %s", manifest_path, exc)
            result.malformed_paths.append(manifest_path)
            continue

        # Skip manifests older than the (existing) artifact — those
        # candidates were already folded in.
        if artifact_mtime is not None and mtime <= artifact_mtime:
            log.debug(
                "skipping %s (mtime %.0f <= artifact mtime %.0f)",
                manifest_path,
                mtime,
                artifact_mtime,
            )
            continue

        try:
            payload = json.loads(manifest_path.read_text())
        except (json.JSONDecodeError, OSError) as exc:
            log.warning("could not parse manifest %s: %s", manifest_path, exc)
            result.malformed_paths.append(manifest_path)
            continue

        candidates = payload.get("candidates") if isinstance(payload, dict) else None
        if not isinstance(candidates, list):
            log.warning(
                "manifest %s missing top-level 'candidates' list; skipping",
                manifest_path,
            )
            result.malformed_paths.append(manifest_path)
            continue

        result.scanned_manifests.append(manifest_path)
        for cand in candidates:
            if not isinstance(cand, dict):
                continue
            if _is_accepted(cand):
                result.count += 1
                feature_name = cand.get("feature_name", "<unknown>")
                result.accepted_features.append(str(feature_name))

    return result


def _format_summary(result: BacklogResult, threshold: int) -> str:
    """Stable, grep-friendly summary line."""
    lines = [
        f"backlog={result.count} threshold={threshold} "
        f"scanned={len(result.scanned_manifests)} "
        f"malformed={len(result.malformed_paths)}"
    ]
    if result.count >= threshold and threshold > 0:
        lines.append(
            f"READY — {result.count} accepted candidates queued (>= threshold {threshold}). "
            "Hand-merge into `build_compile_set()` in "
            "`src/data/causal_role_classifier.py` and re-run "
            "`scripts/compile_causal_role_classifier.py`."
        )
    elif result.count == 0:
        lines.append("No accepted candidates queued — nothing to compile.")
    else:
        lines.append(
            f"backlog below threshold ({result.count} < {threshold}); "
            "keep curating or wait for more disagreements."
        )
    if result.accepted_features:
        # Stable lex-sort for the summary line so repeated runs are
        # diff-friendly.
        feats = sorted(result.accepted_features)
        preview = ", ".join(feats[:5])
        if len(feats) > 5:
            preview += f", … (+{len(feats) - 5} more)"
        lines.append(f"accepted features: {preview}")
    if result.malformed_paths:
        bad = ", ".join(str(p) for p in result.malformed_paths)
        lines.append(f"warning: malformed manifests skipped: {bad}")
    return "\n".join(lines)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--candidates-dir",
        type=Path,
        default=Path("./candidates"),
        help="Directory of curate_compile_set_candidates JSON manifests.",
    )
    parser.add_argument(
        "--artifact",
        type=Path,
        default=Path("artifacts/dspy/causal_role_classifier.json"),
        help="Path to the compiled DSPy classifier artifact.",
    )
    parser.add_argument(
        "--threshold",
        type=int,
        default=DEFAULT_THRESHOLD,
        help=(f"Backlog count at which the READY signal fires (default: {DEFAULT_THRESHOLD})."),
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help=(
            "Pre-flight mode: exit non-zero when backlog == 0. The "
            "compile script wires this in to refuse no-op recompiles."
        ),
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    logging.basicConfig(level=args.log_level, format="%(levelname)s %(name)s: %(message)s")
    logger = logging.getLogger("check_compile_set_candidate_backlog")

    result = count_accepted_candidates(
        candidates_dir=args.candidates_dir,
        compiled_artifact_path=args.artifact,
        logger=logger,
    )

    print(_format_summary(result, args.threshold))

    if args.strict and result.count == 0:
        print(
            "STRICT: refusing to proceed — no new accepted candidates "
            "since last compile. Run `make curate-candidates`, fill in "
            "the 4 required fields per accepted row, or pass --force to "
            "the compile script to override.",
            file=sys.stderr,
        )
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
