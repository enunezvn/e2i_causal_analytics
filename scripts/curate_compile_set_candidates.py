#!/usr/bin/env python3
"""Curate compile-set candidates from the Layer-4 evaluator audit trail.

Plan: .claude/plans/layer4_evaluator_audit_consumer.md.

Trust model (Codex Gate-2 LOW-1): this CLI assumes ``--artifacts-dir``
and ``--output-dir`` are operator-controlled paths inside the operator's
trust boundary (set in ``.env`` / docker-compose / passed by hand). The
CLI does not run with elevated privileges and does not validate against
symlink-traversal. If a future scheduled job runs this CLI against an
untrusted artifacts directory, add ``Path.resolve()`` + reject
``is_symlink()`` outputs.

Reads ``adaptive_verdicts_*.json`` files under
``$ADAPTIVE_VALIDITY_ARTIFACTS_DIR`` (or a path passed via
``--artifacts-dir``), filters for evaluator-disagreement events, dedups
by feature name (keeping the latest critique), and emits both a
markdown report and a JSON manifest to ``--output-dir``.

Output filenames are stamped with the run timestamp so re-running over
the same window doesn't overwrite prior outputs.

The human reviewer reads the markdown, hand-merges accepted candidates
into ``build_compile_set()`` in ``src/data/causal_role_classifier.py``,
fills in the ``expected_*`` fields, and re-runs
``scripts/compile_causal_role_classifier.py`` to produce a new compiled
artifact.

Usage:
    python scripts/curate_compile_set_candidates.py \\
        --artifacts-dir /app/data/audit_artifacts \\
        --output-dir ./candidates \\
        --since 2026-05-01 \\
        --until 2026-05-31
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.audit_candidate_formatter import (  # noqa: E402
    format_json_manifest,
    format_markdown_report,
)
from src.data.audit_sidecar_reader import (  # noqa: E402
    SidecarReader,
    VerdictRecord,
    dedup_disagreements,
    extract_disagreements,
)

# Issue #240 Stage 2: map a promotion-rule id to the ``VerdictRecord``
# predicate that is true iff that Stage-1 shadow rule fired on the row. R1
# fires when ``would_promote_severity`` is set; R2 when ``would_flag_for_review``
# is True; R3 when ``rationale_incomplete_flag`` is True. The predicate is
# evaluated on the record (not the DisagreementEvent) because R2/R3 flags live
# only on VerdictRecord. Design ref:
# ``docs/plans/240-audit-evaluator-gate-promotion.md`` §3 Stage 2 + §4 R1/R2/R3.
_PROMOTION_RULE_PREDICATES: dict[str, Callable[[VerdictRecord], bool]] = {
    "R1": lambda r: r.would_promote_severity is not None,
    "R2": lambda r: r.would_flag_for_review is True,
    "R3": lambda r: r.rationale_incomplete_flag is True,
}


def _parse_date(value: str) -> datetime:
    """Accept YYYY-MM-DD or full ISO8601. Naive dates → UTC."""
    try:
        dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"could not parse date {value!r}: {exc}") from exc
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description=("Curate compile-set candidates from the adaptive-validity audit trail.")
    )
    parser.add_argument(
        "--artifacts-dir",
        type=Path,
        default=None,
        help=(
            "Directory containing adaptive_verdicts_*.json files. "
            "Defaults to $ADAPTIVE_VALIDITY_ARTIFACTS_DIR."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Where to write the markdown report + JSON manifest.",
    )
    parser.add_argument(
        "--since",
        type=_parse_date,
        default=None,
        help="Inclusive lower bound on sidecar written_at (UTC).",
    )
    parser.add_argument(
        "--until",
        type=_parse_date,
        default=None,
        help="Inclusive upper bound on sidecar written_at (UTC).",
    )
    parser.add_argument(
        "--filter-promotion-rule",
        choices=sorted(_PROMOTION_RULE_PREDICATES),
        default=None,
        help=(
            "Issue #240 Stage 2: restrict candidates to disagreement rows where "
            "the named Stage-1 shadow promotion rule fired. R1 = "
            "would_promote_severity set (moderate→high escalation candidate); "
            "R2 = would_flag_for_review; R3 = rationale_incomplete_flag. "
            "Omit to include every evaluator-disagreement row."
        ),
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=args.log_level,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    logger = logging.getLogger("curate_compile_set_candidates")

    artifacts_dir = args.artifacts_dir
    if artifacts_dir is None:
        env_dir = os.environ.get("ADAPTIVE_VALIDITY_ARTIFACTS_DIR")
        if not env_dir:
            parser.error("neither --artifacts-dir nor $ADAPTIVE_VALIDITY_ARTIFACTS_DIR is set")
        artifacts_dir = Path(env_dir)

    if not artifacts_dir.exists():
        logger.warning(
            "artifacts-dir %s does not exist; emitting empty report",
            artifacts_dir,
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)

    reader = SidecarReader(
        artifacts_dir=artifacts_dir,
        since=args.since,
        until=args.until,
    )
    records = list(reader.iter_verdict_records())
    # Issue #240 Stage 2: optionally restrict to rows where one Stage-1 shadow
    # rule fired. The predicate runs on VerdictRecord (R2/R3 flags live there,
    # not on DisagreementEvent) BEFORE extraction. extract_disagreements still
    # applies its own satisfied=False filter afterwards, so a row that fired a
    # rule but is not a disagreement (e.g. R3 on a satisfied verdict) is
    # correctly dropped — the curation flow only surfaces disagreements.
    if args.filter_promotion_rule is not None:
        predicate = _PROMOTION_RULE_PREDICATES[args.filter_promotion_rule]
        before = len(records)
        records = [r for r in records if predicate(r)]
        logger.info(
            "--filter-promotion-rule %s: %d of %d verdict records matched",
            args.filter_promotion_rule,
            len(records),
            before,
        )
    disagreements = list(extract_disagreements(records))
    deduped = list(dedup_disagreements(disagreements))

    generated_at = datetime.now(timezone.utc)
    # Microsecond precision so two reruns in the same second don't
    # overwrite each other (codex review LOW-7, 2026-05-15).
    stamp = generated_at.strftime("%Y%m%dT%H%M%S%fZ")

    md_path = args.output_dir / f"compile_set_candidates_{stamp}.md"
    json_path = args.output_dir / f"compile_set_candidates_{stamp}.json"

    md_path.write_text(format_markdown_report(deduped, generated_at=generated_at))
    json_path.write_text(
        json.dumps(
            format_json_manifest(deduped, generated_at=generated_at),
            indent=2,
        )
    )

    logger.info(
        "wrote %d candidates: %s (%d verdict records scanned, %d disagreements before dedup)",
        len(deduped),
        md_path,
        len(records),
        len(disagreements),
    )
    print(f"Markdown: {md_path}")
    print(f"JSON:     {json_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
