"""Plan v4 Gate G2 / N1 — append G2 manifest to gate_history.

Codex MED-10 fix: previously the G2 workflow uploaded the manifest as
a CI artifact only — the regulatory eligibility audit trail
(``gate_history``) never saw the G2 outcome. This script reads the
manifest emitted by ``run_tier1b_b2_experiment.py``, builds three
``GateEvaluationEntry``-shaped records (one per pre-spec threshold
T1/T2/T3 + the combined ``g2_passes_pre_spec``), and writes them to
the ``gate_history`` JSONL file for downstream ingest by the N1 audit
trail.

Usage
-----

    python scripts/append_g2_to_gate_history.py \\
        --manifest g2_run_manifest.json \\
        --tag-ref refs/tags/tier1b-b2-experiment-1 \\
        --tag-sha abc123def \\
        --s-prespec-sha 7f616f6f \\
        --workflow-run-id 12345 \\
        --output g2_gate_history_entry.json

Schema
------

The output is a list of ``GateEvaluationEntry`` dicts (matching
``src.agents.ml_foundation.model_deployer.regulatory_audit.GateEvaluationEntry``)
with extra G2-specific provenance fields:

    {
        "timestamp": "2026-05-10T17:55:00Z",
        "gate_name": "G2_T1" | "G2_T2" | "G2_T3" | "G2",
        "threshold": <pre-spec threshold value>,
        "value": <observed value>,
        "outcome": "pass" | "fail",
        "threshold_provenance": "literature_anchored",
        "reason": <optional rationale string>,
        "g2_provenance": {
            "tag_ref": "refs/tags/tier1b-b2-experiment-1",
            "tag_sha": "abc123def...",
            "s_prespec_sha": "7f616f6f...",
            "workflow_run_id": "12345",
            "cohort_label": "optum_initiation_default",
            "dataset_hashes": {...},
        }
    }

The N1 audit trail consumer (regulatory_audit.RegulatoryEligibilityAudit)
appends each entry to ``gate_history`` via ``audit.append_gate_entry``.

This script is stdlib-only by design.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


def _utc_timestamp() -> str:
    """Return the current UTC time as an RFC-3339 string."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _build_provenance_block(
    *,
    tag_ref: str,
    tag_sha: str,
    s_prespec_sha: str,
    workflow_run_id: str,
    manifest: Dict[str, Any],
) -> Dict[str, Any]:
    """Common provenance block attached to every entry."""
    return {
        "tag_ref": tag_ref,
        "tag_sha": tag_sha,
        "s_prespec_sha": s_prespec_sha,
        "workflow_run_id": workflow_run_id,
        "cohort_label": manifest.get("cohort_label", "unknown"),
        "cohort_data_dir": manifest.get("cohort_data_dir", "unknown"),
        "cohort_data_snooped": manifest.get("cohort_data_snooped", None),
        "dataset_hashes": manifest.get("dataset_hashes", {}),
        "lifecycle_state": manifest.get("lifecycle_state", "unknown"),
        "experiment_commit_sha": manifest.get("experiment_commit_sha", tag_sha),
    }


def _build_threshold_entry(
    *,
    gate_name: str,
    threshold_dict: Dict[str, Any],
    timestamp: str,
    provenance: Dict[str, Any],
) -> Dict[str, Any]:
    """Build a single per-threshold entry from one element of
    ``manifest['thresholds']`` (T1, T2, or T3)."""
    passes = bool(threshold_dict.get("passes", False))
    outcome = "pass" if passes else "fail"
    return {
        "timestamp": timestamp,
        "gate_name": gate_name,
        "threshold": threshold_dict.get("threshold"),
        "value": threshold_dict.get("delta"),
        "outcome": outcome,
        "threshold_provenance": "literature_anchored",
        "reason": threshold_dict.get("rationale"),
        "g2_provenance": provenance,
    }


def _build_combined_entry(
    *,
    manifest: Dict[str, Any],
    timestamp: str,
    provenance: Dict[str, Any],
) -> Dict[str, Any]:
    """Build the combined G2 verdict entry (T1 AND T2 AND T3)."""
    passes = bool(manifest.get("g2_passes_pre_spec", False))
    return {
        "timestamp": timestamp,
        "gate_name": "G2",
        "threshold": "T1 AND T2 AND T3 all pass",
        "value": "pass" if passes else "fail",
        "outcome": "pass" if passes else "fail",
        "threshold_provenance": "literature_anchored",
        "reason": (
            "G2 combined verdict: pre-specified ΔAUC + ECE + CV-stability "
            "threshold conjunction. Pre-spec memo at S_prespec is the "
            "load-bearing artifact."
        ),
        "g2_provenance": provenance,
    }


def build_audit_entries(
    *,
    manifest: Dict[str, Any],
    tag_ref: str,
    tag_sha: str,
    s_prespec_sha: str,
    workflow_run_id: str,
    timestamp: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Build the list of audit-trail entries (T1 + T2 + T3 + combined)."""
    timestamp = timestamp or _utc_timestamp()
    provenance = _build_provenance_block(
        tag_ref=tag_ref,
        tag_sha=tag_sha,
        s_prespec_sha=s_prespec_sha,
        workflow_run_id=workflow_run_id,
        manifest=manifest,
    )
    entries: List[Dict[str, Any]] = []
    thresholds = manifest.get("thresholds", []) or []
    for threshold_dict in thresholds:
        gate_name = f"G2_{threshold_dict.get('name', 'UNKNOWN')}"
        entries.append(
            _build_threshold_entry(
                gate_name=gate_name,
                threshold_dict=threshold_dict,
                timestamp=timestamp,
                provenance=provenance,
            )
        )
    entries.append(
        _build_combined_entry(
            manifest=manifest,
            timestamp=timestamp,
            provenance=provenance,
        )
    )
    return entries


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        required=True,
        help="Path to the G2 run manifest JSON emitted by run_tier1b_b2_experiment.py.",
    )
    parser.add_argument("--tag-ref", required=True, help="Refs/tags/<tag_name>.")
    parser.add_argument("--tag-sha", required=True, help="Tag commit SHA.")
    parser.add_argument(
        "--s-prespec-sha",
        required=True,
        help="The S_prespec SHA (introducing commit of the pre-spec memo).",
    )
    parser.add_argument(
        "--workflow-run-id",
        required=True,
        help="GitHub Actions workflow run ID for cross-reference.",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output path for the gate_history entries JSON.",
    )
    args = parser.parse_args(argv)

    manifest_path = Path(args.manifest)
    if not manifest_path.exists():
        print(f"FATAL: manifest not found at {manifest_path}", file=sys.stderr)
        return 1
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        print(f"FATAL: manifest at {manifest_path} is not valid JSON: {exc}", file=sys.stderr)
        return 1

    entries = build_audit_entries(
        manifest=manifest,
        tag_ref=args.tag_ref,
        tag_sha=args.tag_sha,
        s_prespec_sha=args.s_prespec_sha,
        workflow_run_id=args.workflow_run_id,
    )
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(entries, indent=2, sort_keys=True), encoding="utf-8")
    print(f"[OK] wrote {len(entries)} gate_history entries to {output_path}")

    # Exit code mirrors the combined G2 verdict so the workflow's
    # `set -euo pipefail` behavior is preserved.
    if not bool(manifest.get("g2_passes_pre_spec", False)):
        # G2 failed — return 0 anyway (we always want the audit append
        # to succeed even when G2 fails; the failed outcome is captured
        # IN the entry).
        return 0
    return 0


if __name__ == "__main__":
    sys.exit(
        main(sys.argv[1:] if len(sys.argv) > 1 else os.environ.get("G2_AUDIT_ARGV", "").split())
    )
