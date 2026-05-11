"""Plan v4 Gate G2 / N1 — append G2 manifest to gate_history.

MED-10 (pass-2): the G2 workflow previously uploaded the manifest as
a CI artifact only — the regulatory eligibility audit trail
(``gate_history``) never saw the G2 outcome.  This script now calls
``RegulatoryEligibilityAudit.append_gate_evaluation`` for each
per-threshold record, making the N1 audit trail the **load-bearing**
append path.  The standalone JSON artifact is kept as a back-compat
shim for downstream consumers that read the raw entry list.

Usage
-----

    python scripts/append_g2_to_gate_history.py \\
        --manifest g2_run_manifest.json \\
        --tag-ref refs/tags/tier1b-b2-experiment-1 \\
        --tag-sha abc123def \\
        --s-prespec-sha 7f616f6f \\
        --workflow-run-id 12345 \\
        --output g2_gate_history_entry.json \\
        --audit-output updated_audit.json \\
        [--audit-state existing_audit.json]

``--audit-output`` (required): path to write the updated
``audit.to_dict()`` snapshot after appending G2 entries.  Required so
the N1 audit trail is durably persisted on disk, not only in memory.
Written even when G2 fails — the failed outcome is captured inside the
entry itself.

``--audit-state`` (optional): path to an existing
``RegulatoryEligibilityAudit.to_dict()`` JSON checkpoint. When
supplied, G2 entries are appended to that audit object. When omitted,
a fresh audit is created.

Schema
------

The JSON artifact (``--output``) is a list of
``GateEvaluationEntry``-shaped dicts with extra G2-specific provenance:

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

The ``g2_provenance`` field is stored verbatim in the ``reason``
parameter of ``append_gate_evaluation`` so N1 can surface it without
schema changes.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

# ---------------------------------------------------------------------------
# N1 audit API — load-bearing import.  The script intentionally imports from
# src/ so the canonical ``RegulatoryEligibilityAudit`` append-only guard is
# the single writer for gate_history.  ``--audit-output`` persists the updated
# audit snapshot; ``--output`` retains the flat entry list for back-compat.
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from src.agents.ml_foundation.model_deployer.regulatory_audit import (  # noqa: E402
    RegulatoryEligibilityAudit,
)


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


def append_to_n1_audit(
    audit: "RegulatoryEligibilityAudit",
    entries: List[Dict[str, Any]],
) -> None:
    """Call ``audit.append_gate_evaluation`` for each entry in *entries*.

    This is the load-bearing N1 API path.  Each entry's ``g2_provenance``
    dict is serialised into the ``reason`` field so the provenance block
    is preserved inside the append-only guard without widening the
    ``GateEvaluationEntry`` schema.

    ``threshold_provenance`` is always ``"literature_anchored"`` for G2
    entries — the thresholds are fixed in the S_prespec memo which is
    the canonical sign-off document.
    """
    for entry in entries:
        # Encode g2_provenance as a JSON string inside reason so it
        # survives the frozen-dataclass serialisation round-trip.
        reason_parts: List[str] = []
        raw_reason = entry.get("reason")
        if raw_reason:
            reason_parts.append(str(raw_reason))
        prov = entry.get("g2_provenance")
        if prov:
            reason_parts.append(
                "g2_provenance=" + json.dumps(prov, sort_keys=True, default=str)
            )
        combined_reason: Optional[str] = "; ".join(reason_parts) if reason_parts else None

        audit.append_gate_evaluation(
            timestamp=entry["timestamp"],
            gate_name=entry["gate_name"],
            threshold=entry.get("threshold"),
            value=entry.get("value"),
            outcome=entry["outcome"],
            threshold_provenance=entry.get("threshold_provenance"),
            reason=combined_reason,
        )


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
        help="Output path for the back-compat flat gate_history entries JSON.",
    )
    parser.add_argument(
        "--audit-state",
        default=None,
        help=(
            "Optional path to an existing RegulatoryEligibilityAudit.to_dict() "
            "JSON checkpoint. When supplied, G2 entries are appended to the "
            "loaded audit. When omitted, a fresh audit is created."
        ),
    )
    parser.add_argument(
        "--audit-output",
        required=True,
        help=(
            "Path to write the updated audit.to_dict() snapshot after "
            "appending G2 entries. Written even when G2 fails — the failed "
            "outcome is captured inside the entry. Required so the N1 audit "
            "trail is durably persisted, not only held in memory."
        ),
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

    # --- N1 audit API (load-bearing path) -----------------------------------
    if args.audit_state:
        audit_state_path = Path(args.audit_state)
        if not audit_state_path.exists():
            print(
                f"FATAL: audit-state checkpoint not found at {audit_state_path}",
                file=sys.stderr,
            )
            return 1
        try:
            audit_payload = json.loads(audit_state_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            print(
                f"FATAL: audit-state at {audit_state_path} is not valid JSON: {exc}",
                file=sys.stderr,
            )
            return 1
        audit = RegulatoryEligibilityAudit.from_dict(audit_payload)
        print(
            f"[INFO] loaded audit checkpoint from {audit_state_path} "
            f"({len(audit.gate_history)} existing gate entries)"
        )
    else:
        audit = RegulatoryEligibilityAudit()

    append_to_n1_audit(audit, entries)
    print(f"[OK] appended {len(entries)} G2 entries to N1 RegulatoryEligibilityAudit")

    # --- Atomic writes: serialise both outputs before touching the filesystem.
    # Both JSON payloads are prepared in memory first; then written via temp
    # files + Path.replace() so a mid-write crash cannot leave either file in
    # a partial state that mismatches the other.
    audit_output_path = Path(args.audit_output)
    output_path = Path(args.output)
    audit_json = json.dumps(audit.to_dict(), indent=2, sort_keys=True)
    shim_json = json.dumps(entries, indent=2, sort_keys=True)

    # Serialisation succeeded for both — now write atomically.
    audit_output_path.parent.mkdir(parents=True, exist_ok=True)
    _tmp_audit = audit_output_path.parent / (audit_output_path.name + ".tmp")
    _tmp_audit.write_text(audit_json, encoding="utf-8")
    _tmp_audit.replace(audit_output_path)
    print(f"[OK] wrote updated audit snapshot to {audit_output_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    _tmp_shim = output_path.parent / (output_path.name + ".tmp")
    _tmp_shim.write_text(shim_json, encoding="utf-8")
    _tmp_shim.replace(output_path)
    print(f"[OK] wrote {len(entries)} gate_history entries (shim) to {output_path}")

    # Exit code is always 0 — we always want the audit append to succeed
    # even when G2 fails; the failed outcome is captured IN the entries.
    return 0


if __name__ == "__main__":
    sys.exit(
        main(sys.argv[1:] if len(sys.argv) > 1 else os.environ.get("G2_AUDIT_ARGV", "").split())
    )
