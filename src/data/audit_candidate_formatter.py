"""Formatters that turn DisagreementEvent rows into curation outputs.

Plan: .claude/plans/layer4_evaluator_audit_consumer.md.

Produces two artifacts per CLI run:
- Markdown report (human-reviewed; accept/reject checkboxes per row).
- JSON manifest (machine-parseable; round-trippable; ``expected_*``
  fields nulled, to be filled in by the human during review).
"""

from __future__ import annotations

from datetime import datetime
from typing import Iterable

from src.data.audit_sidecar_reader import DisagreementEvent


def format_markdown_report(
    events: Iterable[DisagreementEvent],
    *,
    generated_at: datetime,
) -> str:
    """Render the human-readable compile-set candidate report."""
    events_list = list(events)
    lines = [
        "# Compile-set candidates",
        "",
        f"Generated at: {generated_at.isoformat()}",
        f"Candidates: {len(events_list)}",
        "",
        "Each candidate below is a feature where the Layer-4 evaluator "
        "judged the worker's rationale inadequate. The evaluator is NOT "
        "ground truth — review each row, fill in `expected_causal_role` "
        "and `expected_remediation`, and (if accepted) hand-merge into "
        "`build_compile_set()` in `src/data/causal_role_classifier.py`.",
        "",
    ]
    if not events_list:
        lines.append("No candidates in the requested time window.")
        return "\n".join(lines) + "\n"

    for idx, e in enumerate(events_list, start=1):
        missed_str = (
            ", ".join(e.missed_considerations) if e.missed_considerations else "(none specified)"
        )
        lines.extend(
            [
                f"## {idx}. `{e.feature}`",
                "",
                f"- **Source run:** `{e.experiment_id}` ({e.written_at.isoformat()})",
                f"- **Worker verdict:** severity=`{e.worker_severity}`, "
                f"remediation=`{e.worker_remediation}`",
                f"- **Rationale complete:** `{e.rationale_complete}`",
                f"- **Missed considerations:** {missed_str}",
                f"- **Evaluator notes:** {e.notes or '(empty)'}",
                f"- **Evaluator model:** `{e.evaluator_model}`",
                "",
                "**Decision (check one):**",
                "- [ ] accept — add to compile set after filling expected_* fields",
                "- [ ] reject — false-positive disagreement; evaluator was wrong",
                "",
            ]
        )
    return "\n".join(lines) + "\n"


def format_json_manifest(
    events: Iterable[DisagreementEvent],
    *,
    generated_at: datetime,
) -> dict:
    """Render the machine-parseable manifest. Each candidate has
    nullable ``expected_*`` fields the human fills in at review time."""
    candidates = []
    for e in events:
        candidates.append(
            {
                "feature_name": e.feature,
                "derivation_pseudocode": None,
                "dataset_context": None,
                "expected_causal_role": None,
                "expected_remediation": None,
                "evaluator_audit": {
                    "satisfied": False,
                    "rationale_complete": e.rationale_complete,
                    "missed_considerations": list(e.missed_considerations),
                    "notes": e.notes,
                    "model": e.evaluator_model,
                },
                "source_run_id": e.experiment_id,
                "source_written_at": e.written_at.isoformat(),
                "source_path": str(e.source_path),
            }
        )
    return {
        "generated_at": generated_at.isoformat(),
        "schema_version": 1,
        "candidates": candidates,
    }
