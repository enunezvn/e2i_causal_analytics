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

_REQUIRED_FILL_INS = (
    "expected_causal_role",
    "expected_remediation",
    "derivation_pseudocode",
    "dataset_context",
)


def format_markdown_report(
    events: Iterable[DisagreementEvent],
    *,
    generated_at: datetime,
) -> str:
    """Render the human-readable compile-set candidate report.

    Per codex review MED-8 (2026-05-15), the markdown explicitly lists
    ALL four required fill-ins (``expected_causal_role``,
    ``expected_remediation``, ``derivation_pseudocode``,
    ``dataset_context``) — not just the two ``expected_*`` fields. The
    derivation/context pair is not recoverable from sidecar payload
    alone (the sidecar carries worker OUTPUTS, not inputs); the reviewer
    must recover them from the original pipeline run via the
    ``source_run_id`` and ``source_path`` breadcrumbs.
    """
    events_list = list(events)
    fill_in_list = ", ".join(f"`{k}`" for k in _REQUIRED_FILL_INS)
    lines = [
        "# Compile-set candidates",
        "",
        f"Generated at: {generated_at.isoformat()}",
        f"Candidates: {len(events_list)}",
        "",
        "Each candidate below is a feature where the Layer-4 evaluator "
        "judged the worker's rationale inadequate. The evaluator is NOT "
        "ground truth — review each row, fill in ALL of the required "
        f"fields ({fill_in_list}), and (if accepted) hand-merge into "
        "`build_compile_set()` in `src/data/causal_role_classifier.py`. "
        "A merged `dspy.Example` with `derivation_pseudocode=None` or "
        "`dataset_context=None` is malformed and will be skipped during "
        "compile.",
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
                f"- **Source sidecar:** `{e.source_path}`",
                f"- **Worker verdict:** severity=`{e.worker_severity}`, "
                f"remediation=`{e.worker_remediation}`",
                f"- **Rationale complete:** `{e.rationale_complete}`",
                f"- **Missed considerations:** {missed_str}",
                f"- **Evaluator notes:** {e.notes or '(empty)'}",
                f"- **Evaluator model:** `{e.evaluator_model}`",
                "",
                "**Required fill-ins before accepting:**",
                "- [ ] `expected_causal_role` — confounder | mediator | "
                "collider | descendant | iv | proxy_confounder",
                "- [ ] `expected_remediation` — keep | keep_with_caveat | "
                "drop | post_index_only_drop",
                "- [ ] `derivation_pseudocode` — recover from pipeline log at source_run_id above",
                "- [ ] `dataset_context` — target + dataset family + "
                'anchor (e.g. "CSU target ON_180")',
                "",
                "**Decision (check one):**",
                "- [ ] accept — all 4 fill-ins above completed",
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
    """Render the machine-parseable manifest. Each candidate has four
    nullable fields the human MUST fill in before merging into the
    compile set: ``expected_causal_role``, ``expected_remediation``,
    ``derivation_pseudocode``, ``dataset_context``. The two derivation/
    context fields are not in the sidecar JSON (the sidecar carries
    worker OUTPUTS, not the inputs); recover them from the original
    pipeline run via ``source_run_id`` and ``source_path``. The manifest
    top-level carries an explicit ``required_fill_ins`` list so any
    downstream tooling that validates accepted-candidate manifests can
    refuse to merge entries with these still ``None`` (codex review
    MED-8)."""
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
        "required_fill_ins": list(_REQUIRED_FILL_INS),
        "candidates": candidates,
    }
