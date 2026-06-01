"""Human/audit-facing rendering of the tier0 regulatory authorization manifest.

The deployer builds a frozen v5-C1 ``RegulatoryDeploymentManifest`` (a SHA256
payload + Gate-N1 audit) inside ``validate_promotion``, but historically it only
reached humans through a test-only env-gated JSON dump — the tier0 console
summary printed a different, simpler ``deployment_manifest`` (id/env/status/
endpoint) and never the compliance artifact (gap G6). Separately, which quality
gates actually *enforce* promotion vs. merely *advise* lived only in scattered
code comments (gap G12): only ``minimum_auc`` (Gate N1) is ENFORCED — and it
gates ``regulatory_eligible``, NOT model promotion directly (``promotion_allowed``
is governed separately by success_criteria + deployment-path/shadow-mode
validation). The honest-AUC band, permutation-anchored AUC floor, Layer-3
ablation, and the T2.6a signal/calibration/CV bands are advisory pending T2.6c
graduation — they mutate neither ``regulatory_eligible`` nor ``promotion_allowed``.

This module renders both as a markdown/console report so the always-printed
tier0 summary (and a durable markdown file) surface them. Pure functions — no
I/O, no side effects — so they are cheap to unit-test.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

# Advisory-vs-enforced gate map (gap G12). The ONLY enforced regulatory gate is
# minimum_auc (registry_manager.py: N1_REQUIRED_REGULATORY_GATES == ["minimum_auc"]).
# IMPORTANT (audit accuracy): Gate N1 gates ``regulatory_eligible``, NOT model
# promotion directly — registry_manager.py:29-30 ("the deployer MUST evaluate
# before granting regulatory_eligible=True"). ``promotion_allowed`` is decided
# separately (success_criteria + deployment-path / shadow-mode validation).
# Everything else is computed for observability and explicitly mutates neither
# regulatory_eligible nor promotion_allowed — the documented T2.6c data-snooping
# discipline (advisory until an untouched cohort + signoff), NOT a bug.
#
# (gate, status, note)
ADVISORY_VS_ENFORCED: Tuple[Tuple[str, str, str], ...] = (
    (
        "minimum_auc",
        "ENFORCED",
        "Gate N1 — the only ENFORCED regulatory-eligibility gate: denies "
        "regulatory_eligible=True (registry_manager.py: "
        "N1_REQUIRED_REGULATORY_GATES). It does NOT directly deny model "
        "promotion; promotion_allowed is governed separately.",
    ),
    (
        "honest_auc_band",
        "ADVISORY",
        "Cohort-derived honest AUC band; observability-only, does not block "
        "(T2.6c-pending).",
    ),
    (
        "permutation_anchored_auc",
        "ADVISORY",
        "Permutation-null AUC floor; advisory until an untouched cohort lands.",
    ),
    (
        "layer3_ablation",
        "ADVISORY",
        "Per-OHE-category leakage ablation (model_eval_ablation); default-off, "
        "advisory.",
    ),
    (
        "signal_genuineness",
        "ADVISORY",
        "T2.6a permutation-p band; advisory.",
    ),
    (
        "calibration_quality",
        "ADVISORY",
        "T2.6a ECE band; advisory.",
    ),
    (
        "cv_stability",
        "ADVISORY",
        "T2.6a CV std/mean band; advisory.",
    ),
)


def _fmt(value: Any) -> str:
    """Compact, stable string for a manifest scalar (None → 'n/a')."""
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def _enforcement_map_lines() -> List[str]:
    lines = ["## Gate enforcement map (advisory vs enforced)", ""]
    lines.append(
        "_ENFORCED here means Gate N1 regulatory **eligibility** "
        "(`regulatory_eligible`), not model promotion — `promotion_allowed` is "
        "decided separately by success_criteria + deployment-path/shadow-mode "
        "validation. Advisory signals mutate neither._"
    )
    lines.append("")
    lines.append("| Gate | Status | Note |")
    lines.append("|---|---|---|")
    for gate, status, note in ADVISORY_VS_ENFORCED:
        lines.append(f"| {gate} | {status} | {note} |")
    return lines


def format_regulatory_report(manifest: Optional[Dict[str, Any]]) -> str:
    """Render the regulatory authorization manifest + enforcement map as text.

    Args:
        manifest: ``RegulatoryDeploymentManifest.to_dict()`` (or None when the
            deployer did not run / promotion was not reached).

    Returns:
        A markdown/console string. The enforcement map is always included (it is
        static and useful regardless of whether a manifest was produced).
    """
    lines: List[str] = ["## Tier0 Regulatory Authorization Manifest", ""]

    if not manifest:
        lines.append(
            "No regulatory_deployment_manifest in state — the deployer did not "
            "reach validate_promotion (e.g. QC gate blocked, training failed, or "
            "promotion was not attempted). Manifest not available."
        )
        lines.append("")
        lines.extend(_enforcement_map_lines())
        return "\n".join(lines)

    status = manifest.get("t2_6c_authorization_status", "unknown")
    reasons = manifest.get("t2_6c_authorization_reasons") or []

    lines.append(f"- Authorization status (T2.6c): **{status}**")
    if reasons:
        lines.append(f"  - reasons: {'; '.join(str(r) for r in reasons)}")
    lines.append(f"- Cohort: {manifest.get('cohort', 'n/a')} "
                 f"(in C1 scope: {manifest.get('in_c1_scope')})")
    lines.append(
        f"- Held-out AUC: {_fmt(manifest.get('roc_auc'))} "
        f"| honest band: {_fmt(manifest.get('honest_auc_band'))} "
        f"| in band: {manifest.get('auc_in_band')} (advisory)"
    )
    lines.append(
        f"- Permutation p-value: {_fmt(manifest.get('permutation_pvalue'))} "
        f"| signal genuineness: {manifest.get('signal_genuineness_category', 'n/a')}"
    )
    lines.append(
        f"- Calibration ECE: {_fmt(manifest.get('calibration_ece'))} "
        f"→ calibrated: {_fmt(manifest.get('calibrated_ece'))} "
        f"({manifest.get('calibration_method_resolved') or 'no post-hoc'}; "
        f"quality: {manifest.get('calibration_quality_category', 'n/a')})"
    )
    lines.append(
        f"- CV stability (std/mean): {_fmt(manifest.get('cv_stability_std_over_mean'))} "
        f"({manifest.get('cv_stability_category', 'n/a')})"
    )
    lines.append(f"- Feature surface count: {_fmt(manifest.get('feature_surface_count'))}")
    lines.append(
        f"- Gate N1 — regulatory_eligible: {manifest.get('regulatory_eligible')} "
        f"| adapted candidate: {manifest.get('adapted_regulatory_candidate')} "
        f"| audit present: {manifest.get('regulatory_eligibility_audit_present')}"
    )
    lines.append(f"- Manifest SHA256: `{manifest.get('manifest_sha256', 'n/a')}`")
    lines.append(
        f"- Audit fingerprint: "
        f"`{manifest.get('regulatory_eligibility_audit_fingerprint') or 'n/a'}`"
    )
    lines.append(f"- Emitted at: {manifest.get('emitted_at', 'n/a')}")
    lines.append("")
    lines.extend(_enforcement_map_lines())
    return "\n".join(lines)
