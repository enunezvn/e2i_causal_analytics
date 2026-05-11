"""G1 lineage-audit sweep — Plan v4 §2 Gate G1 acceptance criterion.

Closes the **NEW (codex-rescue MEDIUM-4)** sub-criterion of Plan v4 §2
Gate G1 (Tier 1B Gate B1):

    "Derivation-lineage audit on every feature surfaced by HBLP relaxation
    — ``derivation_inputs ⊆ pre-anchor`` per ``MANIFEST_SOURCES``. v3 §6
    Gate B1 named this; v4-draft inadvertently dropped it. Re-included
    here for completeness. PR #127's ``lineage_audit_declared_path``
    helper is the implementation; G1 acceptance demands it run on every
    relaxed feature in the regression sweep, not just standalone."

Why this is a SEPARATE test (not part of the CSU/Optum integration tests)
-------------------------------------------------------------------------
The CSU + Optum integration tests pin pipeline outputs (val_AUC, perm p,
test AUC). Those values depend on real cohort data and skip when data
is missing.

The lineage-audit sweep is **declarative**: it runs against the
``MANIFEST_SOURCES`` registry, not against pipeline output. It pins:
  1. Every CSU + Optum manifest feature's ``derivation_inputs ⊆ pre-anchor``
  2. The ``lineage_audit_declared_path`` helper agrees with
     ``KnowableAt.is_pre_or_at_index()`` on all manifest features
  3. HBLP-relaxed features (those that would survive a relaxed Layer 3
     z-threshold) all have valid declared paths

This means the sweep runs in CI without requiring real cohort data.
When real-cohort runs DO occur (CSU/Optum tests above), the same
helper is invoked against the surviving feature set; this test
provides the SHIFT-LEFT version that catches manifest-level
violations BEFORE a pipeline is even spun up.

What "HBLP relaxation surfaces" means
--------------------------------------
HBLP (``hblp_classify``) inflates the Layer 3 z-threshold for low-N
cohorts and Layer-1-cleared features. A feature whose Layer 3 z-score
is between the legacy 5σ floor and the inflated threshold (e.g., 7.5σ
at n=22) is "relaxed" — it would have been dropped under legacy 5σ but
is kept under HBLP. Per Plan v4 §2 G1, every such feature must have
``derivation_inputs ⊆ pre-anchor`` per the manifest registry. Otherwise
the relaxation is admitting a declared-path-invalid feature into the
trained model.

Sweep semantics:
- Iterate every feature in CSU_FEATURES + OPTUM_FEATURES that has
  ``knowable_at.is_pre_or_at_index() == True`` (i.e., Layer 1 cleared).
- For each, call ``lineage_audit_declared_path(feature, source)`` and
  assert ``declared_path_valid is True``.
- Additionally, for each feature, walk its ``derivation_inputs`` and
  for any input that is itself a manifest-declared feature, recursively
  audit it. This is the "transitive ⊆ pre-anchor" check that v3 §6
  Gate B1 originally named.

This test is what v3 §6 Gate B1 actually demanded but never landed.

References
----------
- Plan v4 §2 Gate G1 acceptance criterion (codex-rescue MEDIUM-4)
- Plan v3 §6 Tier 1B Gate B1
- ``src/agents/ml_foundation/data_preparer/nodes/adaptive_validity_check.py``
  (``lineage_audit_declared_path`` — PR #127)
- Companion: ``test_csu_negative_control_20260510.py``
- Companion: ``test_optum_held_out_noninferiority_20260510.py``
"""

from __future__ import annotations

import pytest

from src.agents.ml_foundation.data_preparer.nodes.adaptive_validity_check import (
    hblp_classify,
    hblp_effective_z_threshold,
    lineage_audit_declared_path,
)
from src.data.manifests import MANIFEST_SOURCES
from src.data.manifests.csu_feature_manifest import CSU_FEATURES
from src.data.manifests.optum_feature_manifest import OPTUM_FEATURES

# --------------------------------------------------------------------------- #
# Sweep 1: every manifest feature with knowable_at.is_pre_or_at_index() must  #
# pass lineage_audit_declared_path                                            #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "data_source,features",
    [
        ("csu", CSU_FEATURES),
        ("optum", OPTUM_FEATURES),
    ],
)
def test_g1_every_pre_anchor_feature_passes_lineage_audit(
    data_source: str,
    features: list,
) -> None:
    """Plan v4 §2 G1 sweep: every Layer-1-cleared manifest feature must
    pass ``lineage_audit_declared_path``.

    For each feature in CSU_FEATURES / OPTUM_FEATURES whose
    ``knowable_at.is_pre_or_at_index()`` is True, the lineage audit
    must return ``declared_path_valid=True``. Any divergence here means
    either:

    (a) ``lineage_audit_declared_path``'s ``pre_anchor_refs`` set has
        drifted from ``KnowableAt.is_pre_or_at_index()``, or
    (b) a manifest entry uses a knowable_at reference value that the
        audit doesn't know how to interpret.

    Both are blockers for HBLP wiring (Phase C, G3).
    """
    failures: list[tuple[str, str | None]] = []
    for contract in features:
        if not contract.knowable_at.is_pre_or_at_index():
            continue
        result = lineage_audit_declared_path(contract.name, data_source=data_source)
        if not result["contract_found"]:
            failures.append(
                (
                    contract.name,
                    f"contract_found=False; rationale={result['rationale']!r}",
                )
            )
            continue
        if result["declared_path_valid"] is not True:
            failures.append(
                (
                    contract.name,
                    f"declared_path_valid={result['declared_path_valid']!r}; "
                    f"knowable_at_reference={result['knowable_at_reference']!r}; "
                    f"rationale={result['rationale']!r}",
                )
            )

    assert not failures, (
        f"G1 lineage-audit sweep ({data_source}): "
        f"{len(failures)} pre-anchor manifest features failed "
        f"declared-path audit:\n  - "
        + "\n  - ".join(f"{name}: {detail}" for name, detail in failures)
        + "\nFix: align lineage_audit's pre_anchor_refs with "
        "KnowableAt.is_pre_or_at_index() OR fix the manifest entries."
    )


@pytest.mark.parametrize(
    "data_source,features",
    [
        ("csu", CSU_FEATURES),
        ("optum", OPTUM_FEATURES),
    ],
)
def test_g1_every_post_anchor_feature_fails_lineage_audit(
    data_source: str,
    features: list,
) -> None:
    """Negative control: every ``knowable_at=post_index`` feature must
    FAIL the lineage audit.

    Plan v4 §2 G1 requires ``derivation_inputs ⊆ pre-anchor``; the
    contrapositive is "every post-anchor feature must be flagged
    declared_path_valid=False". This negative control catches
    audit-helper regressions where a relaxation accidentally lets
    post-anchor through.

    Empirically: CSU manifest declares ~5 ``post_index`` features
    (journey_status, journey_duration_days, etc.); Optum manifest
    declares the targets (initiated_biologic_180d, etc.) as
    post_index. All must FAIL.
    """
    post_anchor = [c for c in features if not c.knowable_at.is_pre_or_at_index()]
    assert post_anchor, (
        f"{data_source} manifest has no post_index features; "
        "negative-control test invariant is broken."
    )

    leaks: list[str] = []
    for contract in post_anchor:
        result = lineage_audit_declared_path(contract.name, data_source=data_source)
        if not result["contract_found"]:
            # Manifest contract is supposed to exist; missing is itself a fail.
            leaks.append(f"{contract.name}: contract_found=False")
            continue
        if result["declared_path_valid"] is not False:
            leaks.append(
                f"{contract.name}: declared_path_valid="
                f"{result['declared_path_valid']!r} (expected False); "
                f"knowable_at_reference={result['knowable_at_reference']!r}"
            )

    assert not leaks, (
        f"G1 lineage-audit negative control ({data_source}): "
        f"{len(leaks)} post-anchor manifest features did NOT fail audit:\n  - "
        + "\n  - ".join(leaks)
        + "\nThis is a relaxation regression — the audit should "
        "flag post_index features as declared_path_valid=False."
    )


# --------------------------------------------------------------------------- #
# Sweep 2: derivation_inputs ⊆ pre-anchor (transitive)                         #
# --------------------------------------------------------------------------- #


# Codex pass-1 MED-8 (PR #137 v4 G1): per-source pre-anchor raw-column
# registry. A derivation input that is NOT a manifest-declared feature
# MUST be an entry in this registry — otherwise the audit fails loudly.
# The registry encodes the converter's promise that these raw columns
# are read from upstream data BEFORE the anchor and therefore
# observable-at-prediction-time. Adding a new raw input to a manifest
# entry that is not in this registry will fail this sweep, forcing the
# author to either add it here (with rationale) or re-shape the
# derivation to use a manifest-declared intermediate.
#
# These registries are intentionally minimal — only columns the
# production converters in scripts/convert_csu_rwd.py and
# scripts/convert_optum_rwd.py emit, all of which are read from raw
# upstream tables before the pipeline assembles the anchor cohort.
#
# Date-shaped columns (medication_date, proc_date, etc.) are listed
# here because the WINDOWED aggregations in the manifest already enforce
# pre-anchor selection (window_days param). The raw column itself is
# not pre/post-anchor; the windowed feature is.
#
# Codex pass-2 MED-8 PARTIAL closure (PR #137 v4 G1): post-anchor
# target columns (``treatment_initiated``, ``discontinuation_flag``,
# ``initiated_biologic_180d``) MUST NOT appear here. They live in
# POST_ANCHOR_DERIVATION_INPUTS (below) and are accepted as derivation
# inputs ONLY when the audited PARENT feature is itself post_index.
# Putting them in PRE_ANCHOR_RAW_COLUMNS allowed an undeclared
# derivation input that referenced a target column to silently pass
# the audit even on a pre-anchor parent — that is the exact leak this
# audit must catch.
PRE_ANCHOR_RAW_COLUMNS: dict[str, frozenset[str]] = {
    "csu": frozenset(
        {
            # Demographics (read from member-eligibility table at enrollment)
            "age",
            "gdr_cd",
            "zipcode_5",
            "bus",
            "diagcode",
            # Enrollment span (member-eligibility table)
            "eligeff",
            "eligend",
            # Index-date source column
            "indexdt",
            # Date-shaped event columns (windowed by manifest derivations)
            "medication_date",
            "proc_date",
            "fst_dt",
            "days_sup",
            # Provider / brand normalisation columns
            "npi",
            "brand_normalised",
            "proc_code",
            "abnl_cd",
        }
    ),
    "optum": frozenset(
        {
            # Demographics
            "age",
            "gdr_cd",
            "zipcode_5",
            "zip5",
            "zip3",
            "bus",
            "product",
            "diagcode",
            "diagcode_raw",
            # Enrollment + index
            "eligeff",
            "eligend",
            "indexdt",
            "index_date",
            # Date-shaped event columns
            "medication_date",
            "drug_name",
            "proc_date",
            "fst_dt",
            "days_sup",
            "admit_date",
            "diag1",
            "diag2",
            "diag3",
            "diag4",
            "diag5",
            "tos_cd",
            "loinc_cd",
            "result",
            # Provider / brand
            "npi",
            "brand_normalised",
            "proc_code",
            "abnl_cd",
        }
    ),
}


# Codex pass-2 MED-8 PARTIAL closure (PR #137 v4 G1): post-anchor
# target columns that may legitimately appear as derivation inputs
# of POST-INDEX features ONLY (e.g., ``journey_status`` uses
# ``treatment_initiated`` to compute its label-derived value). Those
# parent features are themselves correctly flagged post_index by
# Layer 1 and are dropped before training; the audit's job here is
# merely to confirm the post-anchor input is a recognized target
# column rather than an arbitrary undeclared post-anchor leak.
#
# An undeclared derivation input that names one of these target
# columns ON A PRE-ANCHOR PARENT feature MUST FAIL the audit (because
# a pre-anchor feature reading a target column is exactly the leak
# pattern v4 G1 forbids). The recursive auditor enforces the
# parent-must-be-post_index gate at the call site.
POST_ANCHOR_DERIVATION_INPUTS: dict[str, frozenset[str]] = {
    "csu": frozenset(
        {
            "treatment_initiated",
            "discontinuation_flag",
        }
    ),
    "optum": frozenset(
        {
            "treatment_initiated",
            "discontinuation_flag",
            "initiated_biologic_180d",
        }
    ),
}


def _audit_derivation_inputs_recursively(
    feature_name: str,
    data_source: str,
    contracts_by_name: dict,
    visited: set | None = None,
) -> tuple[bool, list[str]]:
    """Walk a feature's derivation_inputs and audit each input that is
    itself a manifest-declared feature.

    Returns (all_inputs_pre_anchor, list_of_violation_messages).

    Codex pass-1 MED-8 (PR #137 v4 G1): an input that is NOT a
    manifest-declared feature MUST be an entry in
    ``PRE_ANCHOR_RAW_COLUMNS[data_source]`` — otherwise the audit
    fails. Previously such inputs were silently skipped, masking
    undeclared post-anchor leaks.

    Codex pass-2 MED-8 PARTIAL closure: post-anchor target columns
    (``treatment_initiated``, ``discontinuation_flag``,
    ``initiated_biologic_180d``) live in
    ``POST_ANCHOR_DERIVATION_INPUTS`` and are accepted ONLY when the
    audited PARENT feature is itself post_index. A pre-anchor parent
    that names a target column as a derivation input MUST FAIL — that
    is the exact leak pattern v4 G1 forbids.
    """
    if visited is None:
        visited = set()
    if feature_name in visited:
        return True, []
    visited.add(feature_name)

    contract = contracts_by_name.get(feature_name)
    if contract is None:
        # Not a manifest-declared feature; nothing to walk.
        return True, []

    raw_registry = PRE_ANCHOR_RAW_COLUMNS.get(data_source, frozenset())
    post_anchor_targets = POST_ANCHOR_DERIVATION_INPUTS.get(data_source, frozenset())

    # Codex pass-2 MED-8: parent's pre/post-anchor status governs
    # whether post-anchor target columns are admissible as
    # derivation inputs. Pre-anchor parents MUST NOT reference any
    # post-anchor target column; post-anchor parents may.
    parent_is_pre_anchor = (
        contract.knowable_at is not None and contract.knowable_at.is_pre_or_at_index()
    )

    violations: list[str] = []
    for input_name in contract.derivation_inputs:
        input_contract = contracts_by_name.get(input_name)
        if input_contract is None:
            # Codex pass-1 MED-8: previously raw columns were assumed
            # safe-by-convention. Now we require every undeclared input
            # to be in the source-level raw-column registry OR (when
            # the parent is post-anchor) the post-anchor target
            # registry — otherwise an undeclared post-anchor column
            # would silently slip past the audit.
            if input_name in raw_registry:
                # Pre-anchor raw column — admissible for any parent.
                continue
            if input_name in post_anchor_targets:
                # Codex pass-2 MED-8 PARTIAL closure: post-anchor
                # target columns admissible ONLY on post-anchor
                # parents. A pre-anchor parent reading a target
                # column is the exact leak v4 G1 forbids.
                if parent_is_pre_anchor:
                    violations.append(
                        f"{feature_name} -> {input_name}: pre-anchor "
                        f"parent feature references a POST-ANCHOR "
                        f"TARGET column from POST_ANCHOR_DERIVATION_INPUTS"
                        f"[{data_source!r}]. This is exactly the leak "
                        f"pattern v4 G1 forbids: pre-anchor features "
                        f"MUST NOT reference target columns. Either "
                        f"re-declare the parent as post_index, OR "
                        f"re-shape the derivation to use a "
                        f"pre-anchor intermediate."
                    )
                # Else parent is post-anchor → admissible; the parent
                # feature is itself dropped by Layer 1 before training.
                continue
            violations.append(
                f"{feature_name} -> {input_name}: undeclared "
                f"derivation input — not a manifest-declared "
                f"feature AND not in PRE_ANCHOR_RAW_COLUMNS"
                f"[{data_source!r}] AND not in "
                f"POST_ANCHOR_DERIVATION_INPUTS[{data_source!r}]. "
                f"Either declare a manifest contract for it, or "
                f"add it to the appropriate raw-column registry "
                f"with rationale."
            )
            continue
        # Recursively audit the input.
        if not input_contract.knowable_at.is_pre_or_at_index():
            violations.append(
                f"{feature_name} -> {input_name}: input has "
                f"knowable_at={input_contract.knowable_at} (post-anchor)"
            )
            continue
        # Use the helper to confirm.
        result = lineage_audit_declared_path(input_name, data_source=data_source)
        if result["declared_path_valid"] is not True:
            violations.append(
                f"{feature_name} -> {input_name}: helper returned "
                f"declared_path_valid={result['declared_path_valid']!r}"
            )
            continue
        # Recurse into the input's inputs.
        ok, sub_violations = _audit_derivation_inputs_recursively(
            input_name, data_source, contracts_by_name, visited
        )
        if not ok:
            violations.extend(sub_violations)

    return not violations, violations


@pytest.mark.parametrize(
    "data_source,features",
    [
        ("csu", CSU_FEATURES),
        ("optum", OPTUM_FEATURES),
    ],
)
def test_g1_derivation_inputs_subset_of_pre_anchor(
    data_source: str,
    features: list,
) -> None:
    """Plan v4 §2 G1 (codex-rescue MEDIUM-4): every Layer-1-cleared
    manifest feature has ``derivation_inputs ⊆ pre-anchor`` per
    ``MANIFEST_SOURCES``.

    Walks the contract chain — for each pre-anchor feature, every input
    that is itself a manifest-declared feature must also be pre-anchor.
    Inputs that are NOT in the manifest registry (raw columns, external
    timestamps like ``"birth_date"``) are skipped per the
    ``validate_contract_chain`` convention.

    This is the transitive lineage check the v3 §6 Gate B1 spec named
    but never landed; it's the v4 G1 closure.
    """
    contracts_by_name = {c.name: c for c in features}

    all_violations: list[str] = []
    for contract in features:
        if not contract.knowable_at.is_pre_or_at_index():
            continue
        ok, violations = _audit_derivation_inputs_recursively(
            contract.name, data_source, contracts_by_name
        )
        if not ok:
            all_violations.extend(violations)

    assert not all_violations, (
        f"G1 transitive lineage sweep ({data_source}): "
        f"{len(all_violations)} ``derivation_inputs ⊆ pre-anchor`` "
        f"violations:\n  - " + "\n  - ".join(all_violations) + "\n"
        "A pre-anchor feature claims an input that is post-anchor. "
        "Either the input's manifest entry is wrong, or the parent "
        "feature must be re-declared as post_index."
    )


# --------------------------------------------------------------------------- #
# Sweep 3: HBLP relaxation surfaces only declared-path-valid features         #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "data_source,features,n_positives_anchor",
    [
        # CSU n_train_pos≈98 — HBLP variance-inflation factor = 1.0 (no
        # relaxation). Sweep is degenerate but pinned for completeness.
        ("csu", CSU_FEATURES, 98),
        # Optum n_train_pos≈22 — HBLP inflation factor ~1.51, i.e.,
        # threshold rises from 5σ to ~7.54σ. The relaxation IS active.
        ("optum", OPTUM_FEATURES, 22),
    ],
)
def test_g1_hblp_relaxation_surfaces_only_declared_path_valid_features(
    data_source: str,
    features: list,
    n_positives_anchor: int,
) -> None:
    """Plan v4 §2 G1 (the load-bearing acceptance criterion):

        every feature surfaced by HBLP relaxation has
        ``derivation_inputs ⊆ pre-anchor`` per ``MANIFEST_SOURCES``

    "Surfaced by HBLP relaxation" ≜ a feature whose Layer 3 z-score
    falls in the band ``[5σ, hblp_effective_z]`` AND whose Layer 1
    ``declared_safe`` flag is True. Under legacy 5σ this would be
    severity=high (drop); under HBLP it's severity=moderate or info
    (keep). The v4 G1 invariant: every such feature must pass the
    lineage audit.

    This test SIMULATES the relaxation by:
    1. For each Layer-1-cleared feature in the manifest,
    2. Constructing a synthetic z-score in the relaxation band
       (e.g., 6σ — between legacy 5σ and HBLP-effective ~7.54σ at
       Optum n=22),
    3. Running ``hblp_classify`` to confirm severity ≠ "high"
       (i.e., the feature WOULD be relaxed in a real run),
    4. Calling ``lineage_audit_declared_path`` and asserting
       ``declared_path_valid is True``.

    A failure here means: under HBLP relaxation, the manifest registry
    would admit a declared-path-invalid feature into the trained model
    set. That's the precondition the G1 invariant forbids.
    """
    # Confirm HBLP relaxation IS active for this anchor.
    effective_z = hblp_effective_z_threshold(
        n_positives=n_positives_anchor,
        layer_1_declared_safe=True,
    )
    if effective_z <= 5.0:
        # Codex pass-1 HIGH-5 (PR #137 v4 G1): replace skip with
        # explicit assertion. CSU at n_train_pos≈98 has HBLP
        # variance-inflation factor = 1.0 (no relaxation), therefore
        # ``csu_relaxed_features == []`` is the load-bearing
        # invariant the v4 G1 spec demands. A future HBLP redesign
        # that activated relaxation at n=98 would silently void the
        # CSU negative-control invariant; this assertion catches
        # that case.
        relaxed_features = [
            c.name
            for c in features
            if c.knowable_at.is_pre_or_at_index()
            and hblp_classify(
                effective_z + 0.01,  # any z above effective_z is "relaxed"
                n_positives=n_positives_anchor,
                layer_1_declared_safe=True,
            )["hblp_relaxed"]
        ]
        # When effective_z ≤ 5.0, hblp_relaxed is False by definition
        # (relaxation factor = 1.0). So relaxed_features should be [].
        assert relaxed_features == [], (
            f"G1 HBLP-relaxation invariant violated for {data_source} "
            f"at n_positives={n_positives_anchor}: HBLP-effective z = "
            f"{effective_z} ≤ legacy 5σ but {len(relaxed_features)} "
            f"features were marked hblp_relaxed=True. Expected empty "
            f"list. Features: {relaxed_features}"
        )
        # CSU degenerate sweep: nothing to audit beyond this point.
        # Static lineage sweep (test_g1_every_pre_anchor_feature_passes_lineage_audit)
        # exercises the full Layer-1-cleared feature set.
        return

    # Pick a z in the relaxation band — between legacy 5σ and HBLP-effective.
    relaxed_z = 0.5 * (5.0 + effective_z)
    assert 5.0 < relaxed_z < effective_z, (
        f"Relaxation-band z out of range: 5 < {relaxed_z} < {effective_z}"
    )

    failures: list[str] = []
    for contract in features:
        if not contract.knowable_at.is_pre_or_at_index():
            continue
        # Layer 1 cleared this feature; HBLP relaxation applies.
        cls = hblp_classify(
            relaxed_z,
            n_positives=n_positives_anchor,
            layer_1_declared_safe=True,
        )
        # The relaxation MUST keep this feature (severity != high).
        # If hblp_classify says high, our z choice was wrong.
        assert cls["severity"] != "high", (
            f"Test setup: chose z={relaxed_z} thinking it's relaxation-band, "
            f"but hblp_classify returned severity=high for "
            f"n={n_positives_anchor}. Re-check thresholds."
        )
        # The G1 invariant: every relaxed feature has valid declared path.
        result = lineage_audit_declared_path(contract.name, data_source=data_source)
        if not result["contract_found"]:
            failures.append(f"{contract.name}: contract_found=False under HBLP relaxation")
            continue
        if result["declared_path_valid"] is not True:
            failures.append(
                f"{contract.name}: declared_path_valid="
                f"{result['declared_path_valid']!r} under HBLP relaxation; "
                f"knowable_at_reference={result['knowable_at_reference']!r}"
            )

    assert not failures, (
        f"G1 HBLP-relaxation lineage sweep ({data_source}, "
        f"n_positives={n_positives_anchor}): "
        f"{len(failures)} features WOULD be admitted by HBLP relaxation "
        f"but FAIL the lineage audit:\n  - " + "\n  - ".join(failures) + "\n"
        "This is the precondition v4 G1 forbids: HBLP wiring "
        "(Phase C, G3) cannot land while these features remain in "
        "the manifest as Layer-1-cleared."
    )


# --------------------------------------------------------------------------- #
# Sweep 4: lineage audit on ACTUAL relaxed features from a captured artifact  #
# --------------------------------------------------------------------------- #


# Codex pass-2 HIGH-4 (PR #137 v4 G1): per-source identity expectations the
# captured pipeline artifact MUST match. Used by the relaxed-feature audit
# to confirm the artifact in the registry actually came from the cohort
# we're trying to audit (rather than a stray same-pytest-run artifact from
# an unrelated test).
G1_ARTIFACT_IDENTITY = {
    "csu": {
        "feature_manifest_source": "csu",
        "indication": "Chronic Spontaneous Urticaria (CSU)",
        "expected_cohort_size": 9607,
    },
    "optum": {
        "feature_manifest_source": "optum",
        "indication": "initiation",
        "expected_cohort_size": 1294,
    },
}


@pytest.mark.slow
@pytest.mark.integration
@pytest.mark.real_data
@pytest.mark.timeout(2000)
@pytest.mark.parametrize("data_source", ["csu", "optum"])
def test_g1_lineage_audit_on_actual_relaxed_features(
    data_source: str,
    g1_artifact_registry: dict,
    request: pytest.FixtureRequest,
) -> None:
    """Codex pass-1 HIGH-4 + pass-2 HIGH-4 NOT-CLOSED (PR #137 v4 G1):
    audit ACTUAL relaxed features from a captured pipeline run, not
    synthetic z-scores.

    Reads the captured pipeline artifact JSON from the SESSION-SHARED
    ``g1_artifact_registry`` populated by the CSU + Optum integration
    test fixtures in this same pytest run. There is NO env-var
    indirection; the registry is the single source of truth so the
    captured-artifact path is always exercised when the upstream
    fixtures ran.

    Codex pass-2 HIGH-4 closure
    ---------------------------
    The pass-1 implementation skipped when ``G1_CSU_ARTIFACT_PATH`` /
    ``G1_OPTUM_ARTIFACT_PATH`` env vars were unset. CI configurations
    rarely set those env vars, so the captured-artifact half could go
    green without auditing any actual relaxed feature. The pass-2 fix
    (a) consumes artifacts from the SAME pytest run via a
    session-shared registry, (b) defaults to HARD-FAIL when no artifact
    is registered (the symmetric pair of HIGH-1's data-fixture policy),
    and (c) opt-in skip via ``ALLOW_MISSING_REAL_DATA=1`` only.

    Codex pass-2 HIGH-5 closure
    ---------------------------
    Beyond the CSU empty-set assertion (the original HIGH-5 invariant),
    we additionally:
    - Assert artifact identity (``feature_manifest_source``,
      ``indication``, ``cohort_size``) so a stray artifact from an
      unrelated run cannot satisfy this gate.
    - Assert ``adaptive_verdicts`` is present AND non-empty (a
      vacuous-pass on empty/malformed verdicts is itself a regression).
    - For Optum, assert the relaxed-feature set is non-empty (HBLP
      relaxation IS active at n_train_pos≈22). A captured Optum run
      that surfaces zero relaxed features means HBLP isn't firing or
      the verdict payload schema regressed.

    HIGH-5 invariant for CSU: the captured artifact's relaxed-feature
    set MUST be ``[]`` (CSU n_train_pos≈98 has no HBLP relaxation
    active by construction).
    """
    import json
    import os

    artifact_path = g1_artifact_registry.get(data_source)
    if artifact_path is None:
        # Codex pass-2 HIGH-4: default-hard-fail when no upstream
        # fixture registered an artifact. Mirrors the
        # ALLOW_MISSING_REAL_DATA=1 policy of the CSU/Optum fixtures
        # (codex pass-1 HIGH-1) so writer + reader share the same
        # escape hatch.
        if os.environ.get("ALLOW_MISSING_REAL_DATA") == "1":
            pytest.skip(
                f"g1_artifact_registry has no '{data_source}' artifact "
                f"and ALLOW_MISSING_REAL_DATA=1; G1 captured-artifact "
                f"audit skipped explicitly. Locally: run "
                f"{data_source.upper()} integration test in the same "
                f"pytest invocation."
            )
        pytest.fail(
            f"G1 HIGH-4: g1_artifact_registry has no '{data_source}' "
            f"artifact registered. The {data_source.upper()} integration "
            f"test fixture either did not run in this pytest invocation "
            f"or did not register its artifact path. Run the CSU + Optum "
            f"integration tests in the SAME pytest invocation as this "
            f"sweep, OR set ALLOW_MISSING_REAL_DATA=1 to opt into a "
            f"skip."
        )

    if not artifact_path.exists():
        pytest.fail(
            f"G1 HIGH-4: registered artifact path {artifact_path} does "
            f"not exist. The upstream fixture wrote the path but the "
            f"file is missing. Either pytest's tmp_path was cleaned up "
            f"between fixtures (re-order test files so the producer "
            f"runs BEFORE this consumer in the same pytest session) or "
            f"the runner deleted the artifact post-emission."
        )

    artifact = json.loads(artifact_path.read_text())

    # Codex pass-2 HIGH-5: artifact identity assertions. Without these,
    # a stray artifact from an unrelated cohort run could satisfy this
    # gate (vacuous pass).
    identity = G1_ARTIFACT_IDENTITY[data_source]
    actual_manifest = artifact.get("feature_manifest_source")
    assert actual_manifest == identity["feature_manifest_source"], (
        f"G1 HIGH-5: artifact at {artifact_path} has "
        f"feature_manifest_source={actual_manifest!r}, expected "
        f"{identity['feature_manifest_source']!r}. The artifact is from "
        f"a different cohort than this {data_source} sweep iteration."
    )
    actual_indication = artifact.get("indication")
    assert actual_indication == identity["indication"], (
        f"G1 HIGH-5: artifact at {artifact_path} has "
        f"indication={actual_indication!r}, expected "
        f"{identity['indication']!r}. Wrong-cohort artifact."
    )
    actual_cohort_size = artifact.get("cohort_size")
    assert actual_cohort_size == identity["expected_cohort_size"], (
        f"G1 HIGH-5: artifact at {artifact_path} has "
        f"cohort_size={actual_cohort_size}, expected "
        f"{identity['expected_cohort_size']} for {data_source}. "
        f"Cohort-build regression OR wrong-cohort artifact."
    )

    # Codex pass-2 HIGH-5: adaptive_verdicts MUST be present + non-empty.
    # A vacuous pass on empty/malformed verdicts is itself a regression
    # — Layer 5 should always emit verdicts on a real-data run.
    verdicts = artifact.get("adaptive_verdicts")
    assert verdicts is not None, (
        f"G1 HIGH-5: artifact at {artifact_path} has no "
        f"adaptive_verdicts key. The runner's audit emission regressed."
    )
    assert isinstance(verdicts, list), (
        f"G1 HIGH-5: artifact at {artifact_path} adaptive_verdicts is "
        f"not a list (got {type(verdicts).__name__}). Schema regression."
    )
    assert len(verdicts) > 0, (
        f"G1 HIGH-5: artifact at {artifact_path} adaptive_verdicts is "
        f"empty. Layer 5 did not produce a single verdict on the real "
        f"{data_source} run — vacuous-pass guard fires here."
    )

    # A "relaxed feature" in the captured artifact is a layer="3"
    # verdict whose hblp_relaxed=True (the helper marked it relaxed
    # in the actual pipeline run).
    relaxed_feature_names: set[str] = set()
    for v in verdicts:
        if not isinstance(v, dict):
            continue
        if v.get("layer") != "3":
            continue
        feat = v.get("feature")
        if not feat:
            continue
        if v.get("hblp_relaxed") is True:
            relaxed_feature_names.add(feat)

    if data_source == "csu":
        # Codex pass-1 HIGH-5: CSU n=98 → no HBLP relaxation → the
        # relaxed-feature set in any real CSU artifact MUST be empty.
        assert relaxed_feature_names == set(), (
            f"G1 HIGH-5: CSU artifact at {artifact_path} reports "
            f"{len(relaxed_feature_names)} HBLP-relaxed features "
            f"({sorted(relaxed_feature_names)}); expected [] because "
            f"CSU n_train_pos≈98 has variance-inflation factor=1.0. "
            f"This indicates either (a) HBLP wiring activated unexpectedly "
            f"on CSU, (b) the artifact is from a non-CSU run, or (c) the "
            f"verdict payload schema regressed."
        )
        return  # CSU sweep is degenerate by construction.

    # Optum: audit each relaxed feature.
    # Codex pass-2 HIGH-4 + HIGH-5: assert non-empty relaxed-feature
    # set. Optum n_train_pos≈22 has HBLP variance-inflation factor
    # ~1.51 (5σ → ~7.54σ effective threshold); the relaxed band IS
    # active and a real Optum run MUST surface at least one
    # hblp_relaxed feature. Vacuous pass = explicit fail.
    assert relaxed_feature_names, (
        f"G1 HIGH-4 (Optum, captured artifact at {artifact_path}): "
        f"zero hblp_relaxed=True features in adaptive_verdicts. HBLP "
        f"relaxation MUST be active on Optum at n_train_pos≈22 "
        f"(effective z-threshold ≈ 7.54σ vs legacy 5σ). Either (a) HBLP "
        f"wiring did not fire on this run, (b) all Layer 3 features "
        f"scored below 5σ (the legacy floor — no relaxation needed), "
        f"or (c) the verdict payload schema regressed (hblp_relaxed "
        f"key not emitted). The captured-artifact audit is vacuous "
        f"unless this set is non-empty."
    )

    failures: list[str] = []
    for feat_name in sorted(relaxed_feature_names):
        result = lineage_audit_declared_path(feat_name, data_source=data_source)
        if not result["contract_found"]:
            failures.append(f"{feat_name}: contract_found=False under HBLP relaxation")
            continue
        if result["declared_path_valid"] is not True:
            failures.append(
                f"{feat_name}: declared_path_valid="
                f"{result['declared_path_valid']!r}; "
                f"knowable_at_reference={result['knowable_at_reference']!r}"
            )

    assert not failures, (
        f"G1 HIGH-4 (Optum, captured artifact at {artifact_path}): "
        f"{len(failures)} ACTUAL HBLP-relaxed features FAIL the "
        f"lineage audit:\n  - " + "\n  - ".join(failures) + "\n"
        "This is the load-bearing v4 G1 invariant on real-pipeline "
        "output. HBLP wiring (Phase C, G3) cannot land while these "
        "features fail the audit."
    )


# --------------------------------------------------------------------------- #
# Helper sanity: MANIFEST_SOURCES coverage                                    #
# --------------------------------------------------------------------------- #


def test_g1_manifest_sources_registry_covers_csu_optum() -> None:
    """Sanity: ``MANIFEST_SOURCES`` registry has both CSU and Optum
    declared. If the registry shape changes, the parametrized sweep
    above would silently skip a cohort; this test catches that.
    """
    assert "csu" in MANIFEST_SOURCES
    assert "optum" in MANIFEST_SOURCES
    # Each entry must be a callable (matches the
    # `Mapping[str, Callable[[str], FeatureContract | None]]` type).
    assert callable(MANIFEST_SOURCES["csu"])
    assert callable(MANIFEST_SOURCES["optum"])


def test_g1_undeclared_derivation_input_fails_med_8() -> None:
    """Codex pass-1 MED-8 (PR #137 v4 G1): an undeclared derivation
    input — neither a manifest-declared feature NOR in the
    ``PRE_ANCHOR_RAW_COLUMNS`` registry — MUST fail the audit.

    Builds a synthetic in-memory manifest naming an input that is
    not in either catalog and verifies the recursive auditor
    surfaces a violation. Also verifies that an input listed in
    ``PRE_ANCHOR_RAW_COLUMNS`` does NOT fail.
    """
    from src.data.feature_contract import FeatureContract, KnowableAt

    # Synthetic source name not in MANIFEST_SOURCES; the helper that
    # _audit_derivation_inputs_recursively calls (lineage_audit_declared_path)
    # short-circuits unknown sources, so our test exercises only the
    # raw-column registry branch.
    fake_source = "med8_synthetic"

    # Build a contract that references an undeclared raw input.
    bad_contract = FeatureContract(
        name="med8_undeclared_input_feature",
        knowable_at=KnowableAt(reference="index_date"),
        source="synthetic",
        derivation_inputs=("__med8_definitely_not_in_registry__",),
    )

    # And one that references a known raw column (should not fail
    # the registry branch).
    good_contract = FeatureContract(
        name="med8_declared_input_feature",
        knowable_at=KnowableAt(reference="index_date"),
        source="synthetic",
        derivation_inputs=("medication_date",),  # in CSU registry
    )

    # Use the CSU registry by piggy-backing on the fake_source key.
    # We modify the registry transiently with monkeypatching; simpler
    # to override locally with the global patch via a try/finally.
    original = PRE_ANCHOR_RAW_COLUMNS.get(fake_source)
    PRE_ANCHOR_RAW_COLUMNS[fake_source] = PRE_ANCHOR_RAW_COLUMNS["csu"]
    try:
        contracts_by_name_bad = {bad_contract.name: bad_contract}
        ok_bad, violations_bad = _audit_derivation_inputs_recursively(
            bad_contract.name, fake_source, contracts_by_name_bad
        )
        assert not ok_bad
        assert any("__med8_definitely_not_in_registry__" in v for v in violations_bad), (
            f"expected violation message to name the offending input, got: {violations_bad}"
        )
        assert any("PRE_ANCHOR_RAW_COLUMNS" in v for v in violations_bad), (
            f"expected violation message to point at PRE_ANCHOR_RAW_COLUMNS, got: {violations_bad}"
        )

        contracts_by_name_good = {good_contract.name: good_contract}
        ok_good, violations_good = _audit_derivation_inputs_recursively(
            good_contract.name, fake_source, contracts_by_name_good
        )
        assert ok_good, f"expected pass, got violations: {violations_good}"
        assert violations_good == []
    finally:
        if original is None:
            PRE_ANCHOR_RAW_COLUMNS.pop(fake_source, None)
        else:
            PRE_ANCHOR_RAW_COLUMNS[fake_source] = original


def test_g1_pre_anchor_feature_with_target_input_fails_med_8_partial() -> None:
    """Codex pass-2 MED-8 PARTIAL closure (PR #137 v4 G1): a pre-anchor
    feature whose ``derivation_inputs`` reference a post-anchor target
    column MUST FAIL the audit.

    This is the load-bearing v4 G1 invariant the pass-1 fix did NOT
    cover: putting target columns (``treatment_initiated``,
    ``discontinuation_flag``, ``initiated_biologic_180d``) in
    ``PRE_ANCHOR_RAW_COLUMNS`` allowed an undeclared input on a
    pre-anchor parent to silently pass — exactly the leak pattern
    v4 G1 forbids. The pass-2 fix splits the registry: target columns
    move to ``POST_ANCHOR_DERIVATION_INPUTS`` and are accepted as
    inputs ONLY when the audited parent is itself post_index.

    This regression test enforces the invariant by:
    1. Building a synthetic feature with ``knowable_at=enrollment``
       (i.e., pre-anchor) whose ``derivation_inputs=("treatment_initiated",)``.
    2. Asserting the auditor emits a violation that points at the
       POST_ANCHOR_DERIVATION_INPUTS gate.

    Under the OLD (pass-1) registry this case would have PASSED
    silently because ``treatment_initiated`` was whitelisted in
    PRE_ANCHOR_RAW_COLUMNS. Under the NEW (pass-2) split, it FAILS
    loudly.
    """
    from src.data.feature_contract import FeatureContract, KnowableAt

    fake_source = "med8_partial_synthetic"

    # Synthetic pre-anchor feature with a target-column input. This
    # is the exact leak pattern v4 G1 forbids.
    leaky_pre_anchor = FeatureContract(
        name="leaky_demographic_referencing_target",
        knowable_at=KnowableAt(reference="enrollment"),
        source="synthetic",
        derivation_inputs=("treatment_initiated",),  # post-anchor target
    )

    # Synthetic post-anchor feature with the same input. This SHOULD
    # pass — the parent is post-anchor and the input is in the
    # POST_ANCHOR_DERIVATION_INPUTS registry. This mirrors the real
    # csu/optum manifests (e.g., journey_status, journey_end_date).
    post_anchor_using_target = FeatureContract(
        name="legitimate_journey_status",
        knowable_at=KnowableAt(reference="post_index"),
        source="synthetic",
        derivation_inputs=("treatment_initiated",),
    )

    # Piggy-back on the CSU registries — they're the single source of
    # truth for which target columns are recognized.
    original_pre = PRE_ANCHOR_RAW_COLUMNS.get(fake_source)
    original_post = POST_ANCHOR_DERIVATION_INPUTS.get(fake_source)
    PRE_ANCHOR_RAW_COLUMNS[fake_source] = PRE_ANCHOR_RAW_COLUMNS["csu"]
    POST_ANCHOR_DERIVATION_INPUTS[fake_source] = POST_ANCHOR_DERIVATION_INPUTS["csu"]
    try:
        # Bad: pre-anchor parent reading a target column.
        contracts_bad = {leaky_pre_anchor.name: leaky_pre_anchor}
        ok_bad, violations_bad = _audit_derivation_inputs_recursively(
            leaky_pre_anchor.name, fake_source, contracts_bad
        )
        assert not ok_bad, (
            "MED-8 PARTIAL regression: pre-anchor feature referencing a "
            "post-anchor target column passed the audit. This is the exact "
            "leak pattern v4 G1 forbids; the auditor must catch it."
        )
        assert any("treatment_initiated" in v for v in violations_bad), (
            f"expected violation to name the offending target column, got: "
            f"{violations_bad}"
        )
        assert any("POST_ANCHOR_DERIVATION_INPUTS" in v for v in violations_bad), (
            f"expected violation to point at POST_ANCHOR_DERIVATION_INPUTS "
            f"gate, got: {violations_bad}"
        )
        assert any("pre-anchor parent" in v.lower() for v in violations_bad), (
            f"expected violation to surface 'pre-anchor parent' framing for "
            f"triage, got: {violations_bad}"
        )

        # Good: post-anchor parent reading a target column → admissible.
        contracts_good = {post_anchor_using_target.name: post_anchor_using_target}
        ok_good, violations_good = _audit_derivation_inputs_recursively(
            post_anchor_using_target.name, fake_source, contracts_good
        )
        assert ok_good, (
            f"MED-8 PARTIAL regression: post-anchor parent legitimately "
            f"referencing a target column failed the audit. Got violations: "
            f"{violations_good}. Real-manifest equivalents (journey_status, "
            f"journey_end_date) would also fail spuriously."
        )
        assert violations_good == []
    finally:
        # Restore registries to avoid test-order pollution.
        if original_pre is None:
            PRE_ANCHOR_RAW_COLUMNS.pop(fake_source, None)
        else:
            PRE_ANCHOR_RAW_COLUMNS[fake_source] = original_pre
        if original_post is None:
            POST_ANCHOR_DERIVATION_INPUTS.pop(fake_source, None)
        else:
            POST_ANCHOR_DERIVATION_INPUTS[fake_source] = original_post


def test_g1_target_columns_excluded_from_pre_anchor_raw_columns_med_8() -> None:
    """Codex pass-2 MED-8 PARTIAL closure (PR #137 v4 G1): no
    post-anchor target column appears in ``PRE_ANCHOR_RAW_COLUMNS``.

    The pass-1 registry whitelisted ``treatment_initiated``,
    ``discontinuation_flag``, and ``initiated_biologic_180d`` in
    ``PRE_ANCHOR_RAW_COLUMNS`` — that is the exact code path that
    allowed a pre-anchor feature to reference a target column and
    silently pass. The pass-2 split moves them to
    ``POST_ANCHOR_DERIVATION_INPUTS``. This test pins the invariant
    so a future PR cannot regress the registry.
    """
    forbidden_in_pre_anchor = {
        "treatment_initiated",
        "discontinuation_flag",
        "initiated_biologic_180d",
    }
    for source, raw_cols in PRE_ANCHOR_RAW_COLUMNS.items():
        regressed = forbidden_in_pre_anchor & raw_cols
        assert not regressed, (
            f"MED-8 PARTIAL regression: PRE_ANCHOR_RAW_COLUMNS[{source!r}] "
            f"contains post-anchor target columns: {sorted(regressed)}. "
            f"These belong in POST_ANCHOR_DERIVATION_INPUTS instead. A "
            f"target column in PRE_ANCHOR_RAW_COLUMNS allows a pre-anchor "
            f"feature to silently reference a label-derived value — the "
            f"exact leak pattern v4 G1 forbids."
        )

    # Spot-check the post-anchor registry is populated for both
    # production sources (otherwise the legitimate post-anchor
    # parents like journey_status would fail their own audit).
    assert "treatment_initiated" in POST_ANCHOR_DERIVATION_INPUTS["csu"]
    assert "discontinuation_flag" in POST_ANCHOR_DERIVATION_INPUTS["csu"]
    assert "treatment_initiated" in POST_ANCHOR_DERIVATION_INPUTS["optum"]
    assert "discontinuation_flag" in POST_ANCHOR_DERIVATION_INPUTS["optum"]
    assert "initiated_biologic_180d" in POST_ANCHOR_DERIVATION_INPUTS["optum"]


def test_g1_lineage_helper_signature_unchanged_from_pr_127() -> None:
    """Plan v4 §2 G1 references PR #127's ``lineage_audit_declared_path``
    helper signature. If a future refactor changes the public surface
    (e.g., adds a required positional arg, or removes the
    ``data_source: Optional[str]`` accept-None contract), this test
    fires.

    Codex pass-1 INFO-13: pin the RETURN SCHEMA in addition to the
    parameter signature. The G1 lineage-audit sweep and the
    downstream pipeline node both read specific keys from the
    returned dict; a key rename or removal would silently break
    callers.

    Mirrors the discipline of pinning helper signatures used by
    cross-PR contracts (matches the pattern in PRs #105/#106 for
    ``data_loader._drop_unhashable_columns``).
    """
    import inspect

    sig = inspect.signature(lineage_audit_declared_path)
    params = sig.parameters
    assert "feature_name" in params
    assert "data_source" in params
    # Both should be positional-or-keyword (no required-keyword shift).
    assert params["feature_name"].kind == inspect.Parameter.POSITIONAL_OR_KEYWORD
    assert params["data_source"].kind == inspect.Parameter.POSITIONAL_OR_KEYWORD

    # Codex pass-1 INFO-13: return-schema pin. Probe with a real
    # source (CSU manifest's ``age_continuous`` Layer-1-cleared feature
    # used elsewhere in the sweep) so the schema reflects the
    # contract-found branch — the most-exercised return shape.
    result = lineage_audit_declared_path("age_continuous", data_source="csu")
    required_keys = {
        "contract_found",
        "declared_path_valid",
        "knowable_at_reference",
        "rationale",
    }
    missing = required_keys - set(result.keys())
    assert not missing, (
        f"INFO-13 return-schema regression: required keys missing "
        f"from lineage_audit_declared_path() return: {sorted(missing)}. "
        f"Actual keys: {sorted(result.keys())}"
    )
    # Type pins on the contract-found branch
    assert isinstance(result["contract_found"], bool)
    # declared_path_valid is bool when contract_found, None otherwise
    assert isinstance(result["declared_path_valid"], (bool, type(None)))
    # knowable_at_reference is str when contract_found, may be None
    assert isinstance(result["knowable_at_reference"], (str, type(None)))
    assert isinstance(result["rationale"], str)

    # Probe the contract-not-found branch — feature_name that doesn't
    # exist in any manifest. The schema MUST still carry the same keys
    # so callers can rely on key presence regardless of branch.
    result_missing = lineage_audit_declared_path(
        "__info13_definitely_not_a_real_feature__", data_source="csu"
    )
    missing_missing = required_keys - set(result_missing.keys())
    assert not missing_missing, (
        f"INFO-13 return-schema regression on contract-not-found branch: "
        f"required keys missing: {sorted(missing_missing)}. "
        f"Actual keys: {sorted(result_missing.keys())}"
    )
    assert result_missing["contract_found"] is False
    # declared_path_valid must be None when contract_found is False
    # (the audit can't validate something that doesn't exist)
    assert result_missing["declared_path_valid"] is None
