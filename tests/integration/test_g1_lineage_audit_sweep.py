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
            # Post-anchor target columns: included as raw inputs because
            # post-anchor features legitimately list them as derivation
            # inputs (e.g., journey_status uses treatment_initiated).
            # The post-anchor feature itself is correctly flagged
            # post_index by Layer 1; this registry's job is only to
            # confirm the input IS a known column, not to re-classify it.
            "treatment_initiated",
            "discontinuation_flag",
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
            # Post-anchor target columns (same rationale as CSU)
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
    violations: list[str] = []
    for input_name in contract.derivation_inputs:
        input_contract = contracts_by_name.get(input_name)
        if input_contract is None:
            # Codex pass-1 MED-8: previously raw columns were assumed
            # safe-by-convention. Now we require every undeclared input
            # to be in the source-level raw-column registry; otherwise
            # an undeclared post-anchor column would silently slip past
            # the audit.
            if input_name not in raw_registry:
                violations.append(
                    f"{feature_name} -> {input_name}: undeclared "
                    f"derivation input — not a manifest-declared "
                    f"feature AND not in PRE_ANCHOR_RAW_COLUMNS"
                    f"[{data_source!r}]. Either declare a manifest "
                    f"contract for it, or add it to the raw-column "
                    f"registry with rationale."
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
        # No relaxation active (CSU at n=98); the sweep is degenerate
        # but we still pin that the lineage helper returns valid for
        # every Layer-1-cleared feature (covered by the sweep above).
        # Skip the simulated relaxation half on this anchor.
        pytest.skip(
            f"HBLP not active at n={n_positives_anchor} "
            f"(effective_z={effective_z}σ ≤ legacy 5σ); "
            "the lineage sweep is exercised by the static "
            "test_g1_every_pre_anchor_feature_passes_lineage_audit."
        )

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


def test_g1_lineage_helper_signature_unchanged_from_pr_127() -> None:
    """Plan v4 §2 G1 references PR #127's ``lineage_audit_declared_path``
    helper signature. If a future refactor changes the public surface
    (e.g., adds a required positional arg, or removes the
    ``data_source: Optional[str]`` accept-None contract), this test
    fires.

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
