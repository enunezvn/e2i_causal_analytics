"""Tests for the canonical model-name → Feast feature-refs registry.

Block 6B / 3A-M-1: ``src/feature_store/model_feature_refs.py`` is the
single source of truth for the predictions.py + explain.py registries
that previously duplicated each other.
"""

from __future__ import annotations


def test_canonical_registry_covers_all_known_model_types():
    """``MODEL_FEATURE_REFS`` must enumerate every ``ModelType`` value
    used by the explanation route — keeping the two in lockstep is the
    whole point of the consolidation."""
    from src.api.routes.explain import ModelType
    from src.feature_store.model_feature_refs import MODEL_FEATURE_REFS

    expected_keys = {m.value for m in ModelType}
    registry_keys = set(MODEL_FEATURE_REFS.keys())
    missing = expected_keys - registry_keys
    assert not missing, (
        f"MODEL_FEATURE_REFS is missing entries for ModelType values: "
        f"{sorted(missing)}; registry currently has {sorted(registry_keys)}"
    )


def test_predictions_route_imports_canonical_registry():
    """``predictions._MODEL_FEATURE_REFS`` must BE the canonical
    registry, not a local copy. We compare object identity to catch
    accidental future copy-paste regressions."""
    from src.api.routes import predictions
    from src.feature_store.model_feature_refs import MODEL_FEATURE_REFS

    assert predictions._MODEL_FEATURE_REFS is MODEL_FEATURE_REFS, (
        "predictions._MODEL_FEATURE_REFS must be the same object as "
        "src.feature_store.model_feature_refs.MODEL_FEATURE_REFS — a "
        "fresh copy means the two will silently drift apart over time."
    )


def test_feature_refs_for_model_falls_back_to_propensity():
    """Unknown model name → propensity refs (Feast rejects empty refs)."""
    from src.feature_store.model_feature_refs import (
        MODEL_FEATURE_REFS,
        feature_refs_for_model,
    )

    refs = feature_refs_for_model("totally_unknown_model_xyz")
    assert refs == MODEL_FEATURE_REFS["propensity"]


def test_feature_refs_for_model_known_models():
    """Known models return their registered refs verbatim."""
    from src.feature_store.model_feature_refs import (
        MODEL_FEATURE_REFS,
        feature_refs_for_model,
    )

    for model_name, expected_refs in MODEL_FEATURE_REFS.items():
        actual = feature_refs_for_model(model_name)
        assert actual == expected_refs, (
            f"Mismatch for {model_name!r}: got {actual!r}, expected {expected_refs!r}"
        )


def test_patient_goldstd_cohorts_fetch_enriched_base7():
    """The 3 patient gold-standard cohorts must fetch the 7 ``_BASE7`` raw
    covariates (T9/T11 enrichment), not just the legacy base 3 — else the
    7-covariate serving bundle gets an incomplete vector (#576 null-trap) and
    the live Feature-Importance page silently shows 3 covariates."""
    from src.feature_store.model_feature_refs import MODEL_FEATURE_REFS

    base7 = {
        "goldstd_cohort_features:disease_severity",
        "goldstd_cohort_features:academic_hcp",
        "goldstd_cohort_features:geographic_region",
        "goldstd_cohort_features:insurance_type",
        "goldstd_cohort_features:age_at_diagnosis",
        "goldstd_cohort_features:comorbidity_burden",
        "goldstd_cohort_features:prior_therapy_lines",
    }
    # COMM-ARMS Phase 1/2: the three cohorts are no longer identical. persistence and
    # discontinuation additionally fetch the commercial arms copay_support (Phase 1) +
    # psp_enrolled (Phase 2), which both enter the discontinuation logit; initiation
    # does not, because neither is in the treatment_initiated equation. Asserted PER
    # COHORT rather than as one shared set so this still fails on drift in either
    # direction — a shared set would have had to be loosened to accommodate the split,
    # which would stop catching a cohort that silently loses a ref.
    commercial = {
        "goldstd_cohort_features:copay_support",
        "goldstd_cohort_features:psp_enrolled",
    }
    expected_by_cohort = {
        "initiation": base7,
        "persistence": base7 | commercial,
        "discontinuation": base7 | commercial,
    }
    for cohort, expected in expected_by_cohort.items():
        assert set(MODEL_FEATURE_REFS[cohort]) == expected, (
            f"{cohort} must fetch exactly {sorted(expected)}; "
            f"got {sorted(MODEL_FEATURE_REFS[cohort])}"
        )


def test_patient_cohort_refs_match_the_cohort_spec_covariates():
    """The registry and the model spec must agree on the covariate set.

    This is the guard the #576 null-trap actually needed: the serving bundle builds
    its vector from ``spec.base_covariates``, so a spec declaring N covariates while
    the refs fetch N-1 produces an incomplete vector at serving time and a 503. The
    pre-existing test above pins the refs to a literal, which catches a change to the
    refs but NOT a change to the spec that forgets them. Deriving one from the other
    closes that gap."""
    from src.feature_store.model_feature_refs import MODEL_FEATURE_REFS
    from src.mlops.gold_standard_eval.cohort_spec import BRANDS, make_patient_spec

    for cohort in ("initiation", "persistence", "discontinuation"):
        refs = {r.split(":", 1)[1] for r in MODEL_FEATURE_REFS[cohort]}
        for brand in BRANDS:
            covariates = set(make_patient_spec(cohort, brand).base_covariates)
            assert refs == covariates, (
                f"{cohort}/{brand}: feature refs {sorted(refs)} != spec covariates "
                f"{sorted(covariates)}; the serving bundle would get an incomplete "
                "vector (#576 null-trap)."
            )


def test_explain_route_uses_canonical_registry():
    """``explain.py:_get_feature_refs_for_model`` must return the same
    feature_refs as the canonical registry for known model types,
    preserving the legacy ``[]`` fallback for unknowns."""
    # Surface the inner method without instantiating the heavy class —
    # walk the class definition to grab the unbound method.
    from src.api.routes.explain import ExplainRequest, ModelType, RealTimeSHAPService  # noqa: F401
    from src.feature_store.model_feature_refs import MODEL_FEATURE_REFS

    service = RealTimeSHAPService.__new__(RealTimeSHAPService)
    for model_type in ModelType:
        refs = service._get_feature_refs_for_model(model_type)
        expected = MODEL_FEATURE_REFS[model_type.value]
        assert refs == expected, (
            f"explain.py registry diverged for {model_type!r}: got {refs!r}, expected {expected!r}"
        )


def test_canonical_registry_no_src_imports():
    """Architectural constraint: model_feature_refs.py must not import
    from ``src.*`` so it can be bundled into the BentoML container.

    Probe by reading the source file (not via ast) — the module is
    already imported by sibling tests, so it must work; the question
    is what it depends on."""
    import re
    from pathlib import Path

    module_path = (
        Path(__file__).resolve().parents[3] / "src" / "feature_store" / "model_feature_refs.py"
    )
    source = module_path.read_text(encoding="utf-8")
    # Strip docstrings and comments for the purposes of the import check.
    src_imports = re.findall(r"^\s*(?:from|import)\s+src\.", source, flags=re.MULTILINE)
    assert not src_imports, (
        f"model_feature_refs.py must not import from src.*; found: "
        f"{src_imports}. This module is bundled into the BentoML "
        f"container and pulling in src.* would balloon the image."
    )
