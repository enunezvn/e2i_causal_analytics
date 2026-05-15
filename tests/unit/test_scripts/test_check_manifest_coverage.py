"""Tests for ``scripts/check_manifest_coverage.py`` (Phase 1.5 Layer 1
manifest-coverage CI guard).

Test surface:

  * **Real cohort coverage** — CSU + 3 Optum cohorts pass cleanly
    against the live converter + manifest as-of-test-run.
  * **Synthetic missing-column scenario** — when a fake converter
    emits a column with no manifest entry and no allowlist match,
    the reconciler reports it as unmapped.
  * **Allowlist semantics** — both the literal ``AUDIT_COLUMN_ALLOWLIST``
    membership and the ``ALLOWED_PREFIXES`` prefix-match branches are
    exercised.
  * **Discovery sanity** — known feature names (``treatment_initiated``,
    ``age_at_index``, comorbidity expansion outputs) appear in the
    discovered set, so a converter refactor that renames the output
    dict would surface here.
  * **F-string allowed-prefix tolerance** — an unresolved f-string
    whose literal prefix matches ``ALLOWED_PREFIXES`` does NOT produce
    a hard error, even though no concrete column is enumerated.
  * **F-string non-allowed-prefix failure** — an unresolved f-string
    with an arbitrary prefix DOES produce a hard error.
  * **CLI integration** — main() with ``--only-cohort csu`` returns
    exit 0 in the happy path and exit 1 when a synthetic cohort
    config produces an unmapped column.
"""

from __future__ import annotations

import ast
import sys
import textwrap
from pathlib import Path

import pytest

# Ensure repo root on sys.path so the script imports cleanly.
_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts import check_manifest_coverage as cmc  # noqa: E402

# ---------------------------------------------------------------------------
# Real cohort coverage
# ---------------------------------------------------------------------------


def test_csu_manifest_fully_covers_csu_output() -> None:
    """CSU patient_journeys output must be fully covered by either the
    CSU manifest or the audit-column allowlist. Any failure here means
    a new CSU column landed without a manifest entry.
    """
    exit_code, reports, errors = cmc.check_all(
        _REPO_ROOT, only_cohorts=("csu",)
    )
    assert errors == [], f"discovery/manifest errors: {errors}"
    assert exit_code == 0, f"CSU coverage failure: {reports}"
    assert len(reports) == 1
    assert reports[0].cohort == "csu"
    assert reports[0].passed


def test_optum_manifest_fully_covers_all_three_cohorts() -> None:
    """The Optum manifest is shared across initiation / discontinuation
    / persistence. All three must pass coverage.
    """
    optum_cohorts = (
        "optum-initiation",
        "optum-discontinuation",
        "optum-persistence",
    )
    exit_code, reports, errors = cmc.check_all(
        _REPO_ROOT, only_cohorts=optum_cohorts
    )
    assert errors == [], f"discovery/manifest errors: {errors}"
    assert exit_code == 0, f"Optum coverage failure: {reports}"
    assert {r.cohort for r in reports} == set(optum_cohorts)
    for r in reports:
        assert r.passed, f"{r.cohort} unmapped={r.unmapped}"


def test_all_default_cohorts_pass() -> None:
    """The default (all-cohorts) invocation must exit 0 on a clean
    repo.
    """
    exit_code, reports, errors = cmc.check_all(_REPO_ROOT)
    assert errors == [], f"discovery/manifest errors: {errors}"
    assert exit_code == 0, f"coverage failure: {reports}"
    # Confirm all four cohorts ran.
    assert {r.cohort for r in reports} == {c.name for c in cmc.COHORTS}


# ---------------------------------------------------------------------------
# Discovery sanity
# ---------------------------------------------------------------------------


def test_csu_discovery_finds_known_columns() -> None:
    """The static AST walker must enumerate the CSU
    ``treatment_initiated`` and ``patient_journey_id`` columns. A
    converter refactor that renamed the output dict (``journey_dict``)
    would silently zero-out the discovered set; this guards against
    that drift.
    """
    discovered, errors = cmc.discover_columns_for_cohort(
        _REPO_ROOT, cmc.COHORTS[0]
    )
    assert errors == []
    for canonical in (
        "treatment_initiated",
        "patient_journey_id",
        "discontinuation_flag",
        "primary_diagnosis_code",
        "journey_status",
    ):
        assert canonical in discovered, f"missing canonical column {canonical!r}"


def test_optum_discovery_finds_loop_expanded_columns() -> None:
    """The loop-expansion path must produce ``has_atopic_dermatitis``,
    ``ige_total_tested``, ``h1_1g_ever_filled``, etc. — these come from
    ``for X in COMORBIDITY_CODES.items()`` etc. patterns. A regression
    in the loop-iterable resolver would zero out the expanded set.
    """
    optum_init = next(c for c in cmc.COHORTS if c.name == "optum-initiation")
    discovered, errors = cmc.discover_columns_for_cohort(_REPO_ROOT, optum_init)
    assert errors == []
    for canonical in (
        "has_atopic_dermatitis",
        "atopic_dermatitis_claim_count",
        "ige_total_tested",
        "ige_total_result_last",
        "ige_total_abnormal_flag",
        "h1_1g_ever_filled",
        "h1_1g_fill_count",
        "h1_1g_days_supply_total",
        "h1_1g_days_since_last_fill",
    ):
        assert canonical in discovered, f"missing loop-expanded column {canonical!r}"


def test_csu_discovery_ignores_non_output_dict_assignments() -> None:
    """Static analyser must NOT misattribute helper-dict subscript
    assignments to the journey output. The CSU converter has a
    ``type_counts = {"A": 0, "B": 0, "C": 0}`` literal that earlier
    iterations of this guard mistakenly collected. Verify the fix
    holds.
    """
    discovered, _ = cmc.discover_columns_for_cohort(_REPO_ROOT, cmc.COHORTS[0])
    # The single-character archetype labels MUST NOT appear in the
    # discovered surface — they're internal aggregation keys.
    assert "A" not in discovered
    assert "B" not in discovered
    assert "C" not in discovered


def test_optum_discovery_ignores_intermediate_dict_assignments() -> None:
    """Same property for Optum: ``l50_counts`` is an internal
    aggregation dict, NOT the journey output. Earlier iterations
    leaked its ``"total"`` key into the discovered set.
    """
    optum_init = next(c for c in cmc.COHORTS if c.name == "optum-initiation")
    discovered, _ = cmc.discover_columns_for_cohort(_REPO_ROOT, optum_init)
    # "total" is the l50_counts aggregation key; "L501" / "L508" / "L509"
    # are the per-prefix counters. None of these are journey columns.
    assert "total" not in discovered
    assert "L501" not in discovered
    assert "L508" not in discovered
    assert "L509" not in discovered


# ---------------------------------------------------------------------------
# Allowlist
# ---------------------------------------------------------------------------


def test_allowlist_literal_match() -> None:
    """Each canonical audit-column name is accepted via the literal
    allowlist."""
    for name in (
        "patient_journey_id",
        "patient_id",
        "patient_hash",
        "_patid",
        "created_at",
        "updated_at",
        "ingestion_timestamp",
        "source_timestamp",
        "data_source",
        "data_sources_matched",
        "primary_diagnosis_desc",
        "secondary_diagnosis_codes",
        "comorbidities",
        "data_quality_score",
        "risk_score",
        "state",
        "payer_bus_raw",
        "payer_product_raw",
    ):
        assert cmc._is_allowed(name), f"allowlist should accept {name!r}"


def test_allowlist_prefix_match() -> None:
    """The ``demo_*`` runtime pass-through prefix matches any
    ``demo_<anything>`` key."""
    assert cmc._is_allowed("demo_pat_typ")
    assert cmc._is_allowed("demo_state_2020")
    assert cmc._is_allowed("demo_arbitrary_runtime_label")


def test_allowlist_rejects_unknown_columns() -> None:
    """A column that's not in the literal allowlist and doesn't match
    any prefix is rejected."""
    assert not cmc._is_allowed("a_brand_new_feature")
    assert not cmc._is_allowed("ed_visits_total")  # this IS in manifest, not allowlist


def test_allowlist_accepts_visitor_sentinel() -> None:
    """The ``<allowed-prefix>`` sentinel emitted by the visitor when
    an f-string subscript matches ALLOWED_PREFIXES but the variable
    can't be resolved is accepted by ``_is_allowed``."""
    sentinel = cmc._ColumnDiscoveryVisitor._ALLOWED_PREFIX_SENTINEL
    assert cmc._is_allowed(sentinel)


# ---------------------------------------------------------------------------
# Synthetic missing-column scenario
# ---------------------------------------------------------------------------


_SYNTHETIC_CONVERTER_OK = textwrap.dedent(
    """
    from __future__ import annotations
    from typing import Any

    KNOWN_FAMILIES: tuple[str, ...] = ("foo", "bar")

    class Converter:
        def _build_record(self) -> dict[str, Any]:
            record: dict[str, Any] = {
                "patient_journey_id": "PJ_X",
                "age_at_index": 0,
            }
            for fam in KNOWN_FAMILIES:
                record[f"has_{fam}"] = 0
            return record
    """
).strip() + "\n"


_SYNTHETIC_CONVERTER_MISSING_COLUMN = textwrap.dedent(
    """
    from __future__ import annotations
    from typing import Any

    KNOWN_FAMILIES: tuple[str, ...] = ("foo", "bar")

    class Converter:
        def _build_record(self) -> dict[str, Any]:
            record: dict[str, Any] = {
                "patient_journey_id": "PJ_X",
                "age_at_index": 0,
                "newly_added_unmapped_column": 0,
            }
            for fam in KNOWN_FAMILIES:
                record[f"has_{fam}"] = 0
            return record
    """
).strip() + "\n"


_SYNTHETIC_MANIFEST = textwrap.dedent(
    """
    from src.data.feature_contract import FeatureContract, KnowableAt

    SYNTHETIC_TEST_FEATURES: list[FeatureContract] = [
        FeatureContract(
            name="age_at_index",
            knowable_at=KnowableAt(reference="enrollment"),
            source="demo",
            derivation_inputs=("age",),
        ),
        FeatureContract(
            name="has_foo",
            knowable_at=KnowableAt(reference="enrollment"),
            source="derived",
        ),
        FeatureContract(
            name="has_bar",
            knowable_at=KnowableAt(reference="enrollment"),
            source="derived",
        ),
    ]
    """
).strip() + "\n"


def _write_synthetic_repo(
    tmp_path: Path, converter_src: str, manifest_src: str
) -> Path:
    """Materialise a tiny repo layout that ``check_all`` can consume."""
    (tmp_path / "scripts").mkdir()
    (tmp_path / "scripts" / "synthetic_converter.py").write_text(converter_src)
    # We re-use the real ``src.data.manifests.synthetic_feature_manifest``
    # module path so the existing manifest stays untouched; instead
    # we write the synthetic manifest into a sibling file and import
    # by absolute file path.
    (tmp_path / "_synthetic_manifest.py").write_text(manifest_src)
    return tmp_path


def _make_synthetic_cohort(
    repo_root: Path, converter_rel: str, manifest_attr: str
) -> cmc.CohortConfig:
    return cmc.CohortConfig(
        name="synthetic-test",
        converter_rel_path=converter_rel,
        discovery_funcs=(
            cmc.DiscoveryFunc(
                func_name="_build_record",
                output_dict_names=("record",),
            ),
        ),
        manifest_module="_synthetic_manifest",  # loaded by absolute import
        manifest_attr=manifest_attr,
    )


def test_synthetic_missing_column_triggers_failure(tmp_path: Path) -> None:
    """A converter that introduces a new column NOT in the manifest
    AND NOT in the allowlist must be flagged as unmapped.
    """
    repo = _write_synthetic_repo(
        tmp_path,
        _SYNTHETIC_CONVERTER_MISSING_COLUMN,
        _SYNTHETIC_MANIFEST,
    )

    # The synthetic manifest lives at the repo root rather than under
    # ``src.data.manifests``; we need it importable so the loader sees
    # it. Insert ``repo`` onto sys.path for the duration of the test.
    old_path = list(sys.path)
    sys.path.insert(0, str(repo))
    try:
        cohort = _make_synthetic_cohort(
            repo, "scripts/synthetic_converter.py", "SYNTHETIC_TEST_FEATURES"
        )
        discovered, disc_errs = cmc.discover_columns_for_cohort(repo, cohort)
        assert disc_errs == []
        manifest, man_errs = cmc.load_manifest_names(repo, cohort)
        assert man_errs == []

        report = cmc.reconcile_cohort(discovered, manifest, cohort.name)
        assert not report.passed
        assert "newly_added_unmapped_column" in report.unmapped
        # Confirm the allowlisted ID + the loop-expanded families are
        # NOT in unmapped — only the genuinely new column should
        # surface.
        assert "patient_journey_id" not in report.unmapped
        assert "has_foo" not in report.unmapped
        assert "has_bar" not in report.unmapped
    finally:
        sys.path[:] = old_path


def test_synthetic_clean_converter_passes(tmp_path: Path) -> None:
    """A converter whose every column has either a manifest entry or
    an allowlist hit must pass.
    """
    repo = _write_synthetic_repo(
        tmp_path,
        _SYNTHETIC_CONVERTER_OK,
        _SYNTHETIC_MANIFEST,
    )
    old_path = list(sys.path)
    sys.path.insert(0, str(repo))
    try:
        cohort = _make_synthetic_cohort(
            repo, "scripts/synthetic_converter.py", "SYNTHETIC_TEST_FEATURES"
        )
        discovered, disc_errs = cmc.discover_columns_for_cohort(repo, cohort)
        assert disc_errs == []
        manifest, man_errs = cmc.load_manifest_names(repo, cohort)
        assert man_errs == []

        report = cmc.reconcile_cohort(discovered, manifest, cohort.name)
        assert report.passed, f"unexpected unmapped={report.unmapped}"
    finally:
        sys.path[:] = old_path


# ---------------------------------------------------------------------------
# F-string special cases
# ---------------------------------------------------------------------------


def test_fstring_allowed_prefix_does_not_error(tmp_path: Path) -> None:
    """An f-string whose literal prefix matches ``ALLOWED_PREFIXES``
    is tolerated even when the loop variable can't be statically
    bound. This is the CSU ``extra_demo[f"demo_{col_name}"] = ...``
    case where ``col_name`` ranges over runtime input data.
    """
    src = textwrap.dedent(
        """
        from typing import Any

        class C:
            def _build_record(self) -> dict[str, Any]:
                record: dict[str, Any] = {"x": 0}
                # `col_name` comes from a runtime input (un-bindable).
                for col_name in self._dynamic_cols():
                    record[f"demo_{col_name}"] = 0
                return record
        """
    ).strip()
    (tmp_path / "scripts").mkdir()
    (tmp_path / "scripts" / "synthetic_converter.py").write_text(src)

    cohort = _make_synthetic_cohort(
        tmp_path, "scripts/synthetic_converter.py", "SYNTHETIC_TEST_FEATURES"
    )
    discovered, errors = cmc.discover_columns_for_cohort(tmp_path, cohort)
    assert errors == [], f"unexpected errors: {errors}"
    # The static keys still present:
    assert "x" in discovered
    # The sentinel is emitted so the per-cohort summary reflects the
    # prefix-bound dynamic surface.
    assert (
        cmc._ColumnDiscoveryVisitor._ALLOWED_PREFIX_SENTINEL in discovered
    )


def test_fstring_unbound_var_no_prefix_triggers_error(tmp_path: Path) -> None:
    """An f-string with an unbindable loop var AND a non-allowed
    prefix MUST fail discovery — otherwise the guard would silently
    miss a class of dynamically generated columns.
    """
    src = textwrap.dedent(
        """
        from typing import Any

        class C:
            def _build_record(self) -> dict[str, Any]:
                record: dict[str, Any] = {"x": 0}
                # `col_name` comes from a runtime input; prefix is NOT
                # on the allowlist.
                for col_name in self._dynamic_cols():
                    record[f"feature_{col_name}"] = 0
                return record
        """
    ).strip()
    (tmp_path / "scripts").mkdir()
    (tmp_path / "scripts" / "synthetic_converter.py").write_text(src)

    cohort = _make_synthetic_cohort(
        tmp_path, "scripts/synthetic_converter.py", "SYNTHETIC_TEST_FEATURES"
    )
    discovered, errors = cmc.discover_columns_for_cohort(tmp_path, cohort)
    assert errors, "unbound f-string with non-allowed prefix must error"
    # The error must name the offending function so the developer
    # can find the source-line fast.
    assert any("_build_record" in e for e in errors)


# ---------------------------------------------------------------------------
# Module iterables resolver
# ---------------------------------------------------------------------------


def test_module_iterables_extracts_dict_keys() -> None:
    """Module-level dict literals must yield their string keys as the
    iterable's value tuple. This is the underlying mechanism for the
    Optum loop expansion."""
    tree = ast.parse(
        textwrap.dedent(
            """
            FOO: dict[str, int] = {"a": 1, "b": 2, "c": 3}
            BAR = ("x", "y")
            """
        )
    )
    result = cmc._extract_module_iterables(tree)
    assert result["FOO"] == ("a", "b", "c")
    assert result["BAR"] == ("x", "y")


def test_module_iterables_skips_non_string_keys() -> None:
    """A dict literal with non-string keys is skipped — the resolver
    only handles string-keyed iterables."""
    tree = ast.parse(
        textwrap.dedent(
            """
            INTS = {1: "a", 2: "b"}
            MIXED = {"x": 1, 2: "y"}
            """
        )
    )
    result = cmc._extract_module_iterables(tree)
    assert "INTS" not in result
    assert "MIXED" not in result


# ---------------------------------------------------------------------------
# Manifest entry for payer_category survives chain validation
# ---------------------------------------------------------------------------


def test_payer_category_manifest_entry_is_valid() -> None:
    """The ``payer_category`` entry added to the Optum manifest as part
    of Phase 1.5 must pass FeatureContract construction (already tested
    by the manifest's own test suite) AND must appear in the manifest
    registry under the canonical name."""
    from src.data.manifests.optum_feature_manifest import (
        OPTUM_FEATURES,
        OPTUM_SAFE_FEATURES,
    )

    names = {c.name for c in OPTUM_FEATURES}
    assert "payer_category" in names
    # payer_category claims enrollment-knowable, so it must appear in
    # the SAFE view (knowable_at <= index_date).
    assert "payer_category" in OPTUM_SAFE_FEATURES


# ---------------------------------------------------------------------------
# CLI integration
# ---------------------------------------------------------------------------


def test_cli_only_cohort_csu_returns_zero(capsys: pytest.CaptureFixture[str]) -> None:
    """``main(["--only-cohort", "csu"])`` exits with code 0 on the
    real repo."""
    rc = cmc.main(["--only-cohort", "csu"])
    assert rc == 0


def test_cli_unknown_cohort_returns_two() -> None:
    """Argparse rejects an unknown ``--only-cohort`` choice; this
    propagates as SystemExit before we ever reach ``check_all``.
    """
    with pytest.raises(SystemExit) as exc_info:
        cmc.main(["--only-cohort", "definitely-not-a-cohort"])
    # argparse's exit code is 2 by convention.
    assert exc_info.value.code == 2


def test_cli_default_all_cohorts_returns_zero() -> None:
    """The default invocation (all cohorts) returns 0 on a clean repo."""
    rc = cmc.main([])
    assert rc == 0
