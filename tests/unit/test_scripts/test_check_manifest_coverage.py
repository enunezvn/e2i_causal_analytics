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
    exit_code, reports, errors = cmc.check_all(_REPO_ROOT, only_cohorts=("csu",))
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
    exit_code, reports, errors = cmc.check_all(_REPO_ROOT, only_cohorts=optum_cohorts)
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
    discovered, errors = cmc.discover_columns_for_cohort(_REPO_ROOT, cmc.COHORTS[0])
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


_SYNTHETIC_CONVERTER_OK = (
    textwrap.dedent(
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
    ).strip()
    + "\n"
)


_SYNTHETIC_CONVERTER_MISSING_COLUMN = (
    textwrap.dedent(
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
    ).strip()
    + "\n"
)


_SYNTHETIC_MANIFEST = (
    textwrap.dedent(
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
    ).strip()
    + "\n"
)


def _write_synthetic_repo(tmp_path: Path, converter_src: str, manifest_src: str) -> Path:
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
    assert cmc._ColumnDiscoveryVisitor._ALLOWED_PREFIX_SENTINEL in discovered


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


# ---------------------------------------------------------------------------
# Codex-rescue HIGH-1: fail-closed on unsupported output-dict writes
# ---------------------------------------------------------------------------
#
# Each parametrized case below exercises one bypass shape that an
# earlier iteration of the guard silently dropped. The expectation is
# that the discovery walker now either (a) catches the column via the
# alias-propagation path (making it unmapped — exit 1), OR (b) records
# the expression as an ``unsupported_writes`` entry (exit 2). Both are
# acceptable PR-blocking outcomes; what we MUST NOT see is a clean PASS.


def _write_synthetic_converter(tmp_path: Path, body_src: str) -> cmc.CohortConfig:
    """Helper: write a converter with the given ``_build_record`` body
    and return a CohortConfig pointing at it. The synthetic manifest
    has 3 entries (``known``, ``has_foo``, ``has_bar``) so a real
    bypass would land as ``unmapped`` IF the visitor catches it; the
    fail-closed surface lands the bypass as ``unsupported_writes``.
    """
    (tmp_path / "scripts").mkdir(exist_ok=True)
    converter = textwrap.dedent(
        f"""
        from typing import Any

        class C:
            def _build_record(self) -> dict[str, Any]:
                record: dict[str, Any] = {{"known": 1}}
                {body_src}
                return record
        """
    ).strip()
    (tmp_path / "scripts" / "synthetic_converter.py").write_text(converter)
    (tmp_path / "_synthetic_manifest.py").write_text(_SYNTHETIC_MANIFEST)
    return _make_synthetic_cohort(
        tmp_path, "scripts/synthetic_converter.py", "SYNTHETIC_TEST_FEATURES"
    )


@pytest.mark.parametrize(
    "bypass_body",
    [
        # Alias propagation — the visitor's alias tracker turns this
        # into a discovered subscript on the alias. The bypass key
        # ends up in ``unmapped`` (not unsupported_writes), which is
        # equivalent — both fail the guard. NOTE: must be on its own
        # line because Python lexer treats ``;`` after annotated
        # assignments in dedented synthetic source.
        'alias = record\n                alias["unmapped_alias_key"] = 1',
        # BinOp key — statically unenumerable; unsupported_writes.
        'record["prefix" + "_suffix"] = 1',
        # Walrus inside subscript — unsupported_writes.
        'record[(k := "walrus_key")] = 1',
        # setdefault — unsupported_writes.
        'record.setdefault("setdefault_key", 1)',
        # __setitem__ — unsupported_writes.
        'record.__setitem__("setitem_key", 1)',
        # .update(non_dict_literal) where the arg is NOT a sibling
        # output dict — unsupported_writes.
        'extra = {"a": 1}\n                record.update(extra)',
    ],
)
def test_unsupported_output_dict_writes_fail_closed(
    tmp_path: Path,
    bypass_body: str,
) -> None:
    """Each unsupported output-dict write shape must produce either an
    ``unmapped`` column (alias case) or a discovery error
    (``unsupported_writes``). Either is acceptable — both block PR
    merge. The original codex-rescue HIGH-1 finding was that ALL of
    these silently dropped.
    """
    cohort = _write_synthetic_converter(tmp_path, bypass_body)
    old_path = list(sys.path)
    sys.path.insert(0, str(tmp_path))
    try:
        discovered, disc_errs = cmc.discover_columns_for_cohort(tmp_path, cohort)
        manifest, _ = cmc.load_manifest_names(tmp_path, cohort)

        # Either discovery errored (unsupported_writes path) OR the
        # column was discovered + is unmapped (alias path). What we
        # MUST NOT see is "no errors AND empty unmapped".
        report = cmc.reconcile_cohort(discovered, manifest, cohort.name)
        passed_cleanly = report.passed and not disc_errs
        assert not passed_cleanly, (
            f"BYPASS NOT CAUGHT: body={bypass_body!r}, "
            f"discovered={sorted(discovered)}, errors={disc_errs}, "
            f"unmapped={report.unmapped}"
        )
    finally:
        sys.path[:] = old_path


def test_dict_unpack_in_output_literal_fails_closed(tmp_path: Path) -> None:
    """``record = {"known": 1, **unmapped_unpack}`` was a bypass in the
    initial guard — the **unpack's keys were silently ignored. Verify
    the fail-closed path now triggers an unsupported_write."""
    converter = textwrap.dedent(
        """
        from typing import Any

        class C:
            def _build_record(self) -> dict[str, Any]:
                extra = {"unmapped_unpack_key": 1}
                record: dict[str, Any] = {"known": 1, **extra}
                return record
        """
    ).strip()
    (tmp_path / "scripts").mkdir()
    (tmp_path / "scripts" / "synthetic_converter.py").write_text(converter)
    (tmp_path / "_synthetic_manifest.py").write_text(_SYNTHETIC_MANIFEST)

    cohort = _make_synthetic_cohort(
        tmp_path, "scripts/synthetic_converter.py", "SYNTHETIC_TEST_FEATURES"
    )
    old_path = list(sys.path)
    sys.path.insert(0, str(tmp_path))
    try:
        discovered, disc_errs = cmc.discover_columns_for_cohort(tmp_path, cohort)
        assert disc_errs, "**unpack must produce a discovery error"
        assert any("unsupported" in e.lower() for e in disc_errs)
    finally:
        sys.path[:] = old_path


def test_safe_spread_record_update_feats_does_not_error(tmp_path: Path) -> None:
    """``record.update(feats)`` where ``feats`` is a sibling
    DiscoveryFunc's output dict must NOT trigger an unsupported_write.
    The Optum converter relies on this pattern: ``_build_journey_record``
    spreads ``_compute_features``'s ``feats`` dict into ``record``.
    """
    converter = textwrap.dedent(
        """
        from typing import Any

        class C:
            def _build_record(self) -> dict[str, Any]:
                feats = self._compute_features()
                record: dict[str, Any] = {"known": 1}
                record.update(feats)
                return record

            def _compute_features(self) -> dict[str, Any]:
                feats: dict[str, Any] = {}
                feats["safe_feature_a"] = 1
                feats["safe_feature_b"] = 1
                return feats
        """
    ).strip()
    (tmp_path / "scripts").mkdir()
    (tmp_path / "scripts" / "synthetic_converter.py").write_text(converter)
    (tmp_path / "_synthetic_manifest.py").write_text(_SYNTHETIC_MANIFEST)

    # Two DiscoveryFunc entries — _build_record + _compute_features.
    cohort = cmc.CohortConfig(
        name="synthetic-spread",
        converter_rel_path="scripts/synthetic_converter.py",
        discovery_funcs=(
            cmc.DiscoveryFunc(
                func_name="_build_record",
                output_dict_names=("record",),
            ),
            cmc.DiscoveryFunc(
                func_name="_compute_features",
                output_dict_names=("feats",),
            ),
        ),
        manifest_module="_synthetic_manifest",
        manifest_attr="SYNTHETIC_TEST_FEATURES",
    )

    old_path = list(sys.path)
    sys.path.insert(0, str(tmp_path))
    try:
        discovered, disc_errs = cmc.discover_columns_for_cohort(tmp_path, cohort)
        assert disc_errs == [], f"safe spread should not error: {disc_errs}"
        # All three keys should be discovered.
        assert {"known", "safe_feature_a", "safe_feature_b"}.issubset(discovered)
    finally:
        sys.path[:] = old_path


def test_alias_propagation_catches_aliased_write(tmp_path: Path) -> None:
    """``alias = record; alias["unmapped"] = 1`` — the alias propagation
    in the visitor turns this into a discovered subscript on the alias.
    Result: the bypass key ``unmapped`` lands in the discovered set
    and is reported as ``unmapped`` by reconcile_cohort (not in
    manifest, not in allowlist).
    """
    converter = textwrap.dedent(
        """
        from typing import Any

        class C:
            def _build_record(self) -> dict[str, Any]:
                record: dict[str, Any] = {"known": 1}
                alias = record
                alias["alias_unmapped_key"] = 1
                return record
        """
    ).strip()
    (tmp_path / "scripts").mkdir()
    (tmp_path / "scripts" / "synthetic_converter.py").write_text(converter)
    (tmp_path / "_synthetic_manifest.py").write_text(_SYNTHETIC_MANIFEST)

    cohort = _make_synthetic_cohort(
        tmp_path, "scripts/synthetic_converter.py", "SYNTHETIC_TEST_FEATURES"
    )

    old_path = list(sys.path)
    sys.path.insert(0, str(tmp_path))
    try:
        discovered, disc_errs = cmc.discover_columns_for_cohort(tmp_path, cohort)
        assert disc_errs == []
        # The aliased key WAS discovered.
        assert "alias_unmapped_key" in discovered
        manifest, _ = cmc.load_manifest_names(tmp_path, cohort)
        report = cmc.reconcile_cohort(discovered, manifest, cohort.name)
        # It's not in manifest, not in allowlist → unmapped.
        assert "alias_unmapped_key" in report.unmapped
    finally:
        sys.path[:] = old_path


# ---------------------------------------------------------------------------
# Codex-rescue HIGH-2: required-column sanity check (rename-collapses-discovery)
# ---------------------------------------------------------------------------


def test_output_dict_rename_collapses_to_required_column_error(tmp_path: Path) -> None:
    """If the converter's output dict is renamed (e.g., ``record`` →
    ``journey``) and the cohort config is NOT updated, the visitor
    discovers ZERO columns. Without HIGH-2's required_columns sanity
    check, this would PASS coverage silently (no unmapped). With the
    sanity check, the cohort errors out — exit 2.
    """
    converter = textwrap.dedent(
        """
        from typing import Any

        class C:
            def _build_record(self) -> dict[str, Any]:
                # Rename: the canonical local var was ``record``;
                # this code uses ``journey`` instead. The cohort
                # config still says output_dict_names=("record",), so
                # the visitor sees nothing.
                journey: dict[str, Any] = {
                    "renamed_output_key_a": 1,
                    "renamed_output_key_b": 2,
                }
                return journey
        """
    ).strip()
    (tmp_path / "scripts").mkdir()
    (tmp_path / "scripts" / "synthetic_converter.py").write_text(converter)
    (tmp_path / "_synthetic_manifest.py").write_text(_SYNTHETIC_MANIFEST)

    # Cohort config has required_columns set — the rename will trip it.
    cohort = cmc.CohortConfig(
        name="synthetic-rename",
        converter_rel_path="scripts/synthetic_converter.py",
        discovery_funcs=(
            cmc.DiscoveryFunc(
                func_name="_build_record",
                output_dict_names=("record",),
                required_columns=("known", "age_at_index"),
            ),
        ),
        manifest_module="_synthetic_manifest",
        manifest_attr="SYNTHETIC_TEST_FEATURES",
    )

    old_path = list(sys.path)
    sys.path.insert(0, str(tmp_path))
    try:
        discovered, disc_errs = cmc.discover_columns_for_cohort(tmp_path, cohort)
        # The visitor found nothing under output_dict_names=("record",).
        assert discovered == frozenset()
        # The required-column sanity check fires.
        assert disc_errs, "rename without config update must error"
        assert any("did NOT produce the canonical columns" in e for e in disc_errs)
    finally:
        sys.path[:] = old_path


def test_real_cohort_required_columns_satisfied() -> None:
    """The real cohorts' required_columns lists must be satisfied by
    the live converter at HEAD. Any failure here means either (a) the
    required_columns list drifted ahead of the converter, or (b) the
    converter was refactored without updating either the
    output_dict_names or the manifest.
    """
    for cohort in cmc.COHORTS:
        discovered, errors = cmc.discover_columns_for_cohort(_REPO_ROOT, cohort)
        assert errors == [], f"{cohort.name}: discovery errors: {errors}"
        for df in cohort.discovery_funcs:
            for req in df.required_columns:
                assert req in discovered, (
                    f"{cohort.name}/{df.func_name}: required column "
                    f"{req!r} not in discovered surface"
                )


# ---------------------------------------------------------------------------
# Codex-rescue pass-2: HIGH-3 (helper-call bypass), HIGH-4 (tuple-target
# subscript), HIGH-5 (shadowed safe-spread name), MEDIUM-2 (conditional alias)
# ---------------------------------------------------------------------------


def test_helper_call_with_output_arg_fails_closed(tmp_path: Path) -> None:
    """Codex pass-2 HIGH-3: ``self._add_columns(record)`` passes the
    output dict to a helper that could mutate it. The static analyser
    cannot see what the helper does, so the call MUST be recorded as
    ``unsupported_writes`` (exit 2).
    """
    converter = textwrap.dedent(
        """
        from typing import Any

        class C:
            def _build_record(self) -> dict[str, Any]:
                record: dict[str, Any] = {"known": 1}
                self._add_columns(record)
                return record

            def _add_columns(self, out: dict[str, Any]) -> None:
                out["unmapped_helper_key"] = 1
        """
    ).strip()
    (tmp_path / "scripts").mkdir()
    (tmp_path / "scripts" / "synthetic_converter.py").write_text(converter)
    (tmp_path / "_synthetic_manifest.py").write_text(_SYNTHETIC_MANIFEST)

    cohort = _make_synthetic_cohort(
        tmp_path, "scripts/synthetic_converter.py", "SYNTHETIC_TEST_FEATURES"
    )

    old_path = list(sys.path)
    sys.path.insert(0, str(tmp_path))
    try:
        discovered, disc_errs = cmc.discover_columns_for_cohort(tmp_path, cohort)
        assert disc_errs, "helper-call bypass must produce a discovery error"
        assert any("self._add_columns" in e for e in disc_errs)
    finally:
        sys.path[:] = old_path


def test_helper_call_with_output_kwarg_fails_closed(tmp_path: Path) -> None:
    """Same as the positional case but the output dict is passed as a
    keyword argument: ``helper(out=record)``.
    """
    converter = textwrap.dedent(
        """
        from typing import Any

        class C:
            def _build_record(self) -> dict[str, Any]:
                record: dict[str, Any] = {"known": 1}
                self._add_columns(out=record)
                return record
        """
    ).strip()
    (tmp_path / "scripts").mkdir()
    (tmp_path / "scripts" / "synthetic_converter.py").write_text(converter)
    (tmp_path / "_synthetic_manifest.py").write_text(_SYNTHETIC_MANIFEST)

    cohort = _make_synthetic_cohort(
        tmp_path, "scripts/synthetic_converter.py", "SYNTHETIC_TEST_FEATURES"
    )

    old_path = list(sys.path)
    sys.path.insert(0, str(tmp_path))
    try:
        discovered, disc_errs = cmc.discover_columns_for_cohort(tmp_path, cohort)
        assert disc_errs, "kwarg helper-call bypass must produce a discovery error"
    finally:
        sys.path[:] = old_path


def test_journeys_append_output_does_not_error(tmp_path: Path) -> None:
    """``journeys.append(journey_dict)`` is the canonical CSU
    converter pattern: the per-patient dict is appended to a list
    after all writes are complete. Pass-6 narrows the append exception
    to receiver Names in ``collector_names``; this test declares
    ``journeys`` as the collector and verifies the legitimate pattern
    still passes.
    """
    converter = textwrap.dedent(
        """
        from typing import Any

        class C:
            def _build_record(self) -> list[dict[str, Any]]:
                journeys: list[dict[str, Any]] = []
                for _ in range(2):
                    record: dict[str, Any] = {"known": 1, "patient_id": 0}
                    journeys.append(record)
                return journeys
        """
    ).strip()
    (tmp_path / "scripts").mkdir()
    (tmp_path / "scripts" / "synthetic_converter.py").write_text(converter)
    (tmp_path / "_synthetic_manifest.py").write_text(_SYNTHETIC_MANIFEST)

    # Pass-6 HIGH: explicit collector_names declaration so the
    # legitimate journeys.append(record) call is accepted.
    cohort = cmc.CohortConfig(
        name="synthetic-collector",
        converter_rel_path="scripts/synthetic_converter.py",
        discovery_funcs=(
            cmc.DiscoveryFunc(
                func_name="_build_record",
                output_dict_names=("record",),
                collector_names=("journeys",),
            ),
        ),
        manifest_module="_synthetic_manifest",
        manifest_attr="SYNTHETIC_TEST_FEATURES",
    )

    old_path = list(sys.path)
    sys.path.insert(0, str(tmp_path))
    try:
        discovered, disc_errs = cmc.discover_columns_for_cohort(tmp_path, cohort)
        assert disc_errs == [], f".append() should not trigger helper-call check: {disc_errs}"
        assert {"known", "patient_id"}.issubset(discovered)
    finally:
        sys.path[:] = old_path


def test_tuple_target_subscript_assign_caught(tmp_path: Path) -> None:
    """Codex pass-2 HIGH-4: ``record["a"], record["b"] = 1, 2`` —
    top-level target is ``ast.Tuple`` whose elements are subscripts.
    The visitor's _walk_assign_target recursion must catch both.
    """
    converter = textwrap.dedent(
        """
        from typing import Any

        class C:
            def _build_record(self) -> dict[str, Any]:
                record: dict[str, Any] = {"known": 1}
                record["tuple_a"], record["tuple_b"] = 1, 2
                return record
        """
    ).strip()
    (tmp_path / "scripts").mkdir()
    (tmp_path / "scripts" / "synthetic_converter.py").write_text(converter)
    (tmp_path / "_synthetic_manifest.py").write_text(_SYNTHETIC_MANIFEST)

    cohort = _make_synthetic_cohort(
        tmp_path, "scripts/synthetic_converter.py", "SYNTHETIC_TEST_FEATURES"
    )

    old_path = list(sys.path)
    sys.path.insert(0, str(tmp_path))
    try:
        discovered, disc_errs = cmc.discover_columns_for_cohort(tmp_path, cohort)
        assert disc_errs == [], f"tuple-target subscript must enumerate cleanly: {disc_errs}"
        assert "tuple_a" in discovered
        assert "tuple_b" in discovered
        # These are not in the manifest → unmapped.
        manifest, _ = cmc.load_manifest_names(tmp_path, cohort)
        report = cmc.reconcile_cohort(discovered, manifest, cohort.name)
        assert "tuple_a" in report.unmapped
        assert "tuple_b" in report.unmapped
    finally:
        sys.path[:] = old_path


def test_starred_target_subscript_caught(tmp_path: Path) -> None:
    """``record["a"], *rest = (1, 2, 3)`` — the visitor must descend
    into the Starred wrapper to find the inner Subscript target.
    """
    converter = textwrap.dedent(
        """
        from typing import Any

        class C:
            def _build_record(self) -> dict[str, Any]:
                record: dict[str, Any] = {"known": 1}
                record["starred_a"], *rest = (1, 2, 3)
                return record
        """
    ).strip()
    (tmp_path / "scripts").mkdir()
    (tmp_path / "scripts" / "synthetic_converter.py").write_text(converter)
    (tmp_path / "_synthetic_manifest.py").write_text(_SYNTHETIC_MANIFEST)

    cohort = _make_synthetic_cohort(
        tmp_path, "scripts/synthetic_converter.py", "SYNTHETIC_TEST_FEATURES"
    )

    old_path = list(sys.path)
    sys.path.insert(0, str(tmp_path))
    try:
        discovered, disc_errs = cmc.discover_columns_for_cohort(tmp_path, cohort)
        assert disc_errs == []
        assert "starred_a" in discovered
    finally:
        sys.path[:] = old_path


def test_shadowed_safe_spread_fails_closed(tmp_path: Path) -> None:
    """Codex pass-2 HIGH-5: if a function locally shadows a safe-spread
    name by assigning a dict literal to it (``feats = {"unmapped": 1}``),
    the subsequent ``record.update(feats)`` MUST NOT be tolerated —
    the spread arg's "trusted" identifier has been rebound to a local
    dict the sibling pass never sees.
    """
    converter = textwrap.dedent(
        """
        from typing import Any

        class C:
            def _build_record(self) -> dict[str, Any]:
                # Shadow the safe-spread name with a fresh local dict.
                feats: dict[str, Any] = {"unmapped_shadow_key": 1}
                record: dict[str, Any] = {"known": 1}
                record.update(feats)
                return record

            def _compute_features(self) -> dict[str, Any]:
                feats: dict[str, Any] = {}
                feats["safe_feature"] = 1
                return feats
        """
    ).strip()
    (tmp_path / "scripts").mkdir()
    (tmp_path / "scripts" / "synthetic_converter.py").write_text(converter)
    (tmp_path / "_synthetic_manifest.py").write_text(_SYNTHETIC_MANIFEST)

    cohort = cmc.CohortConfig(
        name="synthetic-shadow",
        converter_rel_path="scripts/synthetic_converter.py",
        discovery_funcs=(
            cmc.DiscoveryFunc(
                func_name="_build_record",
                output_dict_names=("record",),
            ),
            cmc.DiscoveryFunc(
                func_name="_compute_features",
                output_dict_names=("feats",),
            ),
        ),
        manifest_module="_synthetic_manifest",
        manifest_attr="SYNTHETIC_TEST_FEATURES",
    )

    old_path = list(sys.path)
    sys.path.insert(0, str(tmp_path))
    try:
        discovered, disc_errs = cmc.discover_columns_for_cohort(tmp_path, cohort)
        # The local-shadow of ``feats`` in _build_record means
        # ``record.update(feats)`` cannot use the safe-spread tolerance.
        # Recorded as unsupported_writes.
        assert disc_errs, f"shadowed safe-spread must error: {disc_errs}"
        assert any("record.update(feats)" in e for e in disc_errs)
    finally:
        sys.path[:] = old_path


def test_conditional_alias_assignment_fails_closed(tmp_path: Path) -> None:
    """Codex pass-2 MEDIUM-2: ``alias = record if cond else other``
    is a dynamic alias whose binding the static walker can't resolve.
    Must be recorded as unsupported_writes.
    """
    converter = textwrap.dedent(
        """
        from typing import Any

        class C:
            def _build_record(self, cond: bool) -> dict[str, Any]:
                record: dict[str, Any] = {"known": 1}
                other: dict[str, Any] = {}
                alias = record if cond else other
                alias["ternary_alias_key"] = 1
                return record
        """
    ).strip()
    (tmp_path / "scripts").mkdir()
    (tmp_path / "scripts" / "synthetic_converter.py").write_text(converter)
    (tmp_path / "_synthetic_manifest.py").write_text(_SYNTHETIC_MANIFEST)

    cohort = _make_synthetic_cohort(
        tmp_path, "scripts/synthetic_converter.py", "SYNTHETIC_TEST_FEATURES"
    )

    old_path = list(sys.path)
    sys.path.insert(0, str(tmp_path))
    try:
        discovered, disc_errs = cmc.discover_columns_for_cohort(tmp_path, cohort)
        assert disc_errs, f"ternary alias must error: {disc_errs}"
        assert any("alias = record if cond else other" in e for e in disc_errs)
    finally:
        sys.path[:] = old_path


# ---------------------------------------------------------------------------
# Codex-rescue pass-3: HIGH-6 (wrapper-method bypass), HIGH-7 (output-dict
# reassignment), HIGH-8 (AugAssign on output dicts)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "wrapper_method",
    ["get", "copy", "items", "keys", "values", "fromkeys"],
)
def test_wrapper_method_bypass_fails_closed(tmp_path: Path, wrapper_method: str) -> None:
    """Codex pass-3 HIGH-6: an arbitrary user-defined object's method
    named with ``get`` / ``copy`` / ``items`` / ``keys`` / ``values``
    is just user code that could mutate the dict argument. Pass-2's
    broader ``_NON_MUTATING_METHODS`` exception allowed all of these
    through; pass-3 tightens to only ``append`` / ``extend`` on
    non-output receivers.
    """
    converter = textwrap.dedent(
        f"""
        from typing import Any

        class C:
            def _build_record(self) -> dict[str, Any]:
                record: dict[str, Any] = {{"known": 1}}
                wrapper.{wrapper_method}(record)
                return record
        """
    ).strip()
    (tmp_path / "scripts").mkdir()
    (tmp_path / "scripts" / "synthetic_converter.py").write_text(converter)
    (tmp_path / "_synthetic_manifest.py").write_text(_SYNTHETIC_MANIFEST)

    cohort = _make_synthetic_cohort(
        tmp_path, "scripts/synthetic_converter.py", "SYNTHETIC_TEST_FEATURES"
    )

    old_path = list(sys.path)
    sys.path.insert(0, str(tmp_path))
    try:
        discovered, disc_errs = cmc.discover_columns_for_cohort(tmp_path, cohort)
        assert disc_errs, f"wrapper.{wrapper_method}(record) must error (pass-3 HIGH-6)"
    finally:
        sys.path[:] = old_path


def test_output_dict_reassignment_fromkeys_fails_closed(tmp_path: Path) -> None:
    """Codex pass-3 HIGH-7: ``record = dict.fromkeys(...)`` reassigns
    the output dict to a fresh dict whose keys the visitor can't
    enumerate. The earlier dict-literal keys remain in ``discovered``
    but the actual returned dict is different. Must error.
    """
    converter = textwrap.dedent(
        """
        from typing import Any

        class C:
            def _build_record(self) -> dict[str, Any]:
                record: dict[str, Any] = {"known": 1}
                record = dict.fromkeys(["known", "unmapped_fromkeys"], 1)
                return record
        """
    ).strip()
    (tmp_path / "scripts").mkdir()
    (tmp_path / "scripts" / "synthetic_converter.py").write_text(converter)
    (tmp_path / "_synthetic_manifest.py").write_text(_SYNTHETIC_MANIFEST)

    cohort = _make_synthetic_cohort(
        tmp_path, "scripts/synthetic_converter.py", "SYNTHETIC_TEST_FEATURES"
    )

    old_path = list(sys.path)
    sys.path.insert(0, str(tmp_path))
    try:
        discovered, disc_errs = cmc.discover_columns_for_cohort(tmp_path, cohort)
        assert disc_errs, "output-dict reassignment must error (pass-3 HIGH-7)"
        assert any("dict.fromkeys" in e for e in disc_errs)
    finally:
        sys.path[:] = old_path


def test_output_dict_walrus_reassignment_fails_closed(tmp_path: Path) -> None:
    """Codex pass-3 HIGH-7 (walrus form): ``record = (newrec := {...})``."""
    converter = textwrap.dedent(
        """
        from typing import Any

        class C:
            def _build_record(self) -> dict[str, Any]:
                record: dict[str, Any] = {"known": 1}
                record = (newrec := {"known": 1, "unmapped_walrus": 1})
                return record
        """
    ).strip()
    (tmp_path / "scripts").mkdir()
    (tmp_path / "scripts" / "synthetic_converter.py").write_text(converter)
    (tmp_path / "_synthetic_manifest.py").write_text(_SYNTHETIC_MANIFEST)

    cohort = _make_synthetic_cohort(
        tmp_path, "scripts/synthetic_converter.py", "SYNTHETIC_TEST_FEATURES"
    )

    old_path = list(sys.path)
    sys.path.insert(0, str(tmp_path))
    try:
        discovered, disc_errs = cmc.discover_columns_for_cohort(tmp_path, cohort)
        assert disc_errs, "walrus reassignment must error"
    finally:
        sys.path[:] = old_path


def test_augassign_dict_union_literal_enumerates(tmp_path: Path) -> None:
    """Codex pass-3 HIGH-8: ``record |= {"new_key": 1}`` MUST enumerate
    the literal keys (similar to ``.update({...})``). Without this fix,
    pass-3 had ``visit_AugAssign`` as a no-op.
    """
    converter = textwrap.dedent(
        """
        from typing import Any

        class C:
            def _build_record(self) -> dict[str, Any]:
                record: dict[str, Any] = {"known": 1}
                record |= {"aug_known_key": 1}
                return record
        """
    ).strip()
    (tmp_path / "scripts").mkdir()
    (tmp_path / "scripts" / "synthetic_converter.py").write_text(converter)
    (tmp_path / "_synthetic_manifest.py").write_text(_SYNTHETIC_MANIFEST)

    cohort = _make_synthetic_cohort(
        tmp_path, "scripts/synthetic_converter.py", "SYNTHETIC_TEST_FEATURES"
    )

    old_path = list(sys.path)
    sys.path.insert(0, str(tmp_path))
    try:
        discovered, disc_errs = cmc.discover_columns_for_cohort(tmp_path, cohort)
        assert disc_errs == [], f"AugAssign with dict literal should enumerate cleanly: {disc_errs}"
        assert "aug_known_key" in discovered
        # Not in manifest → unmapped
        manifest, _ = cmc.load_manifest_names(tmp_path, cohort)
        report = cmc.reconcile_cohort(discovered, manifest, cohort.name)
        assert "aug_known_key" in report.unmapped
    finally:
        sys.path[:] = old_path


def test_augassign_dict_union_var_rhs_fails_closed(tmp_path: Path) -> None:
    """``record |= other_dict_var`` with a non-literal, non-safe-spread
    RHS must error."""
    converter = textwrap.dedent(
        """
        from typing import Any

        class C:
            def _build_record(self) -> dict[str, Any]:
                record: dict[str, Any] = {"known": 1}
                other = {"unmapped_aug": 1}
                record |= other
                return record
        """
    ).strip()
    (tmp_path / "scripts").mkdir()
    (tmp_path / "scripts" / "synthetic_converter.py").write_text(converter)
    (tmp_path / "_synthetic_manifest.py").write_text(_SYNTHETIC_MANIFEST)

    cohort = _make_synthetic_cohort(
        tmp_path, "scripts/synthetic_converter.py", "SYNTHETIC_TEST_FEATURES"
    )

    old_path = list(sys.path)
    sys.path.insert(0, str(tmp_path))
    try:
        discovered, disc_errs = cmc.discover_columns_for_cohort(tmp_path, cohort)
        assert disc_errs, f"|= with non-literal RHS must error: {disc_errs}"
    finally:
        sys.path[:] = old_path


def test_augassign_dict_union_safe_spread_does_not_error(tmp_path: Path) -> None:
    """``record |= feats`` where ``feats`` is a safe-spread name must
    NOT error — mirrors the .update(feats) safe path.
    """
    converter = textwrap.dedent(
        """
        from typing import Any

        class C:
            def _build_record(self) -> dict[str, Any]:
                feats = self._compute_features()
                record: dict[str, Any] = {"known": 1}
                record |= feats
                return record

            def _compute_features(self) -> dict[str, Any]:
                feats: dict[str, Any] = {}
                feats["safe_feature"] = 1
                return feats
        """
    ).strip()
    (tmp_path / "scripts").mkdir()
    (tmp_path / "scripts" / "synthetic_converter.py").write_text(converter)
    (tmp_path / "_synthetic_manifest.py").write_text(_SYNTHETIC_MANIFEST)

    cohort = cmc.CohortConfig(
        name="synthetic-aug-safe",
        converter_rel_path="scripts/synthetic_converter.py",
        discovery_funcs=(
            cmc.DiscoveryFunc(
                func_name="_build_record",
                output_dict_names=("record",),
            ),
            cmc.DiscoveryFunc(
                func_name="_compute_features",
                output_dict_names=("feats",),
            ),
        ),
        manifest_module="_synthetic_manifest",
        manifest_attr="SYNTHETIC_TEST_FEATURES",
    )

    old_path = list(sys.path)
    sys.path.insert(0, str(tmp_path))
    try:
        discovered, disc_errs = cmc.discover_columns_for_cohort(tmp_path, cohort)
        assert disc_errs == [], f"|= with safe-spread should not error: {disc_errs}"
        assert {"known", "safe_feature"}.issubset(discovered)
    finally:
        sys.path[:] = old_path


def test_list_collector_append_still_accepted(tmp_path: Path) -> None:
    """After tightening _NON_MUTATING_METHODS → _LIST_COLLECTOR_METHODS
    (pass-3 HIGH-6) AND further restricting the receiver to
    ``collector_names`` (pass-6), the canonical
    ``journeys.append(journey_dict)`` pattern must still work when
    the cohort config declares ``collector_names=("journeys",)``."""
    converter = textwrap.dedent(
        """
        from typing import Any

        class C:
            def _build_record(self) -> list[dict[str, Any]]:
                journeys: list[dict[str, Any]] = []
                journey_dict: dict[str, Any] = {"known": 1}
                journeys.append(journey_dict)
                journeys.extend([journey_dict])
                return journeys
        """
    ).strip()
    (tmp_path / "scripts").mkdir()
    (tmp_path / "scripts" / "synthetic_converter.py").write_text(converter)
    (tmp_path / "_synthetic_manifest.py").write_text(_SYNTHETIC_MANIFEST)

    cohort = cmc.CohortConfig(
        name="synthetic-append",
        converter_rel_path="scripts/synthetic_converter.py",
        discovery_funcs=(
            cmc.DiscoveryFunc(
                func_name="_build_record",
                output_dict_names=("journey_dict",),
                collector_names=("journeys",),
            ),
        ),
        manifest_module="_synthetic_manifest",
        manifest_attr="SYNTHETIC_TEST_FEATURES",
    )

    old_path = list(sys.path)
    sys.path.insert(0, str(tmp_path))
    try:
        discovered, disc_errs = cmc.discover_columns_for_cohort(tmp_path, cohort)
        assert disc_errs == [], f"journeys.append(journey_dict) should not error: {disc_errs}"
        assert "known" in discovered
    finally:
        sys.path[:] = old_path


# ---------------------------------------------------------------------------
# Codex-rescue pass-4: HIGH-9 (safe-spread Call shadow), HIGH-10 (output Name
# reassigned in tuple-unpack), LOW-3 (non-|= AugAssign on output Name)
# ---------------------------------------------------------------------------


def test_safe_spread_shadowed_by_unknown_helper_fails_closed(
    tmp_path: Path,
) -> None:
    """Codex pass-4 HIGH-9: ``feats = self.unknown_helper()`` shadows
    the safe-spread tolerance — the helper's return value carries
    keys the static walker cannot see. Pass-2 only fired on Dict
    literal RHS; pass-4 widens to any non-trusted Call.
    """
    converter = textwrap.dedent(
        """
        from typing import Any

        class C:
            def _build_record(self) -> dict[str, Any]:
                feats = self.unknown_helper()
                record: dict[str, Any] = {"known": 1}
                record.update(feats)
                return record

            def _compute_features(self) -> dict[str, Any]:
                feats: dict[str, Any] = {}
                feats["safe_feature"] = 1
                return feats
        """
    ).strip()
    (tmp_path / "scripts").mkdir()
    (tmp_path / "scripts" / "synthetic_converter.py").write_text(converter)
    (tmp_path / "_synthetic_manifest.py").write_text(_SYNTHETIC_MANIFEST)

    cohort = cmc.CohortConfig(
        name="synthetic-shadow-call",
        converter_rel_path="scripts/synthetic_converter.py",
        discovery_funcs=(
            cmc.DiscoveryFunc(
                func_name="_build_record",
                output_dict_names=("record",),
            ),
            cmc.DiscoveryFunc(
                func_name="_compute_features",
                output_dict_names=("feats",),
            ),
        ),
        manifest_module="_synthetic_manifest",
        manifest_attr="SYNTHETIC_TEST_FEATURES",
    )

    old_path = list(sys.path)
    sys.path.insert(0, str(tmp_path))
    try:
        discovered, disc_errs = cmc.discover_columns_for_cohort(tmp_path, cohort)
        # The unknown_helper Call rebinds `feats`; safe-spread tolerance
        # is shadowed; ``record.update(feats)`` becomes unsupported.
        assert disc_errs, f"feats = self.unknown_helper() must shadow safe-spread: {disc_errs}"
        assert any("record.update(feats)" in e for e in disc_errs)
    finally:
        sys.path[:] = old_path


def test_safe_spread_trusted_helper_does_not_shadow(tmp_path: Path) -> None:
    """``feats = self._compute_features(...)`` is the canonical Optum
    pattern. The helper name IS a sibling DiscoveryFunc, so the
    binding is trusted and the safe-spread tolerance applies.
    """
    converter = textwrap.dedent(
        """
        from typing import Any

        class C:
            def _build_record(self) -> dict[str, Any]:
                # Trusted: _compute_features is a known sibling DiscoveryFunc.
                feats = self._compute_features()
                record: dict[str, Any] = {"known": 1}
                record.update(feats)
                return record

            def _compute_features(self) -> dict[str, Any]:
                feats: dict[str, Any] = {}
                feats["safe_feature"] = 1
                return feats
        """
    ).strip()
    (tmp_path / "scripts").mkdir()
    (tmp_path / "scripts" / "synthetic_converter.py").write_text(converter)
    (tmp_path / "_synthetic_manifest.py").write_text(_SYNTHETIC_MANIFEST)

    cohort = cmc.CohortConfig(
        name="synthetic-trusted-helper",
        converter_rel_path="scripts/synthetic_converter.py",
        discovery_funcs=(
            cmc.DiscoveryFunc(
                func_name="_build_record",
                output_dict_names=("record",),
            ),
            cmc.DiscoveryFunc(
                func_name="_compute_features",
                output_dict_names=("feats",),
            ),
        ),
        manifest_module="_synthetic_manifest",
        manifest_attr="SYNTHETIC_TEST_FEATURES",
    )

    old_path = list(sys.path)
    sys.path.insert(0, str(tmp_path))
    try:
        discovered, disc_errs = cmc.discover_columns_for_cohort(tmp_path, cohort)
        assert disc_errs == [], f"trusted helper binding must not shadow: {disc_errs}"
        assert {"known", "safe_feature"}.issubset(discovered)
    finally:
        sys.path[:] = old_path


def test_output_name_reassigned_in_tuple_unpack_fails_closed(
    tmp_path: Path,
) -> None:
    """Codex pass-4 HIGH-10: ``record, *rest = some_call()`` rebinds
    the output Name inside a tuple unpack. ``record`` is now an
    unenumerable value. Must error.
    """
    converter = textwrap.dedent(
        """
        from typing import Any

        class C:
            def _build_record(self) -> dict[str, Any]:
                record: dict[str, Any] = {"known": 1}
                record, *rest = self.helper_returning_tuple()
                return record
        """
    ).strip()
    (tmp_path / "scripts").mkdir()
    (tmp_path / "scripts" / "synthetic_converter.py").write_text(converter)
    (tmp_path / "_synthetic_manifest.py").write_text(_SYNTHETIC_MANIFEST)

    cohort = _make_synthetic_cohort(
        tmp_path, "scripts/synthetic_converter.py", "SYNTHETIC_TEST_FEATURES"
    )

    old_path = list(sys.path)
    sys.path.insert(0, str(tmp_path))
    try:
        discovered, disc_errs = cmc.discover_columns_for_cohort(tmp_path, cohort)
        assert disc_errs, "tuple-unpack rebinding output Name must error"
    finally:
        sys.path[:] = old_path


@pytest.mark.parametrize("op_src", ["+=", "*="])
def test_non_ior_augassign_on_output_fails_closed(tmp_path: Path, op_src: str) -> None:
    """Codex pass-4 LOW-3: ``record += {...}`` / ``record *= 2`` etc.
    on an output Name are unsupported. Python's dict would raise
    TypeError at runtime, but the static walker treats them as
    unenumerable output-dict mutations and flags them.
    """
    converter = textwrap.dedent(
        f"""
        from typing import Any

        class C:
            def _build_record(self) -> dict[str, Any]:
                record: dict[str, Any] = {{"known": 1}}
                record {op_src} 2
                return record
        """
    ).strip()
    (tmp_path / "scripts").mkdir()
    (tmp_path / "scripts" / "synthetic_converter.py").write_text(converter)
    (tmp_path / "_synthetic_manifest.py").write_text(_SYNTHETIC_MANIFEST)

    cohort = _make_synthetic_cohort(
        tmp_path, "scripts/synthetic_converter.py", "SYNTHETIC_TEST_FEATURES"
    )

    old_path = list(sys.path)
    sys.path.insert(0, str(tmp_path))
    try:
        discovered, disc_errs = cmc.discover_columns_for_cohort(tmp_path, cohort)
        assert disc_errs, f"non-|= AugAssign on output Name must error (op={op_src}): {disc_errs}"
    finally:
        sys.path[:] = old_path


# ---------------------------------------------------------------------------
# Codex-rescue pass-5: HIGH-11 (trusted-helper spoofing — non-self receiver
# / bare Call), HIGH-12 (output-dict reassignment from arbitrary Name RHS)
# ---------------------------------------------------------------------------


def test_trusted_helper_non_self_receiver_shadows(tmp_path: Path) -> None:
    """Codex pass-5 HIGH-11: ``feats = other_obj._compute_features()``
    must NOT be accepted as the trusted helper binding — pass-4 only
    checked the .attr name, not the receiver. Pass-5 narrows to
    ``self.<trusted_helper>()``.
    """
    converter = textwrap.dedent(
        """
        from typing import Any

        class C:
            def _build_record(self, other_obj) -> dict[str, Any]:
                # Receiver is `other_obj`, not `self` — not trusted.
                feats = other_obj._compute_features()
                record: dict[str, Any] = {"known": 1}
                record.update(feats)
                return record

            def _compute_features(self) -> dict[str, Any]:
                feats: dict[str, Any] = {}
                feats["safe_feature"] = 1
                return feats
        """
    ).strip()
    (tmp_path / "scripts").mkdir()
    (tmp_path / "scripts" / "synthetic_converter.py").write_text(converter)
    (tmp_path / "_synthetic_manifest.py").write_text(_SYNTHETIC_MANIFEST)

    cohort = cmc.CohortConfig(
        name="synthetic-non-self",
        converter_rel_path="scripts/synthetic_converter.py",
        discovery_funcs=(
            cmc.DiscoveryFunc(
                func_name="_build_record",
                output_dict_names=("record",),
            ),
            cmc.DiscoveryFunc(
                func_name="_compute_features",
                output_dict_names=("feats",),
            ),
        ),
        manifest_module="_synthetic_manifest",
        manifest_attr="SYNTHETIC_TEST_FEATURES",
    )

    old_path = list(sys.path)
    sys.path.insert(0, str(tmp_path))
    try:
        discovered, disc_errs = cmc.discover_columns_for_cohort(tmp_path, cohort)
        assert disc_errs, "non-self receiver trusted-helper-spoof must shadow safe-spread"
        assert any("record.update(feats)" in e for e in disc_errs)
    finally:
        sys.path[:] = old_path


def test_trusted_helper_bare_call_shadows(tmp_path: Path) -> None:
    """Codex pass-5 HIGH-11: bare ``feats = _compute_features()`` (no
    receiver) must NOT be trusted.
    """
    converter = textwrap.dedent(
        """
        from typing import Any

        def _compute_features() -> dict[str, Any]:
            return {"unmapped_bare": 1}

        class C:
            def _build_record(self) -> dict[str, Any]:
                # Bare function call — not self.<trusted>.
                feats = _compute_features()
                record: dict[str, Any] = {"known": 1}
                record.update(feats)
                return record
        """
    ).strip()
    (tmp_path / "scripts").mkdir()
    (tmp_path / "scripts" / "synthetic_converter.py").write_text(converter)
    (tmp_path / "_synthetic_manifest.py").write_text(_SYNTHETIC_MANIFEST)

    cohort = cmc.CohortConfig(
        name="synthetic-bare-call",
        converter_rel_path="scripts/synthetic_converter.py",
        discovery_funcs=(
            cmc.DiscoveryFunc(
                func_name="_build_record",
                output_dict_names=("record",),
            ),
            cmc.DiscoveryFunc(
                func_name="_compute_features",
                output_dict_names=("feats",),
            ),
        ),
        manifest_module="_synthetic_manifest",
        manifest_attr="SYNTHETIC_TEST_FEATURES",
    )

    old_path = list(sys.path)
    sys.path.insert(0, str(tmp_path))
    try:
        discovered, disc_errs = cmc.discover_columns_for_cohort(tmp_path, cohort)
        assert disc_errs, "bare-Call trusted-helper-spoof must shadow"
    finally:
        sys.path[:] = old_path


def test_output_dict_reassignment_from_arbitrary_name_fails_closed(
    tmp_path: Path,
) -> None:
    """Codex pass-5 HIGH-12: ``record = temp`` where ``temp`` is NOT
    an output-dict alias must error. Pass-3 exempted ALL Name RHS;
    pass-5 narrows to only output-dict alias names.
    """
    converter = textwrap.dedent(
        """
        from typing import Any

        class C:
            def _build_record(self) -> dict[str, Any]:
                record: dict[str, Any] = {"known": 1}
                temp = {"known": 1, "unmapped_temp_key": 1}
                record = temp
                return record
        """
    ).strip()
    (tmp_path / "scripts").mkdir()
    (tmp_path / "scripts" / "synthetic_converter.py").write_text(converter)
    (tmp_path / "_synthetic_manifest.py").write_text(_SYNTHETIC_MANIFEST)

    cohort = _make_synthetic_cohort(
        tmp_path, "scripts/synthetic_converter.py", "SYNTHETIC_TEST_FEATURES"
    )

    old_path = list(sys.path)
    sys.path.insert(0, str(tmp_path))
    try:
        discovered, disc_errs = cmc.discover_columns_for_cohort(tmp_path, cohort)
        assert disc_errs, f"record = arbitrary_temp must error: {disc_errs}"
        assert any("record = temp" in e for e in disc_errs)
    finally:
        sys.path[:] = old_path


@pytest.mark.parametrize(
    "call_shape",
    [
        # Star-args list-unpack: ``some_func(*[record])``
        "some_func(*[record])",
        # Star-args tuple-unpack: ``some_func(*(record,))``
        "some_func(*(record,))",
        # Kwargs dict-unpack: ``some_func(**{\"arg\": record})``
        'some_func(**{"arg": record})',
    ],
)
def test_helper_call_unpacked_output_arg_fails_closed(tmp_path: Path, call_shape: str) -> None:
    """Codex pass-7 HIGH-14: an output-dict Name hidden inside a
    Starred / List / Tuple / Dict unpack passed to a helper still
    delivers the dict by reference at runtime. The recursive
    arg-scan (``_bare_output_names_in_expr``) must surface the
    output Name.
    """
    converter = textwrap.dedent(
        f"""
        from typing import Any

        def some_func(*args, **kwargs) -> None:
            pass

        class C:
            def _build_record(self) -> dict[str, Any]:
                record: dict[str, Any] = {{"known": 1}}
                {call_shape}
                return record
        """
    ).strip()
    (tmp_path / "scripts").mkdir()
    (tmp_path / "scripts" / "synthetic_converter.py").write_text(converter)
    (tmp_path / "_synthetic_manifest.py").write_text(_SYNTHETIC_MANIFEST)

    cohort = _make_synthetic_cohort(
        tmp_path, "scripts/synthetic_converter.py", "SYNTHETIC_TEST_FEATURES"
    )

    old_path = list(sys.path)
    sys.path.insert(0, str(tmp_path))
    try:
        discovered, disc_errs = cmc.discover_columns_for_cohort(tmp_path, cohort)
        assert disc_errs, f"unpack-wrapper hiding output dict must error (shape={call_shape!r})"
    finally:
        sys.path[:] = old_path


def test_wrapper_append_with_output_arg_fails_closed(tmp_path: Path) -> None:
    """Codex pass-6 HIGH: ``wrapper.append(record)`` where ``wrapper`` is
    NOT in the cohort's ``collector_names`` must still be flagged. The
    pass-3 fix only narrowed by METHOD name (append/extend); pass-6
    further requires the RECEIVER to be a known collector identifier.
    """
    converter = textwrap.dedent(
        """
        from typing import Any

        class C:
            def _build_record(self) -> dict[str, Any]:
                record: dict[str, Any] = {"known": 1}
                # ``wrapper.append`` is user-defined code that can
                # mutate ``record``; not a real list-collector.
                wrapper.append(record)
                return record
        """
    ).strip()
    (tmp_path / "scripts").mkdir()
    (tmp_path / "scripts" / "synthetic_converter.py").write_text(converter)
    (tmp_path / "_synthetic_manifest.py").write_text(_SYNTHETIC_MANIFEST)

    # No collector_names declared — wrapper.append(record) must be
    # flagged as unsupported.
    cohort = _make_synthetic_cohort(
        tmp_path, "scripts/synthetic_converter.py", "SYNTHETIC_TEST_FEATURES"
    )

    old_path = list(sys.path)
    sys.path.insert(0, str(tmp_path))
    try:
        discovered, disc_errs = cmc.discover_columns_for_cohort(tmp_path, cohort)
        assert disc_errs, "wrapper.append(record) with no collector_names declared must error"
    finally:
        sys.path[:] = old_path


def test_output_dict_alias_self_assign_does_not_error(tmp_path: Path) -> None:
    """``record = record`` (no-op self-assign) and ``alias = record;
    record = alias`` (alias back to self) must NOT trigger the
    pass-5 HIGH-12 Pattern E check.
    """
    converter = textwrap.dedent(
        """
        from typing import Any

        class C:
            def _build_record(self) -> dict[str, Any]:
                record: dict[str, Any] = {"known": 1}
                alias = record  # alias propagation: alias is added
                record = alias  # alias is in output_dict_names → no-op
                return record
        """
    ).strip()
    (tmp_path / "scripts").mkdir()
    (tmp_path / "scripts" / "synthetic_converter.py").write_text(converter)
    (tmp_path / "_synthetic_manifest.py").write_text(_SYNTHETIC_MANIFEST)

    cohort = _make_synthetic_cohort(
        tmp_path, "scripts/synthetic_converter.py", "SYNTHETIC_TEST_FEATURES"
    )

    old_path = list(sys.path)
    sys.path.insert(0, str(tmp_path))
    try:
        discovered, disc_errs = cmc.discover_columns_for_cohort(tmp_path, cohort)
        assert disc_errs == [], f"alias self-assign should not error: {disc_errs}"
        assert "known" in discovered
    finally:
        sys.path[:] = old_path


def test_subscript_read_in_if_expression_does_not_false_positive(
    tmp_path: Path,
) -> None:
    """The MEDIUM-2 fix must NOT false-positive on a normal subscript
    read inside an IfExp value: this is the Optum
    ``feats["urban_rural_code"] = "urban" if feats["zip3"] in
    URBAN_ZIP3_PREFIXES else "suburban"`` pattern, which is a legitimate
    subscript assignment whose RHS happens to READ the same output
    dict.
    """
    converter = textwrap.dedent(
        """
        from typing import Any

        URBAN: tuple[str, ...] = ("100", "200")

        class C:
            def _build_record(self) -> dict[str, Any]:
                feats: dict[str, Any] = {"zip3": "100"}
                feats["urban_rural_code"] = (
                    "urban" if feats["zip3"] in URBAN else "suburban"
                )
                return feats
        """
    ).strip()
    (tmp_path / "scripts").mkdir()
    (tmp_path / "scripts" / "synthetic_converter.py").write_text(converter)
    (tmp_path / "_synthetic_manifest.py").write_text(
        textwrap.dedent(
            """
            from src.data.feature_contract import FeatureContract, KnowableAt

            SYNTHETIC_TEST_FEATURES: list[FeatureContract] = [
                FeatureContract(
                    name="zip3",
                    knowable_at=KnowableAt(reference="enrollment"),
                    source="demo",
                ),
                FeatureContract(
                    name="urban_rural_code",
                    knowable_at=KnowableAt(reference="enrollment"),
                    source="derived",
                ),
            ]
            """
        ).strip()
    )

    cohort = cmc.CohortConfig(
        name="synthetic-ifexp-read",
        converter_rel_path="scripts/synthetic_converter.py",
        discovery_funcs=(
            cmc.DiscoveryFunc(
                func_name="_build_record",
                output_dict_names=("feats",),
            ),
        ),
        manifest_module="_synthetic_manifest",
        manifest_attr="SYNTHETIC_TEST_FEATURES",
    )

    old_path = list(sys.path)
    sys.path.insert(0, str(tmp_path))
    try:
        discovered, disc_errs = cmc.discover_columns_for_cohort(tmp_path, cohort)
        # The IfExp value reads feats["zip3"], but this is a subscript
        # READ, not an alias assignment. Must NOT trigger MEDIUM-2.
        assert disc_errs == [], f"subscript read in IfExp should not be flagged: {disc_errs}"
        assert "urban_rural_code" in discovered
        assert "zip3" in discovered
    finally:
        sys.path[:] = old_path
