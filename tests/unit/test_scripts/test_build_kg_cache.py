"""Smoke tests for scripts/build_kg_cache.py.

Live KG calls are NOT exercised here — those tests gate on
``UMLS_UTS_API_KEY`` (skipped in CI). The CLI is exercised via its
public functions; HTTP clients are passed as ``None`` for the no-op
case and are not constructed.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def test_cli_help_runs_without_error():
    """`python scripts/build_kg_cache.py --help` exits 0 with usage text."""
    result = subprocess.run(
        [sys.executable, "scripts/build_kg_cache.py", "--help"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "manifest-module" in result.stdout


def test_build_with_no_entity_features_writes_empty_cache(tmp_path: Path):
    """A manifest with zero entity-bearing features produces an empty
    cache file (no-op success).
    """
    from scripts.build_kg_cache import build_cache_for_manifest
    from src.data.feature_contract import FeatureContract, KnowableAt

    features = [
        FeatureContract(
            name="age",
            knowable_at=KnowableAt(reference="enrollment"),
            source="demo",
            derivation_inputs=("age",),
        )
    ]
    out = tmp_path / "kg_cache"
    cache_path = build_cache_for_manifest(
        features=features,
        target_entity_codes=[("RXNORM", "479158")],
        out_dir=out,
    )

    assert cache_path.exists()
    payload = json.loads(cache_path.read_text())
    assert payload == []

    # Companion summary
    summary_path = cache_path.with_suffix(".summary.md")
    assert summary_path.exists()
    summary = summary_path.read_text()
    assert "KG Cache Summary" in summary


def test_build_with_entity_feature_emits_record(tmp_path: Path):
    """A manifest with one entity-bearing feature produces one record."""
    from scripts.build_kg_cache import build_cache_for_manifest
    from src.data.feature_contract import FeatureContract, KnowableAt

    features = [
        FeatureContract(
            name="primary_diagnosis_code",
            knowable_at=KnowableAt(reference="enrollment"),
            source="demo",
            derivation_inputs=("diagcode",),
            kg_entity_codes=(("ICD10CM", "L50.9"), ("UMLS", "C0042109")),
        )
    ]
    out = tmp_path / "kg_cache"
    cache_path = build_cache_for_manifest(
        features=features,
        target_entity_codes=[],
        out_dir=out,
    )

    payload = json.loads(cache_path.read_text())
    assert len(payload) == 1
    assert payload[0]["feature_name"] == "primary_diagnosis_code"
    assert payload[0]["status"] in {"queried_no_edges", "ok"}


def test_parse_target_codes_handles_empty_string():
    """Empty --target-entity-codes parses to []."""
    from scripts.build_kg_cache import _parse_target_codes

    assert _parse_target_codes("") == []
    assert _parse_target_codes("  ") == []


def test_parse_target_codes_parses_multiple():
    from scripts.build_kg_cache import _parse_target_codes

    out = _parse_target_codes("RXNORM:479158,RXNORM:1011295")
    assert out == [("RXNORM", "479158"), ("RXNORM", "1011295")]


def test_parse_target_codes_rejects_malformed():
    """Missing colon → ValueError surfaced to caller."""
    import pytest

    from scripts.build_kg_cache import _parse_target_codes

    with pytest.raises(ValueError, match="exactly SYSTEM:code"):
        _parse_target_codes("RXNORM_no_colon_479158")


def test_parse_target_codes_rejects_empty_system():
    """':479158' (no system) → ValueError."""
    import pytest

    from scripts.build_kg_cache import _parse_target_codes

    with pytest.raises(ValueError, match="non-empty SYSTEM:code"):
        _parse_target_codes(":479158")


def test_parse_target_codes_rejects_empty_code():
    """'RXNORM:' (no code) → ValueError."""
    import pytest

    from scripts.build_kg_cache import _parse_target_codes

    with pytest.raises(ValueError, match="non-empty SYSTEM:code"):
        _parse_target_codes("RXNORM:")


def test_parse_target_codes_rejects_extra_colon():
    """'RXNORM:479158:extra' (>1 colon) → ValueError, not split-on-first."""
    import pytest

    from scripts.build_kg_cache import _parse_target_codes

    with pytest.raises(ValueError, match="exactly SYSTEM:code"):
        _parse_target_codes("RXNORM:479158:extra")


def test_cache_filename_omits_cohort(tmp_path: Path):
    """Disease-agnostic invariant: only the two fingerprints in the path."""
    from scripts.build_kg_cache import build_cache_for_manifest
    from src.data.feature_contract import FeatureContract, KnowableAt

    features = [
        FeatureContract(
            name="age",
            knowable_at=KnowableAt(reference="enrollment"),
            source="demo",
            derivation_inputs=("age",),
        )
    ]
    out = tmp_path / "kg_cache"
    cache_path = build_cache_for_manifest(
        features=features,
        target_entity_codes=[],
        out_dir=out,
    )

    # Cache filename pattern: {manifest_fp}__{target_fp}.json
    assert cache_path.name.endswith(".json")
    assert "__" in cache_path.name
    assert "csu" not in cache_path.name
    assert "optum" not in cache_path.name


def test_build_rejects_partial_live_args(tmp_path: Path):
    """Item B / PR-C: supplying exactly one of (entity_linker, kg_querier)
    is ambiguous (live mode requires both, smoke mode requires neither)
    → raises ValueError. Replaces the prior NotImplementedError gate.
    """
    import pytest

    from scripts.build_kg_cache import build_cache_for_manifest
    from src.data.feature_contract import FeatureContract, KnowableAt

    features = [
        FeatureContract(
            name="age",
            knowable_at=KnowableAt(reference="enrollment"),
            source="demo",
            derivation_inputs=("age",),
        )
    ]
    out = tmp_path / "kg_cache"

    sentinel = object()  # any non-None stand-in
    with pytest.raises(ValueError, match="entity_linker AND.*kg_querier"):
        build_cache_for_manifest(
            features=features,
            target_entity_codes=[],
            out_dir=out,
            entity_linker=sentinel,  # type: ignore[arg-type]
            kg_querier=None,
        )
    with pytest.raises(ValueError, match="entity_linker AND.*kg_querier"):
        build_cache_for_manifest(
            features=features,
            target_entity_codes=[],
            out_dir=out,
            entity_linker=None,
            kg_querier=sentinel,  # type: ignore[arg-type]
        )


def test_build_with_no_clients_records_empty_sources(tmp_path: Path):
    """No clients → sources_attempted is empty (provenance honesty)."""
    import json

    from scripts.build_kg_cache import build_cache_for_manifest
    from src.data.feature_contract import FeatureContract, KnowableAt

    features = [
        FeatureContract(
            name="primary_diagnosis_code",
            knowable_at=KnowableAt(reference="enrollment"),
            source="demo",
            derivation_inputs=("diagcode",),
            kg_entity_codes=(("ICD10CM", "L50.9"),),
        )
    ]
    out = tmp_path / "kg_cache"
    cache_path = build_cache_for_manifest(
        features=features,
        target_entity_codes=[],
        out_dir=out,
    )
    payload = json.loads(cache_path.read_text())
    assert len(payload) == 1
    assert payload[0]["sources_attempted"] == []


def test_summary_report_is_byte_stable_without_timestamp(tmp_path: Path):
    """Two regenerations produce byte-identical .summary.md (no timestamp)."""
    from scripts.build_kg_cache import build_cache_for_manifest
    from src.data.feature_contract import FeatureContract, KnowableAt

    features = [
        FeatureContract(
            name="age",
            knowable_at=KnowableAt(reference="enrollment"),
            source="demo",
            derivation_inputs=("age",),
        )
    ]

    out1 = tmp_path / "kg_cache_1"
    out2 = tmp_path / "kg_cache_2"
    p1 = build_cache_for_manifest(features=features, target_entity_codes=[], out_dir=out1)
    p2 = build_cache_for_manifest(features=features, target_entity_codes=[], out_dir=out2)

    summary1 = p1.with_suffix(".summary.md").read_bytes()
    summary2 = p2.with_suffix(".summary.md").read_bytes()
    assert summary1 == summary2


def test_summary_report_with_explicit_timestamp_includes_it(tmp_path: Path):
    """Passing a generated_at timestamp embeds it in the summary."""
    from datetime import datetime, timezone

    from scripts.build_kg_cache import _write_summary_report

    path = tmp_path / "x.summary.md"
    _write_summary_report(
        path,
        records=[],
        manifest_fp="aaaaaaaa",
        target_fp="bbbbbbbb",
        generated_at=datetime(2026, 5, 8, 12, 0, 0, tzinfo=timezone.utc),
    )
    text = path.read_text()
    assert "Generated:" in text
    assert "2026-05-08T12:00:00+00:00" in text


# =============================================================================
# Item B (PR-C live KG querying) — dispatcher + helper tests
# =============================================================================
#
# These tests exercise the per-entity dispatch + status-precedence logic
# WITHOUT making real HTTP calls. The EntityLinker / KGQuerier collaborators
# are stubbed via simple namespace classes so the test asserts on the
# in-memory dispatch decisions rather than network behavior. Live tests
# (gated on UMLS_UTS_API_KEY) live in the kg integration suite.


class _StubUMLS:
    """Minimal stand-in for ``UMLSClient`` covering only ``cui_lookup``.

    ``known`` is the set of CUIs the stub recognises; lookups outside the
    set raise ``UMLSNotFoundError``."""

    def __init__(self, known: set[str]):
        self.known = known
        self.lookup_calls: list[str] = []

    def cui_lookup(self, cui: str):
        from src.data.kg.types import KGConcept
        from src.data.kg.umls_uts import UMLSNotFoundError

        self.lookup_calls.append(cui)
        if cui not in self.known:
            raise UMLSNotFoundError(f"Unknown CUI: {cui!r}")
        return KGConcept(
            cui=cui, preferred_name=f"name-{cui}", semantic_types=("Disease",), atom_count=1
        )


class _StubEntityLinker:
    """EntityLinker stub: ``resolve(code, system)`` returns a kept set or
    a degenerate EntityLink with ``concept=None`` for misses. The ``umls``
    attribute exposes a ``_StubUMLS`` instance."""

    def __init__(self, *, code_to_cui: dict, umls_known: set[str]):
        self.code_to_cui = code_to_cui  # (system, code) → cui
        self.umls = _StubUMLS(umls_known)
        self.resolve_calls: list = []

    def resolve(self, code: str, system: str):
        from src.data.kg.types import EntityLink, KGConcept

        self.resolve_calls.append((system, code))
        cui = self.code_to_cui.get((system, code))
        if cui is None:
            return EntityLink(input_code=code, input_system=system, error="not found")
        return EntityLink(
            input_code=code,
            input_system=system,
            concept=KGConcept(
                cui=cui,
                preferred_name=f"name-{cui}",
                semantic_types=("Disease",),
                atom_count=1,
            ),
        )


class _StubKGQuerier:
    """KGQuerier stub returning a fixed list of edges per CUI."""

    def __init__(self, edges_by_cui: dict):
        self.edges_by_cui = edges_by_cui  # cui → list[KGEdge]
        self.query_calls: list[str] = []

    def query_disease_hierarchy(self, cui: str):
        self.query_calls.append(cui)
        return list(self.edges_by_cui.get(cui, []))


def _make_kg_edge(subject_id: str, object_id: str, predicate: str = "isa"):
    from src.data.kg.types import KGEdge

    return KGEdge(
        subject_id=subject_id,
        subject_name="",
        predicate=predicate,
        object_id=object_id,
        object_name="",
        evidence_source="umls_uts",
        score=None,
        pmids=(),
        datasource=None,
    )


def test_resolve_entity_to_cui_umls_passthrough():
    """``("UMLS", <CUI>)`` skips EntityLinker.resolve and validates via cui_lookup."""
    from scripts.build_kg_cache import _resolve_entity_to_cui

    linker = _StubEntityLinker(code_to_cui={}, umls_known={"C0042109"})
    cui, err = _resolve_entity_to_cui("UMLS", "C0042109", linker)  # type: ignore[arg-type]
    assert cui == "C0042109"
    assert err is None
    # Critical: resolve was NOT called for UMLS entries (codex B1)
    assert linker.resolve_calls == []
    # cui_lookup was called for validation (codex B3)
    assert linker.umls.lookup_calls == ["C0042109"]


def test_resolve_entity_to_cui_umls_unknown_cui_returns_error():
    """``("UMLS", <bogus>)`` returns an error string, no exception raised."""
    from scripts.build_kg_cache import _resolve_entity_to_cui

    linker = _StubEntityLinker(code_to_cui={}, umls_known={"C0042109"})
    cui, err = _resolve_entity_to_cui("UMLS", "C9999999", linker)  # type: ignore[arg-type]
    assert cui is None
    assert err is not None and "C9999999" in err and "not found" in err


def test_resolve_entity_to_cui_source_vocab_via_entitylinker():
    """ICD10CM / RXNORM / etc. route through EntityLinker.resolve."""
    from scripts.build_kg_cache import _resolve_entity_to_cui

    linker = _StubEntityLinker(
        code_to_cui={("ICD10CM", "L50.9"): "C0042109"},
        umls_known=set(),
    )
    cui, err = _resolve_entity_to_cui("ICD10CM", "L50.9", linker)  # type: ignore[arg-type]
    assert cui == "C0042109"
    assert err is None
    assert ("ICD10CM", "L50.9") in linker.resolve_calls


def test_build_record_live_status_ok_when_edges_returned():
    """At least one edge → status='ok'."""
    from scripts.build_kg_cache import _build_record_live
    from src.data.feature_contract import FeatureContract, KnowableAt

    fc = FeatureContract(
        name="primary_diagnosis_code",
        knowable_at=KnowableAt(reference="enrollment"),
        source="demo",
        derivation_inputs=("diagcode",),
        kg_entity_codes=(("UMLS", "C0042109"),),
    )
    linker = _StubEntityLinker(code_to_cui={}, umls_known={"C0042109"})
    querier = _StubKGQuerier(edges_by_cui={"C0042109": [_make_kg_edge("C0042109", "C0033578")]})
    record = _build_record_live(
        fc,
        manifest_fp="aaaa",
        target_fp="bbbb",
        target_entity_codes=[],
        sources_attempted=("umls_uts",),
        entity_linker=linker,  # type: ignore[arg-type]
        kg_querier=querier,  # type: ignore[arg-type]
    )
    assert record.status == "ok"
    assert len(record.edges) == 1
    assert record.errors == ()


def test_build_record_live_status_queried_no_edges_when_resolved_but_no_edges():
    """Entity resolved → kg_querier returned [] → status='queried_no_edges'."""
    from scripts.build_kg_cache import _build_record_live
    from src.data.feature_contract import FeatureContract, KnowableAt

    fc = FeatureContract(
        name="primary_diagnosis_code",
        knowable_at=KnowableAt(reference="enrollment"),
        source="demo",
        derivation_inputs=("diagcode",),
        kg_entity_codes=(("UMLS", "C0042109"),),
    )
    linker = _StubEntityLinker(code_to_cui={}, umls_known={"C0042109"})
    querier = _StubKGQuerier(edges_by_cui={"C0042109": []})
    record = _build_record_live(
        fc,
        manifest_fp="aaaa",
        target_fp="bbbb",
        target_entity_codes=[],
        sources_attempted=("umls_uts",),
        entity_linker=linker,  # type: ignore[arg-type]
        kg_querier=querier,  # type: ignore[arg-type]
    )
    assert record.status == "queried_no_edges"
    assert record.edges == ()


def test_build_record_live_status_entity_unresolved_when_all_unknown():
    """All entities fail to resolve → status='entity_unresolved'."""
    from scripts.build_kg_cache import _build_record_live
    from src.data.feature_contract import FeatureContract, KnowableAt

    fc = FeatureContract(
        name="primary_diagnosis_code",
        knowable_at=KnowableAt(reference="enrollment"),
        source="demo",
        derivation_inputs=("diagcode",),
        kg_entity_codes=(("UMLS", "C9999999"),),
    )
    linker = _StubEntityLinker(code_to_cui={}, umls_known=set())  # CUI unknown
    querier = _StubKGQuerier(edges_by_cui={})
    record = _build_record_live(
        fc,
        manifest_fp="aaaa",
        target_fp="bbbb",
        target_entity_codes=[],
        sources_attempted=("umls_uts",),
        entity_linker=linker,  # type: ignore[arg-type]
        kg_querier=querier,  # type: ignore[arg-type]
    )
    assert record.status == "entity_unresolved"
    assert len(record.errors) >= 1
    assert record.edges == ()


def test_build_record_live_dedupes_resolved_cuis():
    """codex L4: a feature with two entity codes resolving to the SAME
    CUI must query the upstream KG exactly once (no duplicate edges in
    the cache, no double-counted KG signal downstream)."""
    from scripts.build_kg_cache import _build_record_live
    from src.data.feature_contract import FeatureContract, KnowableAt

    fc = FeatureContract(
        name="primary_diagnosis_code",
        knowable_at=KnowableAt(reference="enrollment"),
        source="demo",
        derivation_inputs=("diagcode",),
        kg_entity_codes=(("ICD10CM", "L50.9"), ("UMLS", "C0042109")),
    )
    # Both entities resolve to the same CUI C0042109.
    linker = _StubEntityLinker(
        code_to_cui={("ICD10CM", "L50.9"): "C0042109"},
        umls_known={"C0042109"},
    )
    querier = _StubKGQuerier(
        edges_by_cui={"C0042109": [_make_kg_edge("C0042109", f"P{i}") for i in range(3)]}
    )
    record = _build_record_live(
        fc,
        manifest_fp="aaaa",
        target_fp="bbbb",
        target_entity_codes=[],
        sources_attempted=("umls_uts",),
        entity_linker=linker,  # type: ignore[arg-type]
        kg_querier=querier,  # type: ignore[arg-type]
    )
    assert record.status == "ok"
    # Three edges total — query was invoked ONCE despite two entity codes.
    assert len(record.edges) == 3
    assert querier.query_calls == ["C0042109"], (
        "Duplicate CUI was re-queried; codex L4 dedup contract violated"
    )


def test_build_record_live_aggregates_edges_across_distinct_cuis():
    """A feature with multiple entity codes resolving to DIFFERENT CUIs
    aggregates edges from each query (no dedup applied; legitimate)."""
    from scripts.build_kg_cache import _build_record_live
    from src.data.feature_contract import FeatureContract, KnowableAt

    fc = FeatureContract(
        name="primary_diagnosis_code",
        knowable_at=KnowableAt(reference="enrollment"),
        source="demo",
        derivation_inputs=("diagcode",),
        kg_entity_codes=(("UMLS", "C0042109"), ("UMLS", "C0033578")),
    )
    linker = _StubEntityLinker(
        code_to_cui={},
        umls_known={"C0042109", "C0033578"},
    )
    querier = _StubKGQuerier(
        edges_by_cui={
            "C0042109": [_make_kg_edge("C0042109", "X")],
            "C0033578": [_make_kg_edge("C0033578", "Y"), _make_kg_edge("C0033578", "Z")],
        }
    )
    record = _build_record_live(
        fc,
        manifest_fp="aaaa",
        target_fp="bbbb",
        target_entity_codes=[],
        sources_attempted=("umls_uts",),
        entity_linker=linker,  # type: ignore[arg-type]
        kg_querier=querier,  # type: ignore[arg-type]
    )
    assert record.status == "ok"
    assert len(record.edges) == 3  # 1 + 2
    assert sorted(querier.query_calls) == ["C0033578", "C0042109"]


def test_build_record_live_partial_failure_preserved_via_errors_field():
    """codex M3: when ONE entity resolves+queries successfully but
    ANOTHER hits a typed transport error, status remains ``ok`` (we got
    edges) and the error is recorded in the ``errors`` field for
    audit. Operators must check BOTH status AND errors. After D1
    (PR #102 H1 follow-up) the catch in ``_query_edges_for_cui``
    narrowed to ``UMLSError`` / ``OpenTargetsError`` so the partial-
    failure error must originate from a typed transport class."""
    from scripts.build_kg_cache import _build_record_live
    from src.data.feature_contract import FeatureContract, KnowableAt
    from src.data.kg.umls_uts import UMLSError

    fc = FeatureContract(
        name="primary_diagnosis_code",
        knowable_at=KnowableAt(reference="enrollment"),
        source="demo",
        derivation_inputs=("diagcode",),
        kg_entity_codes=(("UMLS", "C0042109"), ("UMLS", "C0033578")),
    )
    linker = _StubEntityLinker(
        code_to_cui={},
        umls_known={"C0042109", "C0033578"},
    )

    class _PartialFailQuerier:
        def query_disease_hierarchy(self, cui: str):
            if cui == "C0033578":
                raise UMLSError("transport boom")
            return [_make_kg_edge(cui, "X")]

    record = _build_record_live(
        fc,
        manifest_fp="aaaa",
        target_fp="bbbb",
        target_entity_codes=[],
        sources_attempted=("umls_uts",),
        entity_linker=linker,  # type: ignore[arg-type]
        kg_querier=_PartialFailQuerier(),  # type: ignore[arg-type]
    )
    assert record.status == "ok"
    assert len(record.edges) == 1
    assert any("transport boom" in e for e in record.errors), (
        f"Expected partial-failure error in errors[]; got {record.errors}"
    )


def test_build_record_live_all_queries_failed_yields_source_error():
    """codex M2: when every successfully-resolved entity's KG query
    raises a typed transport error, status MUST be 'source_error' (not
    'queried_no_edges'). After D1 (PR #102 H1 follow-up) the catch in
    ``_query_edges_for_cui`` narrowed to ``UMLSError`` / ``OpenTargetsError``
    so transport failures and programming bugs are no longer
    indistinguishable; this test pins the typed-transport-failure path."""
    from scripts.build_kg_cache import _build_record_live
    from src.data.feature_contract import FeatureContract, KnowableAt
    from src.data.kg.umls_uts import UMLSError

    fc = FeatureContract(
        name="primary_diagnosis_code",
        knowable_at=KnowableAt(reference="enrollment"),
        source="demo",
        derivation_inputs=("diagcode",),
        kg_entity_codes=(("UMLS", "C0042109"),),
    )
    linker = _StubEntityLinker(code_to_cui={}, umls_known={"C0042109"})

    class _AlwaysFailQuerier:
        def query_disease_hierarchy(self, cui: str):
            raise UMLSError("transport boom")

    record = _build_record_live(
        fc,
        manifest_fp="aaaa",
        target_fp="bbbb",
        target_entity_codes=[],
        sources_attempted=("umls_uts",),
        entity_linker=linker,  # type: ignore[arg-type]
        kg_querier=_AlwaysFailQuerier(),  # type: ignore[arg-type]
    )
    assert record.status == "source_error", (
        f"Expected source_error when all queries failed post-resolution; got {record.status}"
    )
    assert any("transport boom" in e for e in record.errors)


def test_build_record_live_propagates_umls_auth_error():
    """Typed-error follow-up (D1 / PR #102 H1): ``UMLSAuthError`` is fatal
    across the entire UMLS surface; ``_query_edges_for_cui`` MUST NOT
    swallow it as ``status=source_error`` because auth failure on one
    CUI implies it's broken for every CUI in the build. The cache builder
    should surface the error so the operator can fix credentials before
    rebuilding."""
    import pytest as _pytest

    from scripts.build_kg_cache import _build_record_live
    from src.data.feature_contract import FeatureContract, KnowableAt
    from src.data.kg.umls_uts import UMLSAuthError

    fc = FeatureContract(
        name="primary_diagnosis_code",
        knowable_at=KnowableAt(reference="enrollment"),
        source="demo",
        derivation_inputs=("diagcode",),
        kg_entity_codes=(("UMLS", "C0042109"),),
    )
    linker = _StubEntityLinker(code_to_cui={}, umls_known={"C0042109"})

    class _AuthFailQuerier:
        def query_disease_hierarchy(self, cui: str):
            raise UMLSAuthError("invalid api key")

    with _pytest.raises(UMLSAuthError):
        _build_record_live(
            fc,
            manifest_fp="aaaa",
            target_fp="bbbb",
            target_entity_codes=[],
            sources_attempted=("umls_uts",),
            entity_linker=linker,  # type: ignore[arg-type]
            kg_querier=_AuthFailQuerier(),  # type: ignore[arg-type]
        )


def test_query_edges_for_cui_propagates_umls_auth_error_at_helper_layer(caplog):
    """Codex LOW-3 follow-up: pin the ordering of ``except UMLSAuthError``
    BEFORE ``except (UMLSError, OpenTargetsError)`` directly at the
    ``_query_edges_for_cui`` boundary. ``UMLSAuthError`` is a UMLSError
    subclass, so without the explicit-first-arm ``except UMLSAuthError``
    the second arm would silently catch auth failures and surface them
    as ``status=source_error`` instead of aborting the build."""
    import logging

    import pytest as _pytest

    from scripts.build_kg_cache import _query_edges_for_cui
    from src.data.kg.umls_uts import UMLSAuthError

    class _AuthRaiseQuerier:
        def query_disease_hierarchy(self, cui: str):
            raise UMLSAuthError("invalid api key")

    with caplog.at_level(logging.WARNING, logger="scripts.build_kg_cache"):
        with _pytest.raises(UMLSAuthError):
            _query_edges_for_cui("C0042109", _AuthRaiseQuerier())  # type: ignore[arg-type]
    # The helper does not log on the auth-fatal path — the propagation
    # itself is the audit signal; no transport-failure-style warning
    # should masquerade as recoverable.
    assert not any("query_disease_hierarchy" in rec.message for rec in caplog.records), (
        f"Auth path should not emit a transport-style warning; got {[r.message for r in caplog.records]}"
    )


def test_query_edges_for_cui_returns_error_for_non_auth_umls_error():
    """Codex LOW-3 follow-up: pin the second-arm catch at the helper
    boundary. A non-auth ``UMLSError`` returns ``([], [error_message])``
    rather than raising. Combined with the auth test above, this proves
    the ordering takes effect (UMLSAuthError raises while UMLSError
    yields error)."""
    from scripts.build_kg_cache import _query_edges_for_cui
    from src.data.kg.umls_uts import UMLSError

    class _TransportRaiseQuerier:
        def query_disease_hierarchy(self, cui: str):
            raise UMLSError("transport boom")

    edges, errors = _query_edges_for_cui("C0042109", _TransportRaiseQuerier())  # type: ignore[arg-type]
    assert edges == ()
    assert len(errors) == 1
    assert "transport boom" in errors[0]
    assert "C0042109" in errors[0]


def test_build_record_live_programming_bug_not_swallowed_as_source_error():
    """Typed-error follow-up (D1 / PR #102 H1): a generic ``RuntimeError``
    (a programming bug elsewhere — e.g., a malformed FeatureContract)
    must NOT be caught and recorded as ``status=source_error``. After
    narrowing the catch to typed transport errors, programming bugs
    propagate as bugs and surface in CI immediately rather than
    silently producing degenerate cache entries."""
    import pytest as _pytest

    from scripts.build_kg_cache import _build_record_live
    from src.data.feature_contract import FeatureContract, KnowableAt

    fc = FeatureContract(
        name="primary_diagnosis_code",
        knowable_at=KnowableAt(reference="enrollment"),
        source="demo",
        derivation_inputs=("diagcode",),
        kg_entity_codes=(("UMLS", "C0042109"),),
    )
    linker = _StubEntityLinker(code_to_cui={}, umls_known={"C0042109"})

    class _BugQuerier:
        def query_disease_hierarchy(self, cui: str):
            raise RuntimeError("programmer-induced bug, not a transport failure")

    with _pytest.raises(RuntimeError, match="programmer-induced bug"):
        _build_record_live(
            fc,
            manifest_fp="aaaa",
            target_fp="bbbb",
            target_entity_codes=[],
            sources_attempted=("umls_uts",),
            entity_linker=linker,  # type: ignore[arg-type]
            kg_querier=_BugQuerier(),  # type: ignore[arg-type]
        )


def test_build_record_live_pagination_warning_at_threshold(caplog):
    """A query returning ≥50 edges should log a WARNING about truncation."""
    import logging

    from scripts.build_kg_cache import _build_record_live
    from src.data.feature_contract import FeatureContract, KnowableAt

    fc = FeatureContract(
        name="primary_diagnosis_code",
        knowable_at=KnowableAt(reference="enrollment"),
        source="demo",
        derivation_inputs=("diagcode",),
        kg_entity_codes=(("UMLS", "C0042109"),),
    )
    linker = _StubEntityLinker(code_to_cui={}, umls_known={"C0042109"})
    edges = [_make_kg_edge("C0042109", f"P{i}") for i in range(50)]
    querier = _StubKGQuerier(edges_by_cui={"C0042109": edges})

    with caplog.at_level(logging.WARNING, logger="scripts.build_kg_cache"):
        _build_record_live(
            fc,
            manifest_fp="aaaa",
            target_fp="bbbb",
            target_entity_codes=[],
            sources_attempted=("umls_uts",),
            entity_linker=linker,  # type: ignore[arg-type]
            kg_querier=querier,  # type: ignore[arg-type]
        )
    assert any("pagination ceiling" in rec.message for rec in caplog.records), (
        f"Pagination warning not emitted; records: {[r.message for r in caplog.records]}"
    )


def test_build_cache_for_manifest_live_path_produces_records(tmp_path: Path):
    """End-to-end live path on stub clients: cache file written, records
    have status='ok'/'queried_no_edges' as appropriate, sources_attempted
    includes 'umls_uts'."""
    import json

    from scripts.build_kg_cache import build_cache_for_manifest
    from src.data.feature_contract import FeatureContract, KnowableAt

    features = [
        FeatureContract(
            name="primary_diagnosis_code",
            knowable_at=KnowableAt(reference="enrollment"),
            source="demo",
            derivation_inputs=("diagcode",),
            kg_entity_codes=(("UMLS", "C0042109"),),
        ),
    ]
    linker = _StubEntityLinker(code_to_cui={}, umls_known={"C0042109"})
    querier = _StubKGQuerier(edges_by_cui={"C0042109": [_make_kg_edge("C0042109", "C0033578")]})

    out = tmp_path / "kg_cache"
    cache_path = build_cache_for_manifest(
        features=features,
        target_entity_codes=[("RXNORM", "479158")],
        out_dir=out,
        entity_linker=linker,  # type: ignore[arg-type]
        kg_querier=querier,  # type: ignore[arg-type]
    )
    payload = json.loads(cache_path.read_text())
    assert len(payload) == 1
    assert payload[0]["status"] == "ok"
    assert payload[0]["sources_attempted"] == ["umls_uts"]
    assert len(payload[0]["edges"]) == 1
