"""KG Layer-2 is ON and produces a real signal (#1607 AC3 + AC4).

Phase 2.9 Stage 2 shipped the KG code but never the data. ``kg_cache_path`` and
``kg_mode`` both defaulted to ``None``, ``_resolve_kg_mode(None)`` returned
``"off"``, and nothing in the repo set either — so ``kg_edges`` was always
``()``, ``classify_kg_signal`` always returned ``no_signal`` and ``decided_by``
was never ``"kg"``.

Building the cache alone did NOT fix that, which is the part the issue did not
anticipate. Measured 2026-08-14: a ``--live`` build produced 74 records with 82
real UMLS edges, and every single feature still classified as ``no_signal`` —
because ``query_disease_hierarchy`` relates a feature concept only to its OWN
parents and children, never to the prediction target, and
``classify_kg_signal._connects`` requires one endpoint in the feature set AND
one in the target set. Re-running with a valid UMLS disease CUI as the target
gave the same 74/74.

What actually lights the signal is the Open Targets drug-disease pass:
"is the target drug APPROVED to treat this feature's disease?". That path was
dead twice over — never called at build time, and broken against the live
GraphQL schema (HTTP 400 on every call) because all its unit tests mock the
transport.

These tests read the COMMITTED cache, so they run offline and pin the artifact
the pipeline actually loads. They are the regression guard for "the KG layer is
quietly dark again".
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.agents.ml_foundation.data_preparer.nodes.adaptive_validity_check import (
    _load_kg_cache,
    _parse_target_entity_codes,
    _resolve_kg_mode,
)
from src.data.kg.activation import KG_ACTIVATIONS, apply_kg_activation
from src.data.kg.ensemble_voter import classify_kg_signal
from src.data.manifests.optum_feature_manifest import OPTUM_FEATURES

pytestmark = [pytest.mark.integration]

_REPO_ROOT = Path(__file__).resolve().parents[3]
_OPTUM = KG_ACTIVATIONS["optum"]


def _cache_path() -> Path:
    return _REPO_ROOT / "data" / "kg_cache" / _OPTUM.cache_filename


# ------------------------------------------------------------------ AC2: artifact


def test_committed_kg_cache_exists_and_is_tracked() -> None:
    """The artifact the pipeline depends on must be IN THE REPO.

    ``data/kg_cache/*.json`` is gitignored; the committed cache carries an
    explicit un-ignore. #600 is the precedent: a gitignored tier0 cache was
    never committed and silently skipped agent execution in CI, presenting as
    "no signal" rather than "missing file".
    """
    path = _cache_path()
    assert path.is_file(), (
        f"committed KG cache {path} is missing — KG Layer 2 silently degrades to "
        "no_signal. Rebuild with scripts/build_kg_cache.py --live."
    )
    records = json.loads(path.read_text())
    assert isinstance(records, list) and records, "committed cache is empty"


# ------------------------------------------------------------------ AC3: wiring


def test_activation_sets_cache_path_and_shadow_mode() -> None:
    """#1607 AC3 — the Optum cohort's scope_spec gets kg_cache_path + shadow."""
    scope_spec: dict = {"feature_manifest_source": "optum"}
    assert apply_kg_activation(scope_spec, "optum") is True

    assert scope_spec["kg_cache_path"].endswith(_OPTUM.cache_filename)
    assert scope_spec["kg_mode"] == "shadow"
    assert _resolve_kg_mode(scope_spec["kg_mode"]) == "shadow"
    assert [tuple(t) for t in scope_spec["target_entity_codes"]] == [("RXNORM", "302379")]


def test_activation_respects_an_explicit_operator_setting() -> None:
    """An operator who turned KG off must stay off."""
    scope_spec: dict = {"feature_manifest_source": "optum", "kg_mode": "off"}
    assert apply_kg_activation(scope_spec, "optum") is False
    assert scope_spec["kg_mode"] == "off"
    assert "kg_cache_path" not in scope_spec


def test_activation_is_a_loud_noop_when_the_cache_is_missing(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """A missing cache must not look like "the KG had nothing to say"."""
    scope_spec: dict = {"feature_manifest_source": "optum"}
    with caplog.at_level("ERROR", logger="src.data.kg.activation"):
        activated = apply_kg_activation(scope_spec, "optum", cache_dir=tmp_path)

    assert activated is False
    assert scope_spec.get("kg_mode") is None
    assert any("MISSING" in message for message in caplog.messages), (
        "a missing cache must log an ERROR — silence makes it indistinguishable "
        "from a genuine no_signal"
    )


def test_unknown_manifest_source_is_not_activated() -> None:
    scope_spec: dict = {}
    assert apply_kg_activation(scope_spec, "synthetic") is False
    assert apply_kg_activation(scope_spec, None) is False
    assert scope_spec == {}


# ------------------------------------------------------------------- AC4: signal


def test_at_least_one_feature_carries_a_real_kg_signal() -> None:
    """#1607 AC4 — the load-bearing assertion this whole issue is about.

    Before the drug-disease pass this was 74/74 ``no_signal``, and it would have
    stayed that way after merely building a cache and flipping the flag.
    """
    scope_spec: dict = {"feature_manifest_source": "optum"}
    assert apply_kg_activation(scope_spec, "optum") is True

    cache = _load_kg_cache(scope_spec)
    assert cache, "activation set kg_cache_path but the loader returned nothing"

    target_ids = _parse_target_entity_codes(scope_spec["target_entity_codes"])
    assert target_ids, "target_entity_codes did not parse into ids"

    by_name = {f.name: f for f in OPTUM_FEATURES}
    signalled: dict[str, str] = {}
    for feature, edges in cache.items():
        contract = by_name.get(feature)
        feature_ids = (
            tuple(code for _system, code in (contract.kg_entity_codes or ()))
            if contract is not None
            else ()
        )
        signal, _considered = classify_kg_signal(
            tuple(edges), feature_entity_ids=feature_ids, target_entity_ids=target_ids
        )
        if signal != "no_signal":
            signalled[feature] = signal

    assert signalled, (
        "every feature classified as no_signal — the KG layer is dark again. "
        "Building the cache is NOT sufficient: only the Open Targets "
        "drug-disease pass produces edges that connect a feature to the target."
    )
    # The urticaria diagnosis-code features are the clinically obvious leak:
    # omalizumab is approved to treat urticaria, so a post-index L50.x claim
    # count leaks treatment.
    assert "dx_total_csu" in signalled, (
        f"expected the CSU diagnosis-count feature to be flagged; got {signalled}"
    )
    assert all(s == "leak_drug_treats_disease" for s in signalled.values()), signalled


def test_signal_is_selective_not_blanket() -> None:
    """A signal on EVERY feature would mean the connect-check had stopped filtering.

    Labs and utilisation counts have no disease counterpart the target drug
    treats, so they must stay ``no_signal``. This is the guard against a future
    change that makes ``_connects`` promiscuous.
    """
    scope_spec: dict = {"feature_manifest_source": "optum"}
    apply_kg_activation(scope_spec, "optum")
    cache = _load_kg_cache(scope_spec) or {}
    target_ids = _parse_target_entity_codes(scope_spec["target_entity_codes"])
    by_name = {f.name: f for f in OPTUM_FEATURES}

    signals = []
    for feature, edges in cache.items():
        contract = by_name.get(feature)
        feature_ids = (
            tuple(code for _s, code in (contract.kg_entity_codes or ()))
            if contract is not None
            else ()
        )
        signal, _c = classify_kg_signal(
            tuple(edges), feature_entity_ids=feature_ids, target_entity_ids=target_ids
        )
        signals.append(signal)

    flagged = [s for s in signals if s != "no_signal"]
    assert 0 < len(flagged) < len(signals), (
        f"expected a selective signal; got {len(flagged)} flagged of {len(signals)}"
    )


def test_open_targets_edges_are_present_and_phase_gated() -> None:
    """The committed cache must actually contain approved-indication edges."""
    records = json.loads(_cache_path().read_text())
    ot_edges = [
        edge
        for record in records
        for edge in (record.get("edges") or [])
        if edge.get("evidence_source") == "open_targets"
    ]
    assert ot_edges, (
        "no open_targets edges in the committed cache — the drug-disease pass "
        "did not run or returned nothing"
    )
    treats = [e for e in ot_edges if e.get("predicate") == "treats"]
    assert treats, "no 'treats' edges — only an approved indication earns that predicate"
    # Provenance survives the cache round-trip via the NAMES (KGEdge.raw is not
    # serialised by cache._kg_edge_to_json).
    assert any(e.get("subject_name") for e in treats), "lost the drug name provenance"
    assert all(e.get("datasource") == "chembl_indications" for e in ot_edges)
