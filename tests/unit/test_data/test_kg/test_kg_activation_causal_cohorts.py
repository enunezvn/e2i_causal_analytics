"""Lane E — KG Layer 2 is bound, in shadow, for the causal cohorts.

Spec §3 Lane E item 1: caches for ``optum_mart`` and ``csu`` built with
``scripts/build_kg_cache.py --live`` against BOTH treatment concepts of the Lane A
contrast (omalizumab ``RXNORM:302379`` and dupilumab, resolved through RxNav
in-lane: ``RXNORM:1876376``, exact match, TTY=IN — evidence
``docs/demos/results/2026-09-22_lane_e_feature_role_voters/rxnav_dupilumab_resolution.txt``),
committed under ``data/kg_cache/`` and activated in ``KG_ACTIVATIONS`` in shadow.
Promotion is an owner decision (spec §7) and is NOT made here.

These tests read the COMMITTED caches so they run offline and fail loudly when a
configured cache is missing or stale (spec §6: "fail-loud on a missing cache").
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.agents.ml_foundation.data_preparer.nodes.adaptive_validity_check import (
    _load_kg_cache,
    _parse_target_entity_codes,
    _resolve_manifest_features,
)
from src.data.kg.activation import (
    DEFAULT_KG_CACHE_DIR,
    DUPILUMAB_RXCUI,
    KG_ACTIVATIONS,
    OMALIZUMAB_RXCUI,
    apply_kg_activation,
)
from src.data.kg.cache import (
    compose_cache_filename,
    compute_manifest_fingerprint,
    compute_target_codes_fingerprint,
)
from src.data.kg.ensemble_voter import classify_kg_signal

_CAUSAL_SOURCES = ("optum_mart", "csu")
_BOTH_DRUGS = [("RXNORM", OMALIZUMAB_RXCUI), ("RXNORM", DUPILUMAB_RXCUI)]


def test_the_rxcuis_are_pinned() -> None:
    assert OMALIZUMAB_RXCUI == "302379"
    assert DUPILUMAB_RXCUI == "1876376"


@pytest.mark.parametrize("source", _CAUSAL_SOURCES)
def test_causal_cohort_is_activated_in_shadow_against_both_drugs(source: str) -> None:
    activation = KG_ACTIVATIONS[source]
    assert activation.mode == "shadow", "promotion is an owner decision (spec §7)"
    assert list(activation.target_entity_codes) == _BOTH_DRUGS


@pytest.mark.parametrize("source", _CAUSAL_SOURCES)
def test_committed_cache_exists_and_loads(source: str) -> None:
    path = DEFAULT_KG_CACHE_DIR / KG_ACTIVATIONS[source].cache_filename
    assert path.is_file(), f"{path} is configured but not committed"
    scope_spec: dict = {"feature_manifest_source": source}
    assert apply_kg_activation(scope_spec, source) is True
    assert scope_spec["kg_mode"] == "shadow"
    cache = _load_kg_cache(scope_spec)
    assert cache is not None, "activation bound a path the loader refused (stale fingerprint?)"


@pytest.mark.parametrize("source", sorted(KG_ACTIVATIONS))
def test_every_activation_names_the_cache_its_manifest_fingerprints_to(source: str) -> None:
    """Generalises the optum-only #1623 staleness guard to every activation."""
    activation = KG_ACTIVATIONS[source]
    features = _resolve_manifest_features(source)
    assert features is not None, f"{source} is activated but not a registered manifest"
    expected = compose_cache_filename(
        compute_manifest_fingerprint(features),
        compute_target_codes_fingerprint([tuple(t) for t in activation.target_entity_codes]),
    )
    assert expected == activation.cache_filename, (
        f"KG_ACTIVATIONS[{source!r}] names {activation.cache_filename!r} but the live "
        f"manifest + targets fingerprint to {expected!r}; rebuild the cache."
    )


def _signals(source: str) -> dict[str, tuple[str, set[str]]]:
    """feature -> (signal, target codes its considered edges connect to)."""
    scope_spec: dict = {"feature_manifest_source": source}
    apply_kg_activation(scope_spec, source)
    cache = _load_kg_cache(scope_spec) or {}
    target_ids = _parse_target_entity_codes(scope_spec["target_entity_codes"])
    by_name = {f.name: f for f in (_resolve_manifest_features(source) or [])}
    out: dict[str, tuple[str, set[str]]] = {}
    for feature, edges in cache.items():
        contract = by_name.get(feature)
        feat_ids = tuple(code for _s, code in (contract.kg_entity_codes if contract else ()))
        signal, considered = classify_kg_signal(
            tuple(edges), feature_entity_ids=feat_ids, target_entity_ids=target_ids
        )
        out[feature] = (
            signal,
            {e.subject_id for e in considered} | {e.object_id for e in considered},
        )
    return out


def test_optum_mart_cache_carries_a_real_signal_from_each_drug() -> None:
    """Both drugs are approved for asthma, which the chronic-pulmonary flags
    name (``MART_COMORBIDITY_KG_CODES``). A signal that connects to only one
    target code would mean the second drug never got its drug-disease pass."""
    signals = _signals("optum_mart")
    assert signals, "the optum_mart cache has no records — the manifest has no entity codes?"
    flagged = {f: s for f, (s, _t) in signals.items() if s != "no_signal"}
    assert "cci_chronic_pulmonary" in flagged, flagged
    assert 0 < len(flagged) < len(signals), "signal must be selective, not blanket"
    _sig, targets = signals["cci_chronic_pulmonary"]
    assert {OMALIZUMAB_RXCUI, DUPILUMAB_RXCUI} <= targets, targets


def test_csu_cache_has_its_one_entity_feature() -> None:
    path = DEFAULT_KG_CACHE_DIR / KG_ACTIVATIONS["csu"].cache_filename
    records = json.loads(Path(path).read_text())
    assert [r["feature_name"] for r in records] == ["primary_diagnosis_code"]


def test_optum_activation_is_unchanged() -> None:
    """The prediction cohort keeps its one-drug cache (built for initiation)."""
    assert KG_ACTIVATIONS["optum"].cache_filename == "1cdaa038__96bfd2e0.json"
    assert list(KG_ACTIVATIONS["optum"].target_entity_codes) == [("RXNORM", "302379")]
