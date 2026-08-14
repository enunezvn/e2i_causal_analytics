"""Phase 2.9 citation-channel wiring — CitationResolver -> EnsembleVoter (#1608).

Europe PMC and Crossref have complete, unit-tested clients and a complete
``CitationResolver`` on top of them, and ``EnsembleVoter`` already accepts
``citation_verdicts``. Nothing called the resolver, so the channel had no
producer: ``_compose_legacy_verdict`` had no ``citation_verdicts`` parameter at
all, and the voter always received the empty default.

The consequence is visible in the voter's own audit text — every Layer-4
verdict carrying PMIDs landed on:

    "LLM cited N PMIDs but no CitationVerdicts supplied; treating as if
     citations were not checked"

so we surfaced LLM-cited PMIDs in the audit trail with no verification that the
abstract behind the PMID actually co-mentions the entities and a causal cue.

Measured 2026-08-14, motivating this wiring:
* The compiled Layer-4 classifier DOES cite — 129 of 192 demo mechanism strings
  match the loader's PMID regex, 57 distinct PMIDs. (The loader docstring
  claiming "the compile-set mechanism strings do not contain citations" is
  stale.)
* Those PMIDs resolve: 14/14 sampled returned real Europe PMC records.
* At least one is topically bogus — PMID 27176981 is "Enhanced absorption of
  graphene monolayer with a single-layer resonant grating", a physics paper, in
  a pharma causal-inference classifier. It RESOLVES, so a naive existence check
  passes it; only the entity co-mention + causal-cue check rejects it. That is
  the signal this channel adds.

These tests use real ``CitationVerdict`` dataclasses and a real
``EnsembleVoter`` — no transport mocking. The live Europe PMC / Crossref paths
are covered in tests/integration/test_kg/.
"""

from __future__ import annotations

from typing import Any

import pytest

from src.data.kg.types import CitationVerdict, LLMVerdict

# The adversarial "moderate" bucket is the Layer-4 trigger; any
# ``adversarial_input`` passed directly must carry the routing-guard tag.
_ADVERSARIAL_MODERATE: dict[str, Any] = {
    "layer": "3",
    "severity": "moderate",
    "remediation": "ambiguous",
    "evidence": "test moderate signal",
    "z_score": 3.5,
    "actual_auc": 0.62,
    "null_mean": 0.50,
    "null_std": 0.035,
    "p_value": 0.001,
    "n_permutations": 200,
    "_hblp_classified": True,
}


def _verified_verdict(identifier: str = "39021347") -> CitationVerdict:
    """A citation that passes the Phase 2.6 bar (resolved + 2 entities + cue)."""
    return CitationVerdict(
        identifier=identifier,
        identifier_kind="pmid",
        abstract_resolved=True,
        entities_found=("galectin-9", "urticaria"),
        causal_cue_found="associated with",
        overall_confidence=1.0,
    )


def _unverified_verdict(identifier: str = "27176981") -> CitationVerdict:
    """A citation that RESOLVES but is topically wrong (the graphene paper)."""
    return CitationVerdict(
        identifier=identifier,
        identifier_kind="pmid",
        abstract_resolved=True,
        entities_found=(),
        causal_cue_found=None,
        overall_confidence=0.0,
    )


def _llm_verdict(cited: tuple[str, ...]) -> LLMVerdict:
    return LLMVerdict(
        causal_role="ancestor",
        mechanism="pre-index inflammatory marker; see PMID: " + ", PMID: ".join(cited),
        recommended_remediation="keep",
        cited_pmids=cited,
    )


# ---------------------------------------------------------------- AC1: the seam


@pytest.fixture
def llm_decides(monkeypatch: pytest.MonkeyPatch) -> None:
    """Enable the LLM-decides path so ``_score_llm_verdict`` actually runs.

    IMPORTANT: citations only *modulate* confidence under
    ``ADAPTIVE_LAYER4_LLM_DECIDES=1``. The production default is Plan v4
    Phase 1 "audit-only", where the LLM verdict is recorded but does not
    decide, and ``_score_llm_verdict`` is never called. Both modes are
    covered here — the scoring assertions use this fixture; the audit-only
    assertions deliberately do not.
    """
    monkeypatch.setenv("ADAPTIVE_LAYER4_LLM_DECIDES", "1")


def _compose(**kwargs: Any) -> dict[str, Any]:
    from src.agents.ml_foundation.data_preparer.nodes.adaptive_validity_check import (
        _compose_legacy_verdict,
    )
    from src.data.kg.ensemble_voter import EnsembleVoter

    return _compose_legacy_verdict(
        "serum_tnf_alpha_baseline",
        voter=EnsembleVoter(),
        adversarial_input=dict(_ADVERSARIAL_MODERATE),
        **kwargs,
    )


def test_compose_legacy_verdict_accepts_and_forwards_citation_verdicts(
    llm_decides: None,
) -> None:
    """#1608 AC1 — the parameter exists and reaches the voter.

    Without it the voter always received the empty default and the audit trail
    recorded "no CitationVerdicts supplied" on every cited Layer-4 verdict.
    """
    verdict = _compose(
        llm_verdict=_llm_verdict(("39021347",)),
        citation_verdicts=(_verified_verdict(),),
    )

    evidence = str(verdict.get("evidence", ""))
    assert "no CitationVerdicts supplied" not in evidence, (
        "citation_verdicts did not reach the voter — the channel still has no producer"
    )
    assert "1 of 1 citation(s) verified" in evidence, (
        f"voter did not record the verified citation; evidence was: {evidence!r}"
    )


def test_all_citations_failing_is_recorded_and_penalised(llm_decides: None) -> None:
    """A resolvable-but-topically-wrong citation must NOT count as evidence.

    This is the graphene-paper case: the PMID exists and the abstract resolves,
    so any existence-only check passes it. Verification must still fail it.
    """
    verdict = _compose(
        llm_verdict=_llm_verdict(("27176981",)),
        citation_verdicts=(_unverified_verdict(),),
    )

    evidence = str(verdict.get("evidence", ""))
    assert "failed verification" in evidence, (
        f"an unverified citation must be recorded as failing; got: {evidence!r}"
    )


def test_default_citation_verdicts_preserves_prior_behaviour(llm_decides: None) -> None:
    """Omitting the parameter must leave existing callers byte-identical."""
    verdict = _compose(llm_verdict=_llm_verdict(("39021347",)))
    assert "no CitationVerdicts supplied" in str(verdict.get("evidence", "")), (
        "the no-citations path must be unchanged when the caller supplies nothing"
    )


# ------------------------------------------------- audit trail (default mode)


def test_citation_counts_reach_the_legacy_dict_in_audit_only_mode() -> None:
    """The counts must survive the legacy adaptation in the PRODUCTION default.

    ``ADAPTIVE_LAYER4_LLM_DECIDES`` is OFF by default, so the LLM verdict is
    audit-only and citations do not move severity. They must still be RECORDED:
    the harm #1608 describes is an unverified citation sitting in the audit
    trail looking like evidence. ``_ensemble_to_legacy_dict`` previously carried
    no citation fields at all, so verification would have run and then been
    discarded at this boundary.
    """
    verdict = _compose(
        llm_verdict=_llm_verdict(("39021347", "27176981")),
        citation_verdicts=(_verified_verdict("39021347"), _unverified_verdict("27176981")),
    )

    assert verdict["citations_checked"] == 2
    assert verdict["citations_verified"] == 1
    assert verdict["citations_unverified"] == 1
    assert verdict["verified_citation_ids"] == ["39021347"]
    assert verdict["cited_pmids"] == ["39021347", "27176981"]


def test_zero_cited_pmids_is_distinguishable_from_all_failed() -> None:
    """``citations_checked`` separates "cited nothing" from "all failed"."""
    none_cited = _compose(llm_verdict=_llm_verdict(()))
    all_failed = _compose(
        llm_verdict=_llm_verdict(("27176981",)),
        citation_verdicts=(_unverified_verdict(),),
    )

    assert none_cited["citations_checked"] == 0
    assert none_cited["cited_pmids"] == []
    assert all_failed["citations_checked"] == 1
    assert all_failed["citations_verified"] == 0


# ------------------------------------------------- AC2/AC3/AC4: the producer


def test_resolve_citation_verdicts_returns_empty_without_resolver() -> None:
    """No resolver configured (no UMLS key / import failure) → no verdicts, no raise."""
    from src.agents.ml_foundation.data_preparer.nodes.adaptive_validity_check import (
        _CitationBudget,
        _resolve_citation_verdicts,
    )

    assert (
        _resolve_citation_verdicts(
            _llm_verdict(("39021347",)),
            feature="serum_tnf_alpha_baseline",
            contract=None,
            target="responder",
            resolver=None,
            budget=_CitationBudget(limit=10),
        )
        == ()
    )


def test_resolve_citation_verdicts_fails_open_when_resolver_raises() -> None:
    """#1608 AC3 — any Europe PMC / Crossref error yields no verdict, never blocks.

    Mirrors the Layer-4 try/except: the pipeline must continue with an empty
    citation channel rather than propagating a network failure.
    """
    from src.agents.ml_foundation.data_preparer.nodes.adaptive_validity_check import (
        _CitationBudget,
        _resolve_citation_verdicts,
    )

    class _ExplodingResolver:
        def verify_citation(self, *args: Any, **kwargs: Any) -> CitationVerdict:
            raise RuntimeError("Europe PMC is down")

    result = _resolve_citation_verdicts(
        _llm_verdict(("39021347",)),
        feature="serum_tnf_alpha_baseline",
        contract=None,
        target="responder",
        resolver=_ExplodingResolver(),
        budget=_CitationBudget(limit=10),
    )
    assert result == (), "a resolver failure must degrade to no verdicts, not raise"


def test_resolution_is_bounded_per_run() -> None:
    """#1608 AC4 — a wide feature sweep must not fan out into hundreds of calls."""
    from src.agents.ml_foundation.data_preparer.nodes.adaptive_validity_check import (
        _CitationBudget,
        _resolve_citation_verdicts,
    )

    calls: list[str] = []

    class _CountingResolver:
        def verify_citation(self, identifier: str, **kwargs: Any) -> CitationVerdict:
            calls.append(identifier)
            return _verified_verdict(identifier)

    budget = _CitationBudget(limit=3)
    resolver = _CountingResolver()
    # Five features, two PMIDs each = 10 candidate resolutions against a budget of 3.
    for i in range(5):
        _resolve_citation_verdicts(
            _llm_verdict((f"1000000{i}", f"2000000{i}")),
            feature=f"feature_{i}",
            contract=None,
            target="responder",
            resolver=resolver,
            budget=budget,
        )

    assert len(calls) == 3, f"budget of 3 was exceeded: {len(calls)} resolutions made"


def test_per_feature_pmid_cap_limits_one_features_fanout() -> None:
    """One feature citing many PMIDs must not consume the whole run budget."""
    from src.agents.ml_foundation.data_preparer.nodes.adaptive_validity_check import (
        _CITATION_MAX_PMIDS_PER_FEATURE,
        _CitationBudget,
        _resolve_citation_verdicts,
    )

    calls: list[str] = []

    class _CountingResolver:
        def verify_citation(self, identifier: str, **kwargs: Any) -> CitationVerdict:
            calls.append(identifier)
            return _verified_verdict(identifier)

    many = tuple(f"3000000{i}" for i in range(10))
    _resolve_citation_verdicts(
        _llm_verdict(many),
        feature="chatty_feature",
        contract=None,
        target="responder",
        resolver=_CountingResolver(),
        budget=_CitationBudget(limit=100),
    )
    assert len(calls) == _CITATION_MAX_PMIDS_PER_FEATURE


# ------------------------------------------- CitationResolver constructibility


def test_citation_resolver_is_constructible_without_umls_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``umls=None`` must mean "no synonym expansion", as the docstring says.

    ``CitationResolver.__init__`` did ``self.umls = umls if umls is not None
    else UMLSClient()``, and ``UMLSClient()`` RAISES ``UMLSAuthError`` without a
    key — so the documented degraded mode was unreachable and the resolver was
    unconstructible in any environment lacking ``UMLS_UTS_API_KEY`` (including
    CI). That directly defeats #1608 AC3's fail-open requirement.
    """
    from src.data.kg.citation_resolver import CitationResolver

    monkeypatch.delenv("UMLS_UTS_API_KEY", raising=False)
    with CitationResolver() as resolver:
        assert resolver.umls is None, "expected UMLS to be absent without a key"
        # Europe PMC / Crossref must still be usable — they are zero-auth.
        assert resolver.europe_pmc is not None
        assert resolver.crossref is not None
