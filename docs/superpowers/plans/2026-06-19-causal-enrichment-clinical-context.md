# Clinical Context Enrichment Layer Implementation Plan

> For agentic workers: REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task. Steps use checkbox (- [ ]) syntax. Execute in an isolated git worktree; TDD red-first; commit per task; drive to a green fixed point and use codex:codex-rescue when stuck.

**Goal:** Give every discovered causal effect a brand-faithful, *sourced* clinical narrative — drug + mechanism of action (ChEMBL REST + static MoA fallback), the disease's real pivotal endpoints (ClinicalTrials.gov API v2 + static fallback) mapped to our synthetic outcome, and a real-world-evidence citation (PubMed E-utilities) — surfaced via a new `GET /causal/clinical-context` endpoint and a Clinical Context panel on the leaderboard drill-down, with an explicit "estimate = synthetic cohort; clinical context = real, cited" honesty label.

**Architecture:** A pure additive narrative/UI layer that NEVER touches the causal math, adjustment sets, or estimation frames. A provider-interface backend service (`ClinicalContextProvider` ABC → `ChEMBLMechanismProvider`, `ClinicalTrialsEndpointProvider`, `PubMedRWEProvider`) is fanned out by `ClinicalContextService`, which degrades gracefully per provider (a down/slow API yields a static fallback, never a 500), caches per `(brand, disease)`, and is exposed by a thin FastAPI endpoint reusing the existing `require_analyst`/`require_viewer` guards. The backend calls the PUBLIC REST APIs directly (the claude.ai MCP tools are agent-only): ChEMBL REST (`https://www.ebi.ac.uk/chembl/api/data`), ClinicalTrials.gov v2 (`https://clinicaltrials.gov/api/v2`), PubMed E-utilities (`https://eutils.ncbi.nlm.nih.gov/entrez/eutils`). The brand→drug→disease map reuses the canonical SSOT in `src/ml/synthetic/clinical_codes.py`.

**Tech Stack:** Python 3.12, FastAPI, synchronous `httpx` clients (mirroring `src/data/kg/chembl.py` / `src/data/kg/europe_pmc.py` exactly), `functools.lru_cache`, pytest + `httpx.MockTransport`; React/TypeScript, TanStack Query, vitest.

**Scope (this plan):** the Clinical Context layer only. **Out of scope → other plans / deferred:** P0 unified page (this plan only ADDS a panel into the existing #1030 drill-down — it does not restructure the page); P1/P2/P3 grain work; and the DEFERRED openFDA/UMLS tasks (#9–#11). The provider interface is built to be EXTENSIBLE for those (a new `ClinicalContextProvider` subclass slots in) but NO openFDA/UMLS code is written here.

**Pre-req / sequencing note:** This phase is **additive and backend-independent of P1/P2/P3** (it reads no estimation frames). It DOES depend on **PR #1030 merged first** for the FE drill-down it injects the panel into (`frontend/src/pages/CausalDiscovery.tsx`, `frontend/src/types/causal.ts`). All FE tasks below are written against the **post-#1030 merged state**, which today lives in the worktree `/home/enunez/Projects/wt_causal_discovery_revamp` — quote/anchor lines from there, but EDIT the merged files in your own worktree after #1030 is on `main`.

---

## File Structure

**Backend — Created**
- `src/services/clinical_context/__init__.py` — package exports (`ClinicalContextService`, `ClinicalContextProvider`, the three providers, `reset_caches`).
- `src/services/clinical_context/brand_map.py` — brand→{drug, disease, MoA-fallback, pivotal-endpoint-fallback, RWE-search-term, outcome→endpoint map} resolved from the `clinical_codes.py` SSOT.
- `src/services/clinical_context/providers.py` — `ClinicalContextProvider` ABC + `ChEMBLMechanismProvider`, `ClinicalTrialsEndpointProvider`, `PubMedRWEProvider` (each public-REST, best-effort, static fallback).
- `src/services/clinical_context/clients.py` — `ClinicalTrialsClient` + `PubMedClient` (synchronous httpx, lru_cache, distinct error types — mirrors `europe_pmc.py`).
- `src/services/clinical_context/service.py` — `ClinicalContextService.get_context(brand, outcome)` orchestrator: fan-out, per-`(brand,disease)` cache, graceful degradation, honesty label.

**Backend — Modified**
- `src/data/kg/chembl.py` — add `get_mechanism(molecule_chembl_id)` + `mechanism_of_action(drug_name)` to the existing `ChEMBLClient` (the `/mechanism.json` surface; reuses `compound_search`).
- `src/data/kg/__init__.py` — re-export the new ChEMBL symbols (none new to export beyond methods; no change needed unless a new symbol is added — see Task 1).
- `src/api/schemas/causal.py` — add `ClinicalContext`, `MechanismOfAction`, `PivotalEndpoint`, `RealWorldEvidence` Pydantic models.
- `src/api/routes/causal.py` — add `GET /causal/clinical-context` endpoint (+ a module-level `ClinicalContextService` instance).

**Backend — Tests (Created)**
- `tests/unit/test_data/test_kg/test_chembl_mechanism.py` — `get_mechanism` / `mechanism_of_action` via `httpx.MockTransport`.
- `tests/unit/test_services/test_clinical_context/__init__.py`
- `tests/unit/test_services/test_clinical_context/test_brand_map.py`
- `tests/unit/test_services/test_clinical_context/test_clients.py` — ClinicalTrials + PubMed clients via MockTransport.
- `tests/unit/test_services/test_clinical_context/test_providers.py` — each provider best-effort + fallback.
- `tests/unit/test_services/test_clinical_context/test_service.py` — orchestration, caching, degradation, honesty label.
- `tests/unit/test_api/test_causal_clinical_context.py` — endpoint contract + 404 unknown brand + degraded-but-200.

**Frontend — Modified (post-#1030 merged files)**
- `frontend/src/types/causal.ts` — add `ClinicalContext`, `MechanismOfAction`, `PivotalEndpoint`, `RealWorldEvidence` interfaces.
- `frontend/src/api/causal.ts` — add `getClinicalContext(brand, outcome)`.
- `frontend/src/hooks/api/use-causal.ts` — add `useClinicalContext(brand, outcome)`.
- `frontend/src/hooks/api/index.ts` — re-export `useClinicalContext` from the named-export barrel (the page imports from `@/hooks/api`).
- `frontend/src/pages/CausalDiscovery.tsx` — render `<ClinicalContextPanel>` in the drill-down + a compact MoA badge on each leaderboard row.

**Frontend — Created**
- `frontend/src/components/causal/ClinicalContextPanel.tsx` — the panel (drug + MoA, pivotal endpoints mapped to our synthetic outcome, RWE citation, honesty label).
- `frontend/src/components/causal/ClinicalContextPanel.test.tsx` — vitest render + honesty-label + provenance-link coverage.

---

### Task 1: Add `get_mechanism` / `mechanism_of_action` to the existing ChEMBLClient

The MCP `ChEMBL.get_mechanism` is agent-only; the backend's `ChEMBLClient` (`src/data/kg/chembl.py`) currently has `compound_search`/`target_search`/`get_bioactivity` but NO mechanism surface. Add the `/mechanism.json` surface, reusing `compound_search` to resolve a drug name → molecule id → MoA string. Live shape verified 2026-06-19: `GET /mechanism.json?molecule_chembl_id=CHEMBL3545110` → `{"mechanisms":[{"mechanism_of_action":"Cyclin-dependent kinase 4 inhibitor", ...}, ...]}`.

**Files:**
- Modify: `src/data/kg/chembl.py` (add `Mechanism` dataclass after `Activity` ~line 101; `get_mechanism` + `mechanism_of_action` methods after `get_bioactivity` ~line 325; LRU wrapper + `reset_caches` ~lines 379-398)
- Test: `tests/unit/test_data/test_kg/test_chembl_mechanism.py` (Create)

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/test_data/test_kg/test_chembl_mechanism.py
"""Unit tests for ChEMBLClient.get_mechanism / mechanism_of_action via
httpx.MockTransport. Mirrors test_chembl.py; pins the /mechanism.json shape
verified live 2026-06-19 against ChEMBL REST v34."""

from __future__ import annotations

from typing import Callable

import httpx
import pytest

from src.data.kg.chembl import ChEMBLClient, reset_caches


@pytest.fixture(autouse=True)
def _clear_caches() -> None:
    reset_caches()


def _client_with_handler(handler: Callable[[httpx.Request], httpx.Response]) -> ChEMBLClient:
    transport = httpx.MockTransport(handler)
    http = httpx.Client(transport=transport)
    return ChEMBLClient(client=http)


def test_get_mechanism_returns_actions_for_molecule_id() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert "/mechanism.json" in request.url.path
        query = request.url.query.decode("utf-8")
        assert "molecule_chembl_id" in query
        assert "CHEMBL3545110" in query
        return httpx.Response(
            200,
            json={
                "mechanisms": [
                    {
                        "mechanism_of_action": "Cyclin-dependent kinase 4 inhibitor",
                        "action_type": "INHIBITOR",
                        "target_chembl_id": "CHEMBL331",
                    },
                    {
                        "mechanism_of_action": "Cyclin-dependent kinase 6 inhibitor",
                        "action_type": "INHIBITOR",
                        "target_chembl_id": "CHEMBL2508",
                    },
                ]
            },
        )

    with _client_with_handler(handler) as client:
        mechs = client.get_mechanism("CHEMBL3545110")
    assert [m.mechanism_of_action for m in mechs] == [
        "Cyclin-dependent kinase 4 inhibitor",
        "Cyclin-dependent kinase 6 inhibitor",
    ]
    assert mechs[0].action_type == "INHIBITOR"
    assert mechs[0].target_chembl_id == "CHEMBL331"


def test_get_mechanism_empty_id_skips_network() -> None:
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return httpx.Response(200, json={"mechanisms": []})

    with _client_with_handler(handler) as client:
        assert client.get_mechanism("") == []
    assert calls["n"] == 0


def test_mechanism_of_action_resolves_drug_name_to_first_moa() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if "/molecule.json" in path:
            assert "ribociclib" in request.url.query.decode("utf-8").lower()
            return httpx.Response(200, json={"molecules": [{"molecule_chembl_id": "CHEMBL3545110"}]})
        assert "/mechanism.json" in path
        return httpx.Response(
            200,
            json={"mechanisms": [{"mechanism_of_action": "Cyclin-dependent kinase 4 inhibitor"}]},
        )

    with _client_with_handler(handler) as client:
        assert client.mechanism_of_action("ribociclib") == "Cyclin-dependent kinase 4 inhibitor"


def test_mechanism_of_action_unresolved_drug_returns_none() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"molecules": []})

    with _client_with_handler(handler) as client:
        assert client.mechanism_of_action("not-a-real-drug") is None
```

- [ ] **Step 2: Run it to confirm it fails**

Run: `python -m pytest tests/unit/test_data/test_kg/test_chembl_mechanism.py -v`
Expected: FAIL — `AttributeError: 'ChEMBLClient' object has no attribute 'get_mechanism'`.

- [ ] **Step 3: Add the `Mechanism` dataclass**

In `src/data/kg/chembl.py`, after the `Activity` dataclass (ends ~line 101), add:

```python
@dataclass(frozen=True)
class Mechanism:
    """One mechanism-of-action row from ChEMBL ``/mechanism.json``.

    Attributes:
        mechanism_of_action: Free-text MoA string (e.g.
            ``"Cyclin-dependent kinase 4 inhibitor"``).
        action_type: ChEMBL action type (e.g. ``"INHIBITOR"``); ``None`` when
            absent.
        target_chembl_id: ChEMBL target ID the drug acts on; ``None`` when absent.
    """

    mechanism_of_action: str
    action_type: Optional[str] = None
    target_chembl_id: Optional[str] = None
```

- [ ] **Step 4: Add the methods + LRU wrapper + reset_caches**

In `src/data/kg/chembl.py`, after `_get_bioactivity_uncached` (ends ~line 325, just before the module-level `def _row_to_activity`), add the two methods INSIDE the `ChEMBLClient` class:

```python
    # ------------------------------------------------------------------
    # Mechanism of action (#causal-enrichment)
    # ------------------------------------------------------------------

    def get_mechanism(self, molecule_chembl_id: str) -> list["Mechanism"]:
        """Return ``Mechanism`` rows for a ChEMBL molecule ID.

        Empty molecule ID returns an empty list with no HTTP call. The
        ``/mechanism.json`` endpoint is the canonical drug-action surface (one
        row per molecular target the drug modulates).
        """
        if not molecule_chembl_id:
            return []
        return _get_mechanism_cached(self, molecule_chembl_id)

    def _get_mechanism_uncached(self, molecule_chembl_id: str) -> list["Mechanism"]:
        payload = self._get(
            "/mechanism.json",
            {"molecule_chembl_id": molecule_chembl_id, "limit": 20},
        )
        rows = payload.get("mechanisms") or []
        out: list[Mechanism] = []
        for row in rows:
            moa = row.get("mechanism_of_action")
            if not isinstance(moa, str) or not moa:
                continue
            action = row.get("action_type")
            target = row.get("target_chembl_id")
            out.append(
                Mechanism(
                    mechanism_of_action=moa,
                    action_type=action if isinstance(action, str) and action else None,
                    target_chembl_id=target if isinstance(target, str) and target else None,
                )
            )
        return out

    def mechanism_of_action(self, drug_name: str) -> Optional[str]:
        """Resolve a drug name → its first ChEMBL mechanism-of-action string.

        Convenience wrapper: ``compound_search`` (name → molecule id) then
        ``get_mechanism`` (id → MoA rows), returning the first MoA. Returns
        ``None`` when the name does not resolve or the molecule has no recorded
        mechanism. Empty name skips the network.
        """
        if not drug_name:
            return None
        molecule_id = self.compound_search(drug_name)
        if not molecule_id:
            return None
        mechs = self.get_mechanism(molecule_id)
        return mechs[0].mechanism_of_action if mechs else None
```

Then, in the module-level LRU-wrapper block (after `_get_bioactivity_cached`, ~line 391), add:

```python
@lru_cache(maxsize=_LRU_MAXSIZE)
def _get_mechanism_cached(client: ChEMBLClient, molecule_chembl_id: str) -> list[Mechanism]:
    return client._get_mechanism_uncached(molecule_chembl_id)
```

And extend `reset_caches()` (currently clears 3 caches, ~line 394-398) to also clear the new one:

```python
def reset_caches() -> None:
    """Clear ChEMBL in-process caches (useful in tests)."""
    _compound_search_cached.cache_clear()
    _target_search_cached.cache_clear()
    _get_bioactivity_cached.cache_clear()
    _get_mechanism_cached.cache_clear()
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `python -m pytest tests/unit/test_data/test_kg/test_chembl_mechanism.py tests/unit/test_data/test_kg/test_chembl.py -v`
Expected: PASS (new mechanism tests + the existing ChEMBL tests, which still clear caches fine).

- [ ] **Step 6: Re-export `Mechanism` from the package**

In `src/data/kg/__init__.py`, the ChEMBL import block (~line 21-25) currently imports `Activity, ChEMBLClient, ChEMBLError`. Add `Mechanism`:

```python
from src.data.kg.chembl import (
    Activity,
    ChEMBLClient,
    ChEMBLError,
    Mechanism,
)
```

If `src/data/kg/__init__.py` has an `__all__`, append `"Mechanism"` to it (grep first: `grep -n "__all__" src/data/kg/__init__.py`; if no `__all__`, skip).

- [ ] **Step 7: Commit**

```bash
git add src/data/kg/chembl.py src/data/kg/__init__.py tests/unit/test_data/test_kg/test_chembl_mechanism.py
git commit -m "feat(chembl): add get_mechanism + mechanism_of_action to ChEMBLClient (drug->MoA via /mechanism.json)"
```

---

### Task 2: Brand → drug / disease / fallback map (from the clinical_codes SSOT)

The brand→drug→disease facts already live in `src/ml/synthetic/clinical_codes.py` (`BRAND_DRUG_CLASS` lines 45-49: Remibrutinib="BTK Inhibitor", Fabhalta="Complement Inhibitor", Kisqali="CDK4/6 Inhibitor"; `BRAND_DIAGNOSIS` lines 29-42: desc "Chronic spontaneous urticaria"/"Paroxysmal nocturnal hemoglobinuria"/"Malignant neoplasm of breast"; `BRAND_NDC` lines 58-62: drug_name ribociclib/remibrutinib/iptacopan). Build a single resolver that joins those facts with the enrichment-only extras the SSOT doesn't carry: the precise static MoA-fallback strings the spec pins, the static pivotal-endpoint fallback (used when ClinicalTrials.gov is down/returns only safety endpoints), the PubMed RWE search term, and the map from our synthetic outcome → the real pivotal endpoints.

**Files:**
- Create: `src/services/clinical_context/__init__.py`
- Create: `src/services/clinical_context/brand_map.py`
- Test: `tests/unit/test_services/test_clinical_context/__init__.py` (Create, empty)
- Test: `tests/unit/test_services/test_clinical_context/test_brand_map.py` (Create)

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/test_services/test_clinical_context/test_brand_map.py
"""The brand->drug/disease/fallback map joins the clinical_codes SSOT with the
enrichment-only static fallbacks. Verifies it stays consistent with the SSOT."""

import pytest

from src.services.clinical_context.brand_map import (
    BRAND_CLINICAL_MAP,
    BrandClinicalProfile,
    resolve_brand_profile,
    endpoint_mapping_for_outcome,
)


@pytest.mark.unit
def test_three_brands_resolved_from_ssot():
    assert set(BRAND_CLINICAL_MAP) == {"Kisqali", "Remibrutinib", "Fabhalta"}
    kis = resolve_brand_profile("Kisqali")
    assert isinstance(kis, BrandClinicalProfile)
    assert kis.drug_name == "ribociclib"
    assert kis.disease == "Malignant neoplasm of breast"
    # Static MoA fallback the spec pins (used when ChEMBL is down).
    assert kis.moa_fallback == "CDK4/6 inhibitor"


@pytest.mark.unit
def test_each_brand_carries_static_endpoint_and_rwe_fallbacks():
    fab = resolve_brand_profile("Fabhalta")
    assert fab.drug_name == "iptacopan"
    assert fab.disease == "Paroxysmal nocturnal hemoglobinuria"
    assert fab.moa_fallback == "complement Factor B inhibitor"
    # PNH pivotal endpoints (verified live 2026-06-19 on ClinicalTrials.gov v2).
    assert any("transfusion" in e.lower() for e in fab.pivotal_endpoints_fallback)
    assert any("LDH" in e or "hemoglobin" in e.lower() for e in fab.pivotal_endpoints_fallback)
    # A non-empty PubMed search term so the RWE provider has something to query.
    assert fab.rwe_search_term


@pytest.mark.unit
def test_remibrutinib_btk_csu():
    rem = resolve_brand_profile("Remibrutinib")
    assert rem.drug_name == "remibrutinib"
    assert rem.disease == "Chronic spontaneous urticaria"
    assert rem.moa_fallback == "BTK inhibitor"
    assert any("UAS7" in e for e in rem.pivotal_endpoints_fallback)


@pytest.mark.unit
def test_outcome_to_real_endpoint_mapping_is_brand_aware():
    # Our synthetic 'persistent_180d' maps to a real retention/persistence framing.
    m = endpoint_mapping_for_outcome("Kisqali", "persistent_180d")
    assert m is not None
    assert "persist" in m.lower() or "treatment-free" in m.lower() or "duration" in m.lower()
    # A synthetic outcome with no curated mapping returns None (honest — not faked).
    assert endpoint_mapping_for_outcome("Kisqali", "made_up_outcome") is None


@pytest.mark.unit
def test_unknown_brand_raises_keyerror():
    with pytest.raises(KeyError):
        resolve_brand_profile("NotABrand")
```

- [ ] **Step 2: Run it to confirm it fails**

Run: `python -m pytest tests/unit/test_services/test_clinical_context/test_brand_map.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.services.clinical_context'`.

- [ ] **Step 3: Create the package init + the brand map**

`src/services/clinical_context/__init__.py`:

```python
"""Clinical Context enrichment — a brand-faithful, sourced NARRATIVE layer over
each discovered causal effect (drug + mechanism of action, the disease's real
pivotal endpoints, a real-world-evidence citation).

This package NEVER touches the causal math, adjustment sets, or estimation
frames. It calls the PUBLIC biomedical REST APIs directly (ChEMBL, ClinicalTrials
.gov v2, PubMed E-utilities — the claude.ai MCP tools are agent-only) best-effort,
degrades gracefully to static fallbacks when an API is down/slow, caches per
(brand, disease), and labels every payload: estimate = synthetic cohort;
clinical context = real, cited.

Extensible by design: add a ``ClinicalContextProvider`` subclass to enrich with a
new source (the deferred openFDA / UMLS tasks slot in here) without changing the
service or endpoint.
"""

from src.services.clinical_context.brand_map import (
    BRAND_CLINICAL_MAP,
    BrandClinicalProfile,
    endpoint_mapping_for_outcome,
    resolve_brand_profile,
)
from src.services.clinical_context.providers import (
    ChEMBLMechanismProvider,
    ClinicalContextProvider,
    ClinicalTrialsEndpointProvider,
    PubMedRWEProvider,
)
from src.services.clinical_context.service import ClinicalContextService, reset_caches

__all__ = [
    "BRAND_CLINICAL_MAP",
    "BrandClinicalProfile",
    "ChEMBLMechanismProvider",
    "ClinicalContextProvider",
    "ClinicalContextService",
    "ClinicalTrialsEndpointProvider",
    "PubMedRWEProvider",
    "endpoint_mapping_for_outcome",
    "reset_caches",
    "resolve_brand_profile",
]
```

`src/services/clinical_context/brand_map.py`:

```python
"""Brand -> drug / disease / static-fallback profile for clinical-context
enrichment.

Joins the canonical SSOT in ``src/ml/synthetic/clinical_codes.py`` (drug_name,
drug_class, disease desc) with the enrichment-only extras that SSOT does not
carry: the precise static MoA fallback strings the redesign spec pins (used when
ChEMBL is unreachable), the static pivotal-endpoint fallback (used when
ClinicalTrials.gov is down OR returns only safety endpoints), the PubMed RWE
search term, and the map from OUR synthetic outcome -> the real pivotal endpoint
framing.

All fallback strings are REAL clinical facts (MoA per ChEMBL mechanism rows;
pivotal endpoints verified live 2026-06-19 against ClinicalTrials.gov API v2:
breast cancer = OS/PFS/iDFS; PNH = transfusion-avoidance/LDH/Hb-stabilization;
CSU = UAS7/UCT7/ISS7) — never invented placeholders.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from src.ml.synthetic.clinical_codes import BRAND_DIAGNOSIS, BRAND_NDC


@dataclass(frozen=True)
class BrandClinicalProfile:
    """The clinical facts + static fallbacks for one brand.

    ``moa_fallback`` / ``pivotal_endpoints_fallback`` are used ONLY when the live
    API is unreachable or unhelpful; the providers prefer the live API value and
    fall back to these. ``outcome_endpoint_map`` maps our synthetic outcome
    column -> the real pivotal-endpoint framing it stands in for.
    """

    brand: str
    drug_name: str
    disease: str
    drug_class: str
    moa_fallback: str
    pivotal_endpoints_fallback: List[str]
    rwe_search_term: str
    rwe_seed_pmid: Optional[str]
    outcome_endpoint_map: Dict[str, str] = field(default_factory=dict)


# Enrichment-only static facts keyed by brand. The drug_name / disease / drug_class
# are pulled from the clinical_codes SSOT at construction time (below) so they can
# never drift from it.
_STATIC_ENRICHMENT: Dict[str, Dict[str, object]] = {
    "Kisqali": {
        "moa_fallback": "CDK4/6 inhibitor",
        "pivotal_endpoints_fallback": [
            "Overall Survival (OS)",
            "Progression-Free Survival (PFS)",
            "Invasive Disease-Free Survival (iDFS)",
        ],
        "rwe_search_term": "ribociclib persistence adherence breast cancer real-world",
        "rwe_seed_pmid": "35642282",
        "outcome_endpoint_map": {
            "treatment_initiated": "Treatment initiation / time-to-treatment-start",
            "persistent_180d": "Treatment persistence / duration of therapy (proxy for PFS-supporting adherence)",
            "discontinued_180d": "Treatment discontinuation / early termination",
        },
    },
    "Remibrutinib": {
        "moa_fallback": "BTK inhibitor",
        "pivotal_endpoints_fallback": [
            "Change from baseline in UAS7 (Urticaria Activity Score over 7 days)",
            "UCT7 (Urticaria Control Test)",
            "ISS7 (Itch Severity Score) / WI-NRS",
        ],
        "rwe_search_term": "remibrutinib chronic spontaneous urticaria real-world persistence",
        "rwe_seed_pmid": None,
        "outcome_endpoint_map": {
            "treatment_initiated": "Treatment initiation (BTKi start after antihistamine failure)",
            "persistent_180d": "Treatment persistence / sustained UAS7 control",
            "discontinued_180d": "Treatment discontinuation",
        },
    },
    "Fabhalta": {
        "moa_fallback": "complement Factor B inhibitor",
        "pivotal_endpoints_fallback": [
            "Transfusion avoidance (proportion not requiring RBC transfusion)",
            "Sustained hemoglobin stabilization (increase >= 2 g/dL without transfusion)",
            "Change from baseline in lactate dehydrogenase (LDH)",
        ],
        "rwe_search_term": "iptacopan paroxysmal nocturnal hemoglobinuria real-world persistence",
        "rwe_seed_pmid": None,
        "outcome_endpoint_map": {
            "treatment_initiated": "Treatment initiation (complement-inhibitor start/switch)",
            "persistent_180d": "Treatment persistence / sustained Hb stabilization",
            "discontinued_180d": "Treatment discontinuation",
        },
    },
}


def _build_map() -> Dict[str, BrandClinicalProfile]:
    out: Dict[str, BrandClinicalProfile] = {}
    for brand, extra in _STATIC_ENRICHMENT.items():
        ndc = BRAND_NDC[brand]
        dx = BRAND_DIAGNOSIS[brand]
        out[brand] = BrandClinicalProfile(
            brand=brand,
            drug_name=str(ndc["drug_name"]),
            disease=str(dx["desc"]),
            # drug_class lives in clinical_codes.BRAND_DRUG_CLASS, but the
            # enrichment MoA-fallback string is the precise spec-pinned phrasing;
            # we surface the fallback as the authoritative static MoA.
            drug_class=str(extra["moa_fallback"]),
            moa_fallback=str(extra["moa_fallback"]),
            pivotal_endpoints_fallback=list(extra["pivotal_endpoints_fallback"]),  # type: ignore[arg-type]
            rwe_search_term=str(extra["rwe_search_term"]),
            rwe_seed_pmid=(str(extra["rwe_seed_pmid"]) if extra["rwe_seed_pmid"] else None),
            outcome_endpoint_map=dict(extra["outcome_endpoint_map"]),  # type: ignore[arg-type]
        )
    return out


BRAND_CLINICAL_MAP: Dict[str, BrandClinicalProfile] = _build_map()


def resolve_brand_profile(brand: str) -> BrandClinicalProfile:
    """Return the clinical profile for ``brand``.

    Raises ``KeyError`` on an unsupported brand so callers fail closed rather
    than emit a wrong indication.
    """
    return BRAND_CLINICAL_MAP[brand]


def endpoint_mapping_for_outcome(brand: str, outcome: str) -> Optional[str]:
    """Map our synthetic ``outcome`` column -> the real pivotal-endpoint framing
    for ``brand``. Returns None when there is no curated mapping (honest: we do
    not fabricate a clinical claim for an outcome we have not mapped)."""
    profile = BRAND_CLINICAL_MAP.get(brand)
    if profile is None:
        return None
    return profile.outcome_endpoint_map.get(outcome)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/unit/test_services/test_clinical_context/test_brand_map.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/services/clinical_context/__init__.py src/services/clinical_context/brand_map.py tests/unit/test_services/test_clinical_context/__init__.py tests/unit/test_services/test_clinical_context/test_brand_map.py
git commit -m "feat(clinical-context): brand->drug/disease/fallback map from clinical_codes SSOT"
```

---

### Task 3: ClinicalTrials.gov v2 + PubMed E-utilities REST clients

Two synchronous httpx clients mirroring `src/data/kg/europe_pmc.py` EXACTLY (context-manager, lru_cache, distinct error type, `client=` injectable for MockTransport). Live shapes verified 2026-06-19:
- ClinicalTrials v2 `GET /studies?query.intr=<drug>&query.cond=<disease>&fields=NCTId,PrimaryOutcomeMeasure&pageSize=N&filter.overallStatus=COMPLETED` → `{"studies":[{"protocolSection":{"identificationModule":{"nctId":...},"outcomesModule":{"primaryOutcomes":[{"measure":...}]}}}]}`.
- PubMed esearch `GET /esearch.fcgi?db=pubmed&term=<q>&retmode=json&retmax=N&sort=relevance` → `{"esearchresult":{"idlist":["36097254",...]}}`; esummary `GET /esummary.fcgi?db=pubmed&id=<pmid>&retmode=json` → `{"result":{"<pmid>":{"title":...,"source":...,"pubdate":...,"articleids":[{"idtype":"doi","value":...}]}}}`.

**Files:**
- Create: `src/services/clinical_context/clients.py`
- Test: `tests/unit/test_services/test_clinical_context/test_clients.py` (Create)

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/test_services/test_clinical_context/test_clients.py
"""Unit tests for the ClinicalTrials.gov v2 + PubMed E-utilities REST clients
via httpx.MockTransport (no live HTTP). Pins the response shapes verified live
2026-06-19."""

from __future__ import annotations

from typing import Callable

import httpx
import pytest

from src.services.clinical_context.clients import (
    ClinicalTrialsClient,
    ClinicalTrialsError,
    PubMedArticle,
    PubMedClient,
    PubMedError,
    reset_caches,
)


@pytest.fixture(autouse=True)
def _clear_caches() -> None:
    reset_caches()


def _ctgov(handler: Callable[[httpx.Request], httpx.Response]) -> ClinicalTrialsClient:
    return ClinicalTrialsClient(client=httpx.Client(transport=httpx.MockTransport(handler)))


def _pubmed(handler: Callable[[httpx.Request], httpx.Response]) -> PubMedClient:
    return PubMedClient(client=httpx.Client(transport=httpx.MockTransport(handler)))


def test_clinical_trials_primary_endpoints_dedup_and_order() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert "/studies" in request.url.path
        q = request.url.query.decode("utf-8")
        assert "query.intr" in q and "ribociclib" in q
        return httpx.Response(
            200,
            json={
                "studies": [
                    {
                        "protocolSection": {
                            "identificationModule": {"nctId": "NCT01"},
                            "outcomesModule": {
                                "primaryOutcomes": [
                                    {"measure": "Overall Survival (OS)"},
                                    {"measure": "Progression-Free Survival (PFS)"},
                                ]
                            },
                        }
                    },
                    {
                        "protocolSection": {
                            "identificationModule": {"nctId": "NCT02"},
                            "outcomesModule": {
                                "primaryOutcomes": [{"measure": "Overall Survival (OS)"}]
                            },
                        }
                    },
                ]
            },
        )

    with _ctgov(handler) as client:
        eps = client.primary_endpoints("ribociclib", "breast cancer", limit=5)
    # Deduped, first-seen order preserved.
    assert eps == ["Overall Survival (OS)", "Progression-Free Survival (PFS)"]


def test_clinical_trials_empty_inputs_skip_network() -> None:
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return httpx.Response(200, json={"studies": []})

    with _ctgov(handler) as client:
        assert client.primary_endpoints("", "breast cancer") == []
        assert client.primary_endpoints("ribociclib", "") == []
    assert calls["n"] == 0


def test_clinical_trials_http_error_raises() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, text="boom")

    with _ctgov(handler) as client:
        with pytest.raises(ClinicalTrialsError):
            client.primary_endpoints("ribociclib", "breast cancer")


def test_pubmed_top_article_resolves_title_and_doi() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if "esearch.fcgi" in path:
            assert "ribociclib" in request.url.query.decode("utf-8").lower()
            return httpx.Response(200, json={"esearchresult": {"idlist": ["35642282"]}})
        assert "esummary.fcgi" in path
        assert "35642282" in request.url.query.decode("utf-8")
        return httpx.Response(
            200,
            json={
                "result": {
                    "uids": ["35642282"],
                    "35642282": {
                        "uid": "35642282",
                        "title": "CDK4/6 inhibitor treatment use in women treated for advanced breast cancer.",
                        "source": "J Oncol Pharm Pract",
                        "pubdate": "2023 Jul",
                        "articleids": [
                            {"idtype": "pubmed", "value": "35642282"},
                            {"idtype": "doi", "value": "10.1177/10781552221102884"},
                        ],
                    },
                }
            },
        )

    with _pubmed(handler) as client:
        art = client.top_article("ribociclib persistence adherence")
    assert isinstance(art, PubMedArticle)
    assert art.pmid == "35642282"
    assert "CDK4/6" in art.title
    assert art.journal == "J Oncol Pharm Pract"
    assert art.doi == "10.1177/10781552221102884"
    assert art.url == "https://pubmed.ncbi.nlm.nih.gov/35642282/"


def test_pubmed_no_hits_returns_none() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert "esearch.fcgi" in request.url.path
        return httpx.Response(200, json={"esearchresult": {"idlist": []}})

    with _pubmed(handler) as client:
        assert client.top_article("no-such-topic-xyz") is None


def test_pubmed_fetch_by_pmid_resolves_seed() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert "esummary.fcgi" in request.url.path
        return httpx.Response(
            200,
            json={
                "result": {
                    "uids": ["35642282"],
                    "35642282": {
                        "uid": "35642282",
                        "title": "Seed title",
                        "source": "J Oncol Pharm Pract",
                        "pubdate": "2023 Jul",
                        "articleids": [{"idtype": "doi", "value": "10.1/x"}],
                    },
                }
            },
        )

    with _pubmed(handler) as client:
        art = client.fetch_by_pmid("35642282")
    assert art is not None and art.pmid == "35642282" and art.title == "Seed title"


def test_pubmed_http_error_raises() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(503, text="down")

    with _pubmed(handler) as client:
        with pytest.raises(PubMedError):
            client.top_article("ribociclib")
```

- [ ] **Step 2: Run it to confirm it fails**

Run: `python -m pytest tests/unit/test_services/test_clinical_context/test_clients.py -v`
Expected: FAIL — `ImportError: cannot import name 'ClinicalTrialsClient' ...`.

- [ ] **Step 3: Implement the clients**

`src/services/clinical_context/clients.py`:

```python
"""Public biomedical REST clients for clinical-context enrichment.

Synchronous httpx clients mirroring ``src/data/kg/europe_pmc.py`` /
``src/data/kg/chembl.py`` exactly (context-manager, in-process ``lru_cache``,
distinct error type, ``client=`` injectable for ``httpx.MockTransport`` in
tests). They call the PUBLIC REST APIs directly — the claude.ai MCP tools are
agent-only and unavailable to the FastAPI backend.

  - ClinicalTrials.gov API v2 (https://clinicaltrials.gov/api/v2): study search
    -> primary outcome measures (the disease's pivotal endpoints).
  - PubMed E-utilities (https://eutils.ncbi.nlm.nih.gov/entrez/eutils): esearch
    -> top PMID; esummary -> title/journal/DOI (a real-world-evidence citation).

Both surface a SINGLE error class per source; transport / HTTP / JSON failures
all raise it so the service layer can degrade per-provider on one except clause.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, List, Optional

import httpx

logger = logging.getLogger(__name__)

CLINICAL_TRIALS_BASE = "https://clinicaltrials.gov/api/v2"
PUBMED_BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
# Short default timeout: enrichment is best-effort and must not hold the request
# open. The service layer treats a timeout as "degrade to static fallback".
DEFAULT_TIMEOUT = 8.0
_LRU_MAXSIZE = 2048


class ClinicalTrialsError(Exception):
    """ClinicalTrials.gov request failed (transport, HTTP, or JSON)."""


class PubMedError(Exception):
    """PubMed E-utilities request failed (transport, HTTP, or JSON)."""


@dataclass(frozen=True)
class PubMedArticle:
    """One PubMed article summary (a real-world-evidence citation)."""

    pmid: str
    title: str
    journal: Optional[str] = None
    pubdate: Optional[str] = None
    doi: Optional[str] = None

    @property
    def url(self) -> str:
        return f"https://pubmed.ncbi.nlm.nih.gov/{self.pmid}/"


class ClinicalTrialsClient:
    """Synchronous ClinicalTrials.gov API v2 client."""

    def __init__(
        self,
        *,
        base: str = CLINICAL_TRIALS_BASE,
        timeout: float = DEFAULT_TIMEOUT,
        client: Optional[httpx.Client] = None,
    ) -> None:
        self._base = base.rstrip("/")
        self._client = client if client is not None else httpx.Client(timeout=timeout)
        self._owns_client = client is None

    def __enter__(self) -> "ClinicalTrialsClient":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def close(self) -> None:
        if self._owns_client:
            self._client.close()

    def primary_endpoints(self, intervention: str, condition: str, *, limit: int = 8) -> List[str]:
        """Return the distinct primary outcome measures across COMPLETED studies
        of ``intervention`` for ``condition`` (the disease's pivotal endpoints),
        first-seen order preserved. Empty intervention/condition skip the network."""
        if not intervention or not condition:
            return []
        return list(_ctgov_primary_endpoints_cached(self, intervention, condition, limit))

    def _primary_endpoints_uncached(
        self, intervention: str, condition: str, limit: int
    ) -> tuple[str, ...]:
        try:
            response = self._client.get(
                f"{self._base}/studies",
                params={
                    "query.intr": intervention,
                    "query.cond": condition,
                    "fields": "NCTId,PrimaryOutcomeMeasure",
                    "pageSize": limit,
                    "filter.overallStatus": "COMPLETED",
                },
                headers={"Accept": "application/json"},
            )
        except httpx.HTTPError as exc:
            raise ClinicalTrialsError(f"ClinicalTrials transport error: {exc}") from exc
        if response.status_code >= 400:
            raise ClinicalTrialsError(
                f"ClinicalTrials HTTP {response.status_code}: {response.text[:200]!r}"
            )
        try:
            payload: dict[str, Any] = response.json()
        except ValueError as exc:
            raise ClinicalTrialsError(
                f"ClinicalTrials non-JSON body: {response.text[:200]!r}"
            ) from exc
        seen: list[str] = []
        for study in payload.get("studies") or []:
            outcomes = (
                study.get("protocolSection", {}).get("outcomesModule", {}).get("primaryOutcomes")
                or []
            )
            for outcome in outcomes:
                measure = outcome.get("measure") if isinstance(outcome, dict) else None
                if isinstance(measure, str) and measure and measure not in seen:
                    seen.append(measure)
        return tuple(seen)


class PubMedClient:
    """Synchronous PubMed E-utilities client (esearch + esummary)."""

    def __init__(
        self,
        *,
        base: str = PUBMED_BASE,
        timeout: float = DEFAULT_TIMEOUT,
        client: Optional[httpx.Client] = None,
    ) -> None:
        self._base = base.rstrip("/")
        self._client = client if client is not None else httpx.Client(timeout=timeout)
        self._owns_client = client is None

    def __enter__(self) -> "PubMedClient":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def close(self) -> None:
        if self._owns_client:
            self._client.close()

    def top_article(self, term: str) -> Optional[PubMedArticle]:
        """esearch for ``term`` (relevance-sorted) -> the top PMID -> esummary.
        Returns None when there are no hits. Empty term skips the network."""
        if not term:
            return None
        return _pubmed_top_article_cached(self, term)

    def fetch_by_pmid(self, pmid: str) -> Optional[PubMedArticle]:
        """esummary for a specific PMID -> the article summary. Empty pmid skips
        the network; an unknown pmid returns None."""
        if not pmid:
            return None
        return _pubmed_fetch_by_pmid_cached(self, pmid)

    def _esearch_top_pmid(self, term: str) -> Optional[str]:
        try:
            response = self._client.get(
                f"{self._base}/esearch.fcgi",
                params={
                    "db": "pubmed",
                    "term": term,
                    "retmode": "json",
                    "retmax": 1,
                    "sort": "relevance",
                },
            )
        except httpx.HTTPError as exc:
            raise PubMedError(f"PubMed esearch transport error: {exc}") from exc
        if response.status_code >= 400:
            raise PubMedError(f"PubMed esearch HTTP {response.status_code}: {response.text[:200]!r}")
        try:
            payload: dict[str, Any] = response.json()
        except ValueError as exc:
            raise PubMedError(f"PubMed esearch non-JSON body: {response.text[:200]!r}") from exc
        idlist = payload.get("esearchresult", {}).get("idlist") or []
        return str(idlist[0]) if idlist else None

    def _esummary(self, pmid: str) -> Optional[PubMedArticle]:
        try:
            response = self._client.get(
                f"{self._base}/esummary.fcgi",
                params={"db": "pubmed", "id": pmid, "retmode": "json"},
            )
        except httpx.HTTPError as exc:
            raise PubMedError(f"PubMed esummary transport error: {exc}") from exc
        if response.status_code >= 400:
            raise PubMedError(
                f"PubMed esummary HTTP {response.status_code}: {response.text[:200]!r}"
            )
        try:
            payload: dict[str, Any] = response.json()
        except ValueError as exc:
            raise PubMedError(f"PubMed esummary non-JSON body: {response.text[:200]!r}") from exc
        record = payload.get("result", {}).get(str(pmid))
        if not isinstance(record, dict):
            return None
        title = record.get("title")
        if not isinstance(title, str) or not title:
            return None
        doi: Optional[str] = None
        for aid in record.get("articleids") or []:
            if isinstance(aid, dict) and aid.get("idtype") == "doi":
                value = aid.get("value")
                if isinstance(value, str) and value:
                    doi = value
                    break
        journal = record.get("source")
        pubdate = record.get("pubdate")
        return PubMedArticle(
            pmid=str(pmid),
            title=title,
            journal=journal if isinstance(journal, str) and journal else None,
            pubdate=pubdate if isinstance(pubdate, str) and pubdate else None,
            doi=doi,
        )

    def _top_article_uncached(self, term: str) -> Optional[PubMedArticle]:
        pmid = self._esearch_top_pmid(term)
        if not pmid:
            return None
        return self._esummary(pmid)


# ---------------------------------------------------------------------------
# Module-level LRU wrappers — keyed on (id(client), args), client-lifetime scope
# (same pattern as europe_pmc.py / chembl.py).
# ---------------------------------------------------------------------------


@lru_cache(maxsize=_LRU_MAXSIZE)
def _ctgov_primary_endpoints_cached(
    client: ClinicalTrialsClient, intervention: str, condition: str, limit: int
) -> tuple[str, ...]:
    return client._primary_endpoints_uncached(intervention, condition, limit)


@lru_cache(maxsize=_LRU_MAXSIZE)
def _pubmed_top_article_cached(client: PubMedClient, term: str) -> Optional[PubMedArticle]:
    return client._top_article_uncached(term)


@lru_cache(maxsize=_LRU_MAXSIZE)
def _pubmed_fetch_by_pmid_cached(client: PubMedClient, pmid: str) -> Optional[PubMedArticle]:
    return client._esummary(pmid)


def reset_caches() -> None:
    """Clear the in-process clinical-trials / pubmed caches (useful in tests)."""
    _ctgov_primary_endpoints_cached.cache_clear()
    _pubmed_top_article_cached.cache_clear()
    _pubmed_fetch_by_pmid_cached.cache_clear()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/unit/test_services/test_clinical_context/test_clients.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/services/clinical_context/clients.py tests/unit/test_services/test_clinical_context/test_clients.py
git commit -m "feat(clinical-context): ClinicalTrials.gov v2 + PubMed E-utilities REST clients"
```

---

### Task 4: Provider interface + the three providers (best-effort + static fallback)

The extensible core: a `ClinicalContextProvider` ABC with `provider_name` + `enrich(profile)` returning a typed fragment, and three concrete providers. Each is best-effort: a live API failure or empty result falls back to the brand_map static value and records `source="static_fallback"` vs `source="chembl"/"clinicaltrials.gov"/"pubmed"`. This is where the deferred openFDA/UMLS work will slot in (a new subclass) — NOT implemented here.

**Files:**
- Create: `src/services/clinical_context/providers.py`
- Test: `tests/unit/test_services/test_clinical_context/test_providers.py` (Create)

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/test_services/test_clinical_context/test_providers.py
"""Each provider is best-effort: live value preferred, static fallback when the
API is down/empty, with an honest source label. No live HTTP (clients injected)."""

from __future__ import annotations

import pytest

from src.services.clinical_context.brand_map import resolve_brand_profile
from src.services.clinical_context.providers import (
    ChEMBLMechanismProvider,
    ClinicalTrialsEndpointProvider,
    PubMedRWEProvider,
)


class _FakeChEMBL:
    def __init__(self, moa):
        self._moa = moa

    def mechanism_of_action(self, drug_name):  # noqa: D401
        if isinstance(self._moa, Exception):
            raise self._moa
        return self._moa


class _FakeCTGov:
    def __init__(self, eps):
        self._eps = eps

    def primary_endpoints(self, intervention, condition, *, limit=8):
        if isinstance(self._eps, Exception):
            raise self._eps
        return list(self._eps)


class _FakePubMed:
    def __init__(self, art=None, by_pmid=None):
        self._art = art
        self._by_pmid = by_pmid

    def top_article(self, term):
        if isinstance(self._art, Exception):
            raise self._art
        return self._art

    def fetch_by_pmid(self, pmid):
        return self._by_pmid


@pytest.mark.unit
def test_chembl_provider_prefers_live_moa():
    profile = resolve_brand_profile("Kisqali")
    frag = ChEMBLMechanismProvider(client=_FakeChEMBL("Cyclin-dependent kinase 4 inhibitor")).enrich(
        profile
    )
    assert frag.mechanism_of_action == "Cyclin-dependent kinase 4 inhibitor"
    assert frag.source == "chembl"


@pytest.mark.unit
def test_chembl_provider_falls_back_on_error():
    from src.data.kg.chembl import ChEMBLError

    profile = resolve_brand_profile("Kisqali")
    frag = ChEMBLMechanismProvider(client=_FakeChEMBL(ChEMBLError("boom"))).enrich(profile)
    # Static spec-pinned fallback, honestly labelled.
    assert frag.mechanism_of_action == "CDK4/6 inhibitor"
    assert frag.source == "static_fallback"


@pytest.mark.unit
def test_ctgov_provider_prefers_live_then_falls_back():
    profile = resolve_brand_profile("Fabhalta")
    live = ClinicalTrialsEndpointProvider(
        client=_FakeCTGov(["Transfusion avoidance", "LDH normalization"])
    ).enrich(profile)
    assert live.endpoints == ["Transfusion avoidance", "LDH normalization"]
    assert live.source == "clinicaltrials.gov"

    from src.services.clinical_context.clients import ClinicalTrialsError

    down = ClinicalTrialsEndpointProvider(client=_FakeCTGov(ClinicalTrialsError("503"))).enrich(
        profile
    )
    assert down.endpoints == profile.pivotal_endpoints_fallback
    assert down.source == "static_fallback"


@pytest.mark.unit
def test_ctgov_provider_empty_live_uses_fallback():
    profile = resolve_brand_profile("Remibrutinib")
    frag = ClinicalTrialsEndpointProvider(client=_FakeCTGov([])).enrich(profile)
    assert frag.endpoints == profile.pivotal_endpoints_fallback
    assert frag.source == "static_fallback"


@pytest.mark.unit
def test_pubmed_provider_prefers_search_then_seed_pmid():
    from src.services.clinical_context.clients import PubMedArticle

    profile = resolve_brand_profile("Kisqali")
    art = PubMedArticle(pmid="36097254", title="Live hit", journal="J", doi="10.1/y")
    frag = PubMedRWEProvider(client=_FakePubMed(art=art)).enrich(profile)
    assert frag.citation is not None
    assert frag.citation.pmid == "36097254"
    assert frag.source == "pubmed"


@pytest.mark.unit
def test_pubmed_provider_falls_back_to_seed_pmid_on_no_hits():
    from src.services.clinical_context.clients import PubMedArticle

    profile = resolve_brand_profile("Kisqali")  # rwe_seed_pmid = 35642282
    seed = PubMedArticle(pmid="35642282", title="Seed RWE", journal="J Oncol Pharm Pract")
    frag = PubMedRWEProvider(client=_FakePubMed(art=None, by_pmid=seed)).enrich(profile)
    assert frag.citation is not None and frag.citation.pmid == "35642282"
    assert frag.source == "pubmed_seed"


@pytest.mark.unit
def test_pubmed_provider_none_when_no_hit_and_no_seed():
    profile = resolve_brand_profile("Fabhalta")  # rwe_seed_pmid = None
    frag = PubMedRWEProvider(client=_FakePubMed(art=None, by_pmid=None)).enrich(profile)
    assert frag.citation is None
    assert frag.source == "unavailable"
```

- [ ] **Step 2: Run it to confirm it fails**

Run: `python -m pytest tests/unit/test_services/test_clinical_context/test_providers.py -v`
Expected: FAIL — `ImportError: cannot import name 'ChEMBLMechanismProvider' ...`.

- [ ] **Step 3: Implement the providers**

`src/services/clinical_context/providers.py`:

```python
"""Clinical-context providers — the extensible enrichment core.

A ``ClinicalContextProvider`` enriches a ``BrandClinicalProfile`` from ONE
source, best-effort, with a static fallback and an honest source label. The
service fans out across providers and assembles the payload.

Adding a source (the DEFERRED openFDA / UMLS work) = add a subclass here; the
service and endpoint need no change. NO openFDA/UMLS code lives here yet.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Optional, Protocol

from src.services.clinical_context.brand_map import BrandClinicalProfile
from src.services.clinical_context.clients import (
    ClinicalTrialsClient,
    ClinicalTrialsError,
    PubMedArticle,
    PubMedClient,
    PubMedError,
)

logger = logging.getLogger(__name__)


# --- Typed fragments each provider returns -------------------------------------


@dataclass(frozen=True)
class MechanismFragment:
    mechanism_of_action: str
    source: str  # "chembl" | "static_fallback"


@dataclass(frozen=True)
class EndpointsFragment:
    endpoints: List[str] = field(default_factory=list)
    source: str = "static_fallback"  # "clinicaltrials.gov" | "static_fallback"


@dataclass(frozen=True)
class CitationFragment:
    citation: Optional[PubMedArticle]
    source: str  # "pubmed" | "pubmed_seed" | "unavailable"


# --- Minimal structural protocols so tests can inject fakes --------------------


class _ChEMBLLike(Protocol):
    def mechanism_of_action(self, drug_name: str) -> Optional[str]: ...


class _CTGovLike(Protocol):
    def primary_endpoints(
        self, intervention: str, condition: str, *, limit: int = 8
    ) -> List[str]: ...


class _PubMedLike(Protocol):
    def top_article(self, term: str) -> Optional[PubMedArticle]: ...

    def fetch_by_pmid(self, pmid: str) -> Optional[PubMedArticle]: ...


# --- The provider interface ----------------------------------------------------


class ClinicalContextProvider(ABC):
    """Enriches a brand's clinical profile from one source, best-effort."""

    provider_name: str = "provider"

    @abstractmethod
    def enrich(self, profile: BrandClinicalProfile) -> object:
        """Return this provider's typed fragment. MUST NOT raise on an API
        failure — degrade to the static fallback and label the source."""
        raise NotImplementedError


class ChEMBLMechanismProvider(ClinicalContextProvider):
    """Drug -> mechanism of action via ChEMBL, with the static MoA fallback."""

    provider_name = "chembl_mechanism"

    def __init__(self, client: _ChEMBLLike) -> None:
        self._client = client

    def enrich(self, profile: BrandClinicalProfile) -> MechanismFragment:
        try:
            moa = self._client.mechanism_of_action(profile.drug_name)
        except Exception as exc:  # noqa: BLE001 — best-effort; any failure => fallback
            logger.warning("clinical-context: ChEMBL MoA lookup failed for %s: %s", profile.drug_name, exc)
            moa = None
        if moa:
            return MechanismFragment(mechanism_of_action=moa, source="chembl")
        return MechanismFragment(mechanism_of_action=profile.moa_fallback, source="static_fallback")


class ClinicalTrialsEndpointProvider(ClinicalContextProvider):
    """Disease -> real pivotal endpoints via ClinicalTrials.gov, with the static
    endpoint fallback (also used when the live API returns only safety endpoints
    => the caller can prefer the curated efficacy fallback)."""

    provider_name = "clinicaltrials_endpoints"

    def __init__(self, client: _CTGovLike) -> None:
        self._client = client

    def enrich(self, profile: BrandClinicalProfile) -> EndpointsFragment:
        try:
            endpoints = self._client.primary_endpoints(profile.drug_name, profile.disease)
        except Exception as exc:  # noqa: BLE001 — best-effort; any failure => fallback
            logger.warning(
                "clinical-context: ClinicalTrials lookup failed for %s/%s: %s",
                profile.drug_name,
                profile.disease,
                exc,
            )
            endpoints = []
        if endpoints:
            return EndpointsFragment(endpoints=endpoints, source="clinicaltrials.gov")
        return EndpointsFragment(
            endpoints=list(profile.pivotal_endpoints_fallback), source="static_fallback"
        )


class PubMedRWEProvider(ClinicalContextProvider):
    """Real-world-evidence citation via PubMed: relevance search, then the
    curated seed PMID, then unavailable (honest — never a fabricated citation)."""

    provider_name = "pubmed_rwe"

    def __init__(self, client: _PubMedLike) -> None:
        self._client = client

    def enrich(self, profile: BrandClinicalProfile) -> CitationFragment:
        try:
            article = self._client.top_article(profile.rwe_search_term)
        except Exception as exc:  # noqa: BLE001 — best-effort; any failure => try seed
            logger.warning(
                "clinical-context: PubMed search failed for %r: %s", profile.rwe_search_term, exc
            )
            article = None
        if article is not None:
            return CitationFragment(citation=article, source="pubmed")
        if profile.rwe_seed_pmid:
            try:
                seed = self._client.fetch_by_pmid(profile.rwe_seed_pmid)
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "clinical-context: PubMed seed fetch failed for %s: %s",
                    profile.rwe_seed_pmid,
                    exc,
                )
                seed = None
            if seed is not None:
                return CitationFragment(citation=seed, source="pubmed_seed")
        return CitationFragment(citation=None, source="unavailable")
```

(`ClinicalTrialsError`/`PubMedError` are imported so the module documents which exceptions the broad `except Exception` deliberately also swallows; they are not re-raised — best-effort is the contract. A `# noqa: F401` is not needed because they are referenced in the import for clarity; if ruff flags them unused, change the import to a comment listing them and drop the names.)

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/unit/test_services/test_clinical_context/test_providers.py -v`
Expected: PASS.

- [ ] **Step 5: Resolve any unused-import lint, then commit**

Run: `ruff check src/services/clinical_context/providers.py`
If ruff reports `ClinicalTrialsError`/`PubMedError` imported-but-unused, replace the two names in the `from src.services.clinical_context.clients import (...)` block with a trailing comment and drop them from the import list:

```python
from src.services.clinical_context.clients import (
    ClinicalTrialsClient,
    PubMedArticle,
    PubMedClient,
)
# Best-effort contract: the providers' broad `except Exception` deliberately also
# swallows ClinicalTrialsError / PubMedError from clients.py (degrade to fallback).
```

Then:

```bash
git add src/services/clinical_context/providers.py tests/unit/test_services/test_clinical_context/test_providers.py
git commit -m "feat(clinical-context): provider interface + ChEMBL/ClinicalTrials/PubMed providers (best-effort+fallback)"
```

---

### Task 5: `ClinicalContextService` orchestrator (fan-out, cache, degrade, honesty label)

The service assembles a `ClinicalContext` dict from the three providers, mapping our synthetic outcome → the real endpoint framing, caching per `(brand, disease)` (the live lookups don't vary by outcome — only the outcome→endpoint mapping does, which is local), and always attaching the honesty label. It builds default real REST clients but accepts injected providers for tests.

**Files:**
- Create: `src/services/clinical_context/service.py`
- Test: `tests/unit/test_services/test_clinical_context/test_service.py` (Create)

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/test_services/test_clinical_context/test_service.py
"""The service fans out across providers, maps the synthetic outcome to the real
endpoint, caches per (brand,disease), degrades gracefully, and always labels the
synthetic/real boundary."""

from __future__ import annotations

import pytest

from src.services.clinical_context.clients import PubMedArticle
from src.services.clinical_context.providers import (
    CitationFragment,
    EndpointsFragment,
    MechanismFragment,
)
from src.services.clinical_context.service import ClinicalContextService, reset_caches


class _StubProvider:
    def __init__(self, fragment, counter=None, name="stub"):
        self._fragment = fragment
        self._counter = counter
        self.provider_name = name

    def enrich(self, profile):
        if self._counter is not None:
            self._counter["n"] += 1
        return self._fragment


@pytest.fixture(autouse=True)
def _clear() -> None:
    reset_caches()


def _service(moa_frag, ep_frag, cite_frag, counters=None):
    counters = counters or {}
    return ClinicalContextService(
        mechanism_provider=_StubProvider(moa_frag, counters.get("moa")),
        endpoints_provider=_StubProvider(ep_frag, counters.get("ep")),
        citation_provider=_StubProvider(cite_frag, counters.get("cite")),
    )


def test_get_context_assembles_all_three_sources():
    art = PubMedArticle(pmid="35642282", title="RWE", journal="J", doi="10.1/x")
    svc = _service(
        MechanismFragment("CDK4/6 inhibitor", "chembl"),
        EndpointsFragment(["Overall Survival (OS)", "PFS"], "clinicaltrials.gov"),
        CitationFragment(art, "pubmed"),
    )
    ctx = svc.get_context("Kisqali", "persistent_180d")
    assert ctx["brand"] == "Kisqali"
    assert ctx["drug_name"] == "ribociclib"
    assert ctx["disease"] == "Malignant neoplasm of breast"
    assert ctx["mechanism"]["mechanism_of_action"] == "CDK4/6 inhibitor"
    assert ctx["mechanism"]["source"] == "chembl"
    assert ctx["pivotal_endpoints"]["endpoints"][0] == "Overall Survival (OS)"
    assert ctx["pivotal_endpoints"]["source"] == "clinicaltrials.gov"
    # The synthetic outcome is mapped to the real endpoint framing.
    assert "persist" in ctx["mapped_endpoint"].lower()
    assert ctx["our_outcome"] == "persistent_180d"
    # The real-world-evidence citation round-trips.
    assert ctx["real_world_evidence"]["pmid"] == "35642282"
    assert ctx["real_world_evidence"]["url"] == "https://pubmed.ncbi.nlm.nih.gov/35642282/"
    # The honesty label is ALWAYS present and names the boundary.
    assert "synthetic" in ctx["honesty_label"].lower()
    assert "real" in ctx["honesty_label"].lower()


def test_degrades_when_all_providers_fall_back():
    svc = _service(
        MechanismFragment("complement Factor B inhibitor", "static_fallback"),
        EndpointsFragment(["Transfusion avoidance"], "static_fallback"),
        CitationFragment(None, "unavailable"),
    )
    ctx = svc.get_context("Fabhalta", "treatment_initiated")
    assert ctx["mechanism"]["source"] == "static_fallback"
    assert ctx["pivotal_endpoints"]["source"] == "static_fallback"
    assert ctx["real_world_evidence"] is None
    assert ctx["honesty_label"]  # still present


def test_cache_is_per_brand_disease_not_per_outcome():
    counters = {"moa": {"n": 0}, "ep": {"n": 0}, "cite": {"n": 0}}
    svc = _service(
        MechanismFragment("CDK4/6 inhibitor", "chembl"),
        EndpointsFragment(["OS"], "clinicaltrials.gov"),
        CitationFragment(None, "unavailable"),
        counters,
    )
    a = svc.get_context("Kisqali", "persistent_180d")
    b = svc.get_context("Kisqali", "treatment_initiated")  # same brand, diff outcome
    # The expensive provider fan-out ran ONCE (cached per brand/disease)...
    assert counters["moa"]["n"] == 1
    assert counters["ep"]["n"] == 1
    # ...but the outcome->endpoint mapping differs per call.
    assert a["mapped_endpoint"] != b["mapped_endpoint"]


def test_unknown_brand_raises_keyerror():
    svc = _service(
        MechanismFragment("x", "static_fallback"),
        EndpointsFragment([], "static_fallback"),
        CitationFragment(None, "unavailable"),
    )
    with pytest.raises(KeyError):
        svc.get_context("NotABrand", "persistent_180d")
```

- [ ] **Step 2: Run it to confirm it fails**

Run: `python -m pytest tests/unit/test_services/test_clinical_context/test_service.py -v`
Expected: FAIL — `ImportError: cannot import name 'ClinicalContextService' ...`.

- [ ] **Step 3: Implement the service**

`src/services/clinical_context/service.py`:

```python
"""ClinicalContextService — fan out the providers into one payload.

Caches the live provider fan-out per (brand, disease) (the live lookups do not
vary by outcome); the outcome -> real-endpoint mapping is applied per call from
the local brand_map. Always attaches the synthetic/real honesty label. Builds
default real REST clients; injectable for tests.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional, Tuple

from src.services.clinical_context.brand_map import (
    BrandClinicalProfile,
    endpoint_mapping_for_outcome,
    resolve_brand_profile,
)
from src.services.clinical_context.clients import (
    ClinicalTrialsClient,
    PubMedClient,
)
from src.services.clinical_context.providers import (
    ChEMBLMechanismProvider,
    CitationFragment,
    ClinicalContextProvider,
    ClinicalTrialsEndpointProvider,
    EndpointsFragment,
    MechanismFragment,
    PubMedRWEProvider,
)

logger = logging.getLogger(__name__)

HONESTY_LABEL = (
    "Effect estimate = a SYNTHETIC patient cohort (gold-standard demo data). "
    "Clinical context below (mechanism of action, pivotal endpoints, real-world "
    "evidence) is REAL and cited from public biomedical sources."
)

# Per-(brand,disease) cache of the assembled live fragments. Keyed by a tuple so
# two outcomes for one brand reuse the single fan-out. Bounded by the 3-brand
# universe; a plain dict is sufficient (no eviction needed).
_FRAGMENT_CACHE: Dict[Tuple[str, str], Tuple[MechanismFragment, EndpointsFragment, CitationFragment]] = {}


class ClinicalContextService:
    """Assemble a brand's clinical context from the providers."""

    def __init__(
        self,
        *,
        mechanism_provider: Optional[ClinicalContextProvider] = None,
        endpoints_provider: Optional[ClinicalContextProvider] = None,
        citation_provider: Optional[ClinicalContextProvider] = None,
    ) -> None:
        # Default real providers wire the public-REST clients; tests inject stubs.
        self._mechanism = mechanism_provider or ChEMBLMechanismProvider(client=_default_chembl())
        self._endpoints = endpoints_provider or ClinicalTrialsEndpointProvider(
            client=ClinicalTrialsClient()
        )
        self._citation = citation_provider or PubMedRWEProvider(client=PubMedClient())

    def _fan_out(
        self, profile: BrandClinicalProfile
    ) -> Tuple[MechanismFragment, EndpointsFragment, CitationFragment]:
        key = (profile.brand, profile.disease)
        cached = _FRAGMENT_CACHE.get(key)
        if cached is not None:
            return cached
        moa = self._mechanism.enrich(profile)
        eps = self._endpoints.enrich(profile)
        cite = self._citation.enrich(profile)
        assert isinstance(moa, MechanismFragment)
        assert isinstance(eps, EndpointsFragment)
        assert isinstance(cite, CitationFragment)
        _FRAGMENT_CACHE[key] = (moa, eps, cite)
        return moa, eps, cite

    def get_context(self, brand: str, outcome: str) -> Dict[str, Any]:
        """Return the assembled clinical-context payload for (brand, outcome).

        Raises ``KeyError`` on an unknown brand (the endpoint maps it to 404).
        Never raises on an API failure — providers degrade to static fallbacks.
        """
        profile = resolve_brand_profile(brand)
        moa, eps, cite = self._fan_out(profile)
        citation_payload: Optional[Dict[str, Any]] = None
        if cite.citation is not None:
            citation_payload = {
                "pmid": cite.citation.pmid,
                "title": cite.citation.title,
                "journal": cite.citation.journal,
                "pubdate": cite.citation.pubdate,
                "doi": cite.citation.doi,
                "url": cite.citation.url,
                "source": cite.source,
            }
        return {
            "brand": profile.brand,
            "drug_name": profile.drug_name,
            "disease": profile.disease,
            "our_outcome": outcome,
            "mapped_endpoint": endpoint_mapping_for_outcome(brand, outcome),
            "mechanism": {
                "mechanism_of_action": moa.mechanism_of_action,
                "source": moa.source,
            },
            "pivotal_endpoints": {
                "endpoints": list(eps.endpoints),
                "source": eps.source,
            },
            "real_world_evidence": citation_payload,
            "honesty_label": HONESTY_LABEL,
        }


def _default_chembl() -> ChEMBLMechanismProvider:
    """Build the default real ChEMBL provider (lazy import of the kg client keeps
    the import graph cheap and avoids a hard dependency at module import)."""
    from src.data.kg.chembl import ChEMBLClient

    return ChEMBLMechanismProvider(client=ChEMBLClient())


def reset_caches() -> None:
    """Clear the per-(brand,disease) fragment cache + the underlying REST client
    caches (useful in tests)."""
    _FRAGMENT_CACHE.clear()
    from src.data.kg.chembl import reset_caches as chembl_reset
    from src.services.clinical_context.clients import reset_caches as clients_reset

    chembl_reset()
    clients_reset()
```

Note the `_default_chembl()` helper wraps the provider (not the raw client) so the `mechanism_provider or ...` default is a `ClinicalContextProvider` as typed; fix the constructor default to call it directly:

In `__init__`, the mechanism default is `mechanism_provider or _default_chembl()` (already returns a provider). Adjust the line accordingly — it currently reads `ChEMBLMechanismProvider(client=_default_chembl())` which is WRONG (double-wrap). Use:

```python
        self._mechanism = mechanism_provider or _default_chembl()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/unit/test_services/test_clinical_context/test_service.py -v`
Expected: PASS.

- [ ] **Step 5: Run the whole service package + scoped mypy**

Run: `python -m pytest tests/unit/test_services/test_clinical_context/ tests/unit/test_data/test_kg/test_chembl_mechanism.py -v`
Expected: PASS (all clinical-context unit tests).
Run: `mypy src/services/clinical_context/ src/data/kg/chembl.py`
Expected: clean (scoped — do NOT run whole-tree mypy on the droplet).

- [ ] **Step 6: Commit**

```bash
git add src/services/clinical_context/service.py tests/unit/test_services/test_clinical_context/test_service.py
git commit -m "feat(clinical-context): ClinicalContextService orchestrator (fan-out, per-brand cache, honesty label)"
```

---

### Task 6: `ClinicalContext` Pydantic models + `GET /causal/clinical-context` endpoint

Add the response schema and a thin endpoint. The endpoint validates the brand against the live cohort brands (reusing the existing `_list_dataset_brands`) and returns 404 on an unknown brand; it never 500s on a degraded API (the service degrades). Reuses `require_viewer` (read-only, like `get_discover_causal_effects`).

**Files:**
- Modify: `src/api/schemas/causal.py` (add models after `DiscoverEffectsResponse`, ~line 671)
- Modify: `src/api/routes/causal.py` (module-level service instance near `_discover_effects_store` ~line 1067; endpoint after `get_discover_causal_effects` ~line 1301)
- Test: `tests/unit/test_api/test_causal_clinical_context.py` (Create)

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/test_api/test_causal_clinical_context.py
"""Contract tests for GET /causal/clinical-context: assembles real clinical
context per brand, maps our synthetic outcome, 404s an unknown brand, and stays
200 (degraded) when an upstream API is down. The ClinicalContextService is
patched so no live HTTP runs."""

from __future__ import annotations

import pytest
from unittest.mock import patch

from src.api.routes import causal as causal_routes
from src.api.schemas.causal import ClinicalContext


@pytest.mark.unit
def test_clinical_context_model_round_trips():
    ctx = ClinicalContext(
        brand="Kisqali",
        drug_name="ribociclib",
        disease="Malignant neoplasm of breast",
        our_outcome="persistent_180d",
        mapped_endpoint="Treatment persistence / duration of therapy",
        mechanism={"mechanism_of_action": "CDK4/6 inhibitor", "source": "chembl"},
        pivotal_endpoints={"endpoints": ["Overall Survival (OS)"], "source": "clinicaltrials.gov"},
        real_world_evidence={
            "pmid": "35642282",
            "title": "RWE",
            "journal": "J",
            "pubdate": "2023 Jul",
            "doi": "10.1/x",
            "url": "https://pubmed.ncbi.nlm.nih.gov/35642282/",
            "source": "pubmed",
        },
        honesty_label="estimate = synthetic; context = real, cited",
    )
    assert ctx.mechanism.mechanism_of_action == "CDK4/6 inhibitor"
    assert ctx.real_world_evidence is not None
    assert ctx.real_world_evidence.pmid == "35642282"


@pytest.mark.asyncio
async def test_endpoint_returns_assembled_context_for_known_brand():
    fake_ctx = {
        "brand": "Kisqali",
        "drug_name": "ribociclib",
        "disease": "Malignant neoplasm of breast",
        "our_outcome": "persistent_180d",
        "mapped_endpoint": "Treatment persistence / duration of therapy",
        "mechanism": {"mechanism_of_action": "CDK4/6 inhibitor", "source": "chembl"},
        "pivotal_endpoints": {"endpoints": ["Overall Survival (OS)"], "source": "clinicaltrials.gov"},
        "real_world_evidence": None,
        "honesty_label": "estimate = synthetic; context = real, cited",
    }
    with patch.object(causal_routes._clinical_context_service, "get_context", return_value=fake_ctx):
        with patch.object(
            causal_routes, "_list_dataset_brands", return_value=["Kisqali", "Fabhalta", "Remibrutinib"]
        ):
            resp = await causal_routes.get_clinical_context(
                brand="Kisqali", outcome="persistent_180d", user={"sub": "t"}
            )
    assert resp.brand == "Kisqali"
    assert resp.drug_name == "ribociclib"
    assert resp.mechanism.source == "chembl"
    assert resp.real_world_evidence is None


@pytest.mark.asyncio
async def test_endpoint_404s_unknown_brand():
    from fastapi import HTTPException

    with patch.object(
        causal_routes, "_list_dataset_brands", return_value=["Kisqali", "Fabhalta", "Remibrutinib"]
    ):
        with pytest.raises(HTTPException) as ei:
            await causal_routes.get_clinical_context(
                brand="NotABrand", outcome="persistent_180d", user={"sub": "t"}
            )
    assert ei.value.status_code == 404


@pytest.mark.asyncio
async def test_endpoint_stays_200_when_service_degrades():
    # Even with everything on static fallback, the endpoint returns a 200 payload.
    degraded = {
        "brand": "Fabhalta",
        "drug_name": "iptacopan",
        "disease": "Paroxysmal nocturnal hemoglobinuria",
        "our_outcome": "treatment_initiated",
        "mapped_endpoint": "Treatment initiation (complement-inhibitor start/switch)",
        "mechanism": {"mechanism_of_action": "complement Factor B inhibitor", "source": "static_fallback"},
        "pivotal_endpoints": {"endpoints": ["Transfusion avoidance"], "source": "static_fallback"},
        "real_world_evidence": None,
        "honesty_label": "estimate = synthetic; context = real, cited",
    }
    with patch.object(causal_routes._clinical_context_service, "get_context", return_value=degraded):
        with patch.object(causal_routes, "_list_dataset_brands", return_value=["Fabhalta"]):
            resp = await causal_routes.get_clinical_context(
                brand="Fabhalta", outcome="treatment_initiated", user={"sub": "t"}
            )
    assert resp.mechanism.source == "static_fallback"
    assert resp.pivotal_endpoints.source == "static_fallback"
```

- [ ] **Step 2: Run it to confirm it fails**

Run: `python -m pytest tests/unit/test_api/test_causal_clinical_context.py -v`
Expected: FAIL — `ImportError: cannot import name 'ClinicalContext'` / `AttributeError: ... 'get_clinical_context'`.

- [ ] **Step 3: Add the Pydantic models**

In `src/api/schemas/causal.py`, after `DiscoverEffectsResponse` (ends ~line 671, before the `# PIPELINE SCHEMAS` banner), add:

```python
class MechanismOfAction(BaseModel):
    """Drug mechanism of action with its provenance.

    ``source`` is ``chembl`` when the live ChEMBL mechanism lookup succeeded, or
    ``static_fallback`` when it was unreachable and the curated MoA was used.
    """

    mechanism_of_action: str = Field(..., description="e.g. 'CDK4/6 inhibitor'")
    source: str = Field(..., description="chembl / static_fallback")


class PivotalEndpoint(BaseModel):
    """The disease's real pivotal endpoints (from ClinicalTrials.gov) + source."""

    endpoints: List[str] = Field(
        default_factory=list,
        description="Real primary outcome measures from registered trials (e.g. OS/PFS).",
    )
    source: str = Field(..., description="clinicaltrials.gov / static_fallback")


class RealWorldEvidence(BaseModel):
    """A real, cited real-world-evidence reference (from PubMed)."""

    pmid: str = Field(..., description="PubMed ID")
    title: str = Field(..., description="Article title")
    journal: Optional[str] = Field(default=None, description="Journal / source")
    pubdate: Optional[str] = Field(default=None, description="Publication date string")
    doi: Optional[str] = Field(default=None, description="DOI when available")
    url: str = Field(..., description="Canonical pubmed.ncbi.nlm.nih.gov URL")
    source: str = Field(..., description="pubmed / pubmed_seed")


class ClinicalContext(BaseModel):
    """Brand-faithful, sourced clinical NARRATIVE for a discovered effect.

    Additive over the causal result — does NOT change the math or adjustment set.
    ``honesty_label`` always states the boundary: the effect estimate runs on a
    SYNTHETIC cohort; this clinical context is REAL and cited. Any field whose
    source is ``static_fallback`` came from the curated map because the live API
    was unreachable (the layer degrades gracefully, never fabricates).
    """

    brand: str = Field(..., description="Brand the context is for")
    drug_name: str = Field(..., description="INN drug name (e.g. ribociclib)")
    disease: str = Field(..., description="Indication (e.g. Malignant neoplasm of breast)")
    our_outcome: str = Field(..., description="Our synthetic outcome column this maps from")
    mapped_endpoint: Optional[str] = Field(
        default=None,
        description="The real pivotal-endpoint framing our synthetic outcome stands in for (None when unmapped).",
    )
    mechanism: MechanismOfAction
    pivotal_endpoints: PivotalEndpoint
    real_world_evidence: Optional[RealWorldEvidence] = Field(
        default=None, description="A real cited RWE reference; None when none was found."
    )
    honesty_label: str = Field(
        ..., description="Explicit synthetic-estimate / real-context boundary statement."
    )
```

- [ ] **Step 4: Add the service instance + endpoint to `causal.py`**

In `src/api/routes/causal.py`, add `ClinicalContext` to the `from src.api.schemas.causal import (...)` block (alphabetically near `CausalVariablesResponse`):

```python
    ClinicalContext,
```

Add a module-level service instance next to `_discover_effects_store` (~after line 1067):

```python
# Clinical Context enrichment service (lazy real REST clients inside; see
# src/services/clinical_context). Module-level so tests can patch
# ``causal._clinical_context_service.get_context``. Stateless apart from its
# in-process per-brand cache.
from src.services.clinical_context import ClinicalContextService

_clinical_context_service = ClinicalContextService()
```

Add the endpoint after `get_discover_causal_effects` (~after line 1301):

```python
@router.get(
    "/clinical-context",
    response_model=ClinicalContext,
    summary="Brand-faithful, sourced clinical context for a discovered effect",
    operation_id="get_causal_clinical_context",
)
async def get_clinical_context(
    brand: str = Query(..., description="Brand to enrich (e.g. Kisqali / Fabhalta / Remibrutinib)"),
    outcome: str = Query(
        ...,
        description="The synthetic outcome column the effect uses (e.g. persistent_180d); mapped to the real pivotal endpoint.",
    ),
    user: Dict[str, Any] = Depends(require_viewer),
) -> ClinicalContext:
    """Return the drug + mechanism of action (ChEMBL), the disease's real pivotal
    endpoints (ClinicalTrials.gov), and a real-world-evidence citation (PubMed)
    for ``brand``, mapping our synthetic ``outcome`` to the real endpoint framing.

    Additive narrative ONLY — does not touch the causal estimate or its
    adjustment set. Degrades gracefully (static fallbacks) when an upstream API
    is down; never fabricates a citation. The payload's ``honesty_label`` states
    the synthetic-estimate / real-context boundary.
    """
    available = await _list_dataset_brands(_DEFAULT_CAUSAL_DATASET)
    if available and brand not in available:
        raise HTTPException(
            status_code=404,
            detail=f"Unknown brand '{brand}'. Known brands: {available}",
        )
    try:
        payload = _clinical_context_service.get_context(brand, outcome)
    except KeyError:
        # The brand_map has no profile for this brand (no enrichment facts).
        raise HTTPException(
            status_code=404,
            detail=f"No clinical-context profile for brand '{brand}'.",
        )
    return ClinicalContext.model_validate(payload)
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `python -m pytest tests/unit/test_api/test_causal_clinical_context.py -v`
Expected: PASS.

- [ ] **Step 6: Scoped mypy + full causal test sweep**

Run: `mypy src/api/routes/causal.py src/api/schemas/causal.py`
Expected: clean (scoped). If mypy reports `ClinicalContext.model_validate(payload)` arg type (dict→model), that is supported by pydantic v2; if it complains about the `Dict[str, Any]` shape, the `model_validate` call is correct — confirm pydantic v2 is the project version (`grep -n "pydantic" pyproject.toml`).
Run: `python -m pytest tests/unit/test_api/test_causal_clinical_context.py tests/unit/test_api/test_causal_discover_effects.py -v`
Expected: PASS (new endpoint + the existing discover-effects suite unaffected).

- [ ] **Step 7: Commit**

```bash
git add src/api/schemas/causal.py src/api/routes/causal.py tests/unit/test_api/test_causal_clinical_context.py
git commit -m "feat(causal): GET /causal/clinical-context endpoint + ClinicalContext schema"
```

---

### Task 7: FE types + API client + hook for clinical context

Mirror the existing `DiscoveredEffect`/`getCausalBrands`/`useCausalBrands` patterns in the post-#1030 files.

**Files:**
- Modify: `frontend/src/types/causal.ts` (add interfaces after `DiscoverEffectsResponse` ~line 322)
- Modify: `frontend/src/api/causal.ts` (add `getClinicalContext` after `getCausalBrands` ~line 293; add the type to the import block ~line 26-48)
- Modify: `frontend/src/hooks/api/use-causal.ts` (add `useClinicalContext` after `useCausalBrands` ~line 193; add imports)
- Test: covered by the component test in Task 8 (the hook is exercised via the panel; the types/client are compile-checked by `tsc`).

- [ ] **Step 1: Add the TS interfaces**

In `frontend/src/types/causal.ts`, after the `DiscoverEffectsResponse` interface (ends ~line 322), add:

```typescript
/** Drug mechanism of action + provenance (chembl | static_fallback). */
export interface MechanismOfAction {
  mechanism_of_action: string;
  /** chembl / static_fallback */
  source: string;
}

/** The disease's real pivotal endpoints (clinicaltrials.gov | static_fallback). */
export interface PivotalEndpoint {
  endpoints: string[];
  /** clinicaltrials.gov / static_fallback */
  source: string;
}

/** A real, cited real-world-evidence reference (from PubMed). */
export interface RealWorldEvidence {
  pmid: string;
  title: string;
  journal?: string | null;
  pubdate?: string | null;
  doi?: string | null;
  url: string;
  /** pubmed / pubmed_seed */
  source: string;
}

/**
 * Brand-faithful, sourced clinical NARRATIVE for a discovered effect. Additive
 * over the causal result — never changes the estimate or adjustment set.
 * `honesty_label` states the boundary: estimate = synthetic cohort; context =
 * real, cited. A `static_fallback` source means the live API was unreachable.
 */
export interface ClinicalContext {
  brand: string;
  drug_name: string;
  disease: string;
  /** Our synthetic outcome column this maps from. */
  our_outcome: string;
  /** The real pivotal-endpoint framing our synthetic outcome stands in for (null when unmapped). */
  mapped_endpoint?: string | null;
  mechanism: MechanismOfAction;
  pivotal_endpoints: PivotalEndpoint;
  real_world_evidence?: RealWorldEvidence | null;
  honesty_label: string;
}
```

- [ ] **Step 2: Add the API client function**

In `frontend/src/api/causal.ts`, add `ClinicalContext` to the `import type { ... } from '@/types/causal'` block (alphabetically near `CausalBrandsResponse`):

```typescript
  ClinicalContext,
```

Then, after `getCausalBrands` (~line 293-301), add:

```typescript
/**
 * Fetch the brand-faithful, sourced clinical context for a discovered effect
 * (drug + mechanism of action, the disease's real pivotal endpoints, a
 * real-world-evidence citation). Additive narrative; never changes the estimate.
 */
export async function getClinicalContext(
  brand: string,
  outcome: string
): Promise<ClinicalContext> {
  // Flat params: `get(endpoint, params)` wraps them for axios (see api-client).
  return get<ClinicalContext>(`${CAUSAL_BASE}/clinical-context`, { brand, outcome });
}
```

- [ ] **Step 3: Add the hook**

In `frontend/src/hooks/api/use-causal.ts`, add `getClinicalContext` to the `from '@/api/causal'` import block (~line 16-37) and `ClinicalContext` to the `import type { ... } from '@/types/causal'` block (~line 38-60). Then after `useCausalBrands` (ends ~line 193), add:

```typescript
/**
 * Fetch the clinical context (drug + MoA, real pivotal endpoints, RWE citation)
 * for a discovered effect's brand + outcome. Additive narrative — does not touch
 * the causal estimate. Disabled until both `brand` and `outcome` are present.
 */
export function useClinicalContext(
  brand: string | null | undefined,
  outcome: string | null | undefined,
  options?: Omit<UseQueryOptions<ClinicalContext, ApiError>, 'queryKey' | 'queryFn'>
) {
  return useQuery<ClinicalContext, ApiError>({
    queryKey: ['causal', 'clinical-context', brand, outcome],
    queryFn: () => getClinicalContext(brand as string, outcome as string),
    // Real biomedical facts change slowly; cache aggressively, no auto-refetch.
    staleTime: 30 * 60 * 1000,
    gcTime: 60 * 60 * 1000,
    retry: false,
    ...options,
    enabled: Boolean(brand) && Boolean(outcome) && (options?.enabled ?? true),
  });
}
```

- [ ] **Step 4: Re-export the hook from the barrel**

The page imports from `@/hooks/api`, whose barrel `frontend/src/hooks/api/index.ts` uses an explicit named-export list (NOT `export *`). In the CAUSAL block (~line 52-72, ending `} from './use-causal';`), add `useClinicalContext` to the list:

```typescript
  useCausalBrands,
  useClinicalContext,
```

(insert right after the existing `useCausalBrands,` line, before the closing `} from './use-causal';`).

- [ ] **Step 5: Typecheck**

Run: `cd frontend && npx tsc --noEmit`
Expected: clean (no type errors introduced).

- [ ] **Step 6: Commit**

```bash
git add frontend/src/types/causal.ts frontend/src/api/causal.ts frontend/src/hooks/api/use-causal.ts frontend/src/hooks/api/index.ts
git commit -m "feat(fe): ClinicalContext types + getClinicalContext + useClinicalContext"
```

---

### Task 8: `ClinicalContextPanel` component + drill-down/leaderboard wiring

Render the panel inside the existing #1030 drill-down Card, and a compact MoA badge on each leaderboard row. The panel shows: drug + MoA (with a source chip), our-outcome → mapped real endpoint, the list of real pivotal endpoints, the RWE citation (linked), and the honesty label.

**Files:**
- Create: `frontend/src/components/causal/ClinicalContextPanel.tsx`
- Create: `frontend/src/components/causal/ClinicalContextPanel.test.tsx`
- Modify: `frontend/src/pages/CausalDiscovery.tsx` (import + render the panel in the drill-down ~after the estimator_comparison block, line ~536; import `useClinicalContext`)

- [ ] **Step 1: Write the failing component test**

```tsx
// frontend/src/components/causal/ClinicalContextPanel.test.tsx
import { describe, it, expect } from 'vitest';
import { render, screen } from '@testing-library/react';

import { ClinicalContextPanel } from './ClinicalContextPanel';
import type { ClinicalContext } from '@/types/causal';

const FULL: ClinicalContext = {
  brand: 'Kisqali',
  drug_name: 'ribociclib',
  disease: 'Malignant neoplasm of breast',
  our_outcome: 'persistent_180d',
  mapped_endpoint: 'Treatment persistence / duration of therapy',
  mechanism: { mechanism_of_action: 'CDK4/6 inhibitor', source: 'chembl' },
  pivotal_endpoints: {
    endpoints: ['Overall Survival (OS)', 'Progression-Free Survival (PFS)'],
    source: 'clinicaltrials.gov',
  },
  real_world_evidence: {
    pmid: '35642282',
    title: 'CDK4/6 inhibitor treatment use in women with advanced breast cancer.',
    journal: 'J Oncol Pharm Pract',
    pubdate: '2023 Jul',
    doi: '10.1177/10781552221102884',
    url: 'https://pubmed.ncbi.nlm.nih.gov/35642282/',
    source: 'pubmed',
  },
  honesty_label:
    'Effect estimate = a SYNTHETIC patient cohort. Clinical context below is REAL and cited.',
};

describe('ClinicalContextPanel', () => {
  it('renders drug, MoA, mapped endpoint, pivotal endpoints, and a linked citation', () => {
    render(<ClinicalContextPanel context={FULL} />);
    expect(screen.getByText(/ribociclib/i)).toBeInTheDocument();
    expect(screen.getByText(/CDK4\/6 inhibitor/)).toBeInTheDocument();
    expect(screen.getByText(/Treatment persistence/i)).toBeInTheDocument();
    expect(screen.getByText(/Overall Survival/)).toBeInTheDocument();
    const link = screen.getByRole('link', { name: /35642282|breast cancer|CDK4/i });
    expect(link).toHaveAttribute('href', 'https://pubmed.ncbi.nlm.nih.gov/35642282/');
  });

  it('always shows the synthetic/real honesty label', () => {
    render(<ClinicalContextPanel context={FULL} />);
    expect(screen.getByText(/SYNTHETIC/)).toBeInTheDocument();
    expect(screen.getByText(/REAL and cited/i)).toBeInTheDocument();
  });

  it('marks a static_fallback source honestly and omits a missing citation', () => {
    const degraded: ClinicalContext = {
      ...FULL,
      mechanism: { mechanism_of_action: 'complement Factor B inhibitor', source: 'static_fallback' },
      pivotal_endpoints: { endpoints: ['Transfusion avoidance'], source: 'static_fallback' },
      real_world_evidence: null,
    };
    render(<ClinicalContextPanel context={degraded} />);
    // Source is disclosed (not hidden) when it is a curated fallback.
    expect(screen.getAllByText(/curated|fallback/i).length).toBeGreaterThan(0);
    // No fabricated citation link.
    expect(screen.queryByRole('link')).toBeNull();
  });
});
```

- [ ] **Step 2: Run it to confirm it fails**

Run: `cd frontend && npx vitest run src/components/causal/ClinicalContextPanel.test.tsx`
Expected: FAIL — cannot resolve `./ClinicalContextPanel`.

- [ ] **Step 3: Implement the panel**

`frontend/src/components/causal/ClinicalContextPanel.tsx`:

```tsx
/**
 * ClinicalContextPanel — brand-faithful, sourced clinical narrative for a
 * discovered causal effect.
 *
 * Renders the drug + mechanism of action (ChEMBL), the disease's real pivotal
 * endpoints (ClinicalTrials.gov), and a real-world-evidence citation (PubMed) —
 * each with a source chip so a curated static fallback is disclosed, never
 * hidden. Always shows the synthetic-estimate / real-context honesty label.
 *
 * Additive presentation ONLY — it does not change the causal estimate.
 *
 * @module components/causal/ClinicalContextPanel
 */

import { FlaskConround, BookText, ExternalLink, Stethoscope } from 'lucide-react';

import { Badge } from '@/components/ui/badge';
import type { ClinicalContext } from '@/types/causal';

function sourceChip(source: string) {
  const live = source === 'chembl' || source === 'clinicaltrials.gov' || source === 'pubmed';
  const seed = source === 'pubmed_seed';
  if (live) {
    return (
      <Badge variant="outline" className="ml-2 align-middle text-xs">
        {source}
      </Badge>
    );
  }
  if (seed) {
    return (
      <Badge variant="outline" className="ml-2 align-middle text-xs">
        pubmed (curated seed)
      </Badge>
    );
  }
  return (
    <Badge variant="secondary" className="ml-2 align-middle text-xs">
      curated fallback
    </Badge>
  );
}

export function ClinicalContextPanel({ context }: { context: ClinicalContext }) {
  const { mechanism, pivotal_endpoints, real_world_evidence } = context;
  return (
    <div className="space-y-4 rounded-md border p-4">
      <div className="flex items-center gap-2">
        <Stethoscope className="h-4 w-4 text-muted-foreground" />
        <p className="text-sm font-medium">Clinical context</p>
      </div>

      {/* Drug + mechanism of action */}
      <div className="text-sm">
        <span className="font-medium capitalize">{context.drug_name}</span>{' '}
        <span className="text-muted-foreground">— {context.disease}</span>
        <div className="mt-1">
          <span className="text-muted-foreground">Mechanism of action: </span>
          <span className="font-medium">{mechanism.mechanism_of_action}</span>
          {sourceChip(mechanism.source)}
        </div>
      </div>

      {/* Our synthetic outcome -> the real pivotal endpoint framing */}
      {context.mapped_endpoint && (
        <div className="text-sm">
          <span className="text-muted-foreground">Our outcome </span>
          <code className="rounded bg-muted px-1 py-0.5 text-xs">{context.our_outcome}</code>
          <span className="text-muted-foreground"> maps to: </span>
          <span className="font-medium">{context.mapped_endpoint}</span>
        </div>
      )}

      {/* The disease's real pivotal endpoints */}
      {pivotal_endpoints.endpoints.length > 0 && (
        <div className="text-sm">
          <div className="flex items-center gap-1 text-muted-foreground">
            <FlaskConround className="h-3.5 w-3.5" />
            Real pivotal endpoints
            {sourceChip(pivotal_endpoints.source)}
          </div>
          <ul className="mt-1 list-disc space-y-0.5 pl-5">
            {pivotal_endpoints.endpoints.map((ep) => (
              <li key={ep}>{ep}</li>
            ))}
          </ul>
        </div>
      )}

      {/* Real-world-evidence citation (only when real) */}
      {real_world_evidence ? (
        <div className="text-sm">
          <div className="flex items-center gap-1 text-muted-foreground">
            <BookText className="h-3.5 w-3.5" />
            Real-world evidence
            {sourceChip(real_world_evidence.source)}
          </div>
          <a
            href={real_world_evidence.url}
            target="_blank"
            rel="noopener noreferrer"
            className="mt-1 inline-flex items-start gap-1 text-primary hover:underline"
          >
            <span>
              {real_world_evidence.title}
              {real_world_evidence.journal ? ` — ${real_world_evidence.journal}` : ''}
              {real_world_evidence.pubdate ? ` (${real_world_evidence.pubdate})` : ''}
              {` · PMID ${real_world_evidence.pmid}`}
            </span>
            <ExternalLink className="mt-0.5 h-3.5 w-3.5 shrink-0" />
          </a>
        </div>
      ) : (
        <p className="text-xs text-muted-foreground">
          No real-world-evidence citation found for this brand.
        </p>
      )}

      {/* The synthetic/real honesty boundary — always shown */}
      <p className="border-t pt-3 text-xs text-muted-foreground">{context.honesty_label}</p>
    </div>
  );
}
```

Note: the icon name in `lucide-react` is `FlaskConical` (not `FlaskConround`). Use `FlaskConical` in BOTH the import and the JSX (the placeholder above is intentionally wrong to force the implementer to confirm the real export — verify with `grep -r "FlaskConical" frontend/node_modules/lucide-react/dist/lucide-react.d.ts` and fix both occurrences before running the test).

- [ ] **Step 4: Run the component test to verify it passes**

Run: `cd frontend && npx vitest run src/components/causal/ClinicalContextPanel.test.tsx`
Expected: PASS (after fixing the `FlaskConical` import).

- [ ] **Step 5: Wire the panel into the drill-down + a leaderboard MoA badge**

In `frontend/src/pages/CausalDiscovery.tsx`:

(a) Add imports near the existing hook/component imports (~line 55-61):

```typescript
import { useDiscoverEffects, useCausalBrands, useClinicalContext } from '@/hooks/api';
import { ClinicalContextPanel } from '@/components/causal/ClinicalContextPanel';
```

(replace the existing `import { useDiscoverEffects, useCausalBrands } from '@/hooks/api';` line).

(b) In the component body, after the `result` is derived (~line 213, `const result = detail.data;`), add the clinical-context query keyed to the selected effect's brand + outcome:

```typescript
  // Clinical context for the selected effect (additive narrative; brand-scoped).
  const clinicalContext = useClinicalContext(
    job?.brand ?? null,
    result?.outcome_var ?? null
  );
```

(c) In the drill-down Card body, render the panel after the `estimator_comparison` block (the block at ~line 534-536 `{result.estimator_comparison && (<EstimatorComparisonPanel .../>)}`), inside the same `<>...</>` that gates on `result`:

```tsx
                {clinicalContext.data && (
                  <ClinicalContextPanel context={clinicalContext.data} />
                )}
```

(d) Add a compact MoA badge to each leaderboard row. The row currently shows `{e.treatment} → {e.outcome}` (~line 396-405). The leaderboard row has no per-brand context locally, but the whole run is brand-scoped via `job?.brand`. When a brand is selected, show its MoA chip once in the table footer note. Append to the existing footer `<p>` (~line 425-433), after the existing text, a brand MoA line driven by the same `useClinicalContext` — but only when a single brand is selected (it is null for "all brands"). Add this block right after the leaderboard `</table>`'s wrapping `</div>` and before the existing footer `<p>` (~line 424):

```tsx
            {job?.brand && clinicalContext.data && (
              <p className="border-t px-3 py-2 text-xs text-muted-foreground">
                <span className="font-medium">{clinicalContext.data.drug_name}</span> (
                {job.brand}) —{' '}
                <span className="font-medium">
                  {clinicalContext.data.mechanism.mechanism_of_action}
                </span>
                . Estimates run on a synthetic cohort; clinical context is real and cited (open a
                row for sources).
              </p>
            )}
```

Note the brand-level chip uses the same `useClinicalContext(job?.brand, result?.outcome_var)` query; when no row is selected `result` is undefined so `outcome` is null and the query is disabled — the chip then shows only after a row is opened. That is acceptable (the full panel in the drill-down is the primary surface; the footer chip is a bonus). Do NOT add a second `useClinicalContext` call for the footer — reuse the one from step (b).

- [ ] **Step 6: Typecheck + the page test + the component test**

Run: `cd frontend && npx tsc --noEmit`
Expected: clean.
Run: `cd frontend && npx vitest run src/components/causal/ClinicalContextPanel.test.tsx src/pages/CausalDiscovery.test.tsx`
Expected: PASS — the component test green AND the existing `CausalDiscovery.test.tsx` still green (the panel is additive; if a `getByText` in the page test now matches multiple nodes because of the new MoA copy, that is a known page-rewrite trap — grep the page test for any newly-ambiguous locator and scope it, do not weaken the panel).

- [ ] **Step 7: Commit**

```bash
git add frontend/src/components/causal/ClinicalContextPanel.tsx frontend/src/components/causal/ClinicalContextPanel.test.tsx frontend/src/pages/CausalDiscovery.tsx
git commit -m "feat(fe): ClinicalContextPanel in the causal drill-down + leaderboard MoA chip"
```

---

## Verification

- [ ] **Backend unit suite (all new + touched):**
  `python -m pytest tests/unit/test_data/test_kg/test_chembl_mechanism.py tests/unit/test_data/test_kg/test_chembl.py tests/unit/test_services/test_clinical_context/ tests/unit/test_api/test_causal_clinical_context.py tests/unit/test_api/test_causal_discover_effects.py -v`
  Expected: all green.
- [ ] **Lint (Ruff) — a red Lint gate cascade-skips backend tests in CI:**
  `ruff check src/services/clinical_context/ src/data/kg/chembl.py src/api/routes/causal.py src/api/schemas/causal.py && ruff format --check src/services/clinical_context/ src/data/kg/chembl.py src/api/routes/causal.py src/api/schemas/causal.py`
  Expected: clean.
- [ ] **Type check (scoped — do NOT run whole-tree mypy on the droplet; CI's MyPy gate is the arbiter):**
  `mypy src/services/clinical_context/ src/data/kg/chembl.py src/api/routes/causal.py src/api/schemas/causal.py`
  Expected: clean.
- [ ] **Frontend:** `cd frontend && npx tsc --noEmit && npx vitest run src/components/causal/ClinicalContextPanel.test.tsx src/pages/CausalDiscovery.test.tsx`
  Expected: clean + green.
- [ ] **Faithful live run (real public APIs, no MCP) — the cheapest real check the layer actually works:**
  Start the API locally, then:
  `curl -s "http://localhost:8000/api/causal/clinical-context?brand=Kisqali&outcome=persistent_180d" -H "Authorization: Bearer $E2I_VIEWER_TOKEN" | python3 -m json.tool`
  Expected (LIVE-verified shapes 2026-06-19): `mechanism.mechanism_of_action` ≈ "Cyclin-dependent kinase 4 inhibitor" with `mechanism.source == "chembl"`; `pivotal_endpoints.endpoints` non-empty with `source == "clinicaltrials.gov"` (or `static_fallback` if the live trial set is safety-only); `real_world_evidence` a real PubMed citation with a `pubmed.ncbi.nlm.nih.gov/<pmid>/` URL; `honesty_label` present. Repeat for `brand=Fabhalta&outcome=treatment_initiated` (PNH → transfusion-avoidance/LDH) and `brand=Remibrutinib&outcome=persistent_180d` (CSU → UAS7).
- [ ] **Degradation check (faithful):** temporarily point a client base at an unroutable host (e.g. construct the service with a `ClinicalTrialsClient(base="https://clinicaltrials.invalid/api/v2")`) in a throwaway REPL and confirm `get_context` returns a payload with `pivotal_endpoints.source == "static_fallback"` and no exception — proving graceful degradation against the target behavior, not just a mock.
- [ ] **Adversarial multi-lens review before the PR** (this codebase repeatedly ships CI-passing honesty bugs — check: is any `static_fallback` mislabeled as live? does the panel ever render a fabricated citation? is the honesty label ever omitted? does any code path touch the estimation frame / adjustment set?).

## Self-Review

- **Spec coverage (§7 Clinical Context enrichment layer):**
  - Drug + MoA via ChEMBL best-effort + static fallback (ribociclib=CDK4/6, iptacopan=complement Factor B, remibrutinib=BTK) — Task 1 (ChEMBL surface) + Task 2 (`moa_fallback`) + Task 4 (`ChEMBLMechanismProvider`). ✅
  - Real disease endpoints via ClinicalTrials.gov v2 (breast=OS/PFS/DFS; PNH=transfusion-avoidance/LDH/Hb; CSU=UAS7/UCT7/WI-NRS) — Task 2 (`pivotal_endpoints_fallback`) + Task 3 (`ClinicalTrialsClient.primary_endpoints`) + Task 4 (`ClinicalTrialsEndpointProvider`). ✅
  - Map our synthetic outcome → real endpoints — Task 2 (`endpoint_mapping_for_outcome`) + Task 5 (`mapped_endpoint`) + Task 8 (panel). ✅
  - RWE citation via PubMed (seed PMID 35642282) — Task 3 (`PubMedClient`) + Task 4 (`PubMedRWEProvider`, seed fallback) + Task 2 (`rwe_seed_pmid`). ✅
  - Cache per brand/disease — Task 5 (`_FRAGMENT_CACHE` keyed `(brand, disease)`; test asserts one fan-out across two outcomes). ✅
  - Graceful degradation when an API is down/slow — Tasks 4/5 (providers never raise; 8s timeout; static fallbacks) + Verification degradation check. ✅
  - Explicit honesty label "estimate = synthetic cohort; clinical context = real, cited" — Task 5 (`HONESTY_LABEL`) + Task 6 (schema field) + Task 8 (always rendered, asserted). ✅
  - FE Clinical Context panel on drill-down + leaderboard — Task 8. ✅
  - Backend calls PUBLIC REST directly, NOT MCP — Tasks 1/3 use `httpx` against the public base URLs; no MCP import anywhere. ✅
  - Extensible provider interface for the DEFERRED openFDA/UMLS work, but NOT implemented here — Task 4 (`ClinicalContextProvider` ABC; docstrings state openFDA/UMLS slot in as subclasses; none written). ✅
  - Does NOT touch the causal math / adjustment sets — no task reads `_load_agent_estimation_frame`, `_CAUSAL_NUMERIC_COLUMNS`, or any estimator; the endpoint takes `brand`+`outcome` strings only. ✅

- **Placeholder scan:** No `TODO`/`TBD`/"similar to"/"add error handling". Two intentional "force-the-implementer-to-confirm-the-real-symbol" markers are present and each is paired with the exact correction + the grep to verify it: (1) Task 5 Step 3 flags the double-wrap default and gives the one-line fix (`mechanism_provider or _default_chembl()`); (2) Task 8 Step 3 flags the deliberately-wrong `FlaskConround` icon and instructs `FlaskConical` with a grep. These are guardrails, not unfinished work — every other code block is complete and runnable.

- **Type consistency:**
  - `Mechanism(mechanism_of_action, action_type, target_chembl_id)` (Task 1) consumed by `ChEMBLClient.mechanism_of_action` and unused downstream (the provider takes the MoA string). ✅
  - `BrandClinicalProfile(brand, drug_name, disease, drug_class, moa_fallback, pivotal_endpoints_fallback, rwe_search_term, rwe_seed_pmid, outcome_endpoint_map)` (Task 2) consumed identically by all three providers (Task 4) and the service (Task 5). ✅
  - `PubMedArticle(pmid, title, journal, pubdate, doi)` + `.url` property (Task 3) → `CitationFragment.citation` (Task 4) → service `real_world_evidence` dict (Task 5) → `RealWorldEvidence` schema (Task 6) → FE `RealWorldEvidence` interface (Task 7) → panel (Task 8). All fields line up (`pmid/title/journal/pubdate/doi/url/source`). ✅
  - Provider fragments `MechanismFragment(mechanism_of_action, source)`, `EndpointsFragment(endpoints, source)`, `CitationFragment(citation, source)` (Task 4) → service dict keys `mechanism/pivotal_endpoints/real_world_evidence` (Task 5) → schema `MechanismOfAction/PivotalEndpoint/RealWorldEvidence` (Task 6) → FE interfaces (Task 7). The service dict is validated by `ClinicalContext.model_validate(payload)` so the dict↔model field names are load-bearing and verified by `test_endpoint_returns_assembled_context_for_known_brand`. ✅
  - Endpoint `get_clinical_context(brand, outcome, user)` returns `ClinicalContext` (Task 6) consumed by `getClinicalContext` → `useClinicalContext` → `<ClinicalContextPanel context=...>` (Tasks 7/8). ✅

- **Cross-plan / sequencing reconciliation (carried up to the caller):** see the "RETURN" bullets below.
