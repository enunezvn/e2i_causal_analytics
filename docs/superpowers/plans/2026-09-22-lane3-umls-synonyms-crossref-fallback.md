# Lane 3: UMLS synonyms and Crossref fallback in citation verification — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Citation verification in the live clinical-context path matches the drug and disease the way abstracts write them (UMLS atoms: "PNH", "CSU", British spellings), and a PMID whose abstract Europe PMC lacks is re-verified through Crossref by DOI.

**Architecture:** Three seams, each behind an existing protocol. (1) `UMLSClient.atoms(cui)` fetches English synonym names. (2) `CitationResolver._candidate_terms` fans out preferred name + atoms behind the unchanged word-boundary matcher. (3) `CausalEvidenceProvider` resolves drug/disease CUIs once per brand via UMLS exact search, passes them to the verifier, and on a settled Europe PMC absence re-verifies through Crossref using the DOI from PubMed esummary. No API schema change (`VerifiedCitation.source` is a free `str`). The Layer‑4 pipeline path already passes CUIs and inherits the fan-out.

**Tech Stack:** httpx (`MockTransport` in tests), functools.lru_cache, pytest.

**Spec:** `docs/superpowers/specs/2026-09-22-public-apis-live-path-design.md` (Lane 3).

**Measured premise (2026-09-22, inside `e2i_api`):** 47 candidate citations across three brands; 3 fail verification; atoms rescue 2 (Fabhalta PMIDs 40447351, 40395624 — abstracts say "PNH", "haemoglobinuria"). PMID 39968604 has no Europe PMC abstract; esummary DOI `10.1111/bjh.20024`; Crossref also has none for it (fallback correct, yield small). UMLS exact search: "chronic spontaneous urticaria"→C0578870, "remibrutinib"→C5446791, "breast cancer"→C0006142, "paroxysmal nocturnal hemoglobinuria"→C0024790.

---

## Worktree

```bash
git -C /home/enunez/Projects/e2i_causal_analytics worktree add \
  /home/enunez/Projects/e2i_causal_analytics/.worktrees/lane3-umls-crossref \
  -b claude/lane3-umls-crossref origin/main
cd /home/enunez/Projects/e2i_causal_analytics/.worktrees/lane3-umls-crossref
git branch --show-current   # claude/lane3-umls-crossref
```

Every `pytest` runs with `-n 0` from this directory; assert `src.__file__` starts with this path in every python invocation. Local pytest runs write to prod Redis via `.env` — none of these tests touch Redis, but do not add any that do.

---

### Task 1: `UMLSClient.atoms`

**Files:**
- Modify: `src/data/kg/umls_uts.py` (method after `cui_lookup`/`_cui_lookup_uncached`; cache helper after `_cui_lookup_cached`; `reset_caches`)
- Test: `tests/unit/test_data/test_kg/test_umls_uts.py` (append)

- [ ] **Step 1: Write the failing tests**

Append to `tests/unit/test_data/test_kg/test_umls_uts.py` (the file already imports `httpx`, `pytest`, `UMLSClient`, `UMLSError`, `reset_caches`; add `UMLSNotFoundError` if absent):

```python
# ------------------------------------------------------------------- atoms


def _atoms_transport(payload: dict, *, status: int = 200) -> httpx.MockTransport:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path.endswith("/CUI/C0578870/atoms")
        assert request.url.params.get("language") == "ENG"
        assert request.url.params.get("pageSize") == "50"
        return httpx.Response(status, json=payload)

    return httpx.MockTransport(handler)


_ATOMS_PAYLOAD = {
    "result": [
        {"name": "Chronic Spontaneous Urticaria", "language": "ENG"},
        {"name": "CSU", "language": "ENG"},
        {"name": "chronic spontaneous urticaria", "language": "ENG"},  # case-dupe
        {"name": "", "language": "ENG"},
        {"notaname": 1},
    ]
}


def test_atoms_returns_unique_english_names_in_order() -> None:
    reset_caches()
    client = UMLSClient(api_key="k", client=httpx.Client(transport=_atoms_transport(_ATOMS_PAYLOAD)))
    assert client.atoms("C0578870") == ["Chronic Spontaneous Urticaria", "CSU"]


def test_atoms_is_empty_for_a_404_rather_than_raising() -> None:
    reset_caches()
    client = UMLSClient(api_key="k", client=httpx.Client(transport=_atoms_transport({}, status=404)))
    assert client.atoms("C0578870") == []


def test_atoms_raises_umls_error_on_other_http_failures() -> None:
    reset_caches()
    client = UMLSClient(api_key="k", client=httpx.Client(transport=_atoms_transport({}, status=503)))
    with pytest.raises(UMLSError):
        client.atoms("C0578870")


def test_atoms_is_empty_for_an_empty_cui_without_a_request() -> None:
    def boom(request: httpx.Request) -> httpx.Response:
        raise AssertionError("no request expected")

    client = UMLSClient(api_key="k", client=httpx.Client(transport=httpx.MockTransport(boom)))
    assert client.atoms("") == []


def test_atoms_is_cached_and_cleared_by_reset() -> None:
    reset_caches()
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return httpx.Response(200, json=_ATOMS_PAYLOAD)

    client = UMLSClient(api_key="k", client=httpx.Client(transport=httpx.MockTransport(handler)))
    client.atoms("C0578870")
    client.atoms("C0578870")
    assert calls["n"] == 1
    reset_caches()
    client.atoms("C0578870")
    assert calls["n"] == 2
```

- [ ] **Step 2: Run to verify they fail**

Run: `pytest -n 0 tests/unit/test_data/test_kg/test_umls_uts.py -k atoms -v`
Expected: 5 FAIL with `AttributeError: 'UMLSClient' object has no attribute 'atoms'`.

- [ ] **Step 3: Implement**

In `src/data/kg/umls_uts.py`, after `_cui_lookup_uncached`:

```python
    def atoms(self, cui: str, *, language: str = "ENG", page_size: int = 50) -> list[str]:
        """Synonym names (atoms) for a CUI, deduplicated case-insensitively.

        This is the "v2 atom-list synonym fan-out" the CitationResolver docstring
        deferred: abstracts write "PNH", "CSU", "haemoglobinuria" where our curated
        search term says the full American name. Measured 2026-09-22: this rescued
        2 of the 3 candidate citations that failed on names alone. A CUI with no
        atoms page (UTS 404) yields ``[]``; other failures raise ``UMLSError``.
        """
        return list(_atoms_cached(self, cui=cui, language=language, page_size=page_size))

    def _atoms_uncached(self, *, cui: str, language: str, page_size: int) -> tuple[str, ...]:
        if not cui:
            return ()
        try:
            payload = self._get(
                f"/content/{self._version}/CUI/{cui}/atoms",
                {"language": language, "pageSize": page_size},
            )
        except UMLSNotFoundError:
            return ()
        rows = payload.get("result") or []
        seen: set[str] = set()
        names: list[str] = []
        for row in rows:
            name = row.get("name") if isinstance(row, dict) else None
            if not isinstance(name, str) or not name.strip():
                continue
            key = name.strip().lower()
            if key in seen:
                continue
            seen.add(key)
            names.append(name.strip())
        return tuple(names)
```

After `_cui_lookup_cached`:

```python
@lru_cache(maxsize=_LRU_MAXSIZE)
def _atoms_cached(
    client: UMLSClient, *, cui: str, language: str, page_size: int
) -> tuple[str, ...]:
    return client._atoms_uncached(cui=cui, language=language, page_size=page_size)
```

In `reset_caches()` add `_atoms_cached.cache_clear()`.

- [ ] **Step 4: Run the whole UMLS test file**

Run: `pytest -n 0 tests/unit/test_data/test_kg/test_umls_uts.py -v`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add src/data/kg/umls_uts.py tests/unit/test_data/test_kg/test_umls_uts.py
git commit -m "feat(umls): atoms — English synonym names for a CUI

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 2: Synonym fan-out in `CitationResolver._candidate_terms`

**Files:**
- Modify: `src/data/kg/citation_resolver.py` (`_candidate_terms`, lines ~292-320; two module constants)
- Test: `tests/unit/test_data/test_kg/test_citation_resolver.py` (extend `_StubUMLS`; append tests)

- [ ] **Step 1: Extend the stub and write the failing tests**

In `tests/unit/test_data/test_kg/test_citation_resolver.py`, change `_StubUMLS`:

```python
class _StubUMLS:
    def __init__(
        self,
        *,
        concepts: Optional[dict[str, KGConcept]] = None,
        atoms: Optional[dict[str, list[str]]] = None,
        raise_auth: bool = False,
        raise_error: bool = False,
        raise_on_atoms: bool = False,
    ) -> None:
        self._concepts = concepts or {}
        self._atoms = atoms or {}
        self._raise_auth = raise_auth
        self._raise_error = raise_error
        self._raise_on_atoms = raise_on_atoms
        self.atom_calls: list[str] = []

    def cui_lookup(self, cui: str) -> KGConcept:
        if self._raise_auth:
            raise UMLSAuthError("simulated auth fail")
        if self._raise_error:
            raise UMLSError("simulated transport fail")
        return self._concepts.get(cui, KGConcept(cui=cui, preferred_name=""))

    def atoms(self, cui: str, *, language: str = "ENG", page_size: int = 50) -> list[str]:
        self.atom_calls.append(cui)
        if self._raise_on_atoms:
            raise UMLSError("simulated atoms fail")
        return list(self._atoms.get(cui, []))

    def close(self) -> None:
        pass
```

Append:

```python
# ----------------------------------------------------- atom synonym fan-out (lane 3)


def test_an_abbreviation_only_abstract_verifies_with_atoms_and_not_without() -> None:
    """The measured Fabhalta case: the abstract says PNH, never the full name."""
    abstract = "Iptacopan improved hemoglobin in PNH patients; PNH outcomes were durable."
    pmc = _StubEuropePMC(abstracts={"40447351": _record(abstract, identifier="40447351")})
    umls = _StubUMLS(
        concepts={"C0024790": KGConcept(cui="C0024790", preferred_name="Paroxysmal nocturnal hemoglobinuria")},
        atoms={"C0024790": ["PNH", "Paroxysmal nocturnal haemoglobinuria"]},
    )
    without = _resolver(europe_pmc=pmc, umls=umls).verify_citation(
        "40447351", subject_name="iptacopan", object_name="paroxysmal nocturnal hemoglobinuria"
    )
    assert without.overall_confidence == 0.0
    with_cui = _resolver(europe_pmc=pmc, umls=umls).verify_citation(
        "40447351",
        subject_name="iptacopan",
        object_name="paroxysmal nocturnal hemoglobinuria",
        object_cui="C0024790",
    )
    assert with_cui.overall_confidence >= 0.5
    assert "PNH" in with_cui.entities_found


def test_atoms_shorter_than_three_characters_never_match() -> None:
    abstract = "The RA cohort received the drug."  # "RA" must not count as the disease
    pmc = _StubEuropePMC(abstracts={"1": _record(abstract, identifier="1")})
    umls = _StubUMLS(atoms={"C0003873": ["RA", "Rheumatoid arthritis"]})
    verdict = _resolver(europe_pmc=pmc, umls=umls).verify_citation(
        "1", subject_name="drug", object_name="rheumatoid arthritis", object_cui="C0003873"
    )
    assert verdict.overall_confidence == 0.0


def test_the_term_list_is_capped_and_deduplicated() -> None:
    umls = _StubUMLS(
        concepts={"C1": KGConcept(cui="C1", preferred_name="Alpha")},
        atoms={"C1": ["alpha", "ALPHA"] + [f"syn{i}" for i in range(100)]},
    )
    resolver = _resolver(umls=umls)
    terms = resolver._candidate_terms("Alpha", "C1")
    assert terms[0] == "Alpha"
    assert len(terms) == resolver.MAX_CANDIDATE_TERMS
    assert len({t.lower() for t in terms}) == len(terms)


def test_an_atoms_failure_keeps_the_names_gathered_so_far() -> None:
    abstract = "Paroxysmal nocturnal hemoglobinuria treated with iptacopan."
    pmc = _StubEuropePMC(abstracts={"1": _record(abstract, identifier="1")})
    umls = _StubUMLS(
        concepts={"C0024790": KGConcept(cui="C0024790", preferred_name="Paroxysmal nocturnal hemoglobinuria")},
        raise_on_atoms=True,
    )
    verdict = _resolver(europe_pmc=pmc, umls=umls).verify_citation(
        "1", subject_name="iptacopan", object_name="PNH", object_cui="C0024790"
    )
    assert verdict.overall_confidence >= 0.5  # preferred name still matched


def test_no_atoms_call_without_a_cui() -> None:
    umls = _StubUMLS()
    _resolver(umls=umls)._candidate_terms("iptacopan", None)
    assert umls.atom_calls == []
```

- [ ] **Step 2: Run to verify they fail**

Run: `pytest -n 0 tests/unit/test_data/test_kg/test_citation_resolver.py -k "atoms or abbreviation or capped or without_a_cui" -v`
Expected: the abbreviation test FAILS (`with_cui.overall_confidence == 0.0`), the cap test FAILS (`AttributeError: MAX_CANDIDATE_TERMS`), the failure-tolerance test PASSES (atoms not called yet), the short-term and no-call tests PASS trivially. Two red is the signal.

- [ ] **Step 3: Implement**

In `src/data/kg/citation_resolver.py`, add module constants near the existing `WEIGHT_*` constants:

```python
# Atom synonym fan-out (lane 3, 2026-09-22). Terms shorter than this are too
# ambiguous for whole-word matching ("RA"); the cap bounds the regex work per
# abstract for concepts with hundreds of atoms.
MIN_CANDIDATE_TERM_LEN = 3
MAX_CANDIDATE_TERMS = 40
```

Add class attributes on `CitationResolver` (right after the class docstring):

```python
    MIN_CANDIDATE_TERM_LEN = MIN_CANDIDATE_TERM_LEN
    MAX_CANDIDATE_TERMS = MAX_CANDIDATE_TERMS
```

Replace `_candidate_terms`:

```python
    def _candidate_terms(
        self,
        primary_name: str,
        cui: Optional[str],
    ) -> list[str]:
        """Build the list of names to match in an abstract.

        ``primary_name`` first, then the UMLS preferred name, then the CUI's
        atoms (synonyms and abbreviations: "PNH", "CSU", British spellings).
        Deduplicated case-insensitively, terms shorter than
        ``MIN_CANDIDATE_TERM_LEN`` dropped, capped at ``MAX_CANDIDATE_TERMS``.
        An atoms lookup failure keeps whatever was gathered; only an auth failure
        is fatal (the resolver cannot do its job without UMLS once asked for it).
        """
        terms: list[str] = []
        seen: set[str] = set()

        def _add(term: str) -> None:
            cleaned = term.strip()
            key = cleaned.lower()
            if len(cleaned) < self.MIN_CANDIDATE_TERM_LEN or key in seen:
                return
            if len(terms) >= self.MAX_CANDIDATE_TERMS:
                return
            seen.add(key)
            terms.append(cleaned)

        if primary_name:
            _add(primary_name)
        if not cui or self.umls is None:
            return terms
        try:
            concept: KGConcept = self.umls.cui_lookup(cui)
        except UMLSAuthError as exc:
            raise CitationResolverError(f"UMLS auth failed: {exc}") from exc
        except UMLSError as exc:
            logger.warning("UMLS cui_lookup failed for synonym expansion of %s: %s", cui, exc)
        else:
            if concept.preferred_name:
                _add(concept.preferred_name)
        try:
            atom_names = self.umls.atoms(cui)
        except UMLSAuthError as exc:
            raise CitationResolverError(f"UMLS auth failed: {exc}") from exc
        except UMLSError as exc:
            logger.warning("UMLS atoms failed for synonym expansion of %s: %s", cui, exc)
            return terms
        for name in atom_names:
            _add(name)
        return terms
```

- [ ] **Step 4: Run the whole resolver file**

Run: `pytest -n 0 tests/unit/test_data/test_kg/test_citation_resolver.py -v`
Expected: all PASS (existing tests included — `_StubUMLS` defaults keep prior behaviour).

- [ ] **Step 5: Commit**

```bash
git add src/data/kg/citation_resolver.py tests/unit/test_data/test_kg/test_citation_resolver.py
git commit -m "feat(citation): fan out UMLS atoms as candidate terms so PNH/CSU abstracts verify

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 3: The evidence provider resolves CUIs and passes them to the verifier

**Files:**
- Modify: `src/services/clinical_context/causal_evidence.py` (protocols ~226-250; `__init__` ~271-283; `_citations` ~385-391)
- Test: `tests/unit/test_services/test_clinical_context/test_causal_evidence.py` (append)

- [ ] **Step 1: Write the failing tests**

Append:

```python
# ------------------------------------------------ UMLS concept ids (lane 3, 2026-09-22)


class _FakeUMLSSearch:
    def __init__(self, hits=None, boom=None):
        self._hits = hits if hits is not None else {
            "ribociclib": [{"ui": "C1", "name": "ribociclib"}],
            "breast cancer": [{"ui": "C0006142", "name": "Malignant neoplasm of breast"}],
        }
        self._boom = boom
        self.calls = []

    def search(self, term, *, page_size=5, search_type="exact"):
        self.calls.append(term)
        if self._boom is not None:
            raise self._boom
        return list(self._hits.get(term, []))


class _RecordingResolver(_FakeResolver):
    def verify_citation(self, identifier, *, identifier_kind="pmid", subject_name, object_name, **kw):
        self.kwargs = kw
        return super().verify_citation(
            identifier, identifier_kind=identifier_kind, subject_name=subject_name, object_name=object_name
        )


@pytest.mark.unit
def test_cuis_are_resolved_once_per_brand_and_passed_to_the_verifier():
    profile = resolve_brand_profile("Kisqali")
    umls = _FakeUMLSSearch()
    resolver = _RecordingResolver({"1": _verdict("1", 1.0)})
    provider = CausalEvidenceProvider(
        open_targets=_FakeOpenTargets(), pubmed=_FakePubMedSearch(pmids=["1"]), resolver=resolver, umls=umls
    )
    for _ in range(2):
        provider.evidence(
            profile,
            outcome="persistent_180d",
            treatment_context=treatment_context_for("Kisqali", "treatment_arm"),
            search_term="ribociclib breast cancer persistence real-world",
        )
    assert resolver.kwargs == {"subject_cui": "C1", "object_cui": "C0006142"}
    assert umls.calls == ["ribociclib", "breast cancer"]  # cached after the first round


@pytest.mark.unit
def test_a_umls_failure_still_verifies_by_name():
    profile = resolve_brand_profile("Kisqali")
    resolver = _RecordingResolver({"1": _verdict("1", 1.0)})
    provider = CausalEvidenceProvider(
        open_targets=_FakeOpenTargets(),
        pubmed=_FakePubMedSearch(pmids=["1"]),
        resolver=resolver,
        umls=_FakeUMLSSearch(boom=RuntimeError("uts down")),
    )
    frag = provider.evidence(
        profile,
        outcome="persistent_180d",
        treatment_context=treatment_context_for("Kisqali", "treatment_arm"),
        search_term="ribociclib breast cancer persistence real-world",
    )
    assert resolver.kwargs == {"subject_cui": None, "object_cui": None}
    assert [c.pmid for c in frag.citations] == ["1"]


@pytest.mark.unit
def test_without_a_umls_client_no_cuis_are_passed():
    profile = resolve_brand_profile("Kisqali")
    resolver = _RecordingResolver({"1": _verdict("1", 1.0)})
    _provider(pubmed=_FakePubMedSearch(pmids=["1"]), resolver=resolver).evidence(
        profile,
        outcome="persistent_180d",
        treatment_context=treatment_context_for("Kisqali", "treatment_arm"),
        search_term="ribociclib breast cancer persistence real-world",
    )
    assert resolver.kwargs == {"subject_cui": None, "object_cui": None}
```

- [ ] **Step 2: Run to verify they fail**

Run: `pytest -n 0 tests/unit/test_services/test_clinical_context/test_causal_evidence.py -k "cuis or umls_failure or without_a_umls" -v`
Expected: first two FAIL with `TypeError: __init__() got an unexpected keyword argument 'umls'`; the third FAILS with `AttributeError: kwargs` or `{} != {...}`.

- [ ] **Step 3: Implement**

In `causal_evidence.py`, after `_PubMedSearchLike`:

```python
class _UMLSSearchLike(Protocol):
    def search(
        self, term: str, *, page_size: int = 5, search_type: str = "exact"
    ) -> List[dict[str, Any]]: ...
```

Extend `_CitationResolverLike.verify_citation` with the two kwargs:

```python
class _CitationResolverLike(Protocol):
    def verify_citation(
        self,
        identifier: str,
        *,
        identifier_kind: str = "pmid",
        subject_name: str,
        object_name: str,
        subject_cui: Optional[str] = None,
        object_cui: Optional[str] = None,
    ) -> CitationVerdict: ...
```

Change `__init__`:

```python
    def __init__(
        self,
        *,
        open_targets: _OpenTargetsLike,
        pubmed: _PubMedSearchLike,
        resolver: _CitationResolverLike,
        umls: Optional[_UMLSSearchLike] = None,
    ) -> None:
        self._open_targets = open_targets
        self._pubmed = pubmed
        self._resolver = resolver
        self._umls = umls
        # (drug_name, disease_search_term) -> (drug_cui, disease_cui); UMLS
        # concepts are stable, and the key space is the 3-brand universe.
        self._concept_ids_cache: dict[tuple[str, str], tuple[Optional[str], Optional[str]]] = {}
```

Add a method after `__init__`:

```python
    def _concept_ids(self, profile: BrandClinicalProfile) -> tuple[Optional[str], Optional[str]]:
        """UMLS CUIs for the drug and the plain-language disease, or ``(None, None)``.

        CUIs let the resolver fan out synonyms ("PNH", "CSU"). Resolution is
        best-effort: any failure means verification proceeds on names alone, as
        it did before lane 3 — never a dropped citation because UMLS blinked.
        """
        if self._umls is None:
            return None, None
        key = (profile.drug_name, profile.disease_search_term)
        cached = self._concept_ids_cache.get(key)
        if cached is not None:
            return cached
        ids: list[Optional[str]] = []
        for term in key:
            try:
                rows = self._umls.search(term)
            except Exception as exc:  # noqa: BLE001 — best-effort
                logger.warning("causal-evidence: UMLS search failed for %r: %s", term, exc)
                ids = [None, None]
                break
            ui = rows[0].get("ui") if rows and isinstance(rows[0], dict) else None
            ids.append(str(ui) if ui else None)
        result = (ids[0], ids[1])
        self._concept_ids_cache[key] = result
        return result
```

In `_citations`, before the `for attempt, pmid in enumerate(pmids):` loop add `subject_cui, object_cui = self._concept_ids(profile)`, and change the verifier call to:

```python
                verdict = self._resolver.verify_citation(
                    pmid,
                    identifier_kind="pmid",
                    subject_name=profile.drug_name,
                    # The plain-language disease term: the SSOT coding string
                    # ("Malignant neoplasm of breast") never appears in an abstract.
                    object_name=profile.disease_search_term,
                    subject_cui=subject_cui,
                    object_cui=object_cui,
                )
```

- [ ] **Step 4: Run the whole file**

Run: `pytest -n 0 tests/unit/test_services/test_clinical_context/test_causal_evidence.py -v`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add src/services/clinical_context/causal_evidence.py tests/unit/test_services/test_clinical_context/test_causal_evidence.py
git commit -m "feat(clinical-context): resolve drug/disease CUIs and hand them to citation verification

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 4: Crossref fallback on a settled Europe PMC absence

**Files:**
- Modify: `src/services/clinical_context/causal_evidence.py` (`_SOURCE_DISPLAY`; `_citations` return shape and loop; `evidence()` where `lit_unreachable` is consumed)
- Test: `tests/unit/test_services/test_clinical_context/test_causal_evidence.py` (append)

- [ ] **Step 1: Write the failing tests**

Append:

```python
# ------------------------------------------------- Crossref fallback (lane 3, 2026-09-22)


class _DoiAwarePubMed(_FakePubMedSearch):
    def __init__(self, pmids=(), doi_by_pmid=None):
        super().__init__(pmids=pmids)
        self._doi = doi_by_pmid or {}

    def fetch_by_pmid(self, pmid):
        from src.services.clinical_context.clients import PubMedArticle

        return PubMedArticle(
            pmid=pmid, title=f"Study {pmid}", journal="J", pubdate="2024", doi=self._doi.get(pmid)
        )


class _FallbackResolver(_FakeResolver):
    """PMIDs come back as a settled absence; DOIs resolve per the table."""

    def __init__(self, doi_verdicts=None, pmid_error=None):
        super().__init__({})
        self._doi_verdicts = doi_verdicts or {}
        self._pmid_error = pmid_error
        self.kinds = []

    def verify_citation(self, identifier, *, identifier_kind="pmid", subject_name, object_name, **kw):
        self.kinds.append((identifier, identifier_kind))
        if identifier_kind == "doi":
            verdict = self._doi_verdicts.get(identifier)
            if verdict is not None:
                return verdict
            return CitationVerdict(
                identifier=identifier, identifier_kind="doi", abstract_resolved=False,
                entities_found=(), causal_cue_found=None, overall_confidence=0.0,
                error="Crossref unreachable: simulated",
            )
        return CitationVerdict(
            identifier=identifier, identifier_kind="pmid", abstract_resolved=False,
            entities_found=(), causal_cue_found=None, overall_confidence=0.0,
            error=self._pmid_error,
        )


def _doi_verdict(doi, confidence):
    return CitationVerdict(
        identifier=doi, identifier_kind="doi", abstract_resolved=True,
        entities_found=("iptacopan", "PNH"), causal_cue_found=None,
        overall_confidence=confidence, error=None,
    )


def _fabhalta_call(provider):
    profile = resolve_brand_profile("Fabhalta")
    return provider.evidence(
        profile,
        outcome="persistent_180d",
        treatment_context=treatment_context_for("Fabhalta", "treatment_arm"),
        search_term="iptacopan paroxysmal nocturnal hemoglobinuria persistence real-world",
    )


@pytest.mark.unit
def test_a_settled_europe_pmc_absence_is_retried_through_crossref_by_doi():
    pubmed = _DoiAwarePubMed(pmids=["39968604"], doi_by_pmid={"39968604": "10.1111/bjh.20024"})
    resolver = _FallbackResolver(doi_verdicts={"10.1111/bjh.20024": _doi_verdict("10.1111/bjh.20024", 1.0)})
    frag = _fabhalta_call(_provider(pubmed=pubmed, resolver=resolver, open_targets=_FakeOpenTargets(payload=_FABHALTA_OT)))
    assert resolver.kinds == [("39968604", "pmid"), ("10.1111/bjh.20024", "doi")]
    assert [(c.pmid, c.source, c.confidence) for c in frag.citations] == [("39968604", "pubmed+crossref", 1.0)]
    assert frag.citations[0].url == "https://pubmed.ncbi.nlm.nih.gov/39968604/"


@pytest.mark.unit
def test_an_outage_is_not_retried_through_crossref():
    """error set == Europe PMC RAISED; that is unknown, not absent, and must be
    reported as unreachable, not papered over by a second source."""
    pubmed = _DoiAwarePubMed(pmids=["1"], doi_by_pmid={"1": "10.1/x"})
    resolver = _FallbackResolver(doi_verdicts={"10.1/x": _doi_verdict("10.1/x", 1.0)}, pmid_error="Europe PMC unreachable: 503")
    frag = _fabhalta_call(_provider(pubmed=pubmed, resolver=resolver, open_targets=_FakeOpenTargets(payload=_FABHALTA_OT)))
    assert resolver.kinds == [("1", "pmid")]
    assert frag.citations == []
    assert "europe_pmc" in frag.sources_unavailable


@pytest.mark.unit
def test_no_doi_means_no_fallback():
    pubmed = _DoiAwarePubMed(pmids=["1"], doi_by_pmid={})
    resolver = _FallbackResolver()
    frag = _fabhalta_call(_provider(pubmed=pubmed, resolver=resolver, open_targets=_FakeOpenTargets(payload=_FABHALTA_OT)))
    assert resolver.kinds == [("1", "pmid")]
    assert frag.citations == []
    assert frag.sources_unavailable == ()


@pytest.mark.unit
def test_a_crossref_outage_is_named_as_crossref():
    pubmed = _DoiAwarePubMed(pmids=["1"], doi_by_pmid={"1": "10.1/unknown"})
    resolver = _FallbackResolver()  # DOI path returns an error verdict
    frag = _fabhalta_call(_provider(pubmed=pubmed, resolver=resolver, open_targets=_FakeOpenTargets(payload=_FABHALTA_OT)))
    assert frag.citations == []
    assert frag.sources_unavailable == ("crossref",)
    assert "Crossref" in frag.note and "unreachable" in frag.note
```

- [ ] **Step 2: Run to verify they fail**

Run: `pytest -n 0 tests/unit/test_services/test_clinical_context/test_causal_evidence.py -k "crossref or no_doi or outage_is_not" -v`
Expected: the first and last FAIL (no fallback exists; `sources_unavailable` lacks `crossref`); the middle two PASS (current behaviour already drops them) — that is expected, they guard against over-reach.

- [ ] **Step 3: Implement**

Add `"crossref": "Crossref"` to `_SOURCE_DISPLAY`.

Change `_citations` to return unreachable **source names** instead of a count. Signature and docstring:

```python
    def _citations(
        self, profile: BrandClinicalProfile, search_term: str
    ) -> tuple[List[VerifiedCitation], List[str], int, int]:
        """Verified literature for this analysis, plus WHY anything is missing.

        Returns ``(citations, unreachable_sources, unchecked_budget, unchecked_local)``:
        - ``unreachable_sources`` — abstract sources that RAISED for some candidate
          (``"europe_pmc"``, ``"crossref"``), each named once. A real outage.
        - ``unchecked_budget`` — our own wall-clock budget stopped us before this
          candidate was examined.
        - ``unchecked_local`` — the verification call blew up locally. The resolver
          swallows source errors itself, so what escapes is not evidence that the
          upstream is down.
        All three mean "not a settled absence", and they must stay apart.
        """
```

Body changes: `unreachable: List[str] = []`; replace the block after the verifier call:

```python
            article: Any = None
            source = "pubmed+europepmc"
            if not verdict.abstract_resolved:
                if verdict.error:
                    if "europe_pmc" not in unreachable:
                        unreachable.append("europe_pmc")
                    continue
                # Europe PMC ANSWERED and holds no abstract (#1767). Many publishers
                # deposit abstracts with Crossref instead; the DOI comes from the
                # PubMed summary we would fetch anyway for a verified citation.
                article = self._summary(pmid, started)
                doi = getattr(article, "doi", None)
                if not doi:
                    continue
                try:
                    verdict = self._resolver.verify_citation(
                        doi,
                        identifier_kind="doi",
                        subject_name=profile.drug_name,
                        object_name=profile.disease_search_term,
                        subject_cui=subject_cui,
                        object_cui=object_cui,
                    )
                except Exception as exc:  # noqa: BLE001
                    logger.warning("causal-evidence: DOI verification failed for %s: %s", doi, exc)
                    unchecked_local += 1
                    continue
                if not verdict.abstract_resolved:
                    if verdict.error and "crossref" not in unreachable:
                        unreachable.append("crossref")
                    continue
                source = "pubmed+crossref"
            if verdict.overall_confidence < _MIN_CITATION_CONFIDENCE:
                continue
            if article is None:
                article = self._summary(pmid, started)
            out.append(
                VerifiedCitation(
                    pmid=pmid,
                    title=(getattr(article, "title", None) or f"PMID {pmid}"),
                    journal=getattr(article, "journal", None),
                    pubdate=getattr(article, "pubdate", None),
                    url=f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
                    entities_found=tuple(verdict.entities_found),
                    confidence=verdict.overall_confidence,
                    source=source,
                )
            )
        return out, unreachable, unchecked_budget, unchecked_local
```

Extract the existing summary fetch into a helper on the class (it was inline; keep its budget semantics exactly):

```python
    def _summary(self, pmid: str, started: float) -> Any:
        """PubMed esummary for ``pmid`` within the summary budget, else ``None``."""
        if time.monotonic() - started >= _SUMMARY_BUDGET_S:
            logger.info("causal-evidence: summary budget exhausted; PMID %s unadorned", pmid)
            return None
        try:
            return self._pubmed.fetch_by_pmid(pmid)
        except Exception as exc:  # noqa: BLE001
            logger.warning("causal-evidence: summary fetch failed for PMID %s: %s", pmid, exc)
            return None
```

In `evidence()`, replace

```python
                if lit_unreachable:
                    # Europe PMC actually failed. Name it.
                    unavailable.append("europe_pmc")
```

with

```python
                for src_name in lit_unreachable:
                    # An abstract source actually failed. Name it.
                    if src_name not in unavailable:
                        unavailable.append(src_name)
```

- [ ] **Step 4: Run the whole file, then the service and fan-out contract tests that consume the fragment**

Run: `pytest -n 0 tests/unit/test_services/test_clinical_context/ -v`
Expected: all PASS. If `test_service.py` or `test_fan_out_contract.py` stub `_citations` with the old 4-tuple `(list, int, int, int)`, update the stub's second element to `[]` / `["europe_pmc"]` — the meaning is unchanged, the type is now a list of names.

- [ ] **Step 5: Commit**

```bash
git add src/services/clinical_context/causal_evidence.py tests/unit/test_services/test_clinical_context/test_causal_evidence.py
git commit -m "feat(clinical-context): re-verify through Crossref by DOI when Europe PMC holds no abstract

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 5: Wire UMLS into the default provider

**Files:**
- Modify: `src/services/clinical_context/causal_evidence.py` (`default_causal_evidence_provider`)
- Test: `tests/unit/test_services/test_clinical_context/test_causal_evidence.py` (append)

- [ ] **Step 1: Write the failing tests**

```python
@pytest.mark.unit
def test_default_provider_carries_a_umls_client_when_the_key_is_present(monkeypatch):
    monkeypatch.setenv("UMLS_UTS_API_KEY", "k")
    from src.services.clinical_context.causal_evidence import default_causal_evidence_provider

    provider = default_causal_evidence_provider()
    assert provider._umls is not None


@pytest.mark.unit
def test_default_provider_degrades_to_no_umls_without_the_key(monkeypatch):
    monkeypatch.delenv("UMLS_UTS_API_KEY", raising=False)
    from src.services.clinical_context.causal_evidence import default_causal_evidence_provider

    provider = default_causal_evidence_provider()
    assert provider._umls is None
```

- [ ] **Step 2: Run to verify they fail**

Run: `pytest -n 0 tests/unit/test_services/test_clinical_context/test_causal_evidence.py -k default_provider -v`
Expected: first FAILS (`AttributeError`/`None`), second PASSES.

- [ ] **Step 3: Implement**

```python
def default_causal_evidence_provider() -> CausalEvidenceProvider:
    """Build the real provider (lazy imports keep the module graph cheap and let the
    service import without the KG stack's optional auth)."""
    from src.data.kg.citation_resolver import CitationResolver
    from src.data.kg.europe_pmc import EuropePMCClient
    from src.data.kg.open_targets import OpenTargetsClient
    from src.data.kg.umls_uts import UMLSAuthError, UMLSClient
    from src.services.clinical_context.clients import PubMedClient

    umls: Optional[UMLSClient]
    try:
        umls = UMLSClient()
    except UMLSAuthError:
        logger.info("causal-evidence: no UMLS_UTS_API_KEY — verifying on names only.")
        umls = None
    return CausalEvidenceProvider(
        open_targets=OpenTargetsClient(timeout=_OPEN_TARGETS_TIMEOUT_S),
        pubmed=PubMedClient(),
        # A tighter Europe PMC timeout than the KG default: this path runs while a
        # user waits on a panel, not in a batch job. The resolver builds Crossref
        # itself and shares this UMLS client for the atom fan-out.
        resolver=CitationResolver(
            europe_pmc=EuropePMCClient(timeout=_EUROPE_PMC_TIMEOUT_S), umls=umls
        ),
        umls=umls,
    )
```

- [ ] **Step 4: Run the file**

Run: `pytest -n 0 tests/unit/test_services/test_clinical_context/test_causal_evidence.py -v`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add src/services/clinical_context/causal_evidence.py tests/unit/test_services/test_clinical_context/test_causal_evidence.py
git commit -m "feat(clinical-context): default evidence provider shares one UMLS client with the resolver

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 6: Live integration tests

**Files:**
- Modify: `tests/integration/test_kg/test_citation_resolver_live.py` (append)

- [ ] **Step 1: Append the two measured cases**

```python
def test_atoms_rescue_the_measured_fabhalta_pnh_abstract() -> None:
    """Measured 2026-09-22: PMID 40447351 names PNH, never the full American name.
    Names alone score 0.0; with the disease CUI (atoms include "PNH") it verifies."""
    with CitationResolver() as resolver:
        without = resolver.verify_citation(
            "40447351",
            subject_name="iptacopan",
            object_name="paroxysmal nocturnal hemoglobinuria",
        )
        with_cui = resolver.verify_citation(
            "40447351",
            subject_name="iptacopan",
            object_name="paroxysmal nocturnal hemoglobinuria",
            object_cui="C0024790",
        )
    assert without.abstract_resolved and with_cui.abstract_resolved
    assert without.overall_confidence == 0.0
    assert with_cui.overall_confidence >= 0.5, with_cui


def test_a_doi_with_a_deposited_abstract_verifies_through_crossref() -> None:
    """10.1186/s13058-023-01623-6 is the record the Crossref client was measured
    against (#1608): a JATS-structured abstract naming CRP."""
    with CitationResolver() as resolver:
        verdict = resolver.verify_citation(
            "10.1186/s13058-023-01623-6",
            identifier_kind="doi",
            subject_name="C-reactive protein",
            object_name="breast cancer",
        )
    assert verdict.abstract_resolved, verdict
    assert verdict.overall_confidence >= 0.5, verdict
```

- [ ] **Step 2: Run (the droplet has the key and network)**

Run: `pytest -n 0 tests/integration/test_kg/test_citation_resolver_live.py -v`
Expected: all PASS. If the Crossref test's object term does not appear in that abstract, read the live abstract once (`CrossrefClient().fetch_doi_metadata(...)`) and pick two terms that do — the test pins the mechanism, not that pair of words.

- [ ] **Step 3: Commit**

```bash
git add tests/integration/test_kg/test_citation_resolver_live.py
git commit -m "test(citation): live atoms rescue and Crossref-by-DOI cases

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 7: Gates, PR, review, merge, deploy, cert

- [ ] **Step 1: Lint whole-tree with CI's commands**

```bash
ruff check --no-cache src/ tests/ && ruff format --check --no-cache src/ tests/
```
Expected: clean. No mypy on the droplet; read CI's `mypy-report` artifact and diff the count against main's (the new `Protocol` kwargs and `Any` returns are the likely places for a new error — fix, do not bump the ceiling).

- [ ] **Step 2: Targeted regression**

```bash
pytest -n 0 tests/unit/test_data/test_kg tests/unit/test_services/test_clinical_context tests/unit/test_api/test_causal_clinical_context.py -q
```
Expected: all PASS.

- [ ] **Step 3: Push, PR, CI, codex review (brief includes the verbatim design-pushback paragraph), merge with `--merge`.** PR body states: the 2/47 measured rescue, why the payload count cannot change (cap of 2), the settled-absence-only fallback rule, and that `sources_unavailable` can now name `crossref`.

- [ ] **Step 4: Live cert after deploy** — before controls recorded 2026-09-22 (names-only verdict on 40447351 = 0.0; three brands return 200 with `citations` ≤ 2):

```bash
docker inspect -f '{{.State.StartedAt}}' e2i_api
docker exec -i -e PYTHONPATH=/app e2i_api python - <<'EOF' | tee docs/demos/results/2026-09-22_public_apis_live_path/lane3_umls_crossref_cert.txt
from src.data.kg.citation_resolver import CitationResolver
r = CitationResolver()
a = r.verify_citation("40447351", subject_name="iptacopan", object_name="paroxysmal nocturnal hemoglobinuria")
b = r.verify_citation("40447351", subject_name="iptacopan", object_name="paroxysmal nocturnal hemoglobinuria", object_cui="C0024790")
print("names only:", a.overall_confidence, a.entities_found)
print("with CUI  :", b.overall_confidence, b.entities_found)
from src.services.clinical_context.causal_evidence import default_causal_evidence_provider
from src.services.clinical_context.brand_map import BRAND_CLINICAL_MAP
p = default_causal_evidence_provider()
print("provider umls:", type(p._umls).__name__)
print("concept ids Fabhalta:", p._concept_ids(BRAND_CLINICAL_MAP["Fabhalta"]))
EOF
```
Then the HTTP route for all three brands (token via the Supabase password grant, path `/api/causal/clinical-context`): each returns 200, `causal_evidence.status == "evidence"`, `citations` length ≤ 2, and every citation `source` is `pubmed+europepmc` or `pubmed+crossref`. Expected cert lines: `names only: 0.0 ()`, `with CUI  : 0.5 ('iptacopan', 'PNH')` (or higher), `provider umls: UMLSClient`, `concept ids Fabhalta: ('C<...>', 'C0024790')`. Commit the cert file on a docs branch.
