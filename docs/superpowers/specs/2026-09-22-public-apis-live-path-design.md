# Public biomedical APIs into the live request path — design

**Date:** 2026-09-22
**Status:** approved by owner in session (three lanes, in this order)
**Evidence:** every claim below was measured inside the deployed `e2i_api`
container (same keys, same network) on 2026-09-22 unless marked otherwise.

## Why

Seven public APIs are implemented under `src/data/kg/` and
`src/services/clinical_context/clients.py`. Six are exercised by production
traffic through `GET /api/causal/clinical-context`, the chat
`clinical_context_tool`, the strategic-insight routes and the causal-impact
interpretation node. Three are implemented but never reached by a live
request:

| API | Implemented in | Reached today only by |
|---|---|---|
| RxNav | `src/data/kg/rxnav.py` | `EntityLinker` → `scripts/build_kg_cache.py` (offline) |
| UMLS UTS | `src/data/kg/umls_uts.py` | the offline cache build; the Layer‑4 pipeline path (dark, see lane 1) |
| Crossref | `src/data/kg/crossref.py` | `CitationResolver` only when `identifier_kind="doi"`; every live caller passes PMIDs |

Separately, the Layer‑4 causal-role classifier artifact
`artifacts/dspy/causal_role_classifier.json` is committed but absent from the
image. The artifact was absent only because no COPY existed; `.dockerignore`
never excluded it (a slash-less pattern like `*.json` matches the
build-context root only — measured on the production image 2026-09-22 via
`scripts/benchmarks/routing/data/agent_contracts.json`, present in that same
image under that same rule). So Layer 4 silently skips in production. That is
the #1607 / #600 shape already fixed once for the KG cache, minus the
`.dockerignore` part of that fix — this artifact never needed one.

The owner asked for all three APIs in the live request path and for the
Layer‑4 artifact to ship.

## Measured premises (cheapest disproof, run first)

- **RxNav.** Chat entity extraction uses a hand-curated alias table
  (`src/rag/entity_extractor.py`). Measured through the real extractor:
  `"iptacopan TRx in the northeast"` and `"Rhapsido NBRx last quarter"`
  resolve to **no brand**; `"ribociclib share trend"` resolves to Kisqali.
  RxNav knows all three pairs: ribociclib→Kisqali (RxCUI 1873916/1873984),
  iptacopan→Fabhalta (2671061/2671075), remibrutinib→Rhapsido (2724365/2724393).
- **UMLS.** `CitationResolver.verify_citation` already accepts
  `subject_cui`/`object_cui`, but the live caller
  (`causal_evidence.py`) passes names only. Over 47 candidate citations across
  the three brands, 3 fail verification today; UMLS atoms rescue **2**
  (Fabhalta PMIDs 40447351 and 40395624: abstracts say "PNH" and
  "haemoglobinuria"). The atoms endpoint
  (`/content/current/CUI/{cui}/atoms?language=ENG`) returned, for CSU
  C0578870: CIU, CSU, Chronic Idiopathic Urticaria, Chronic urticaria, …
- **Crossref.** Of the 47 candidates, one (PMID 39968604) has no Europe PMC
  abstract; PubMed esummary carries its DOI (10.1111/bjh.20024), Crossref has
  no abstract for it either. The fallback is correct and cheap but its measured
  yield is small. Crossref returns `None` **by design** when a publisher
  deposits no abstract (NEJM, Wiley) — not an outage.
- **Layer 4.** Inside the container `/app/artifacts` does not exist;
  `_try_load_layer_4_classifier()` therefore returns `None`. Both
  `ANTHROPIC_API_KEY` and `OPENAI_API_KEY` are set in the container. The
  production trigger is `execute_model_retraining` in
  `src/tasks/drift_monitoring_tasks.py` (runs `MLFoundationPipeline`), which
  the drift sweep evaluates every 6 hours.

## Lane 1 — ship the Layer‑4 classifier artifact

**Goal.** The compiled classifier becomes LOADABLE in the image —
`load_compiled_classifier` returns a classifier instead of raising / the
loader returning `None`. This does NOT activate Layer 4 and incurs no
classifier-call cost; see "What this does and does not change" below. No
runtime code changes.

**Changes.**
1. `docker/Dockerfile`: `COPY artifacts/dspy/causal_role_classifier.json ./artifacts/dspy/`
   in **both** the `development` and `production` stages, beside the existing
   `COPY data/kg_cache/` lines. The loader's `PROJECT_ROOT` resolves to `/app`
   in the image, so the path lands at `DEFAULT_ARTIFACT_PATH`.
2. `.dockerignore`: NOT NEEDED — premise disproved 2026-09-22. Measured on the
   production image: a slash-less pattern like `*.json` matches the
   build-context root only, so it never excluded the artifact; the only cause
   of the missing file was the absent `COPY` in item 1.
3. Guard test `tests/unit/test_data/test_causal_role_classifier_packaging.py`,
   mirroring `test_kg_cache_packaging.py`: the artifact at
   `DEFAULT_ARTIFACT_PATH` is committed, survives `.dockerignore` under
   last-match-wins, and is COPYed in both Dockerfile stages. Red first: the
   test fails on `main` today.
4. `.github/workflows/deploy.yml`: the artifact becomes an `on.push.paths`
   entry (#1783 guard).
5. `.github/workflows/deploy.yml`'s `build-and-push` job gains an in-image
   assertion step, right after "Build and push": pulls the exact digest just
   pushed and checks the baked inputs (this artifact, the KG caches) actually
   exist in the IMAGE, not just the Dockerfile text — the static packaging
   guards prove the text; this proves what the droplet will pull (#1607
   shape).

**What this does and does not change.** This lane makes the artifact
LOADABLE — `load_compiled_classifier(strict=True)` returns a classifier and
`_try_load_layer_4_classifier()` is no longer `None`. It does **not** activate
Layer 4 and incurs no classifier-call cost. Two gates still keep it dark:

- The LLM-call gate defaults OFF:
  `src/agents/ml_foundation/data_preparer/nodes/adaptive_validity_check.py:3776`
  — `layer4_enabled = bool(state.get("adaptive_layer4_enabled", False))`. Set
  deliberately by commit `d97f52dd5` (2026-05-27, "demote the LLM to
  audit-only in the voter + gate the call") after the classifier scored 0.633
  on hard roles (#242); `ADAPTIVE_LAYER4_LLM_DECIDES` is also OFF (audit-only
  even when the call gate is on). Guarded by
  `tests/unit/test_data_preparer/test_adaptive_validity_check_layer_4.py::test_layer4_llm_not_called_by_default`.
- There is no live consumer of the retrain path at all:
  `src/workers/celery_app.py:176` routes `execute_model_retraining` to the
  `ml` queue; `worker_heavy` (the only `ml`-queue consumer) ships at
  `replicas: 0` (`docker/docker-compose.yml:1174`) and is not running live
  (measured 2026-09-22: `docker ps --filter name=worker_heavy -q` → empty).

When it IS enabled, the trigger set is
`severity_pre_joint_check == "moderate"` OR (`== "high"` AND
`layer_1_declared_safe`) — not "ambiguous only". LLM calls run serially with
NO per-node call budget (`ADAPTIVE_CITATION_RESOLUTION_BUDGET` bounds
citation lookups only, not classifier calls); a provider failure returns
`None` and the retrain continues without Layer-4 evidence (fail-open).

Activating Layer 4 is a separate owner decision, not part of this lane. It
would need: (a) declaring and propagating `adaptive_layer4_enabled` from
`PipelineConfig` through `DataPreparerAgent` into `DataPreparerState`; (b)
scaling `worker_heavy` above zero (or routing the task to a served queue);
(c) a per-node LLM-call budget/timeout plus attempted/succeeded/skipped
telemetry; (d) a test that exercises the real retraining path, not just the
loader.

**Live cert.** (i) Before control (already recorded 2026-09-22): the file is
absent and the loader returns `None`. After deploy, inside `e2i_api`:
`load_compiled_classifier(strict=True)` returns a classifier and
`_try_load_layer_4_classifier()` is not `None`. Record container `StartedAt`.
(ii) Negative control — the call gate is still OFF:
`tests/unit/test_data_preparer/test_adaptive_validity_check_layer_4.py::test_layer4_llm_not_called_by_default`
passes, and `docker ps --filter name=worker_heavy -q` prints nothing
(`worker_heavy` still not running).

## Lane 2 — RxNav-backed brand aliases for chat entity extraction

**Goal.** A typed ingredient or marketed name resolves to the canonical brand
without a hand-maintained alias list, degrading to today's table when RxNav is
unavailable.

**Changes.**
1. `RxNavClient.related_names(rxcui, *, ttys=("IN", "BN", "PIN")) -> list[str]`
   over `/rxcui/{rxcui}/related.json?tty=…`, `lru_cache`d like its siblings,
   covered by `reset_caches()`.
2. New module `src/rag/brand_aliases.py`:
   `rxnav_brand_aliases(brands: Iterable[str], *, client=None) -> dict[str, list[str]]`.
   For each canonical brand: `rxcui_for_name(brand)` → `related_names` →
   lowercase, strip, dedupe, drop the brand's own name and any name shorter
   than 4 characters. Process-level cache (module dict + lock) with a
   24‑hour TTL on success and a 10‑minute negative TTL on failure. Client
   timeout 2 s. The first `RxNavError` in a round stops the round (no
   6 × 2 s worst case) and returns what was gathered.
3. `EntityVocabulary.from_default()` merges `curated ∪ rxnav` per brand.
   Conflict rule: an RxNav name already claimed by another canonical brand's
   curated list is dropped; curated always wins.

**Cost.** The extractor is built once per process (`get_rag_deps` caches it;
the explainer hook is lazy-once), so this is ≈ 6 calls of ≈ 0.1 s at first
use, then cached.

**Tests (red first).**
- Unit, fake client: `iptacopan`→Fabhalta, `rhapsido`→Remibrutinib,
  `ribociclib` still →Kisqali; client raising → vocabulary byte-identical to
  curated; negative TTL respected; conflict rule.
- Existing `test_no_brand_found` ("Show overall market trends") stays green.
- Integration (network-marked, skipped without outbound network): the three
  real pairs above.

**Live cert.** Through the deployed chat path or `EntityExtractor()` inside
`e2i_api`: the two queries that resolve to no brand today resolve to Fabhalta
and Remibrutinib.

## Lane 3 — UMLS synonyms and Crossref fallback in citation verification

**Goal.** Verification matches the disease and drug the way abstracts write
them (abbreviations, British spelling), and a PMID whose abstract Europe PMC
lacks is still verifiable through its DOI.

**Changes.**
1. `UMLSClient.atoms(cui, *, language="ENG", page_size=50) -> list[str]`:
   English atom names for a CUI, `lru_cache`d; 404 → `[]`.
2. `CitationResolver._candidate_terms`: primary name, then the CUI's preferred
   name, then its atoms; case-insensitive dedupe, drop terms shorter than 3
   characters, cap at 40 terms. `UMLSError` on the atoms call logs a warning
   and keeps the terms gathered so far (`UMLSAuthError` still raises
   `CitationResolverError`, as today). The existing word-boundary matcher is
   unchanged, so `CSU`/`PNH` match only as whole words.
3. `CausalEvidenceProvider`:
   - new optional `umls: _UMLSSearchLike | None` (protocol: `search(term) -> list[dict]`);
   - `_concept_ids(profile) -> tuple[str|None, str|None]` resolves
     `drug_name` and `disease_search_term` by exact search (first row's
     `ui`), cached per profile on the instance; any failure → `(None, None)`;
   - `verify_citation(..., subject_cui=…, object_cui=…)`; the
     `_CitationResolverLike` protocol gains the two optional kwargs;
   - **Crossref fallback**: when a PMID verdict is `abstract_resolved=False`
     with `error=None` (settled absence, not an outage), fetch
     `self._pubmed.fetch_by_pmid(pmid)`; if it carries a DOI, call
     `verify_citation(doi, identifier_kind="doi", …)`; a resolved verdict
     yields a `VerifiedCitation` with `source="pubmed+crossref"`, reusing the
     already-fetched summary for title/journal/pubdate. A raised
     `CrossrefError` counts as `unreachable`, mirroring Europe PMC. The
     existing wall-clock budget covers the extra calls.
   - `default_causal_evidence_provider()` passes `UMLSClient()` when the key
     is present (catch `UMLSAuthError` → `None`).
4. No API schema change: `VerifiedCitation.source` is already a free `str`
   (`src/api/schemas/causal.py:1144`), so the OpenAPI/`api.ts` byte-diff gate
   is untouched. The Layer‑4 pipeline path (`_resolve_citation_verdicts`)
   already passes CUIs and gains the synonym fan-out with no edit.

**Tests (red first).**
- Resolver units (stub UMLS with `atoms`): a "PNH"-only abstract verifies at
  ≥ 0.5 with CUIs and at 0.0 without; atoms failure tolerated; cap and
  short-term filter; auth failure still fatal.
- Provider units (fake resolver records kwargs): CUIs are passed through;
  CUI lookup failure still verifies by name; fallback fires only on
  `error=None`, never on an outage; `source` label; unreachable accounting.
- Integration (network-marked): Fabhalta PMID 40447351 with real CUIs
  verifies (measured rescue); DOI `10.1186/s13058-023-01623-6` (the record
  the Crossref client was measured against in #1608) resolves through Crossref.

**Live cert.** The payload cannot show the gain: `_MAX_CITATIONS` is 2 and
Fabhalta already fills both slots at confidence 1.0, and only
`_MAX_CANDIDATE_PMIDS` = 3 candidates are examined per query. So the cert is
measured one layer down, inside the deployed `e2i_api` container, through the
production provider: (a) `CitationResolver(...).verify_citation("40447351",
subject_name="iptacopan", subject_cui=<iptacopan CUI>,
object_name="paroxysmal nocturnal hemoglobinuria", object_cui="C0024790")`
resolves at ≥ 0.5 where the same call without CUIs scores 0.0 (before
control, recorded 2026-09-22); (b) `default_causal_evidence_provider()`
evidence for Fabhalta shows `source="pubmed+europepmc"` citations with CUIs
passed (asserted via a logging probe on the resolver kwargs); (c) the HTTP
route for all three brands still returns 200 with unchanged shape. A
`VerifiedCitation` with `source="pubmed+crossref"` is certified by the
integration test's DOI, not by the live brands, whose Europe PMC coverage is
complete today.

## Non-goals

- Exposing RxCUI/CUI identifiers in the API payload (no consumer asked).
- Changing the composed PubMed queries (Remibrutinib's empty real-world slot
  is honest sparsity: the curated query has zero PubMed hits).
- Switching the Layer‑4 model provider.
- Any change to `scripts/build_kg_cache.py` or the offline KG cache.

## Execution protocol

One worktree, one PR, one deploy per lane, in order 1 → 2 → 3. TDD red-first,
`pytest -n 0`, no mocks in production paths, codex review brief includes the
design-pushback paragraph, live cert with a before control recorded in
`docs/demos/results/2026-09-22_public_apis_live_path/`.
