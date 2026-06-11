# Audit — Cognitive RAG Hybrid Search Mechanics

**Date:** 2026-06-08
**Auditor:** Claude (Opus 4.8), faithful runtime probing + 11-agent adversarial verification
**Environment:** prod droplet `e2i-analytics-prod` — live Supabase pgvector (:5433/:6543), FalkorDB (:6381), real Claude (`anthropic/claude-sonnet-4-20250514`) via DSPy. **No mocks.**
**Scope:** the 4‑step cognitive RAG pipeline — DSPy query rewrite → parallel hybrid search across 3 backends → Reciprocal Rank Fusion → per‑result evidence scoring — the primary data‑retrieval mechanism for KPI, explanation, and context queries.
**Disposition:** audit only. No code changed, no deploy, no DDL. Repro + raw evidence in `docs/reports/rag-hybrid-audit-20260608-repro/`.

---

## 1. Executive summary

The pipeline is **wired and runs green**, but in production it does **not** behave like the 4‑step hybrid system it advertises. Faithful runs against the live backends show that, for real KPI/explanation/context queries:

- the **graph backend never fires** (mis‑wiring),
- the **sparse/full‑text backend returns nothing** (the corpus it indexes has none of the queried vocabulary),
- the **dense backend is the only contributor**, and it searches an **agent‑internal bookkeeping corpus** that contains no KPI/commercial/explanatory answers,
- so **RRF degenerates to an identity re‑ranking of a single list** — the "hybrid + fusion + graph‑boost" machinery is inert,
- the run costs **~17–18 s** (vs **1.3 s** for the non‑LLM fallback) because of **5 sequential** evidence‑scoring LLM calls, and
- the typical result — **one generic agent‑memory row** — is then handed to a DSPy synthesizer that has **no grounding guard**, creating a **fabrication risk** for KPI/causal answers.

**Intent vs reality.** The code's own contract (`causal_rag.py:28-30`) is *"Only indexes operational data… NEVER clinical trials/literature/regulatory."* The pipeline was meant to retrieve **operational analytics knowledge** (KPIs, causal paths, triggers, agent activity). The reality is that the only indexed-and-reachable substrate is **agent self-tracking + synthetic seed data**, which contains none of the brand/KPI/region vocabulary users ask about. **The dominant problem is the data substrate, not the fusion algorithm.** The retrieval/fusion code is mostly correct; the fixes are **wire + populate**, not rewrite.

**Verification.** Each headline finding below was put to an adversarial agent instructed to *refute* it against the real code + SQL + probe traces. 5 of 6 confirmed; 1 (sparse) returned **partial** — the *symptom* (sparse → 0) is real and user‑visible, but my first‑pass *mechanism* ("tsvector indexes the wrong tokens") was overstated and is corrected below.

---

## 2. The pipeline as built — exact production path

The live chatbot retrieval node is:

```
CopilotKit chat endpoint (/api/copilotkit, main.py:1070-1073)
  → run_chatbot / stream_chatbot (chatbot_graph.py:1941, 2127)
    → compiled e2i_chatbot_graph
      → retrieve_rag_node (chatbot_graph.py:524)
        → cognitive_rag_retrieve (chatbot_dspy.py:1491)   ← THE 4-STEP PIPELINE
           step 1  rewrite_query_dspy        (chatbot_dspy.py:1518)
           step 2  hybrid_search             (retriever.py:386 → HybridRetriever.search:245)
           step 3  _reciprocal_rank_fusion   (retriever.py:325)   [inside hybrid_search]
           step 4  score_evidence_dspy × N   (chatbot_dspy.py:1541, sequential loop)
        └ on exception → _execute_basic_rag (chatbot_graph.py:620, no LLM)
```

Default‑enabled: `CHATBOT_COGNITIVE_RAG=true` (`chatbot_dspy.py:1015`). The basic path is only a fallback.

> Note there are **three** different "HybridRetriever"/cognitive-RAG implementations in the tree (`retriever.py`, `hybrid_retriever.py`, `cognitive_rag_dspy.py`) plus a dead duplicate (`e2i_cognitive_rag_dspy.py`). The chatbot uses `retriever.py`. The others are audited in §6.

---

## 3. Faithful evidence (what actually happened)

Three representative queries (KPI / explanation / context) through the **real** `cognitive_rag_retrieve`, plus a control:

| query | rewrite | backends fired | DENSE n | SPARSE n | GRAPH | scoring calls | evidence kept | latency |
|---|---|---|---|---|---|---|---|---|
| "TRx trend for Kisqali in Northeast" | dspy | DENSE, SPARSE | 6 | **0** | **never** | 5 sequential | **1** | **17.4 s** |
| "Why did Kisqali adoption increase…" | dspy | DENSE, SPARSE | 6 | **0** | **never** | 5 sequential | **1** | **17.3 s** |
| "Summarize recent activity for Fabhalta" | dspy | DENSE, SPARSE | 6 | **0** | **never** | 5 sequential | **1** | **18.2 s** |
| **control:** basic `hybrid_search(kpi_name='trx')` | — | DENSE, SPARSE, **GRAPH** | 6 | 0 | **fired** | 0 | 5 returned | **1.3 s** |

Latency split (cognitive): rewrite ≈ 6.6–7.2 s · retrieval ≈ 0.3–2.7 s · **evidence scoring ≈ 14.7–17.9 s (5 × ~3 s, serial)**.

The **same generic row won all three distinct queries**: `[PROC] composition_comp_571cb03a: Needs causal then regional analysis` (relevance = **exactly 0.3**, RRF ≈ 0.0082). The top dense hits for a *TRx/Kisqali* query were `Causal analysis: hcp_engagement_level→patient_conversion_rate ATE=0.413`, `[PROC] composition_comp_…`, `Feature Analysis: tier0_e2e_… Selected 0 features`, `QC Report: passed` — **none about TRx, Kisqali, or the Northeast**.

Corpus the pipeline can reach: `episodic_memories`=72, `procedural_memories`=1322 (dense); `agent_activities`=2323, `triggers`=4356, `causal_paths`=50 (sparse/graph). Raw `ILIKE '%kisqali%'/'%trx%'/'%adoption%'` across the indexed columns = **0 rows**.

Full traces: `docs/reports/rag-hybrid-audit-20260608-repro/EVIDENCE.md` and the two `probe_*.py` scripts.

---

## 4. Findings (verified)

Severity reflects impact on the pipeline's stated purpose (serving grounded KPI/explanation/context answers).

### F1 — The graph backend never fires in the cognitive path · **HIGH** · *confirmed*

`cognitive_rag_retrieve` extracts `graph_entities` during the rewrite (`chatbot_dspy.py:1518`) but its `hybrid_search` call **hardcodes `kpi_name=None` and passes no `entities=` kwarg** (`chatbot_dspy.py:1533-1538`, hardcode at line **1536**). In `HybridRetriever.search`, the graph leg runs only `if entities:` or `elif kpi_name:` (`retriever.py:306-309`); with both empty, `graph_results=[]`. The extracted `graph_entities` are placed into the result/training signal (`:1584/:1595`) but **never into the search**.

Ironic consequence: the **basic fallback is stronger on graph** — it derives `kpi_name` from the query and passes it (`chatbot_graph.py:628-640`), which is why the control fired GRAPH. *Either* forwarding `graph_entities` as `entities=` *or* the `kpi_name` would light the graph leg.

### F2 — Sparse/full-text returns nothing for commercial queries · **HIGH** · *partial (symptom confirmed, mechanism corrected)*

**Symptom (confirmed, user-visible):** `hybrid_fulltext_search` returns **0 rows for every commercial term** — `kisqali`, `trx`, `northeast`, `adoption`, `agent` — across 6,729 indexed rows. So the sparse leg contributes nothing to real queries (EVIDENCE probe 1: SPARSE n=0 every call).

**Correction to my first-pass mechanism.** My initial claim — *"the `search_vector` tsvector indexes agent category tokens, not free text"* — is **overstated**. The adversarial check proved the RPC and tsvectors are **functioning correctly**:
- `triggers.trigger_reason` (weight A) and `causal_paths` node/chain columns **do** index real free text and **are** matchable: `@@ 'adherence'` → 1420, `'engagement'` → 2013, `'competitor'` → 841, `'patient'` → 2576; the RPC returns 50 rows for `'patient'`/`'adherence'`.
- The category-token characterization is accurate **only for `agent_activities`**, whose weight-A column is `agent_name` (e.g. `causal_impact`, `gap_analyzer`) — a genuine but *narrower* coverage weakness.

**True root cause:** a **corpus/vocabulary mismatch**. The commercial terms simply are not in the corpus — raw `ILIKE` for `kisqali`/`adoption` = 0 in every indexed column. `triggers` are generic synthetic personas (*"Competitor activity detected in territory"* ×841, *"Declining engagement pattern…non-adherence"* ×1235); `causal_paths` are schematic node names. The full-text engine is healthy; it is querying a corpus that **lacks the queried vocabulary**. This reinforces F3.

> Source columns (`database/memory/011_hybrid_search_functions_fixed.sql:25-50`, GENERATED…STORED, live def matches): `causal_paths{start_node,end_node,method_used,causal_chain}`, `agent_activities{agent_name,activity_type,analysis_results}`, `triggers{trigger_reason,trigger_type,recommended_action}`. RPC at `022_hybrid_search_max_staleness.sql:31-146`.

### F3 — The reachable corpus is the wrong *kind* of data · **HIGH** · *confirmed*

The dense RPC `hybrid_vector_search` reads **only** `episodic_memories` + `procedural_memories` (`011_…sql:117,144`; `022` leaves it "structurally unchanged"; live def == migration). Their content is **100% agent-internal lifecycle/bookkeeping**: episodic = `model_selection_completed`(60), `model_training_completed`(6), QC/feature/scope `*_completed`; procedural = 1315 HPO `optimization` patterns + 7 tool-composition. `ILIKE trx/kisqali/market-share` = **0 rows** in both.

No KPI/commercial corpus is indexed for vector search **anywhere**: the alternate `rag_document_chunks`, `experiment_knowledge_store`, `composer_episodes` tables all have **0 rows / 0 embeddings**. So even a perfectly-wired pipeline returns structurally irrelevant evidence for KPI/explanation/context queries. **This is the deepest finding: the substrate, not the algorithm.**

### F4 — "Hybrid across 3 backends + RRF" degenerates to single-backend dense · **HIGH** · *confirmed*

Synthesis of F1–F3 for the live default path on real queries: graph empty (F1 wiring), sparse empty (F2 corpus), dense the only non-empty list. `_reciprocal_rank_fusion` then scores each item `weight/(60+rank)` (`retriever.py:325-378`); with one non-empty list this is a **strictly monotonic transform of the dense rank → identity re-ordering**. No item appears in two lists, so RRF's cross-list reinforcement (its entire value) **cannot occur**; the 0.3 sparse / 0.2 graph weights multiply empty lists. The 1.3× graph boost is inert. *Scoping:* the code is capable of 3-leg fusion (the basic fallback fires graph for literal-KPI queries; sparse returns rows for agent-internal tokens) — the degeneration is a property of the **wiring + corpus for real pharma queries on the default path**.

### F5 — ~17–18 s latency from 5 sequential evidence-scoring LLM calls · **MEDIUM** · *confirmed*

`cognitive_rag_retrieve` scores each retrieved row with a **separate, awaited** Claude call inside a `for` loop (`chatbot_dspy.py:1541-1546`); there is **no `asyncio.gather` anywhere** on this path. The calls are independent (same goal, different `evidence_item`) — trivially parallelizable. With `k=5` the loop is 5 serial ~3 s calls ≈ 15 s, dominating the ~17 s total. The non-LLM basic fallback does the same retrieval in **1.3 s**. So the "cognitive" enhancement is **~13× slower** and (given F1–F4) returns **5× fewer, less relevant** results than its own fallback.

### F6 — Multi-hop is dead code in this path · **LOW** · *confirmed*

`enable_multi_hop` is declared (`:1497`) and mentioned only in the docstring (`:1511`) — **never referenced in control flow**. `hop_count` is hardcoded to `1` (`:1525`). `ChatbotHopDecider`/`HopDecisionSignature` are defined (`:1057,:1114`) but **never instantiated anywhere in the repo** (grep clean across `src/` + `tests/`). The live caller also passes `enable_multi_hop=False` (`chatbot_graph.py:582`). Vestigial.

### F7 — Evidence scoring silently falls back to 0.5 if DSPy LM unconfigured · **LOW** · *observed*

`score_evidence_dspy` does **not** call `_ensure_dspy_configured()` (unlike `rewrite_query_dspy`); it relies on a prior rewrite having configured the LM in-process. If ever invoked without that, every item scores the `0.5` exception fallback (`chatbot_dspy.py:1473-1475`) — reproduced directly in probe 2. Latent, not currently harmful in the live path (rewrite always runs first).

---

## 5. Downstream consumption — fabrication risk · **HIGH** · *confirmed (C5)*

What happens to the typical "1 generic, irrelevant row" is the most user-facing risk.

- **No grounding guard in the live synthesis path.** `EvidenceSynthesisSignature` (`chatbot_dspy.py:1614-1658`) carries one *soft prose* instruction ("grounded in the retrieved evidence", `:1623`) and **zero hard constraints** — no abstention field, no refuse-on-insufficient-evidence rule, no post-hoc check that claims are supported. The model is free to write a fluent TRx-trend / causal-driver narrative not present in the evidence it was handed.
- **The honest no-evidence branch is bypassed in prod.** "I don't have specific data to answer…" / low-confidence labeling lives **only** in `synthesize_response_hardcoded` (`:1815-1851`), which is skipped whenever DSPy synthesis is enabled (default). The DSPy path instead passes the literal string `"No evidence retrieved."` and still asks for a "well-structured response" (`:1955`) — it does **not** fail closed.
- **The threshold boundary admits the junk.** Retrieval keeps evidence at `score >= 0.3` (`:1550`); synthesis is skipped only if `avg_evidence_score < 0.3` (`chatbot_graph.py:1143`). The generic row scored **exactly 0.3** for all 3 queries, so `0.3 < 0.3` is false → **synthesis runs on a single irrelevant procedural row** for a KPI question. The keep and skip thresholds are the same value — the typical case lands on the worst boundary.
- **Confidence and citations are self-asserted.** Confidence is parsed by substring-matching the LLM's own free-text statement (`:1982-1988`) with no cross-check against `avg_evidence_score`(0.3) or evidence count(1) — a fabrication can be badged "High confidence." `evidence_citations` are LLM free-text split on commas (`:1965-1967`), **never validated** against actual `source_id`s.

This is the anti-mocking concern in user-facing form: **plausible-but-ungrounded KPI/causal answers, surfaced with confidence badges and invented citations.** (`had_hallucination` exists but is a *training* reward, human-supplied, never gating live output — `:1727-1729,2441-2453`.)

---

## 6. Sibling retrieval variants (coverage)

### C1 — `/api/v1/rag` REST router is dead-on-construction · **HIGH (new)** · *partially-functional*
`src/rag/hybrid_retriever.py` is the *better-engineered* HybridRetriever (parallel 3-backend, config-driven weighted RRF, 1.3× graph boost, dedup-by-`id` with metadata merge — all **correct**). It is reached two ways:
- **Orchestrator path (works):** `dependencies/rag.py:53-58` constructs it with correct kwargs; `RAGContextNode` fires all 3 backends; reachable via the live `/api/cognitive` orchestrator (`cognitive.py:443`).
- **REST path (broken):** `RAGService.retriever` calls `HybridRetriever(self.config)` (`rag.py:309`) but the constructor requires `supabase_client` **and** `falkordb_client` (`hybrid_retriever.py:87`) → **TypeError before any search**. The whole `/api/v1/rag/*` router (mounted `main.py:1025`) **500s on every call**: `/search`, `/graph/{entity}`, `/causal-path`. Plus `get_last_query_stats()` (no such method; it's the `last_search_stats` property) and wrong kwargs in `get_causal_subgraph`/`get_causal_path`. All three bugs are **masked by `# type: ignore`** comments. This path also shares the broken substrate: `rag_document_chunks` is **empty**, `rag_fulltext_search` hits the same agent-token tsvectors, graph fires against the 50-row `causal_paths`.

### C2 — `/api/cognitive/rag` 4-phase sequential path · **MEDIUM (new)** · *partially-functional*
Mounted, auth-gated, no feature flag (`cognitive.py:976`, `main.py:1019`). Genuinely **sequential multi-hop** (one backend per hop), and unlike the chatbot path its **graph leg IS wired** (`cognitive_rag_dspy.py:374-393`). **Crash risk:** `MemoryType(decision.next_memory)` (`:353`) raises `ValueError` on any LLM output outside the enum (only `STOP` is guarded, `:334`) — swallowed by `cognitive_search`'s try/except into a degraded dict rather than a clean 500, so it **silently truncates investigations**. Same impoverished corpus; docstring claims "<2s" but the analogous machinery measures 17–18 s. **Not exercised by my probes** (they hit the chatbot path) — verdict is code-inferred + shared-corpus facts.

### C3 — `src/rag/e2i_cognitive_rag_dspy.py` is dead duplicate · **LOW (new)** · *dead-code*
A stale **pre-GEPA fork** of `cognitive_rag_dspy.py` (same origin commit, 0 GEPA refs vs 37 in the live file). **Zero importers** in `src/`/`tests/`; `__init__.py` exports only the live module. DELETE candidate (vestigial, superseded) — but it's named in an anti-resurrection test docstring (`test_no_orphan_cognitive_workflow.py:14`), so confirm with the owner before removal. Risk: two identically-named symbol sets invite an accidental wrong-module import.

### C4 — RRF correctness (`retriever.py`) · **MEDIUM** · *partially-functional*
- **Dedup by `source_id`, not content** (`:350`): two rows with identical content but different ids both survive and both accrue RRF mass (confirmed in DB — duplicate `Causal analysis…ATE=0.413` under 2 distinct `memory_id`s). The inline comment frames this as intentional "boosting"; it's a dedup **gap** that wastes top-k slots.
- **Surfaced `score` is the raw RRF value ~0.0082** (`:368`), an uncalibrated near-zero number, not relevance. *Mitigating:* gating/ordering uses the independent DSPy `relevance_score` (`:1550`), so the RRF score is a **display/payload artifact** — harmless to ranking but misleading to any consumer that reads it as relevance. No `le=1` crash (max ≈ 0.0164).
- **`"Normalized 0-1" contract not honored`** (`types.py:48`, `retrieval_models.py:16` declare `le=1`) — neither RRF normalizes; only the tiny magnitudes keep the model from rejecting.
- **Two divergent RRF impls** (`retriever.py` weights 0.5/0.3/0.2, dedup by `source_id` vs `hybrid_retriever.py` weights 0.4/0.2/0.4, dedup by `id`, graph boost) — fixes to one don't reach the other.

---

## 7. Severity-ranked summary

| # | Finding | Sev | Status |
|---|---|---|---|
| F3 | Reachable corpus is agent-internal/synthetic — no KPI/commercial/explanatory content indexed anywhere | HIGH | confirmed |
| F4 | Hybrid+RRF degenerates to single-backend dense for real queries | HIGH | confirmed |
| F1 | Graph backend never fires in cognitive path (mis-wiring; basic fallback is stronger) | HIGH | confirmed |
| F2 | Sparse leg returns 0 for all commercial queries (corpus vocabulary mismatch; `agent_activities` also under-indexed) | HIGH | partial |
| C5 | Downstream synthesis has no grounding guard → fabrication risk + self-asserted confidence/citations | HIGH | confirmed |
| C1 | `/api/v1/rag/*` REST router 500s on every call (constructor mis-call, masked by `type: ignore`) | HIGH | confirmed |
| F5 | ~17–18 s latency from 5 sequential scoring LLM calls (13× slower than fallback) | MED | confirmed |
| C2 | `/api/cognitive/rag` `MemoryType()` ValueError silently truncates hops; "<2s" docstring false | MED | confirmed |
| C4 | RRF dedups by id not content; surfaced score uncalibrated; "0-1" contract unmet; 2 divergent impls | MED | confirmed |
| F6 | Multi-hop dead code (`enable_multi_hop` ignored, decider never instantiated) | LOW | confirmed |
| F7 | `score_evidence_dspy` silently 0.5-fallbacks if DSPy LM unconfigured | LOW | observed |
| C3 | `e2i_cognitive_rag_dspy.py` dead duplicate (pre-GEPA fork) | LOW | confirmed |

---

## 8. Recommendations (cheapest-disproof first; wire/populate, not rewrite)

The fusion/retrieval code is largely correct. Ordered by leverage:

1. **Validate the premise before any pipeline work (cheapest disproof).** The single assumption everything rests on is *"the vector/sparse/graph stores contain operational analytics knowledge users ask about."* It is **false today** (F3). Before tuning rewrite/RRF/scoring, decide what corpus *should* be indexed (KPI snapshots, causal findings, commercial context) and **index a small real slice**, then re-run `probe_pipeline.py`. If retrieval is still irrelevant, the problem is upstream of this pipeline.
2. **Fix F1 graph wiring (one line):** forward the rewrite's `graph_entities` as `entities=` (and/or derive `kpi_name`) into `hybrid_search` at `chatbot_dspy.py:1536`. Re-run the probe; expect GRAPH to fire as the control did. Low risk, immediately restores 1 of 3 legs.
3. **Fix C1 (REST router):** correct `HybridRetriever(self.config)` → pass `supabase_client`/`falkordb_client`; fix `get_last_query_stats`→`last_search_stats` and the subgraph/path kwargs. Remove the `# type: ignore`s that masked these. Add a smoke test hitting `/api/v1/rag/search` (it currently 500s).
4. **Close the C5 fabrication gap:** make the no-evidence/low-confidence honest branch run in the **DSPy** path too (abstain or hedge when `avg_evidence_score`≤0.3 or evidence count is 1); validate `evidence_citations` against supplied `source_id`s; stop labeling confidence purely from the LLM's own prose. Raise the keep threshold above the skip threshold so the boundary case doesn't auto-synthesize.
5. **Fix F5 latency:** wrap the per-result scoring loop in `asyncio.gather` (independent calls). Expect ~15 s → ~3 s. Or skip LLM scoring when only dense fired (use the cosine score).
6. **F2 sparse:** once a real corpus exists, confirm the indexed columns carry brand/KPI/region text; widen `agent_activities.search_vector` beyond `agent_name`/`activity_type` if that table is meant to be searchable.
7. **Hygiene:** delete the dead duplicate (C3, after owner OK), reconcile the two RRF impls (C4), fix dedup to consider content, and either implement or remove `enable_multi_hop`/`ChatbotHopDecider` (F6).

---

## 9. Reproduction

```
# Faithful pipeline probe (rewrite → hybrid → RRF → scoring, instrumented):
PYTHONPATH=. .venv/bin/dotenv run -- .venv/bin/python \
  docs/reports/rag-hybrid-audit-20260608-repro/probe_pipeline.py

# Corpus / sparse / scoring probe:
PYTHONPATH=. .venv/bin/dotenv run -- .venv/bin/python \
  docs/reports/rag-hybrid-audit-20260608-repro/probe_corpus_sparse_scoring.py
```

- `docs/reports/rag-hybrid-audit-20260608-repro/EVIDENCE.md` — captured raw traces + in-DB SPARSE root-cause checks.
- `probe_pipeline.py`, `probe_corpus_sparse_scoring.py` — the instrumented faithful probes (no mocks).

**Method note:** every headline finding was adversarially re-checked by an independent agent instructed to *refute* it against the real code, the SQL migrations, and the probe traces. F2 was downgraded to *partial* by that process and corrected above — the symptom holds, the mechanism was sharpened to a corpus/vocabulary mismatch.
