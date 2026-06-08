# RAG Corpus-Substrate Owner Decisions + Phase-0 GO/NO-GO (audit F3)

**Date:** 2026-06-08
**Shard:** `00-premise-validation-spike.md` (GATE for `05-corpus-population.md`)
**Outcome:** ✅ **GO** — Phase 5 (full corpus population) is unlocked.

---

## The premise this gate validated

> *"Indexing a small REAL slice of the operational-analytics corpus into the
> table the chatbot's `hybrid_vector_search` already reads makes the live dense
> retrieval return TOPICALLY-RELEVANT rows (mentioning the queried brand/KPI/
> region), instead of the generic agent-internal `[PROC] composition_comp_…`
> bookkeeping rows that won all 3 audit queries at relevance ~0.3."*

**Validated faithfully** (live Supabase + prod OpenAI 1536-dim embeddings, no
mocks) via `tests/rag/test_corpus_substrate_relevance.py` (RED before indexing →
GREEN after) and a human-read probe across three query archetypes.

---

## F3 root cause — refined by live-DB inspection

The audit said "the vector/sparse/graph stores do not contain the operational-
analytics knowledge users ask about." The **precise** finding from live
inspection:

- The reachable dense corpus (`episodic_memories` + `procedural_memories`) is
  ~100% agent-internal lifecycle bookkeeping (`model_selection_completed`,
  `qc_report_completed`, `feature_analysis_completed`, …). Only **2** of the
  episodic rows carry a brand (both `causal_analysis_completed`, brand=`kisqali`,
  region=NULL, schematic text `hcp_engagement_level -> patient_conversion_rate`).
- **The operational KPI corpus DOES exist — as structured rows in the
  `business_metrics` fact table (4,667 rows)** — but it is **never indexed into
  the RAG retrieval substrate**. Columns: `metric_name` (TRx, NBRx,
  Conversion_Rate, ROI, Total Prescriptions, Market Share Percentage, HCP
  Conversion Rate, HCP Engagement Score, …), `brand` (Kisqali=1532, Fabhalta,
  Remibrutinib), `region` (northeast/south/midwest/west), real
  `value`/`target`/`achievement_rate`/`year_over_year_change`/`roi`.

So F3 is **not** "no such data exists" — it is "the real KPI fact table is not
embedded/indexed into the dense retrieval path." The fix (Phase 5) is to index
it. This is **wire + populate**, exactly as the master plan framed it.

---

## Decisions

### Decision A — corpus SOURCE: ✅ `business_metrics` (real KPI fact table)
Real, verifiable, no fabrication. The spike renders each REAL row as analytic
prose with values taken **verbatim** from the fact table (F3 anti-mocking: no
invented KPI numbers). Recommended for Phase 5: start with `business_metrics`
KPI snapshots; expand to real causal-finding prose (the 2 `causal_impact` rows
+ `causal_paths`) and commercial-context narrative as those become available.

### Decision B — target table + chatbot read path: ✅ B2 (`episodic_memories`)
The spike auto-embeds into `episodic_memories` via the existing
`insert_episodic_memory_with_text` path — which `hybrid_vector_search` already
reads — so **no RPC migration** is required to prove the premise. Phase 5 may
keep B2 (simplest) or move to B1 (`rag_document_chunks` + a `041` RPC-widening
migration) if the owner wants the corpus isolated from agent bookkeeping in its
own table.

### Decision C — embedding provider faithfulness: ✅ OpenAI 1536-dim
Confirmed `OPENAI_API_KEY` present; the spike embedded via the prod provider
(`get_embedding_service()` → 1536-dim). A fallback-384 embed would be a FALSE
GREEN and `vector(1536)` would reject it.

### `memory_event_type` enum: reuse `'system_event'` (valid) for the spike
A dedicated `'kpi_snapshot'` value is OPTIONAL for Phase 5 (additive enum
migration, precedent `039`) if the owner wants corpus rows distinguishable by
`event_type` for analytics.

### `e2i_agent_name` enum: ⚠️ caught live — `agent_name` is an ENUM
The faithful insert proved `agent_name` is the `e2i_agent_name` enum (an
arbitrary `'phase0_corpus_spike'` raises Postgres `22P02`). The spike uses the
valid value `'observability_connector'` (closest semantic fit for operational
KPI facts). **Phase 5 should add a dedicated `'corpus_ingestion'` value via an
additive enum migration** (precedent `029`/`039`) for clean attribution.

### F2 secondary — `agent_activities` full-text: leave as-is (DROPPED)
Verified: `agent_activities` has no dedicated free-text narrative column. The
real sparse fix is routing real commercial prose into a weight-A free-text
column (`triggers.trigger_reason`) or `rag_document_chunks.content` (B1) — not
redefining `agent_activities.search_vector`.

---

## GO/NO-GO evidence (faithful, live)

| Query archetype | Top hit (real `business_metrics` row, rendered) | cosine |
|---|---|---|
| KPI (`TRx trend for Kisqali in the Northeast`) | `Market Share Percentage for Kisqali in the northeast …` / `Total Prescriptions for Kisqali in the northeast …` | 0.874 / 0.864 |
| context (`Market share for Fabhalta in the south`) | `Market Share Percentage for Fabhalta in the south …` | 0.921 |
| explanation (`HCP engagement score for Kisqali`) | `HCP Engagement Score (0-10) for Kisqali in the northeast …` | 0.939 |

All far above the **0.5 effective cosine floor** (`011_…sql:126` SQL floor +
`memory_connector.py:152` Python post-filter). Before indexing, the same gate
returned only generic `[PROC] composition_comp_…` rows (RED). **Decision: GO.**

Spike slice: 20 REAL rows (Kisqali + Fabhalta × northeast + south × 5 KPIs),
`session_id=742c70e0-ee29-4956-a097-70ea095bb0db`, `agent_name=observability_connector`,
removable by that session id if the owner picks a different target table.

---

## What Phase 5 should do (now unlocked)

1. Build a durable `src/rag/corpus_ingestion.py` that reads `business_metrics`
   (and, as available, real causal-finding prose) and indexes via the proven
   auto-embed path — replacing the one-off spike script.
2. (Optional) additive migrations: `'kpi_snapshot'` event_type and/or
   `'corpus_ingestion'` agent_name; or B1's `041` RPC-widening if the owner
   chooses `rag_document_chunks`.
3. Multi-archetype relevance gate across the full corpus (KPI / explanation /
   context), same `E2I_RUN_LIVE_RAG=1` droplet-only manual gate.
4. Owner runs the FULL real-corpus ingestion on the droplet (prod embedder).
