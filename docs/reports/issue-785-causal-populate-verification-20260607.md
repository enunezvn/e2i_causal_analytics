# #785 — Tier-1 causal_impact populate verification (e2i_causal CausalPath + episodic)

**Date:** 2026-06-07
**Branch:** `fix/788-causal-impact-episodic-embedding`
**Deploy:** held. **No fabricated nodes** — every node/edge/row below comes from real agent
processing (real DoWhy estimation, real refutation gate, real OpenAI 1536-dim embedding).

## Objective (#785)

Run a real Tier-1+ `causal_impact` pass and verify that `e2i_causal` grows with real
`CausalPath` (Variable + `CAUSES`) + relationships **and** `episodic_memories > 0` (with a
non-null 1536-dim vector). Record counts. Memory-monitored (the droplet is the dev+prod
box; deploy held).

## What unblocked it

`#785` was blocked by the **same** write-path drift `#788` repairs (the two were filed
together). The canonical populate path is `causal_impact.contribute_to_memory` →
`store_causal_analysis` (episodic) + `store_causal_path` (semantic CausalPath). Before the
fix it failed silently at:

- the `memory_event_type` enum (`causal_analysis_completed` absent → 22P02, swallowed) —
  **migration 040** (additive `ADD VALUE`);
- the `memory_outcome_type` enum (`store_causal_analysis` used the non-enum literal
  `causal_analysis_delivered` → 22P02, swallowed) — fixed to a valid enum
  (`success` / `partial_success`);
- and `causal_impact.run()` never called `contribute_to_memory` at all — now wired
  (gated by `enable_memory`, mirrors heterogeneous_optimizer).

(`#784` had already un-broken the `add_e2i_entity` / `add_relationship` semantic shims that
`store_causal_path` depends on.)

## Faithful run

`scripts/run_785_causal_populate.py` run via `dotenv -f .env run` (real `.env` →
real OpenAI embeddings + real local Supabase + real FalkorDB `e2i_causal`).

Real `causal_impact.run(enable_memory=True)` on the canonical query
("What drove Kisqali conversion in the Northeast?",
`hcp_engagement_level → patient_conversion_rate`, confounders
`[geographic_region, hcp_specialty]`):

```
status=completed  ate=0.4130  confidence=0.90  refutation_passed=True  gate_decision=proceed
```

A real, PROCEED-validated estimate (real DoWhy OLS + real refutation suite).

## Counts (delta from real processing)

| store | baseline | after | delta |
|---|---|---|---|
| episodic `causal_analysis_completed` | 0 | 1 | **+1** |
| ↳ with non-null 1536-dim vector | 0 | 1 | **+1** |
| `e2i_causal` Variable nodes | 1 | 3 | **+2** |
| `e2i_causal` CAUSES edges | 6 | 7 | **+1** |
| `e2i_causal` total nodes / edges | 188 / 104 | 190 / 105 | +2 / +1 |

A second run wrote a second episodic `causal_analysis_completed` row (now 2 total, both
1536-dim); the CausalPath nodes/edges are upserted by `var:<name>`, so re-running the same
pair updates rather than duplicates the edge.

## Durable read-back (proof, not projection)

Episodic vectors (Postgres `vector_dims`):

```
event_type=causal_analysis_completed  outcome_type=success  dims=1536  importance=0.85   (x2)
```

CausalPath edge (FalkorDB `e2i_causal`):

```
(hcp_engagement_level) -[CAUSES { ate_estimate=0.413, confidence=0.9,
                                  refutation_passed=true, brand='kisqali' }]-> (patient_conversion_rate)
```

## "At scale" — honest scope note

The `data_source='synthetic'` fixture (`estimation.py`, seeded `np.random.seed(42)`
HCP/conversion data) models **one** real causal relationship. The canonical pair
PROCEED-validates; arbitrary other treatment→outcome pairs correctly **fail refutation**
(the H2 gate fail-closes, and `contribute_to_memory` skips failed analyses by design — **no
node is fabricated**). A 4-pair batch confirmed this: 1 proceed-validated (populated), 3
honestly failed (nothing written). Genuine multi-path at-scale growth requires **real
multi-relationship data** fed via `state['data_cache']['estimation_data']` (the real
patient_journeys / `tool_composer` feed). That path is unblocked by this fix + `#784`; the
mechanism is proven here end-to-end.

## Memory

Single-process, sequential. Available RAM stayed ≈6.0–6.8 GiB throughout (no OOM); each
real DoWhy pass is bounded.

## Conclusion

**#785 verified.** A real Tier-1 `causal_impact` pass grows `e2i_causal` with a real,
branded, validated `CausalPath` and writes an `episodic_memories` row with a real 1536-dim
vector and zero swallowed errors. The populate mechanism (`#784` semantic wiring + `#788`
episodic/outcome/enum repair + run() wiring) is proven on real agent processing.
