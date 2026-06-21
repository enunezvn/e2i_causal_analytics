# Causal Page Redesign — Design Spec

- **Date:** 2026-06-19
- **Status:** DRAFT — awaiting user review
- **Branch (planned):** `feat/causal-unified-agent-led` (build phases in isolated worktrees)
- **Author:** Claude (with E. Nuñez)

## 1. Context & problem

The platform ships **two** causal pages that are now redundant and weak:

- `/causal-discovery` — picks a brand, auto-runs a leaderboard of validated effects (questions enumerated from a **hand-curated allowlist** cross-product), drill-down per row.
- `/causal-analysis` — an **empty form**: the user picks treatment + outcome + estimator + brand, runs one deep analysis.

User verdict (verbatim intent): the pages are redundant; keep one; **stop asking the user to choose treatment/outcome — "asking the user about the data the agent knows best is unproductive"**; the variable choices are "meager"; *"the agent has access to a rich gold-standard dataset and should use all of it with rigor and knowledge users may not have."*

The detail/drill-down view is literally the same endpoint (`GET /causal/agent-analyze/{id}`) in both pages — that is the duplication. Discovery owns *breadth* (agent-proposed leaderboard); analysis owns *depth* (manual single hypothesis + registry/history).

## 2. Decision

**One unified, agent-led page.** Keep the `/causal-analysis` route; retire `/causal-discovery` (redirect). The landing is the **agent-led leaderboard** of validated effects whose questions are **derived from the gold-standard SSOT — never a hand-curated list** — with a secondary "pose your own question" panel for power users. Deliver **all three live-estimable grains** (patient, HCP, trigger). Add a **Clinical Context enrichment layer** grounded in real biomedical APIs.

## 3. Goals / non-goals

**Goals**
- Agent proposes the causal questions; the user does not fill an empty form to start.
- Questions, adjustment sets, and discovery priors all derive from the **gold-standard `causal_paths` SSOT**, with rigor (correct per-question backdoor sets).
- Surface the full *live-estimable* catalog (~14 brand-specific questions across 3 grains) vs. 3 today.
- Make the clinical narrative **brand-faithful and sourced** (ClinicalTrials.gov / PubMed / ChEMBL), with an explicit synthetic-estimate / real-context boundary.

**Non-goals (this effort)**
- Enriching the synthetic *covariate values* to be clinically brand-faithful (they stay generic; we are transparent).
- The Claims/Optum-RWD track (not in-DB; future).
- openFDA indications/limitations + UMLS competitor intel — **deferred** (tasks #9–#11), do not act now.

## 4. Verified gold-standard catalog (live-probed 2026-06-19)

All confirmed by read-only probe against prod (self-hosted Supabase `supabase-db`, FalkorDB `e2i_falkordb_dev`). Flags: `[modeled-in-DGP | in-KG/causal_paths | live-estimable]`.

### Patient grain — `patient_journeys` (25,000 rows; Remibrutinib 8420 / Kisqali 8357 / Fabhalta 8223; `treatment_arm` binary 0/1, ~16.5% treated)

| Question (× each brand) | Adjustment set (modeled) | DGP | KG | LIVE |
|---|---|---|---|---|
| `treatment_arm` → `treatment_initiated` (initiation) | `{disease_severity, academic_hcp, age_at_diagnosis}` | ✅ | ✅ all 3 | ✅ |
| `treatment_arm` → `persistent_180d` / `discontinued_180d` (retention; **exact complements — collapse**) | `{disease_severity, academic_hcp, geographic_region}` | ✅ | ⚠️ persist→Kisqali only, disc→Fabhalta only | ✅ all 9 cells |

- Outcome base rates (live): init ~0.35, persist ~0.50, disc ~0.50 — variance present for **every** brand×outcome.
- Adjustment covariates **100% populated** live: `disease_severity, academic_hcp, age_at_diagnosis, engagement_score, geographic_region`.

### HCP grain — `hcp_brand_adoption` (15,000) ⋈ `hcp_profiles` (5,000)

| Question (× each brand) | Adjustment set | DGP | KG | LIVE |
|---|---|---|---|---|
| `peer_influence_score` (centrality) → `adopted` | **∅ backdoor** (exogenous) | ✅ | ❌ | ✅ (needs JOIN) |
| `treatment_arm` (rep engagement) → `adopted` | `{centrality_z = log1p(influence_network_size)}` | ✅ | ❌ | ✅ (needs JOIN) |

- `adopted` ~0.40/brand (variance ✓); `treatment_arm` on adoption table ✓; centrality cols on profiles 100% populated ✓.
- Brand CATE ordering differs from patient grain (F > R > K).

### Trigger grain — `triggers` table

| Question | Adjustment set | DGP | KG | LIVE |
|---|---|---|---|---|
| `control_group_flag` (randomized holdout) → `action_taken` | **∅ — the only true RCT in the gold standard** | ✅ | ❌ | ⚠️ columns confirmed; row-count/variance to verify at P3 |
| `acceptance_status` (accepted) → `conversion_flag` | priority = effect modifier | ✅ | ❌ | ⚠️ same |

### Confirmed non-questions / hazards
- **Vaporware:** `prior_therapy`, `adherence` — *do not exist* as columns. Never route questions through them. (Real mediators: `engagement_score`, `disease_severity`.)
- **Decoy:** `engagement_score` is a declared-but-ignored treatment — surface only as a covariate.
- **Denylisted-null:** `adherence_rate`, `risk_score`, `refill_count`, `gap_days` present but ~100% null — keep excluded.
- **Disease-specific markers present** (potential per-brand adjustment richness): `hr_status, her2_status, disease_stage, ecog_performance_status, ldh_ratio, complement_inhibitor_status, proteinuria_g_day, egfr, urticaria_severity_uas7, prior_antihistamine_therapy` — **caveat:** populated brand-generically; verify per-brand before using in a brand-specific adjustment set or narrative.

## 5. Architecture

### 5.1 Information architecture (one page)
- Route `/causal-analysis` stays; `/causal-discovery` → redirect.
- **Landing = agent-led leaderboard** (no empty form). Facets: **grain** (Patient / HCP / Trigger) + **brand**.
- **Row → drill-down** = the existing deep view (DAG, per-test refutation, sensitivity, interpretation, estimator-comparison) — reusing #1030's panels.
- **"Pose your own question"** = secondary panel (manual treatment/outcome/brand per grain), sourced from the retained `/variables` dropdown.

### 5.2 Question derivation (replaces hand-curation)
Source of truth = **`causal_paths` table** (carries the edge **and** the modeled `confounders_controlled`, both 100% populated). NOT the KG graph (it encodes confounders as mediators → adjusting on them induces bias). Mechanism:
1. **Enumerate** distinct `(start_node, end_node, brand, grain)` from `causal_paths` (dedup self-pairs + persist/disc complement).
2. **Attach** each row's `confounders_controlled` as the adjustment set (∩ live columns ∩ numeric-coercion set) — replaces the blanket 9-covariate pool. **Core rigor gain.**
3. **Pre-rank** cheaply with the currently-vestigial FWL adjusted-partial-correlation screen (`_adjusted_partial_corr`, folds in the dead `/propose-questions`) before the expensive serial agent runs.
4. **Seed agent priors** — thread `confounders_controlled` into `CausalPriorKnowledge` required edges (confounder→treatment, confounder→outcome), bypassing the generic `KNOWN_CAUSAL_RELATIONSHIPS`/`COMMON_CONFOUNDERS` constants (0/27 overlap with real covariates).
5. **Final rank** by confidence→impact (existing `_rank_effects`).

### 5.3 Grain abstraction
Make `causal_paths` the **universal SSOT for all grains** (patient + HCP + trigger edges, each with its modeled `confounders_controlled`, incl. ∅-backdoor rows). One derivation path then feeds the leaderboard, the KG viz, and the agent priors — all consistent. Each grain is a `dataset` with: an allowlist + a loader.
- Patient: `patient_journeys` (existing loader).
- HCP: `hcp_adoption` → JOIN loader `hcp_brand_adoption ⋈ hcp_profiles`.
- Trigger: `nba_triggers` → loader over `triggers`.

### 5.4 Keep (do NOT delete `_CAUSAL_DATASET_SPECS`)
It wears three hats; replace only one:
- **(1) enumeration** → replaced by the `causal_paths` derivation.
- **(2) `/variables` dropdown source** → keep (for the "pose your own question" panel).
- **(3) column allowlist + 400 guard + numeric coercion** in the data loaders → keep (security gate for `/agent-analyze`).

## 6. Data work + gated prod actions
- **Patient:** fix the lockstep bug (`causal_paths_generator.py:50,55` — brand & outcome both keyed on `i%3`; decouple to emit all 9 cells) + reseed `causal_paths` (**GATED prod write**, recoverable via re-sync; also de-hairballs the KG / helps open #1031). Categorical-encode `geographic_region` so retention can adjust for it.
- **HCP:** add the `hcp_adoption` spec + JOIN loader; add HCP edges to `causal_paths`.
- **Trigger:** live-verify variance first; add `nba_triggers` spec + loader; add trigger edges to `causal_paths`.

## 7. Clinical Context enrichment layer (verified APIs)
A **narrative/UI layer** over each effect — does **not** touch the causal math or adjustment sets (additive, low risk to rigor).
- **Drug + MoA** — ChEMBL (best-effort; static MoA fallback: ribociclib=CDK4/6i, iptacopan=Factor-B inhibitor, remibrutinib=BTK inhibitor). ChEMBL MCP is currently flaky → must degrade gracefully.
- **Real disease endpoints** — ClinicalTrials.gov `analyze_endpoints` (verified rich: breast cancer = OS/PFS/DFS/IBCFS; PNH = transfusion-avoidance/LDH/Hb-stabilization; CSU = UAS7/UCT-7/WI-NRS). Show how our synthetic outcome maps to the real pivotal endpoints.
- **Real-world evidence** — PubMed (e.g., ribociclib persistence/adherence PMID 35642282) grounding our `persistent_180d`/`discontinued_180d` outcomes.
- **Caching** per brand/disease; **graceful degradation** when any API is down.
- **Honesty label (explicit):** *the effect estimate runs on a synthetic cohort; the clinical context is real and cited.*

## 8. Honesty boundaries (surfaced, not hidden)
- Synthetic adjustment-covariate **values** are generic — stated plainly.
- Clinical markers populated brand-generically → adjustment mechanics only, **no brand-specific clinical claim** unless per-brand population is verified.
- Trigger variance is code-inferred until P3's live verify.
- The reseed is a prod data write — gated on explicit user OK.

## 9. Phasing
- **P0** — unified agent-led page shell (leaderboard landing, facets, drill-down reuse, "pose your own", retire `/causal-discovery`).
- **P1** — Patient grain: SSOT extend + lockstep fix + reseed + derived enumeration + per-question adjustment sets + agent-prior seeding + FWL pre-rank + fold #1030.
- **P2** — HCP grain: `hcp_adoption` spec + JOIN loader + causal_paths HCP edges (+6 questions, incl. exogenous-treatment ∅-backdoor).
- **P3** — Trigger grain: verify variance → `nba_triggers` spec + loader + causal_paths trigger edges (the NBA RCT).
- **Enrichment** — Clinical Context layer (after P0; deepened per grain).
- **Deferred** — #9 OpenFDA setup → #10 indications/limitations → #11 competitor intel (openFDA + UMLS).

Each phase is an independent worktree + PR.

## 10. Testing & verification
- **TDD red-first** per phase (write failing tests first).
- **ralph-loop** to drive each phase to a green fixed point.
- **codex:codex-rescue** for stuck points / second-opinion diagnosis.
- **Adversarial multi-lens review** before each PR (has repeatedly caught CI-passing honesty bugs here).
- **Faithful live agent run per grain** on the real backend before a phase is "done" (e.g., a Kisqali patient run must show a connected DAG, non-empty discovered confounders matching the modeled set, and a stable ATE).
- CI is the gate-arbiter; per-commit check-runs pinned to head SHA.

## 11. Risks & mitigations
| Risk | Mitigation |
|---|---|
| Reseed drops/regresses live causal_paths | Recoverable via re-sync; snapshot before; gated on user OK; verify all 9+ cells post-reseed |
| HCP JOIN loader bypasses the column security gate | Apply the same allowlist + numeric coercion to the JOIN loader |
| ChEMBL flakiness | best-effort + static MoA fallback + cache |
| Trigger variance not yet verified | P3 live-verify gate before building |
| Brand-generic clinical markers misread as faithful | explicit caveat label; don't build brand-specific clinical claims on them |
| Page rewrite desyncs e2e POM locators | grep + realign page-objects after the rewrite (known trap) |

## 12. #1030 disposition
Merge **#1030 first** (green/MERGEABLE; connected-DAG fix, estimator-comparison panel, executive_summary fix, insights/recs). The redesign builds on its backend wins and rewrites its FE. Merge needs explicit user merge/deploy authorization.

## 13. Execution methodology (agreed)
Per phase: **isolated git worktree → TDD red-first → ralph-loop to fixed point → codex:codex-rescue when stuck → adversarial review → PR.** CI green + faithful live run before merge. Prod merge/deploy + the reseed are individually gated on explicit user authorization.

## 14. Open items
- Confirm trigger-grain row count + variance (P3 pre-flight).
- Optionally verify per-brand population of disease-specific markers (richer per-brand adjustment sets) — else keep the generic-but-populated covariate set.
