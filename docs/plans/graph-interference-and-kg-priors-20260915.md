# Graph-aware causal inference: interference adjustment + KG discovery priors (2026-09-15)

## Origin

Evaluation of a proposed "FalkorDB Cypher subgraph → BaLu causal engine" workflow
(BaLu: *Joint Graph Learning for Robust Causal Inference over Knowledge Graphs*,
WSDM '26, https://dl.acm.org/doi/10.1145/3773966.3777980). Verdict: do not adopt
the workflow or BaLu (no public code; GNN black box would bypass the refutation /
E-value / expert-review gates; the proposal's wrapper returns a hard-coded
`ate=0.42`). Two of its ideas ARE worth taking, with the engine we already run
(DoWhy + EconML + causal-learn):

1. **Interference (SUTVA) is assumed, never measured.** `interpretation.py:501`
   lists "SUTVA: No interference between units" as an assumption;
   `expert_review_assessment.py:376` hard-codes the SUTVA question as
   `no_evidence` ("never machine-assessable"). No estimator adjusts for
   spillover. Yet the relational structure exists: the Optum HCP shared-patient
   network is already in FalkorDB (`(:HCP)-[:SHARED_PATIENTS]-(:HCP)`, issue
   #169, `scripts/persist_hcp_influence_to_falkordb.py`), and the synthetic
   conversion frame is per-trigger with `patient_id` + `hcp_id` keys
   (`kpi_resolution._TRIGGER_SELECT`).
2. **KG causal edges never reach the DAG builder.** `memory_hooks._get_causal_paths`
   (line 310) already runs `(t:Variable)-[:CAUSES*1..3]->(o:Variable)` but the
   result only lands in context. `graph_builder` builds from
   `KNOWN_CAUSAL_RELATIONSHIPS` + GES/PC discovery; the only KG influence on
   estimation is the Phase-6 role-attribution path (`kg_role_enrichment` →
   `adjustment_set_policy`).

Explicitly OUT of scope: BaLu / GNN imputation (estimation covariates come from
Supabase/parquet frames, not graph node attributes — `imputation_audit.py`
already covers missingness upstream); FalkorDB multi-graph counterfactual
branching (DoWhy counterfactuals + `src/digital_twin` cover what-if).

## Decisions taken (user, 2026-09-15)

| Decision | Call | Consequence in this plan |
|---|---|---|
| Plant a spillover in the synthetic DGP | **Yes, both channels, kept simple** | Phase 3 ships the within-patient trigger spillover AND a minimal shared-patient HCP graph (patients assigned to 1–2 HCPs). Both opt-in, defaults leave today's output byte-identical. |
| KG edges may shape DAGs shipped on real data | **No, at this time** | Phase 5's real-track tier hints are DEFERRED. The synthetic-only gold-standard prior gate stays as a low-priority, no-shipped-output-change item. |
| Optum joins the graph | **Build and validate the join; do NOT re-persist / analyse Optum yet** | Phase 4 is split: 4a (crosswalk property + validation on the in-process FalkorDB fake and the synthetic shared-patient cohort) is in scope; 4b (Optum re-persist with `--replace` and the peer-diffusion report) waits for an explicit go. |

## Two tracks, two roles (user direction 2026-09-15: "use both")

| Track | Role | Why |
|---|---|---|
| **Synthetic** (Supabase, `is_synthetic=true`, DGP in `src/ml/synthetic/generators`) | **Validation environment** — plant a known spillover, prove the estimator recovers it and is biased without adjustment | Ground truth is known. Recoverable-DGP gates already exist (`tests/integration/test_synthetic_causal_gates.py`, gate 3 ATE/CATE recovery). |
| **Optum RWD** (parquet cohort dirs + FalkorDB `SHARED_PATIENTS`) | **Faithful target** — the estimate that matters, with a real network | Real shared-patient cliques (edge weight = distinct shared patients in the leakage-safe lookback window, `convert_optum_rwd.build_hcp_influence_graph`). |

Neither track alone is sufficient: synthetic has truth but (today) no network;
Optum has a network but no truth and no commercial intervention column.

## Cheapest disproofs (Phase 0) — measured facts so far

### Synthetic

| # | Assumption | Status | Evidence |
|---|---|---|---|
| S0.1 | HCP-level interference channel exists in synthetic data | **DISPROVED (code)** | `patient_generator._assign_hcps` draws exactly ONE `hcp_id` per patient; `trigger_generator.py:623` and `treatment_generator.py:125` copy it off the patient. A shared-patient HCP graph over synthetic data is EMPTY by construction. SUTVA at the HCP level holds trivially. |
| S0.2 | Trigger-level (within-patient) interference channel exists | **SUPPORTED (code)** | `trigger_generator.py:221-231`: each patient gets `randint(1, triggers_per_patient+1)` triggers; `kpi_resolution._compute_conversion_outcome` marks a trigger converted if ANY prescription for its PATIENT falls in `[ts, ts+30d]`. Two triggers on the same patient inside 30 days share the outcome event — trigger *i*'s acceptance can flip trigger *j*'s outcome. This IS a SUTVA violation on the per-trigger grain today, and is currently invisible. |
| S0.3 | Magnitude matters | **TO MEASURE** (faithful docker Supabase, `E2I_DB_INTEGRATION=1`) | (a) fraction of triggers whose patient has ≥1 other trigger within ±30d; (b) ATE(`accepted→converted`) on the per-trigger frame vs on a per-patient-collapsed frame vs with a within-patient exposure covariate. If (a) < 5% or (b) moves < 1 SE, trigger-level interference is not binding and Phase 3's DGP plant becomes the only reason to continue on synthetic. |

### Optum

| # | Assumption | Status | Evidence |
|---|---|---|---|
| O0.1 | HCP profile rows can be joined to FalkorDB HCP nodes | **AT RISK** | Graph node `id` = obfuscated raw `medication.npi` (`persist_hcp_influence_to_falkordb.py`); profile `hcp_id` = `HCP_{seq}` with `seq` = rank in `sorted(npi_rx.keys())` and `npi` = the raw value only when it is a real CMS NPI, else `generate_luhn_npi(obf)` (`convert_optum_rwd.py:3925-3966`). No obf→hcp_id crosswalk is persisted. **Fix is small**: stamp `hcp_id` (same `seq` derivation, same sorted key set) on each `(:HCP)` node in the persist script, or emit a crosswalk parquet from the converter. Measure: `|{node.id} ∩ {obf keys}| / |nodes|` must be 1.0 after the fix. |
| O0.2 | The network is non-trivial and exposure has variance | **TO MEASURE** | Degree distribution per cohort; fraction of HCPs with ≥1 neighbor; variance of `share_treated_neighbors` under the chosen treatment. Zero variance ⇒ nothing to estimate. |
| O0.3 | A treatment exists on the Optum HCP grain | **PARTIALLY DISPROVED** | Claims data carry no commercial intervention: `digital_engagement_score`, `interaction_frequency`, `coverage_status`, `sales_rep_id` are all `None` in the Optum profile writer (`convert_optum_rwd.py:4057-4070`). The only treatments are claims-derived: an HCP's own early adoption (`adoption_category`, first-adoption timing) or prescribing volume. The estimand on Optum is therefore **peer diffusion** (does a neighbor's adoption raise mine?), which is confounded by homophily (shared patients ⇒ similar case mix). This must be reported as such; it is NOT the rep-intervention spillover the synthetic DGP can plant. |
| O0.4 | EconML supports clustered cross-fitting | **TO VERIFY** on the droplet env (econml 0.16.0 pinned; not installed in this session) | `_OrthoLearner.fit(..., groups=)` keeps rows of a group in the same fold. If absent, fall back to a manual `GroupKFold` cv object. |

## Design

### Phase 1 — Shared substrate: `src/causal_engine/interference/`

Pure, frame-level functions; no DB; unit-tested on real-shaped frames (no mocks).

- `exposure.py`
  - `NeighborSource` protocol with two implementations:
    - `SharedKeyNeighbors(key_col, time_col, window_days)` — units are neighbors
      when they share `key_col` (e.g. `patient_id`) within `window_days`. Serves
      the synthetic per-trigger frame (S0.2). Also `key_col="hcp_id"` for
      within-HCP clustering.
    - `FalkorDBNeighbors(cohort_id, graph)` — ONE bulk Cypher per cohort:
      `MATCH (a:HCP {cohort_id:$c})-[r:SHARED_PATIENTS]-(b:HCP {cohort_id:$c}) RETURN a.hcp_id, b.hcp_id, r.weight`
      (requires the O0.1 fix). Serves Optum.
  - `build_exposure_map(frame, unit_col, treatment_col, neighbors, *, weighted=True) -> pd.DataFrame`
    adds `n_neighbors`, `n_treated_neighbors`, `share_treated_neighbors`,
    `cluster_id` (connected component; for SharedKey the key itself).
    Leakage rule: only neighbors whose treatment time ≤ the unit's own
    timestamp count (mirrors the #168 temporal gate).
- `sutva_diagnostic.py`
  - `SutvaDiagnostic(shared_unit_fraction, exposure_variance, n_clusters, mean_cluster_size, verdict, evidence)`
    with `verdict ∈ {supports, concern, no_evidence}`:
    `no_evidence` when no cluster key is present; `supports` when
    `shared_unit_fraction < 0.05`; else `concern`.
  - Wire into `expert_review_assessment._fallback_items` for the
    `sutva_plausible` question (replaces the hard-coded `no_evidence`), and into
    `interpretation.py` so the SUTVA assumption line becomes conditional
    ("SUTVA: 3.1% of triggers share a patient within the outcome window —
    exposure-adjusted").

### Phase 2 — Estimation under interference (causal_impact agent)

- New optional state keys (contract update in
  `src/agents/causal_impact/CONTRACT_VALIDATION.md` + `.claude/contracts`):
  `interference_unit_col`, `interference_cluster_col`, `interference_neighbor_source`
  (`"shared_key" | "falkordb"`), `interference_window_days`.
  Output: `interference_adjusted: bool`, `spillover_effect` (coef + CI on
  `share_treated_neighbors`), `sutva_diagnostic`.
- `estimation.py`: when an exposure map is present, append
  `share_treated_neighbors` to every adjustment set (guarantee-channel style,
  as `_apply_adjustment_guarantee` does for declared confounders) and report the
  treatment coefficient as the **direct effect under the exposure mapping**
  (Hudgens–Halloran / Aronow–Samii framing). The estimand label in the output
  must say "direct effect (exposure-adjusted)", never plain ATE.
- Cluster-robust inference:
  - `estimator_selector.LinearDMLWrapper` / `DRLearnerWrapper`: pass
    `groups=cluster_id` to `fit` (O0.4).
  - `OLSWrapper`: `statsmodels` `cov_type="cluster"` on `cluster_id`.
  - `refutation.py` bootstrap refuter: resample **clusters**, not rows, when
    `cluster_id` is present (row bootstrap under-covers with within-cluster
    dependence). Reuse the existing `nuisance_config.py` so the refit mirrors
    production (the #2031 invariant).
- Dispatcher (`_resolve_causal_impact_input`): for the Conversion Rate KPI
  substrate set `interference_unit_col="trigger_id"`,
  `interference_cluster_col="patient_id"`, `neighbor_source="shared_key"`,
  `window_days=30` (the KPI's own window). Nothing fabricated: the keys are
  real frame columns already selected by `_TRIGGER_SELECT`.

### Phase 3 — Synthetic validation with a PLANTED spillover (APPROVED: both channels, keep simple)

Without a planted channel, synthetic data can only show "adjustment changes
nothing", which proves nothing about the mechanism. Proposed DGP change,
opt-in and versioned (mirrors how COMM-ARMS Phase 4 added `trigger_accepted`):

- `trigger_generator`: `spillover_beta` (default 0.0 ⇒ byte-identical output to
  today). For a patient with ≥2 triggers in the window, an accepted trigger
  raises the conversion probability attached to the patient's OTHER triggers by
  `spillover_beta` on the logit scale. Ground truth written to the existing
  ground-truth sidecar (direct effect, spillover effect).
- Second channel, HCP grain (`patient_generator._assign_hcps`): with
  `secondary_hcp_rate` (default 0.0) a patient gets a SECOND HCP drawn from the
  same academic/non-academic pool; `treatment_generator` / `trigger_generator`
  then draw each event's `hcp_id` from the patient's HCP set instead of the
  single column. That is the whole change — the shared-patient graph FOLLOWS
  from it via the existing `build_hcp_influence_graph` (two HCPs treating the
  same patient in the lookback window form an edge), so
  `persist_hcp_influence_to_falkordb.py` can ingest a synthetic cohort
  unchanged and the FalkorDB neighbor source is exercised end-to-end on data
  with known truth. No referral semantics, no new edge type: it is the same
  shared-patient construct Optum uses. Ground-truth spillover on this channel:
  an HCP's accepted-trigger share raises adoption odds of its shared-patient
  neighbors by `hcp_spillover_beta` (default 0.0).
- Acceptance gate (new `test_gate_12_interference_recovery` in
  `test_synthetic_causal_gates.py`): with `spillover_beta>0`, the unadjusted ATE
  is biased by more than 2 SE; the exposure-adjusted direct effect is within
  tolerance of the planted value; the spillover coefficient has the planted sign
  and is significant; cluster-bootstrap CI coverage ≥ nominal over 20 seeds.

Both knobs default to 0.0 so every existing gate (1–11) runs on byte-identical
data; the recovery gate (12) runs on an opt-in load with the knobs set. The
`secondary_hcp_rate` knob must be exercised by the gate with S0.1 re-measured:
the synthetic `SHARED_PATIENTS` graph must be non-empty ONLY when the knob is
set.

### Phase 4a — Graph join functionality (IN SCOPE; validated on synthetic + fake)

- O0.1 fix: `persist_hcp_influence_to_falkordb.py` stamps `hcp_id`
  (`HCP_{seq}`, same `sorted(npi_rx.keys())` derivation as the converter —
  factor the derivation into one shared helper so the two cannot drift) on
  every `(:HCP)` node, and the `FalkorDBNeighbors` source matches on it.
- Validation: (i) unit test against the in-process FalkorDB fake already used
  by `test_persist_hcp_influence_to_falkordb.py` — round-trip parity of
  `hcp_id` for every node; (ii) integration on the synthetic shared-patient
  cohort from Phase 3 — join fraction = 1.0 and exposure map equals the one
  computed in-memory from the frame (byte-for-byte parity, the #169 contract).
- Optum is NOT re-persisted in this phase.

### Phase 4b — Optum application (DEFERRED — waits for an explicit go)

- Re-persist per cohort with `--replace`, then measure O0.2 (degree, exposure
  variance).
- Treatment = HCP's own adoption status at an index time; outcome = neighbor
  adoption in a later window (diffusion estimand). Exposure map from
  `FalkorDBNeighbors`, leakage gate on adoption timing. Adjust for the
  homophily proxies available on the profile (`specialty`, `practice_type`,
  `decile`, `total_patient_volume`, `geographic_region`) and state plainly that
  homophily is not identified away.
- Report naive vs exposure-adjusted side by side, with the SUTVA diagnostic and
  the cluster-robust CI. This is a REPORT, not a new KPI, until an expert
  review accepts the estimand.

### Phase 5 — KG causal edges as discovery priors (real track DEFERRED; synthetic gate only)

- Source: `semantic_memory.list_relationships(relationship_types=["CAUSES"], curated_only=True)`
  filtered to `validation_status == "validated"` (the sync stamps it;
  `refutation.py:160` is what promotes a path to `validated`).
- Guard against self-confirmation: a validated edge is the OUTPUT of a prior
  discovery+refutation run. Feeding it back as a `required_edge` would make the
  gate corroborate its own history. Therefore:
  - Synthetic track: Shard-09 gold-standard chains (`causal_paths_generator`,
    `treatment_arm -> {treatment_initiated, persistent_180d, discontinued_180d}`)
    are DGP truth — allowed as `required_edges`.
  - Real/Optum track (DEFERRED by decision 2): when revisited, KG edges would
    enter only as **tier hints** (`CausalPriorKnowledge.tiers`,
    source-before-target) and as `forbidden_edges` for the reverse direction —
    orientation only, never existence. Nothing in this plan changes a shipped
    real-data DAG.
- Provenance: extend `_compute_edge_provenance` with `kg_prior` (distinct from
  `required_prior`) so the shipped DAG says which edges the KG oriented.
- Gate (synthetic): guided PC with KG tiers recovers the gold-standard edges at
  ≥ the rate of the current `anchored_confounders` priors, and never reverses a
  gold edge; hash stability via `hasher` for the new prior field.

## Tests (red-first, no mocks)

- `tests/unit/test_causal_engine/test_interference/test_exposure.py`: shared-key
  neighbors on real-shaped trigger frames (inside/outside window, leakage gate,
  weighted share, connected components); FalkorDB source against the in-process
  FalkorDB fake already used by `test_persist_hcp_influence_to_falkordb.py`.
- `.../test_sutva_diagnostic.py`: verdict thresholds; `no_evidence` without keys.
- `tests/unit/test_agents/test_causal_impact/test_estimation_interference.py`:
  exposure column joins every adjustment set; estimand label; cluster `groups`
  reach the wrapper; cluster bootstrap resamples clusters.
- `tests/unit/test_insights/test_expert_review_assessment_sutva.py`: diagnostic
  replaces the hard-coded `no_evidence`.
- `tests/unit/test_agents/test_causal_impact/test_graph_builder_priors.py`
  (extend): KG tiers → `BackgroundKnowledge`; `kg_prior` provenance; hash.
- Integration (faithful docker Supabase, `E2I_DB_INTEGRATION=1`): S0.3
  measurements committed as a test that prints the numbers; gate 12 (Phase 3).
- Optum (droplet only, parquet + FalkorDB): O0.1 join = 1.0 after fix; O0.2
  degree/variance report.

## Order of work and stop rules

1. Phase 0 measurements S0.3 and O0.4 (droplet). O0.1/O0.2 on Optum are
   deferred with Phase 4b; O0.1 is instead proven on the synthetic cohort in
   Phase 4a. S0.3 is now informational (the Phase 3 plant is approved), but
   still worth committing as a printed measurement.
2. Phase 1 (pure functions) — no DB, CI-only.
3. Phase 3 DGP knobs (both channels) + gate 12 skeleton (red).
4. Phase 2 wiring; gate 12 goes green on the opt-in load.
5. Phase 4a join functionality, validated on the synthetic shared-patient
   cohort and the FalkorDB fake.
6. Phase 5 synthetic gold-standard prior gate (low priority, optional).
7. Deferred, each behind its own go: Phase 4b Optum re-persist + report;
   Phase 5 real-track tier hints.

## Discipline

Worktree-isolated; targeted pytest on the droplet, whole-suite + mypy in CI
(CI is the mypy arbiter); codex audit briefs carry the mandatory design-pushback
paragraph; no push of a fabricated value anywhere — every new output field is
either measured or absent.
