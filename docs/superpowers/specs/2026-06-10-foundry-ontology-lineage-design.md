# Foundry Ontology + Data Lineage Files — Design

**Date:** 2026-06-10
**Status:** Approved-by-default (autonomous goal session — approval gates adapted into documented decisions)
**Deliverables:** `docs/foundry/e2i_ontology.yaml`, `docs/foundry/e2i_lineage.yaml`, `docs/foundry/README.md`

## Goal

Author (1) an ontology definition file and (2) a data-lineage definition file describing the
E2I Causal Analytics platform in terms directly consumable for onboarding onto Palantir
Foundry — its Ontology (Object Storage V2) and Data Lineage applications.

Source research: the four Palantir doc pages (object-backend overview, ontologies overview,
data-lineage overview, branching-data-lineage) plus three codebase sweeps (core entities,
data-flow, ML/agent layer), run 2026-06-10 via workflow `wf_44beaf70-43a`.

## Approaches considered

### Ontology file format
1. **YAML "ontology-as-code" descriptor using documented Foundry vocabulary** *(chosen)* —
   `objectTypes` (apiName, displayName, pluralDisplayName, description, primaryKey,
   titleProperty, typed `properties`, `datasources`), `linkTypes` (endpoints, cardinality,
   foreignKey or join-dataset backing), `sharedProperties`, `objectTypeGroups`. Human-readable,
   reviewable, maps 1:1 to Ontology Manager fields, mechanically convertible to API payloads.
2. JSON mirroring Ontology Metadata Service request bodies — more literal but verbose and
   brittle; the full import API shape is not stably documented publicly.
3. Markdown documentation only — not machine-usable; rejected.

### Lineage file format
1. **YAML directed-graph descriptor using Foundry's exact node taxonomy** *(chosen)* — nodes
   typed `dataSource | dataset | objectType | artifact` (the four documented kinds), transforms
   modeled JobSpec-style (per-output, branch-scoped, inputs/outputs/code path/trigger), edges
   derived from transforms plus dataset→objectType "defines an object type" provenance, and a
   two-layer branch model (dataset branches + global branch) per the branching docs.
2. JSON adjacency list — equivalent semantics, less readable; rejected.
3. Mermaid diagram — visual only, no metadata capacity; rejected (one already exists in
   `docs/data/00-INDEX.md`).

## Design

### Ontology (single ontology: `e2i-causal-analytics`)
- **19 object types** in 5 groups: core domain (Hcp, Patient, PatientJourney, TreatmentEvent,
  Trigger, BusinessMetric, ReferenceUniverse, HcpIntentSurvey), reference dimensions (Brand,
  Region), causal analytics (CausalPath, Kpi), ML lifecycle (MlExperiment, MlModel, Prediction,
  DataSplitConfig, DriftEvent), agentic layer (Agent, AgentActivity).
- **Object Storage V2 semantics only** (V1/Phonograph is deprecated, gone after 2026-06-30).
- **Canonical vocabulary = PostgreSQL** (live Supabase docker schema + post-#825/#842
  realignment). FalkorDB graph divergences (lowercase brand names, 7-value journey funnel,
  tier_1..tier_3) recorded in property descriptions, not duplicated as separate types.
- **Writeback disabled everywhere; no actionTypes in v1.** This is a read-only analytics
  ontology; operational edits (e.g. trigger acceptance) are deferred deliberately (YAGNI).
- **Every primary key** verified unique + non-null in the backing table (Foundry hard rule).

### Key modeling decisions
1. **Patient is a derived object type** (status: experimental). Postgres has no patient master
   table — `patient_id` is a plain VARCHAR on 5 tables; the Patient entity lives only in
   FalkorDB. The backing dataset `ds-curated-patient-master` is produced by a declared
   `derive_patient_master` transform (distinct patient_ids + latest journey attributes).
2. **Many-to-many links are backed by join datasets** (Foundry hard rule). The seven graph
   edges (TREATED_BY, PRESCRIBED, PRESCRIBES, INFLUENCES, DISCOVERED, GENERATED, IMPACTS) get
   join datasets produced by an `export_falkordb_links` transform; status experimental. The one
   M:N relationship that already has a DB join table — patient↔split via
   `ml_patient_split_assignments` — is backed by that table directly.
3. **Prediction→MlModel is a derived link.** `ml_predictions.model_version` is a loose string
   handle (no FK; UNIQUE on registry is `(model_name, model_version)`); a curation transform
   resolves it to the registry UUID (mirroring
   `src/repositories/drift_monitoring.py::_resolve_model_id`) and emits `model_registry_id`.
4. **Brand dimension carries all 5 PG enum values** (3 brands + competitor + other) so the
   enum-valued FK columns on journeys/events/metrics resolve for every row.
5. **Kpi is config-backed** (`config/kpi_definitions.yaml`, 46 KPIs; stable ids WS1-DQ-001 …
   CM-005), surfaced as a dataset via a `curate_kpi_definitions` transform.
6. **Soft joins become explicit linkTypes** (Foundry links don't require DB FKs) —
   agent_activities.agent_name, patient_journeys.hcp_id, business_metrics.metric_name are
   modeled with their non-enforcement noted.
7. **`dataSplit` and `isSynthetic` are sharedProperties** — they appear on most row-level
   tables and carry platform-wide semantics (split governance, synthetic provenance with
   default-exclude reads).

### Lineage
- **Zones:** raw (Optum mart/claims parquet, CSU workbook, synthetic_CSU snapshot) →
  intermediate (converter outputs, tier0 contract dirs) → operational store (Supabase via
  Data Connection syncs) → curated ontology-backing datasets → feature store (Feast
  offline/online) → model artifacts → outputs (reports, dashboards).
- **External boundary made explicit:** the Optum mart is built by an external Spark 3.5.5 v9.1
  pipeline not in this repo — lineage upstream of `Optum.parquet` terminates at a `dataSource`
  node, and the uncertified comorbidity lookback is flagged.
- **Branching:** `master` is the root branch; the current git branch
  `feat/synthetic-csu-tier0-e2e` is modeled as a global-branch example (holds 3 unmerged
  fixes; merge auto-deploys). Foundry rules encoded: dataset branches never merge (promotion =
  re-run on master), fallback chain `[master]`, fallbacks disabled when a global branch is
  selected, one ontology per global branch.
- **Edges are derived, not duplicated:** transform `inputs`/`outputs` define dataset→dataset
  edges; per-dataset `definesObjectTypes` defines dataset→objectType provenance edges.
- **Honest caveats carried over:** two data_split regimes (chronological DB loads vs
  stratified-random tier0 exports), converter-specific leakage governance (CSU converter has
  no lookback window; Optum converters are allow-list/180-day safe), synthetic provenance
  (is_synthetic + include_synthetic opt-in), business_metrics content is synthetic.

## Verification plan
Adversarial workflow after authoring: (a) every backing table/column/enum in the ontology file
exists in `database/` SQL + data dictionaries; (b) every dataset location/transform code path
in the lineage file exists in the repo; (c) Foundry-concept fidelity against the fetched doc
summaries; (d) cross-file consistency (every ontology datasource is a lineage dataset node;
every M:N linkType's join dataset exists). Findings fixed before completion.

## Out of scope
Action types / writeback flows, interfaces, Foundry API payload generation, actually
provisioning anything in a Foundry instance, restricted views / column-level permissions,
and the ~40 utility ML tables (HPO, A/B, digital twin, tool composer) beyond the modeled set.
