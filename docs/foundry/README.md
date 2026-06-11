# Palantir Foundry Definitions — E2I Causal Analytics

Files for onboarding the E2I platform onto Palantir Foundry's Ontology and
Data Lineage applications.

| File | What it is |
|---|---|
| `e2i_ontology.yaml` | Ontology definition: 19 object types, 2 shared properties, 5 object type groups, ~40 link types (FK-based one-to-many/one-to-one + join-dataset-backed many-to-many), Object Storage V2 semantics. |
| `e2i_lineage.yaml` | Data-lineage graph: 4 data sources, ~35 datasets across 7 zones, 2 artifacts, ~28 transforms (existing pipeline + proposed Foundry onboarding transforms), two-layer branch model. |
| `../superpowers/specs/2026-06-10-foundry-ontology-lineage-design.md` | Design rationale: approaches considered, decisions, verification plan. |

## How the two files join

The **backing datasource is the bridge** (Foundry's "Defines an object type"
relationship):

- Every `objectType.datasources[].dataset` in the ontology file is a dataset
  node id in the lineage file.
- Every `MANY_TO_MANY` link type's join dataset (`ds-link-*`,
  `ds-curated-ml-patient-split-assignments`) is likewise a lineage dataset
  node.
- The lineage file's `datasets[].definesObjectTypes` lists the inverse
  direction.
- `lineage.ontology: e2i-causal-analytics` scopes the graph to the single
  ontology (Foundry: one ontology per global branch).

## Reading order for Foundry onboarding

1. **Lineage file, curated zone** — these are the datasets Foundry needs
   first: 15 straight table syncs from Supabase (`tf-sync-supabase-to-curated`),
   plus 5 derived/seeded datasets (`tf-derive-patient-master`,
   `tf-curate-ml-predictions`, `tf-curate-kpi-definitions`,
   `tf-seed-dimensions`, `tf-export-falkordb-links`) marked `status: proposed`.
2. **Ontology file** — attach each curated dataset as the backing datasource
   of its object type in Ontology Manager; objects only materialize once a
   datasource is attached.
3. **Link types** — FK links need no extra data; M:N links need their join
   datasets built first.

## Key decisions (full rationale in the design doc)

- **Canonical vocabulary = PostgreSQL** (live docker-Supabase schema,
  post-#825/#842 realignment). FalkorDB divergences (lowercase brands,
  7-value funnel, tier_1..3) are noted in property descriptions; export
  transforms normalize to PG spelling.
- **Patient is derived** — no PG master table exists; `patient_id` is a
  plain column on 5 tables. Status `experimental`, backed by a derived
  dataset.
- **Soft DB joins become real Foundry links** (journeys→HCP,
  activities→agent, predictions→model). The predictions→model link requires
  a curation transform that resolves the loose `model_version` string to the
  registry UUID.
- **Brand dimension carries all 5 enum values** (3 brands + competitor +
  other) so enum-valued FK columns resolve for every row.
- **Read-only v1**: writeback disabled everywhere, no actionTypes, no
  interfaces. Inferred graph edges (HIGH_POTENTIAL_PRESCRIBER,
  INDIRECTLY_INFLUENCES, TRANSITIONED_TO) are not modeled.
- **`dataSplit` / `isSynthetic` are shared properties** — platform-wide
  semantics (two split regimes; synthetic rows default-excluded) documented
  once.

## Branching

Mirrors Foundry's two-layer model: dataset branches (linear transactions,
never merged — promotion = re-run on `master`) and global branches (span
datasets + ontology entities; fallbacks disabled while selected). The
current git branch `feat/synthetic-csu-tier0-e2e` is modeled as an example
global branch carrying the regenerated synthetic_CSU zone and three unmerged
JobSpec overrides.

## Sources

Authored 2026-06-10 from: Palantir Foundry docs (object-backend overview,
ontologies overview, data-lineage overview, branching-data-lineage; fetched
2026-06-10), `docs/data/00-08` data dictionaries,
`database/core/e2i_ml_complete_v3_schema.sql` + migrations,
`database/ml/mlops_tables.sql`, `docs/lineage/` artifacts,
`config/kpi_definitions.yaml`, and `src/` repository code. Research and
adversarial verification run via multi-agent workflows (run ids
`wf_44beaf70-43a` + verification run; see design doc).
