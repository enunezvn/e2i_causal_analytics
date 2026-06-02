# Plan — Adapting the E2I Causal Analytics Pipeline to the Palantir Foundry Ontology

| | |
|---|---|
| **Status** | DRAFT — for review |
| **Date** | 2026-06-02 |
| **Scope** | Express the E2I pharma causal-analytics domain as a Palantir Foundry **Ontology** (object / link / action types, functions, models, AIP) and run the existing pipeline against it. |
| **Reference** | Palantir Foundry Ontology docs (links in §16) |
| **Strategy (recommended)** | **Integration-first strangler-fig**, not big-bang re-platform. Wrap & mirror; cut over per-slice behind parity gates. |
| **Tracked path** | `docs/plans/614-palantir-foundry-ontology-adaptation.md` (working mirror in the git-ignored `.claude/plans/` — see §0.1) |

---

## 0. TL;DR

The E2I platform already contains a **hand-rolled ontology** (FalkorDB graph: 8 node types, 15 edge types, 5 inference rules, plus `src/ontology/` compiler/validator/vocabulary). Foundry's Ontology is the enterprise-grade version of the same idea, adding: governed **write-back Actions**, native **Functions / AIP** for the agent layer, **Model integration** (live deployments + model functions) for the causal/ML layer, **granular security** (Markings / Restricted Views) suited to PHI/PII, **lineage**, and a generated **OSDK** app surface.

The work is fundamentally a **mapping + dual-run + cutover** exercise:

1. **Semantic layer** — 19 core Postgres tables + 8 graph nodes → Foundry **object types**; FKs + 15 graph edges → **link types**.
2. **Kinetic layer** — agent/endpoint write-paths → **action types**; the 21 agents → **AIP Logic functions / Agent Studio agents** with object/function/action tools.
3. **Compute layer** — causal engine (DoWhy/EconML/CausalML) + ML models → **Foundry Models + Modeling Objectives + model functions** (function-backed properties); 46 KPIs + 5 inference rules → **Functions**.
4. **Ingestion** — `src/etl/*` + RWD sources + synthetic generators → **Pipeline Builder / Code Repository transforms** with **Ontology outputs**.
5. **App surface** — FastAPI + React → **OSDK (Python + TypeScript)** consumers and/or **Workshop**.

**Do the Phase 0 spike (§2) before building anything.** It can disprove the whole effort in ~1–2 weeks for the cost of a Foundry dev branch.

### 0.1 Read this first — provenance & assumptions

- This is the **tracked** copy in `docs/plans/`. It was authored as `.claude/plans/2026-06-02-palantir-foundry-ontology-adaptation.md`; because `.claude/` is git-ignored (`.gitignore:58`) and the authoring container is ephemeral, the canonical copy was moved here so it is versioned and reviewable.
- This plan describes a **target architecture**. **No Foundry tenant is referenced in this repo today** — confirming tenant + licensing is the first gate (§2).

---

## 1. Reasoning before rules (per `CLAUDE.md` → REASON-BEFORE-RULES)

Before any "migrate/replace" action, answer the four questions for the thing we're changing — the **existing E2I ontology subsystem**.

**1.1 What is it trying to do?**
`src/ontology/` + `config/ontology/*.yaml` + the FalkorDB graph form a domain ontology that today powers four *load-bearing* functions:
- **NLP routing** — `query_extractor.E2IQueryExtractor` uses `vocabulary_registry` to extract brand/region/KPI entities (<50 ms, no LLM) and route to one of 21 agents.
- **Causal reasoning** — `inference_engine` discovers causal paths, confounders, mediators over the graph; materializes inferred edges.
- **Graph RAG** — `src/rag/` traverses the graph for HCP-influence / patient-journey context.
- **Schema governance** — `schema_compiler` compiles YAML → Cypher DDL; `validator` enforces naming/cardinality/parity; `validate_vocabulary_sync()` keeps Python enums ↔ YAML in lockstep.

**1.2 Why does it exist in this shape?** FalkorDB was chosen for fast causal-DAG traversal (Graphity edge-grouping in `grafiti_config.py`); YAML-as-schema for human editability; Postgres enums mirror the YAML for DB-level constraints. This is a deliberate, coherent design — **not** vestigial.

**1.3 Is it causing harm now?** No. It works. The motivation for Foundry is *additive*: enterprise governance, lineage, write-back Actions, a shared semantic layer, and an OSDK app surface — **not** that the current ontology is broken.

**1.4 What does the user actually want?** Adapt the pipeline *to* the Foundry Ontology per the docs — i.e., represent the E2I domain as a Foundry Ontology and run E2I against it. That is an **express-and-mirror** task, not a delete-and-rewrite task.

**1.5 Consequence (the rule that follows from the reasoning):**
> The existing ontology is the **specification source** for the Foundry ontology, not something to delete. We **export** E2I's YAML/graph schema into Foundry ontology-as-code, **dual-run** for parity, and **retire** bespoke pieces only where Foundry demonstrably supersedes them (§11 Phase 7). No node type, edge type, or inference rule is dropped without a documented Foundry equivalent and a passing parity test (§13).

---

## 2. Cheapest-disproof-first — Phase 0 spike (per `CLAUDE.md`; MANDATORY before build)

Do **not** theorize the migration into existence. Name the assumptions the whole plan rests on, then run the cheapest experiment that could **disprove** each — in a **faithful** environment (a real Foundry dev branch with the actual license tier and a PHI-representative synthetic dataset). Proceed only on survival.

| # | Single assumption the plan depends on | Cheapest disproving experiment (faithful) | Cost | Kills the plan if… |
|---|---|---|---|---|
| A0 | **A Foundry tenant exists and licenses the capabilities we need** (Ontology, Pipeline Builder, Model Integration / Foundry ML, AIP Logic + Agent Studio, OSDK). | 30-min check with the Foundry admin: enrollment + enabled apps + AIP entitlement + OSDK/Developer Console access. | ~hours | AIP or Model Integration or OSDK is not licensed → §5/§7 collapse; fall back to "ontology + pipelines only". |
| A1 | **PHI/PII may legally land in Foundry** (data residency, BAA, IRB/retention for IQVIA/HealthVerity/Optum). | Compliance/legal confirmation + check Foundry region vs data-residency obligations. | ~days | If patient-level RWD cannot egress to the tenant → model Patient as **pseudonymized-only** or keep PHI on-prem and federate (Virtual Tables). |
| A2 | **The E2I domain maps cleanly to object/link/action types** end-to-end. | Build **one vertical slice** on a Foundry branch: `HealthcareProvider` object (backed by a `hcp_profiles` dataset) + `treats` link + `deliverTrigger` action + generated Python OSDK; call it from a 30-line script. | 2–4 days | If the round-trip (read object → run action → read back) can't reproduce a `triggers` write → re-scope to read-only ontology. |
| A3 | **Foundry can host the causal/ML compute at acceptable latency/governance** (the riskiest). | Wrap **one** model (the propensity model) as a Foundry Model → Modeling Objective → live deployment → **model function**; re-implement **one** agent (`causal_impact`) as an **AIP Logic** function that calls the model function + an object-query tool and writes a `CausalEstimate` via an Action. Measure latency vs the in-proc `src/causal_engine` path. | 1–2 wks | If model-function latency or AIP governance makes the agent loop unworkable → keep compute in `src/*`, use Foundry only as the semantic/governance store (Functions call back out). |

**Go/No-Go gate:** all four must survive. Record measured results (not projections) in this file before Phase 1. A green run in a *non-faithful* environment (wrong license tier, non-PHI data) is a **false green** — the faithful environment *is* the target tenant.

---

## 3. Conceptual mapping: E2I → Foundry Ontology (the core)

| E2I construct (today) | Where it lives | Foundry Ontology construct |
|---|---|---|
| 19 core Postgres tables; 8 FalkorDB node types | `database/core/e2i_ml_complete_v3_schema.sql`; `config/ontology/node_types.yaml` | **Object types** (backed by datasets) |
| FK relationships; 13 direct + 2 inferred edges | schema FKs; `config/ontology/edge_types.yaml` | **Link types** (FK-backed or join-dataset-backed; inferred → derived) |
| 12 Postgres ENUMs; `domain_vocabulary.yaml` | `vocabulary_registry.py` | **Shared property types** + enum value types |
| Common columns (`brand`, `region`, `created_at`, `confidence`) | across tables | **Interface types** + **shared property types** |
| JSONB columns (`causal_chain`, `hyperparameters`, `fairness_metrics`) | core/ml tables | **Struct** properties / arrays |
| Timestamped metric series (`business_metrics`, drift) | `business_metrics`, monitoring | **Time series** properties |
| 5 inference rules | `config/ontology/inference_rules.yaml`, `inference_engine.py` | **Functions** (scheduled) producing **derived links / objects** |
| 46 KPIs | `config/kpi_definitions.yaml`, `src/kpi/` | **Functions** / **function-backed properties** |
| Agent/endpoint write-paths (deliver trigger, promote model, stop experiment) | `src/agents/*`, `src/api/routes/*` | **Action types** (parameters, rules, submission criteria, side effects, writeback) |
| 21 agents / 6 tiers | `src/agents/factory.py` | **AIP Logic functions** + **AIP Agent Studio** agents (object/function/action tools) |
| ML models (XGBoost/RF/NN), causal estimators (DoWhy/EconML/CausalML) | `src/ml`, `src/causal_engine` | **Foundry Models** + **Modeling Objectives** + live deployments + **model functions** |
| ETL scripts + RWD sources + synthetic generators | `src/etl/*`, `src/data/*`, `src/ml/synthetic/*` | **Data Connection** sources + **Pipeline Builder / Code Repository** transforms → **Ontology outputs** |
| Feast (8 entities, 9 views, 48 features) | `feature_repo/*`, `src/feature_store/*` | Backing/derived **datasets** + **function-backed properties**; online serving via OSDK / live deployment |
| Supabase RLS + audit hash chain | `database/audit/*`, RLS migrations | **Markings**, **Restricted Views**, **granular policies**, **mandatory control properties**, Action audit logs |
| FastAPI (~29 routers) + React (27–30 pages) | `src/api/*`, `frontend/*` | **OSDK** (Python backend, TS frontend) and/or **Workshop** apps |
| YAML schema + `validator.py` + CI | `config/ontology/*`, `src/ontology/validator.py` | **Ontology-as-code** + **ontology branching / proposals** (PR-like) + **Marketplace** package |

The remaining sections (§4–§10) specify each row granularly.

---

## 4. Object-type catalog

**Conventions.** API name = PascalCase; primary key from the table PK; `title` = human display property; every object type implements interfaces from §4.3. Backing dataset = the Pipeline-Builder output that lands the rows (§8).

### 4.1 Fully specified core object types

#### `HealthcareProvider`  (maps `hcp_profiles` + FalkorDB `HCP`)
- **Backing dataset:** `ontology/hcp_profiles` (output of `pb_hcp_profiles`, §8)
- **Primary key:** `hcpId` (string, from `hcp_id` `^HCP[0-9]{8}$`) · **Title:** `displayName` (derived) · **Alt key:** `npi` (`^[0-9]{10}$`)
- **Properties:**
  | API name | Base type | From | Notes |
  |---|---|---|---|
  | `hcpId` | string (PK) | `hcp_id` | |
  | `npi` | string | `npi` | unique index |
  | `specialty` | string (enum) | `specialty` | 13 values (dermatology … other) |
  | `priorityTier` | string (enum) | `priority_tier` | tier_1/2/3 |
  | `adoptionCategory` | string (enum) | `adoption_category` | innovator … laggard |
  | `region` | string (enum) | `region` | **shared property** (§4.3) |
  | `territoryId` | string | `territory_id` | → `practicesInTerritory` link |
  | `decile` | integer | `decile` | |
  | `digitalEngagementScore` | double | `digital_engagement_score` | |
  | `peerInfluenceScore` | double | `peer_influence_score` | |
  | `createdAt` / `updatedAt` | timestamp | `created_at`/`updated_at` | **interface `ITemporal`** |
- **Links:** `treats`→`TreatmentEvent`; `prescribesBrand`↔`Brand`; `practicesInRegion`→`Region`; `practicesInTerritory`→`Territory`; `influences`↔`HealthcareProvider`; `receivedTrigger`↔`Trigger`; `surveyedBy`→`HcpIntentSurvey`.
- **Security:** no PHI; `npi` guarded by property security policy (commercial-sensitive).

#### `PatientJourney`  (maps `patient_journeys` + FalkorDB `Patient`)
> `patient_journeys` is per-(patient×brand) journey; PK is `patient_journey_id`. Model the **journey** as the object; expose a `Patient` **interface** (§4.3) keyed on the pseudonymized `patientId`. If a canonical patient registry dataset exists, add a thin `Patient` object later.
- **Backing dataset:** `ontology/patient_journeys` · **PK:** `patientJourneyId` · **Title:** `patientJourneyId`
- **Properties:** `patientId` (string, pseudonymized), `patientHash` (string — **PHI/PII marking**), `brand` (enum, shared), `journeyStage` (enum: diagnosis…maintained + 7 engagement-funnel values), `journeyStatus` (enum: active/stable/transitioning/completed), `region` (shared), `ageGroup` (enum), `insuranceType` (enum), `diseaseSeverity`, `engagementScore`, `riskScore` (double), `treatmentInitiated` (bool), `daysToTreatment` (int), `dataSplit` (enum: train/validation/test/holdout — **interface `ISplittable`**), `dataSource`, `hcpId` (FK), `createdAt`/`updatedAt`.
- **Links:** `treatedBy`→`HealthcareProvider`; `prescribedBrand`→`Brand`; `locatedInRegion`→`Region`; `hasTreatmentEvent`→`TreatmentEvent`; `transitionedTo`→`PatientJourney` (self, journey state machine).
- **Security:** **Restricted View** + **mandatory control property** (`patientHash`/`patientId`); cell-level — `patientHash` only with PII marking; 365-day retention policy.

#### `TreatmentEvent`  (maps `treatment_events`)
- **PK:** `treatmentEventId` · **Backing:** `ontology/treatment_events`
- **Properties:** `patientJourneyId` (FK), `hcpId` (FK), `eventDate` (timestamp → time-series capable), `brand` (shared), `eventType` (enum: diagnosis/prescription/lab_test/procedure/consultation/hospitalization), `drugNdc`, `drugClass`, `treatmentResponse` (enum: controlled/inadequate/uncontrolled/refractory/discontinued), `dataSplit`.
- **Links:** `forPatientJourney`→`PatientJourney`; `performedBy`→`HealthcareProvider`; `forBrand`→`Brand`.

#### `Trigger`  (maps `triggers` + FalkorDB `Trigger`)
- **PK:** `triggerId` · **Backing:** `ontology/triggers`
- **Properties:** `patientId`, `hcpId` (FK), `triggerType` (enum: alert/recommendation/insight/nba), `priority` (enum: critical/high/medium/low), `status` (enum: pending/delivered/accepted/rejected/completed/expired), `deliveryChannel`, `deliveryStatus`, `acceptanceStatus`, `changeType` (new/update/escalation/downgrade), `message`, `expirationDate` (timestamp), `dataSplit`.
- **Links:** `targetedAtHcp`→`HealthcareProvider`; `targetedAtPatient`→`PatientJourney`; `generatedBy`→`AgentActivity`.
- **Write-back:** mutated only via `deliverTrigger` / `recordTriggerResponse` actions (§6).

#### `CausalEstimate`  (maps `causal_paths` + FalkorDB `CausalPath`)
- **PK:** `pathId` · **Backing:** `ontology/causal_paths`
- **Properties:** `sourceVariable`, `targetVariable`, `causalChain` (**struct[]** from JSONB), `startNode`, `endNode`, `pathLength` (int), `effectSize` (double = ATE), `confidence` (double, shared — **interface `IConfidenceScored`**), `methodUsed` (enum: dowhy_backdoor … difference_in_differences), `validationStatus` (enum: validated/pending/failed/skipped), `gateDecision` (enum: proceed/review/block), `confoundersControlled` (string[]), `mediatorsIdentified` (string[]), `dataSplit`, `createdAt`.
- **Links:** `impactsKpi`↔`Kpi`; `causes`↔(any) (DAG — see `no_causal_cycles` rule, §6/§7); `discoveredBy`→`AgentActivity`.

#### `Experiment`  (maps A/B-testing + `experiment_lifecycle` tables)
- **PK:** `experimentId` · **Properties:** `name`, `state` (enum: 15 states draft…archived), `hypothesis`, `effectSizeEstimate`, `power`, `alpha`, `sampleSize`, `startDate`, `endDate`, `stoppingReason` (enum), `decision` (enum: implement_treatment/implement_control/no_change/run_followup_experiment), `brand`, `region`.
- **Links:** `designedBy`/`monitoredBy`→`AgentActivity`; `evaluatesModel`→`MlModel`; `assignedTo`→`PatientJourney`(arm).
- **Write-back:** `approveExperiment` / `startExperiment` / `stopExperiment` (§6) implement the state machine.

### 4.2 Remaining object types (compact)

| Object type | Maps from | PK | Key properties | Primary links |
|---|---|---|---|---|
| `Brand` | `brand` enum + `Brand` node | `brandName` | `therapeuticArea`, `indication` | ↔ HCP, ← PatientJourney |
| `Region` | `region` enum + `Region` node | `regionName` | `stateCount` | ← HCP/Patient/Territory |
| `Territory` | `territory_metrics` | `territoryId` | `activeHcpCount`, `coveredLives`, `marketPotential`, `resourceAllocationScore` | → Region, ← HCP |
| `MlPrediction` | `ml_predictions` | `predictionId` | `predictionType` (enum), `predictionValue`, `confidenceScore`, `modelAuc`, `brierScore`, `treatmentEffectEstimate`, `heterogeneousEffect`, `actualOutcome` | → HCP, → MlModel |
| `BusinessMetric` | `business_metrics` | `metricId` | `metricDate` (**time series**), `metricType`, `value`, `achievementRate`, `roi`, CI bounds, `brand`, `region` | → Brand/Region/Territory |
| `Kpi` | `KPI` node + `kpi_definitions.yaml` | `kpiId` | `category` (6 workstreams), `name`, `targetValue`, `currentValue`, `unit`, thresholds | ↔ CausalEstimate |
| `MlExperiment` | `ml_experiments` | `id` | `predictionTarget`, `observationWindowDays`, `predictionHorizonDays`, `minimumAuc`, `brand`, `dataSplit` | → MlModel |
| `MlModel` | `ml_model_registry` | `id` | `modelName`, `modelVersion`, `algorithm`, `hyperparameters` (struct), `auc`, `prAuc`, `fairnessMetrics` (struct), `stage` (enum), `isChampion` (bool) | → MlExperiment |
| `AgentActivity` | `agent_activities` + `Agent` node | `activityId` | `agentName` (21), `agentTier` (6), `activityType`, `workstream` (WS1-3), `recommendations`, `impactEstimate`, `roiEstimate` | → produced objects |
| `Agent` | `agent_registry` | `agentName` | `tier`, `version`, `enabled` | → AgentActivity |
| `Cohort` | cohort-constructor tables | `cohortId` | `definition` (struct), `memberCount`, `demographics` (struct) | ↔ Patient/HCP |
| `ReferenceUniverse` | `reference_universe` | `universeId` | `universeType`, `brand`, `region`, `totalCount` | → Brand/Region |
| `HcpIntentSurvey` | `hcp_intent_surveys` | `surveyId` | `brand`, `intentToPrescribe`, `responseDate` | → HCP |
| `DigitalTwinSimulation` | digital-twin tables | `simulationId` | `twinAlgorithm` (enum), `intervention` (enum), `ate`, `recommendation` (DEPLOY/REFINE/SKIP), `fidelityGrade` | → Experiment/CausalEstimate |
| `DataSource` | `data_source_tracking` | `sourceId` | `sourceName` (IQVIA/HealthVerity/Komodo/Veeva), `matchRate`, `stackingLift`, `matchConfidence` | — |

### 4.3 Interface types & shared property types

**Shared property types** (define once, reuse across object types): `brand` (enum), `region` (enum), `createdAt`, `updatedAt`, `confidence` (double), `validationStatus` (enum), `dataSplit` (enum), `patientId` (pseudonymized string), `npi`.

**Interface types** (abstract; enable polymorphic OSDK/AIP access):
- `ITemporal` → `createdAt`, `updatedAt` (most objects).
- `IBrandScoped` → `brand` (HCP-brand, Patient, Treatment, Metric, Trigger…). Enables a single AIP object-query tool over "everything for Brand X".
- `IRegionScoped` → `region`.
- `ISplittable` → `dataSplit` (every ML-relevant object; supports leakage-safe object sets).
- `IConfidenceScored` → `confidence`, `validationStatus` (CausalEstimate, MlPrediction, DigitalTwinSimulation).
- `IPharmaEntity` (root) → `displayTitle`.

These are derived from the columns that recur across `config/ontology/core_attributes.yaml` and the Postgres ENUM reuse table.

---

## 5. Link-type catalog

| Link API name | A → B | Cardinality | Backing | Maps from |
|---|---|---|---|---|
| `treatedBy` | PatientJourney → HealthcareProvider | N:1 | FK `patient_journeys.hcp_id` | edge `TREATED_BY` |
| `hasTreatmentEvent` | PatientJourney → TreatmentEvent | 1:N | FK `treatment_events.patient_journey_id` | (FK) |
| `performedBy` | TreatmentEvent → HealthcareProvider | N:1 | FK `treatment_events.hcp_id` | (FK) |
| `prescribesBrand` | HealthcareProvider ↔ Brand | M:N | join dataset `link_hcp_brand` | edge `PRESCRIBES` |
| `prescribedBrand` | PatientJourney → Brand | N:1 | column `patient_journeys.brand` | edge `PRESCRIBED` |
| `practicesInRegion` | HealthcareProvider → Region | N:1 | column `hcp_profiles.region` | edge `PRACTICES_IN` |
| `practicesInTerritory` | HealthcareProvider → Territory | N:1 | column `hcp_profiles.territory_id` | (FK) |
| `locatedInRegion` | PatientJourney → Region | N:1 | column `patient_journeys.region` | edge `LOCATED_IN` |
| `influences` | HealthcareProvider ↔ HealthcareProvider | M:N | join dataset `link_hcp_influence` | edge `INFLUENCES` |
| `receivedTrigger` | HealthcareProvider ↔ Trigger | M:N | join dataset `link_hcp_trigger` | edge `RECEIVED` |
| `predictedFor` | MlPrediction → HealthcareProvider/PatientJourney | N:1 | FK `ml_predictions.hcp_id` | (FK) |
| `impactsKpi` | CausalEstimate ↔ Kpi | M:N | join dataset `link_causal_kpi` | edge `IMPACTS` |
| `causes` | CausalEstimate ↔ (any) | M:N (DAG) | `causal_chain` struct | edge `CAUSES` |
| `analyzedBy` | (any) → AgentActivity | N:M | `agent_activities` refs | edge `ANALYZES` |
| `discoveredBy` | CausalEstimate → AgentActivity | N:1 | refs | edge `DISCOVERED` |
| `generatedBy` | Trigger → AgentActivity | N:1 | refs | edge `GENERATED` |
| `transitionedTo` | PatientJourney → PatientJourney | self | transition dataset | edge `TRANSITIONED_TO` |
| `surveyedBy` | HealthcareProvider → HcpIntentSurvey | 1:N | FK | (FK) |
| **`indirectlyInfluences`** (derived) | (any) → (any) | M:N | **Function output** | inferred edge `INDIRECTLY_INFLUENCES` |
| **`highPotentialPrescriber`** (derived) | HealthcareProvider → Brand | M:N | **Function output** | inferred edge `HIGH_POTENTIAL_PRESCRIBER` |

> M:N links require **link-definition (join) datasets**; produce them in Pipeline Builder (§8). The two **inferred** edges become **scheduled Functions** (§7.2) writing derived link datasets — the Foundry equivalent of `inference_engine.materialize_inferred_relationships()`.

---

## 6. Action-type catalog (kinetic layer)

Each action = parameters + rules (object/link edits) + submission criteria (governance) + side effects (webhook/notification) + permissions. These replace direct DB writes in `src/api/routes/*` and agent write-paths.

| Action API name | Parameters | Rules (edits) | Submission criteria | Side effects | Replaces |
|---|---|---|---|---|---|
| `deliverTrigger` | `trigger: Trigger`, `channel: enum`, `recipientHcp: HealthcareProvider` | set `Trigger.status='delivered'`, `deliveryChannel`; create `receivedTrigger` link | `Trigger.status='pending'` AND `expirationDate > now()` AND user role ∈ {field, admin} | **Webhook** → Veeva/CRM; notify rep | `triggers` write in trigger flow |
| `recordTriggerResponse` | `trigger`, `acceptance: enum`, `actionTaken: string` | set `acceptanceStatus`, `status` | trigger was `delivered` | feed `feedback_learner` | `/feedback` route |
| `approveExperiment` | `experiment`, `reviewer` | `Experiment.state: draft→approved` | role=admin AND `power ≥ 0.8` | notify owner | experiment design write |
| `startExperiment` | `experiment` | `state: approved→running`, set `startDate` | state=approved | schedule monitor | `/experiments/{id}/randomize` |
| `stopExperiment` | `experiment`, `reason: enum`, `decision: enum` | `state: running→stopped_*`, set `stoppingReason`,`decision` | role∈{analyst,admin}; reason∈allowed | notify; trigger analysis | experiment_monitor |
| `promoteModel` | `model: MlModel`, `targetStage: enum` | `MlModel.stage→production`, `isChampion=true` (single-champion) | `auc ≥ MlExperiment.minimumAuc` AND fairness pass AND expert-gate approved | **Webhook** → BentoML/live-deployment promote | `model_deployer` |
| `recordCausalValidation` | `estimate: CausalEstimate`, `verdict: enum` | set `validationStatus`,`gateDecision` | refutation tests attached | feed expert gate | `validation_outcome.py` |
| `approveCausalClaim` | `estimate`, `expert` | `gateDecision: review→proceed` | role=expert AND `confidence ≥ 0.75` | publish to dashboards | `expert_review_gate.py` |
| `recordResourceAllocation` | `territory`, `plan: struct` | create `ResourceAllocation` object | budget ≤ cap | notify ops | `resource_optimizer` |
| `annotateOutcome` | `journey: PatientJourney`, `label: enum` | upsert `MlAnnotation` | role∈{physician,ds}; IAA logged | feedback_learner | `ml_annotations` write |
| `acknowledgeDriftAlert` | `alert`, `note` | set ack | role∈{ds,admin} | mute window | drift_monitor |
| `defineCohort` | `definition: struct`, `name` | create `Cohort`, compute members | valid filter | materialize cohort dataset | `cohort_constructor` |

**Writeback storage:** enable **Object Storage V2 edits** (toggle) so actions write back without a separate Phonograph dataset. Every action is audited natively (replaces `database/audit/audit_chain_tables.sql` hash chain — keep the hash chain during dual-run, retire in Phase 7).

---

## 7. Function & model catalog (compute layer)

### 7.1 KPI functions (46) → function-backed properties
Port `config/kpi_definitions.yaml` + `src/kpi/` calculators to **Functions** (TypeScript or Python). Two patterns:
- **Aggregation KPIs** (e.g. `source_coverage_patients = covered/reference`) → Functions over object sets → exposed as **function-backed properties** on `Kpi` / `Brand` / `Territory`.
- **Per-object scores** (e.g. HCP `digitalEngagementScore`) → function-backed property on `HealthcareProvider`.
Thresholds (target/warning/critical) become Function outputs + Workshop conditional formatting.

### 7.2 Inference-rule functions (5) → derived links/objects
Each `config/ontology/inference_rules.yaml` rule → a scheduled Function (the Foundry analogue of `inference_engine`):
| Rule | Function output | Schedule |
|---|---|---|
| `indirect_treatment` | derived `prescribesBrand` (min_patient_count ≥ 3) | daily |
| `causal_chain` | derived `indirectlyInfluences` (conf ≥ 0.50→0.25) | weekly |
| `hcp_influence_propagation` | derived `highPotentialPrescriber` | weekly |
| `patient_journey_progression` | `transitionProbability` on `transitionedTo` | monthly |
| `roi_opportunity_flagging` | `Opportunity` objects (conf ≥ 0.75, gap ≥ 0.10) | weekly |

### 7.3 Models → Modeling Objectives + live deployments + model functions
| E2I model | Foundry Model | Modeling Objective (eval gate) | Model function |
|---|---|---|---|
| Propensity / conversion (XGBoost/RF) | `hcp_conversion_model` | gate: `auc ≥ minimum_auc`, fairness | `predictConversion(hcp)` |
| Patient churn | `patient_churn_model` | gate as above | `predictChurn(journey)` |
| Trigger effectiveness | `trigger_effectiveness_model` | gate | `scoreTrigger(trigger)` |
| Causal CATE (EconML CausalForestDML) | `cate_model` | gate: refutation pass | `estimateCate(treatment,outcome,covars)` |
| Digital twin (RF/GBM/XGB/LGBM) | `digital_twin_model` | gate: fidelity ≥ grade | `simulate(intervention,cohort)` |
Model functions are imported into a Functions repo and wrap the **live deployment** REST endpoint — they replace `src/api/dependencies/bentoml_client.py` calls. Champion/challenger + promotion handled by the Modeling Objective + `promoteModel` action (§6).

### 7.4 The 21 agents → AIP Logic functions / Agent Studio agents
Each agent becomes an **AIP Logic function** (deterministic/tool-using) or an **Agent Studio agent** (conversational), composed of three tool kinds:
- **Object query tool** — scoped to the object types it reads (e.g. `causal_impact` → PatientJourney, TreatmentEvent, BusinessMetric).
- **Function tool** — calls model functions (§7.3) and KPI functions (§7.1).
- **Action tool** — performs write-backs (§6), auto or human-confirmed.

| Tier | Agent | AIP shape | Tools |
|---|---|---|---|
| 1 | `orchestrator` | Agent Studio (router) | object-query (interfaces `IBrandScoped`), function (intent classify) |
| 1 | `tool_composer` | AIP Logic | function (tool registry) |
| 2 | `causal_impact` | AIP Logic | object-query, function `estimateCate`, action `recordCausalValidation` |
| 2 | `gap_analyzer` | AIP Logic | object-query (BusinessMetric, ReferenceUniverse), KPI functions |
| 2 | `heterogeneous_optimizer` | AIP Logic | function `estimateCate`, action `recordResourceAllocation` |
| 3 | `drift_monitor` | scheduled Function | object-query (MlPrediction), action `acknowledgeDriftAlert` |
| 3 | `experiment_designer` | AIP Logic + `simulate` | function `simulate`, action `approveExperiment` |
| 3 | `experiment_monitor` | scheduled Function | object-query, action `stopExperiment` |
| 3 | `health_score` | scheduled Function | object-query (DataSource, MlModel) |
| 4 | `prediction_synthesizer` | AIP Logic | function (model fns), action (write `MlPrediction`) |
| 4 | `resource_optimizer` | AIP Logic | function, action `recordResourceAllocation` |
| 5 | `explainer` | AIP Logic | function `shap`, object-query |
| 5 | `feedback_learner` | scheduled Function | object-query (MlAnnotation), action `promoteModel` (retrain trigger) |
| 0 | ML-foundation (8) | mix of Pipeline + Functions + Modeling Objectives | (data/feature/model lifecycle, §8/§7.3) |

> **AIP availability is gated by Phase-0 A0/A3.** If AIP is not licensed or too slow, keep the agents in `src/agents/*` and have them call Foundry via **Python OSDK** + model-function REST (§10) — the ontology still provides the governed read/write surface.

---

## 8. Pipeline / ETL migration

**Goal:** every dataset that backs an object type (§4) or link type (§5) is produced by a Foundry pipeline, replacing `src/etl/*` + the Supabase load path.

### 8.1 Sources → Data Connection
| E2I source | Foundry ingestion |
|---|---|
| IQVIA APLD/LAAD, HealthVerity, Komodo (claims/lab/linkage) | **Data Connection** agent/connector → raw datasets (governed, PHI-marked at ingest) |
| Veeva CRM | Data Connection (API/file) |
| Synthetic generators (`src/ml/synthetic/generators.py`) | Pipeline Builder **generated data** input, or a **Code Repository** transform porting the generator classes (for dev/test branches) |

### 8.2 Transforms → Pipeline Builder / Code Repositories
Mirror the existing ETL as named pipelines, each emitting an **Ontology output**:
| Pipeline | Ports | Output (Ontology) |
|---|---|---|
| `pb_hcp_profiles` | `hcp_profiles` shaping | object `HealthcareProvider` |
| `pb_patient_journeys` | `patient_adherence_etl.py` (adherence_rate, refill_count, gap_days, is_churned) | object `PatientJourney` |
| `pb_treatment_events` | treatment shaping | object `TreatmentEvent` |
| `pb_business_metrics` | `business_metrics_per_hcp_etl.py` | object `BusinessMetric` (+ time series) |
| `pb_territory_metrics` | `territory_metrics_etl.py` | object `Territory` |
| `pb_triggers` | trigger shaping | object `Trigger` |
| `pb_links_*` | FK/M:N joins (`mapping join`) | link datasets `link_hcp_brand`, `link_hcp_influence`, `link_hcp_trigger`, `link_causal_kpi` |
| `pb_features_*` | Feast feature views (9) | feature datasets / function-backed props (§7.1) |

### 8.3 Data health (replace Pandera)
Port `src/data/` Pandera contracts to **Foundry data expectations / checks** on the pipeline outputs (schema, enum domains, FK integrity, journey-stage transition validity, no causal cycles). Wire to the branch CI gate (§11/§12).

### 8.4 Dual-write / sync during transition (strangler)
- **Outbound** (E2I→Foundry): add an optional sink in `src/repositories/base.py` that, behind a feature flag, also writes to the Foundry datasets (or stream) so the ontology stays current while Supabase remains system-of-record.
- **Inbound** (Foundry→E2I): action writeback datasets exported back to Supabase via scheduled sync, until cutover.
- Drop one direction at cutover (Phase 7).

---

## 9. Security & governance (PHI/PII-grade)

| E2I control | Foundry mechanism |
|---|---|
| Supabase RLS by role/brand/territory | **Granular policies** + **object security policies** (row/object-level) |
| Column-level PHI (`patient_hash`) | **Property security policies** + **mandatory control properties** → cell-level security |
| Brand segregation (field teams see only their brand) | **Markings**: `Remibrutinib`, `Fabhalta`, `Kisqali`; users carry brand marking; objects implement `IBrandScoped` |
| PHI/PII classification | **Markings**: `PHI`, `PII`; `PatientJourney.patientHash` requires `PII` |
| 365-day patient retention (Feast tag) | retention policy on `PatientJourney` restricted view |
| Audit hash chain (`database/audit`) | native **Action audit logs** (keep hash chain during dual-run; retire Phase 7) |
| JWT roles (analyst/admin/field/expert) | mapped to Foundry groups; submission criteria reference user attributes (§6) |

**Patient-level RWD** lands in a **Restricted View**; only pseudonymized `patientId` is broadly visible; `patientHash` is cell-secured. Confirm A1 (§2) before any PHI egress; if blocked, federate via **Virtual Tables** and keep PHI on-prem.

---

## 10. App & SDK surface

### 10.1 Generate OSDK (Developer Console)
- **Python OSDK** for backend services/agents (`src/foundry/osdk/`): used by FastAPI routers and (if agents stay in-repo) the agent layer.
- **TypeScript OSDK** for `frontend/`.
Select the object types (§4), link types (§5), and action types (§6) to include.

### 10.2 FastAPI strangler
Keep `src/api/main.py` and the ~29 routers as the stable external contract; swap their internals to OSDK / model-function calls. Example:
```python
# src/api/routes/causal.py  (after)
from src.foundry.osdk import FoundryClient
client = FoundryClient.from_env()

async def run_cate(req):
    cohort = client.ontology.objects.PatientJourney.where(
        brand=req.brand, data_split="train"
    )
    est = client.ontology.functions.estimate_cate(  # model function (§7.3)
        treatment=req.treatment, outcome=req.outcome, object_set=cohort
    )
    client.ontology.actions.record_causal_validation(  # action (§6)
        estimate=est.path_id, verdict="pending"
    )
    return est
```
Router-by-router cutover (each behind a flag): `causal.py`, `kpi.py`, `predictions.py`, `experiments.py`, `gaps.py`, `graph.py`, `monitoring.py`, `explain.py`, `agents.py`, `digital_twin.py`, `resource_optimizer.py`, …

### 10.3 Frontend
Two options per page (decide in §14): (a) keep React pages calling FastAPI (lowest churn), or (b) migrate read-heavy dashboards to **Workshop** and interactive ones to **OSDK-React**. Suggested split — Workshop: `Home`, `Analytics`, `Monitoring`, `SystemHealth`, `KPIDictionary`; OSDK-React: `CausalAnalysis`, `Experiments`, `GapAnalysis`, `KnowledgeGraph`, `AIAgentInsights`.

### 10.4 Dev lifecycle
- **Ontology-as-code** in `config/foundry/` (object/link/action/interface definitions), generated from existing `config/ontology/*.yaml` by a new exporter (§12).
- Promote changes via **ontology branches + proposals** (PR-like): each repo PR that touches the ontology maps to a Foundry proposal with reviewers.
- Package the finished ontology + pipelines + apps as a **Marketplace product** for repeatable deployment across environments/brands.

---

## 11. Phased roadmap

> Estimates are engineering-weeks for a small team (2–3 eng + 1 Foundry SME), **assuming Phase-0 survives**. Each phase has an exit gate; failing it pauses promotion.

| Phase | Goal | Key deliverables | Exit criteria | Rollback |
|---|---|---|---|---|
| **0 — Spike** (1–2 wk) | Disprove A0–A3 (§2) | One vertical slice (HCP+treats+deliverTrigger+OSDK), one model function, one AIP agent, latency numbers | All 4 assumptions survive in faithful tenant | Abandon / re-scope to "ontology-only" |
| **1 — Foundation** (2–4 wk) | Tenant ready | Namespace, branch, Data Connection to 1 source, Markings (PHI/PII/brand), groups↔roles, CI on branch | Source dataset lands + secured; proposal workflow works | — |
| **2 — Semantic layer** (4–6 wk) | Core objects/links | `pb_*` pipelines + object types HCP/PatientJourney/Treatment/Brand/Region/Territory/Trigger; link datasets; data health checks | Parity §13 on object counts + spot rows | keep Supabase SoR |
| **3 — Derived & ML layer** (4–8 wk) | Features + models | Feature pipelines; KPI functions (46); Models+Objectives+live deployments+model functions (§7.3) | KPI parity within tolerance; model AUC matches MLflow | disable function-backed props |
| **4 — Causal layer** (4–8 wk) | Causal as functions | `CausalEstimate` object; `estimateCate` model fn; refutation gate; `recordCausalValidation`/`approveCausalClaim` actions; inference-rule functions (§7.2) | ATE/CATE parity vs `src/causal_engine`; DAG acyclicity check passes | keep in-proc engine |
| **5 — Kinetic layer** (6–10 wk) | Actions + agents | Action types (§6); AIP Logic/Agent Studio for the 21 agents (§7.4); writeback (OSV2 edits) | Agent outputs match e2i on golden queries; actions audited | keep `src/agents/*` calling OSDK |
| **6 — App surface** (6–10 wk) | OSDK/Workshop | Python+TS OSDK; FastAPI routers cut over (flagged); first Workshop + OSDK-React pages | UI parity on 5 flagship pages | flag back to old internals |
| **7 — Cutover & decommission** (4–8 wk) | Flip SoR | Dual-run → flip system-of-record to Foundry; retire superseded FalkorDB/bespoke pieces **only with parity proof**; Marketplace package | 2 weeks clean dual-run; sign-off | re-flip to Supabase |

Phases 2–6 can overlap per object-type slice (vertical slices ship independently).

---

## 12. Changes in THIS repo (`e2i_causal_analytics`)

New/changed code to support the migration without deleting the working system:

```
src/foundry/                      # NEW — Foundry integration package
  __init__.py
  config.py                       # tenant URL, client creds (env), feature flags
  osdk/                           # generated Python OSDK (Developer Console output)
  ontology_sync.py                # dual-write Supabase <-> Foundry datasets (§8.4)
  action_clients.py               # typed wrappers for action types (§6)
  model_functions.py              # adapters for live-deployment model fns (§7.3)
config/foundry/                   # NEW — ontology-as-code (generated, §10.4)
  object_types.yaml  link_types.yaml  action_types.yaml  interfaces.yaml  markings.yaml
```

Reuse, don't rewrite:
- `src/ontology/schema_compiler.py` → **add `FoundryOntologyExporter`** that emits `config/foundry/*.yaml` from the existing `config/ontology/*.yaml` (single source of truth preserved).
- `src/ontology/validator.py` → **add `FoundryParityValidator`** (object/link/enum parity e2i↔Foundry; runs in CI).
- `src/ontology/vocabulary_registry.py` → source of **shared property types / enums** for the exporter.
- `src/repositories/base.py` → optional Foundry sink behind `FOUNDRY_DUAL_WRITE` flag.
- `feature_repo/` → add Foundry dataset materialization target alongside Feast.
- `src/api/routes/*` → per-router OSDK delegation behind `FOUNDRY_BACKEND_<route>` flags (§10.2).
- `.github/workflows/` → new job: run `FoundryParityValidator` + data-health checks; open/update the matching **ontology proposal**.
- `tests/foundry_parity/` → §13 harness.

---

## 13. Parity, testing & acceptance

- **Golden synthetic dataset** (from `src/ml/synthetic/`) loaded identically into Supabase and Foundry.
- **Parity harness** (`tests/foundry_parity/`): for the same inputs, compare e2i vs Foundry on:
  - object/link **counts** and spot-checked rows (Phase 2),
  - **46 KPI** values within tolerance (Phase 3),
  - **model AUC/PR-AUC** vs `ml_model_registry` (Phase 3),
  - **ATE/CATE + refutation verdicts** vs `src/causal_engine` (Phase 4),
  - **agent outputs** on a fixed query suite (Phase 5),
  - **UI snapshots** on flagship pages (Phase 6).
- **Acceptance gate per phase** = parity within tolerance + data-health green + security review (Markings/Restricted Views enforced) + action audit present.
- **No bespoke component is retired** (Phase 7) until its Foundry replacement passes parity — directly enforces §1.5.

---

## 14. Open decisions (need your call)

1. **Scope:** full re-platform (Foundry = system-of-record, Workshop/OSDK frontends, AIP agents) **vs** integration layer (Foundry ontology as governed semantic/compute layer; keep FastAPI/React). *Recommended: integration-first, full re-platform as end-state.*
2. **Compute home:** agents/causal in **AIP/Functions** vs staying in `src/*` calling Foundry via OSDK. *Driven by Phase-0 A3 latency.*
3. **PHI residency:** patient-level RWD into Foundry vs pseudonymized-only vs federated Virtual Tables. *Driven by A1.*
4. **Frontend:** Workshop vs OSDK-React vs keep-React-on-FastAPI, per page (§10.3).
5. **System-of-record timing:** when to flip Supabase→Foundry (Phase 7 trigger).

---

## 15. Risks

| Risk | Impact | Mitigation |
|---|---|---|
| AIP / Model Integration / OSDK not licensed | guts §5/§7 | Phase-0 A0; integration-only fallback |
| PHI cannot egress to tenant | blocks Patient objects | A1; pseudonymize / federate |
| Model-function latency ≫ in-proc causal engine | agent loop unusable | A3; keep compute in `src/*` |
| Python OSDK feature gaps (interfaces still maturing) | backend ergonomics | use TS where needed; REST fallback |
| Dual-write drift Supabase↔Foundry | data divergence | parity harness + scheduled reconciliation |
| Cost (live deployments reserve compute) | budget | batch where possible; scale-to-zero dev |
| "Lift-and-shift" temptation to delete FalkorDB early | regression | §1.5 + Phase-7 parity gate |

---

## 16. References

**Palantir Foundry docs**
- Ontology overview — https://www.palantir.com/docs/foundry/ontology/overview
- Core concepts — https://www.palantir.com/docs/foundry/ontology/core-concepts
- Object & link types — https://www.palantir.com/docs/foundry/object-link-types/object-types-overview · https://www.palantir.com/docs/foundry/object-link-types/link-types-overview · base types https://www.palantir.com/docs/foundry/object-link-types/base-types
- Interfaces — https://www.palantir.com/docs/foundry/interfaces/interface-overview
- Action types — overview https://www.palantir.com/docs/foundry/action-types/overview · parameters https://www.palantir.com/docs/foundry/action-types/parameter-overview · rules https://www.palantir.com/docs/foundry/action-types/rules · submission criteria https://www.palantir.com/docs/foundry/action-types/submission-criteria
- Functions on models — https://www.palantir.com/docs/foundry/functions/functions-on-models
- Model integration — https://www.palantir.com/docs/foundry/model-integration/overview · objectives https://www.palantir.com/docs/foundry/model-integration/objectives · model functions https://www.palantir.com/docs/foundry/model-integration/model-functions-guide
- AIP — overview https://www.palantir.com/docs/foundry/aip/overview · AIP Logic https://www.palantir.com/docs/foundry/logic/overview · Agent Studio tools https://www.palantir.com/docs/foundry/agent-studio/tools
- Pipeline Builder — overview https://www.palantir.com/docs/foundry/pipeline-builder/overview · Ontology output https://www.palantir.com/docs/foundry/pipeline-builder/outputs-add-ontology-output
- Ontology SDK (OSDK) — https://www.palantir.com/docs/foundry/ontology-sdk/overview · Python https://www.palantir.com/docs/foundry/ontology-sdk/python-osdk · Developer Console https://www.palantir.com/docs/foundry/developer-console/overview
- Security — Restricted Views https://www.palantir.com/docs/foundry/security/restricted-views · granular policies https://www.palantir.com/docs/foundry/platform-security-management/manage-granular-policies · object security https://www.palantir.com/docs/foundry/object-permissioning/object-security-policies · mandatory control properties https://www.palantir.com/docs/foundry/object-link-types/mandatory-control-properties
- Branching & proposals — https://www.palantir.com/docs/foundry/ontologies/ontologies-proposals · https://www.palantir.com/docs/foundry/foundry-branching/branching-lifecycle-usage
- Time series / geotemporal / media — https://www.palantir.com/docs/foundry/object-link-types/base-types · https://www.palantir.com/docs/foundry/geotemporal-series/data-modeling

**E2I repo anchors**
- Core schema `database/core/e2i_ml_complete_v3_schema.sql`; ML schema `database/ml/*`
- Existing ontology `src/ontology/{schema_compiler,validator,vocabulary_registry,inference_engine,query_extractor,grafiti_config}.py`; `config/ontology/*.yaml`
- ETL `src/etl/*`; features `feature_repo/*`, `src/feature_store/*`
- Causal engine `src/causal_engine/*`; agents `src/agents/factory.py` + `src/agents/*`
- API `src/api/main.py` + `src/api/routes/*`; frontend `frontend/src/{api,pages}/*`
- Data dictionaries `docs/data/0{2,3,4,5,6}-*.md`; KPIs `config/kpi_definitions.yaml`

---

*This plan honors `CLAUDE.md`: §1 (reason-before-rules — the existing ontology is the spec, not a delete target) and §2 (cheapest-disproof-first — Phase 0 falsifies the core assumptions before any build). Treat §11 estimates as unverified hypotheses until Phase 0 returns measured numbers.*
