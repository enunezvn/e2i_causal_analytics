# KG Layer-2 cache — build, commit, verify

The adaptive-validity node's Layer 2 reads an **offline** knowledge-graph cache
rather than calling UMLS / RxNav / Open Targets on the hot path. This runbook
covers building that cache, why it is committed rather than provisioned, and
how to tell a genuine "no signal" from a dark layer.

## What the cache is for

`EnsembleVoter.classify_kg_signal` answers one question per feature:

> Does the knowledge graph relate **this feature's concept** to **the
> prediction target**?

It has two ways to say yes:

| Signal | Means | Source |
|---|---|---|
| `leak_drug_treats_disease` | the target drug is approved to treat the feature's disease | Open Targets indications |
| `taxonomic_descendant` | the feature concept is a parent/child of the target concept | UMLS relations |

A `treats` edge is a leakage finding: if the target drug treats urticaria, then
a post-index urticaria diagnosis-code count is partly a record of treatment.

## Building it

```bash
export UMLS_UTS_API_KEY=...      # required; RxNav + Open Targets are zero-auth

python scripts/build_kg_cache.py --live \
    --manifest-module src.data.manifests.optum_feature_manifest \
    --features-attr OPTUM_FEATURES \
    --target-entity-codes RXNORM:302379 \
    --out data/kg_cache
```

Writes `{manifest_sha8}__{target_sha8}.json` plus a `.summary.md` report. The
filename embeds fingerprints of **both** the manifest and the target codes, so
changing either produces a different file — two cohorts sharing both
legitimately share one cache.

`--target-entity-codes` is load-bearing, not decoration. The build resolves it
to a drug:

1. **RxNav** turns `RXNORM:302379` into a name (`omalizumab`).
2. **Open Targets** turns that name into a ChEMBL id, and each feature's UMLS
   concept into a MONDO/EFO disease id.
3. `query_drug_disease_edges` asks which of those diseases the drug is approved
   to treat.

If the target does not resolve to a drug, the build still succeeds but logs a
warning and emits **taxonomic edges only** — which, on their own, can never
produce a signal (see below).

### Auditing a flagged feature

The drug-disease pass rewrites edge endpoints to the manifest CUI and the scope's
target code, because `classify_kg_signal._connects` compares against those and
would never match a raw ChEMBL/MONDO id. The identifiers the source actually
spoke are preserved on `source_subject_id` / `source_object_id`, which the cache
file persists.

Use them. The feature's CUI is mapped to a disease by a fuzzy
`open_targets.search_disease(preferred_name)` lookup, and a broad or wrong
EFO/MONDO match still yields a perfectly plausible `object_name` — so a name
alone cannot tell you whether the match was right. These edges drive a leakage
finding that can drop a feature, so check the id:

```bash
jq '.[] | select(.feature_name=="dx_total_csu") | .edges[]
    | select(.evidence_source=="open_targets")
    | {predicate, object_name, source_object_id, source_subject_id}' \
  data/kg_cache/1cdaa038__96bfd2e0.json
```

Cache files written before #1607 have no such keys and load with both as `None`.

### Why taxonomic edges alone are not enough

`query_disease_hierarchy(cui)` returns a concept's own parents and children. It
never relates the feature to the *target*, and `classify_kg_signal._connects`
requires one edge endpoint in the feature set and one in the target set.

Measured 2026-08-14: a taxonomic-only build over the Optum manifest produced 74
records with 82 real UMLS edges, and **all 74 features classified as
`no_signal`** — with an RXNORM target and again with a valid UMLS disease CUI as
the target. Building a cache and setting `kg_mode="shadow"` is therefore *not*
sufficient on its own; the drug-disease pass is what lights the signal.

## Why the cache is committed

`data/kg_cache/*.json` is gitignored, with an explicit un-ignore for the one
artifact the pipeline loads. This is deliberate.

Issue #600 is the precedent: a gitignored tier0 cache was never committed and
silently skipped agent execution in CI. The failure presented as *quiet output*,
not as a missing file. The same shape here would look exactly like "the KG has
nothing to say about these features".

The artifact is ~72 KB and rebuilt rarely (only when the manifest's entity codes
or the cohort's target change), so committing it is cheaper and far more robust
than a provisioning step someone has to remember.

To commit a rebuild under a new fingerprint, add the new filename to the
un-ignore list in `.gitignore` and update `KG_ACTIVATIONS` in
`src/data/kg/activation.py` — the tests in
`tests/integration/test_kg/test_kg_layer2_activation.py` fail loudly if the two
drift apart.

## Activation

`src/data/kg/activation.py` binds a feature-manifest source to its cache and is
applied inside the adaptive-validity node, so every entry point gets the same
binding regardless of which runner assembled the `scope_spec`.

* An explicit `kg_mode="off"` always wins — activation never re-enables a cohort
  an operator turned off. Any *other* explicit mode (`shadow`, `promoted`) is
  preserved but still gets the cache bound: an explicit on-mode is a request to
  turn the layer on, and treating it as "hands off" left `kg_cache_path` unset,
  which is indistinguishable from the KG having nothing to say.
* A configured-but-missing cache logs an **ERROR** and leaves KG off. "No
  cache" and "no signal" must never look alike.

Current bindings:

| Manifest source | Cache | Target | Mode |
|---|---|---|---|
| `optum` | `1cdaa038__96bfd2e0.json` | `RXNORM:302379` (omalizumab) | `shadow` |

### Modes

* `off` — cache not loaded; verdicts never carry a KG signal.
* `shadow` — cache loaded, `decided_by="kg"` and `kg_signal` recorded for
  audit, but severity is capped to `info` so KG cannot drop a feature.
* `promoted` — KG participates in voter precedence normally and can drop
  features.

Promotion is **operator-driven** (`compute_promotion_eligibility`); there is no
auto-promote. `shadow` exists so signal quality can be observed on a real cohort
first.

## Verifying

```bash
# Offline: the committed artifact and its wiring
pytest tests/integration/test_kg/test_kg_layer2_activation.py

# Live: the three upstream APIs still match what the clients expect
pytest tests/integration/test_kg/test_kg_layer2_live_contracts.py
```

Expected on the committed Optum cache: **7 of 74** features carry
`leak_drug_treats_disease` — the urticaria diagnosis-code features
(`dx_l50_1/8/9_count`, `dx_total_csu`, `primary_diagnosis_code`) and the asthma
features (omalizumab is approved for asthma too). The other 67 are `no_signal`,
which is the honest answer for labs and utilisation counts.

A result of *zero* flagged features means something is wrong — most likely the
drug-disease pass did not run. Check the build log for
`drug-disease pass ENABLED`.

## Upstream drift

The live contract tests exist because Open Targets changed its schema under us
and nothing caught it — every unit test mocks the transport, so the drug-disease
query returned **HTTP 400 on every call** while the suite stayed green:

* `ClinicalIndicationFromDrug.maxPhaseForIndication` → `maxClinicalStage`
  (Int 0–4 → String `APPROVAL` / `PHASE_3` / …).
* The top-level `evidences(drugIds:, diseaseIds:)` Query field was **removed**.
  Evidence now hangs off `Disease.evidences`, which requires a gene
  `ensemblIds` argument and so cannot serve a drug→disease lookup. As a result
  edges no longer carry literature PMIDs or per-row scores.

Run the live contract tests after any upstream incident, and treat a failure
there as a real outage rather than a flaky test.
