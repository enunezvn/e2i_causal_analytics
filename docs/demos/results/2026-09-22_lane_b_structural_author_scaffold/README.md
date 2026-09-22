# Lane B — structural author scaffold (2026-09-22)

Spec: `docs/superpowers/specs/2026-09-22-real-data-causal-estimation-design.md`
(worktree `real-data-causal`), "### Lane B — structural author with human
validation" items 1–5, §5, §6 (Lane B line), §7. Branch
`claude/lane-b-structural-author-scaffold`, base `31a4c5b6d` (origin/main).

No paid LLM call and no prod write was made. Every number below cites the
captured file:line in this directory.

## Cheapest disproofs run before building

| # | Assumption the deliverable rests on | Experiment | Result |
|---|---|---|---|
| 1 | A DSPy signature with typed list / bool outputs can be driven by `dspy.utils.dummies.DummyLM` (so the parser and CLI can be tested without a paid LM) | one `dspy.Predict` under `DummyLM` with `edges: list[list[str]]`, `ambiguous: bool` (session probe, dspy 3.1.0) | parsed `[['f','T'],['f','Y'],['T','Y']]` as `list`, `False` as `bool` — survives |
| 2 | The graph builder's backdoor finder can be reused by the assembler | `import src.agents.causal_impact.nodes.graph_builder` timed with `-X importtime` | 14.9 s (the whole agent package) → the criterion was extracted to `src/ml/causal_role_dgp/backdoor.py` (light) and the node delegates to it; the delegation is pinned by `tests/unit/test_agents/test_causal_impact/test_graph_builder_backdoor_shared.py` |
| 3 | `CitationResolver.verify_citation` can be exercised offline | `inspect.signature(CitationResolver.__init__)` | `(*, europe_pmc=None, crossref=None, umls=None)` — clients are injectable; the grader takes any object with `verify_citation` |
| 4 | Under the unit tree's dead-Supabase pin a `--review` cannot silently succeed | `ExpertReviewRepository().create_review(...)` with `SUPABASE_URL=http://127.0.0.1:1` | returns `None` in 0.0 s ("No Supabase client, skipping review creation") → the CLI treats `None` as failure (exit 3, files kept), pinned by `tests/unit/test_scripts/test_author_cohort_dag.py::test_review_under_the_dead_supabase_pin_fails_loudly_and_keeps_files` |
| 5 | The guide can be read from `docs/` at runtime inside the API container | `.dockerignore:76` is `docs/` | the docs tree is NOT in the image → sections 0–6 are embedded (`src/data/kg/_structural_author_guide.py`) and pinned byte-for-byte to the guide by `test_guide_sections_are_verbatim`; likewise the item-5 loader reads roles from the review row, not from `docs/layer4/generated/` |
| 6 | dspy keeps the docstring verbatim as instructions | `StructuralAttestationSignature.instructions` vs the constant | 16545 vs 16546 chars: `inspect.cleandoc` drops the trailing newline only (the test compares to `.strip()`) |

Red proof for the provenance field (T1): `red_t1_provenance.txt:2` —
`TypeError: ... unexpected keyword argument 'provenance'` at `31a4c5b6d`.

## Dry runs with the fake LM (`model_id=dummy`, `provenance=machine`)

### Scorer on the golden fixtures (spec item 3, gate = zero missed leaks)

`python -m scripts.measure_structural_author --lm fake --fake-source replay --cohort CSU_remibrutinib --out measure_fake_csu`
replays the committed CSU blind authored edges through the full pipeline
(parse → `extract_role` → grade → stamp → score):

- `measure_fake_csu/summary.md:9` — `PASS: gate missed_leaks == 0 — missed leaks 0 (rate 0.000 over 31 scored, 0 routed to review, n=31)`
- `measure_fake_csu/summary.md:10` — `exact role agreement 28/31 (0.903); leak-decision agreement 31/31 (1.000)` — identical to the committed record `docs/layer4/csu_golden_validation_review_record.json` (n 31, exact 28, missed 0).

`--cohort all` (91 briefs; the 60 non-CSU briefs get a stand-in confounder
fragment, clearly not an authored claim):

- `measure_fake_all91/summary.md:9` — `FAIL: gate missed_leaks == 0 — missed leaks 28 (rate 0.308 over 91 scored ...)`, exit code 2: the gate has teeth on a stand-in author (28 = every leak-role feature of the PNH and BC cohorts), and "the report is the deliverable" is the exit path.

### Cost estimate for the REAL benchmark (owner decision)

`measure_real_refused/` — `--lm real` without `--i-accept-cost` exits 3 after
writing the estimate (no LM call): `cost_estimate.json:5` prompt tokens
531,375 (measured from the real ChatAdapter prompt at 4 chars/token,
~23.4k chars per brief because the guide's sections 0–6 ride in every call),
output tokens 81,900 assumed (900/brief), `cost_estimate.json:11` ≈ USD 1.72
at ASSUMED 2.00/8.00 USD per Mtok. The rates are placeholders — pass the list
price of `DSPY_LM_MODEL` (`openai/gpt-5.6-terra` in `.env`) with
`--usd-per-mtok-in/--usd-per-mtok-out`.

Command for the real run (not executed):

    python -m scripts.measure_structural_author --lm real --i-accept-cost --resolver live \
        --cohort all --out docs/demos/results/<date>_structural_author_benchmark

### Cohort authoring CLI (spec item 4), fake LM

Run (b) shape, `python -m scripts.author_cohort_dag --manifest optum --treatment biologic_initiation --outcome initiated_biologic_180d --lm fake --diff-manifest-attestations --no-assumption --out-root author_fake`:

- `author_fake/optum_biologic_initiation_initiated_biologic_180d/manifest_diff.json:3-6` — `n_features 110, n_compared 110, edge_exact_agreement 93, role_agreement 93`; the 17 disagreements are exactly the manifest's 17 `_OPTUM_INSTRUMENT_FEATURES` (the fake author draws a confounder for everything), verified in-session against `src/data/manifests/optum_feature_manifest.py`.

Run (a) shape, `--manifest optum_mart --treatment treatment_dupixent --outcome persistent_at_180d_g28 --treatment-label "remibrutinib vs competitor biologic (CSU escalation therapy; rehearsed as Dupixent vs Xolair)" --lm fake`:

- `author_fake/optum_mart_treatment_dupixent_persistent_at_180d_g28/dag.json:9,13,21,22` — `lm "fake"`, 64 features, `is_dag true`, `adjustment_valid true`.
- `.../review.md` — the reviewer checklist item `escalation_decision_point` (the stated assumption), the per-feature table, `## Review items (0)` at line 85 (a stand-in author has nothing ambiguous to say).

Commands for the real runs (not executed; paid LLM + a prod `expert_reviews` write each):

    python -m scripts.author_cohort_dag --manifest optum_mart --treatment treatment_dupixent \
        --outcome persistent_at_180d_g28 \
        --treatment-label "remibrutinib vs competitor biologic (CSU escalation therapy)" \
        --outcome-label "persistence at 180 days (g28)" \
        --panel <Lane E panel.json for optum_mart/treatment_dupixent/persistent_at_180d_g28> \
        --lm real --i-accept-cost --resolver live --brand Remibrutinib --review

    python -m scripts.author_cohort_dag --manifest optum --treatment biologic_initiation \
        --outcome initiated_biologic_180d --lm real --i-accept-cost --resolver live \
        --no-assumption --diff-manifest-attestations

Cost per run at the same measured prompt size: (a) 64 briefs ≈ 0.70 × the
91-brief estimate; (b) 110 briefs ≈ 1.21 × it.

## What is deliberately NOT in this directory

- No real-LM authored fragments, no benchmark on the real author (owner
  decision, spec §3 Lane B item 3).
- No `expert_reviews` row (owner decision, item 4).
- No Lane E panel: its branch (`claude/lane-e-feature-role-voters`) is not on
  main; the author consumes its `FeatureRoleRecord.to_dict()` shape through
  the typed adapter `PanelRecordView` (field names copied from the Lane E
  interface report) and the CLI takes `--panel panel.json`.
