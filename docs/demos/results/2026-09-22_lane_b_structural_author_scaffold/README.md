# Lane B — structural author scaffold (2026-09-22)

Spec: `docs/superpowers/specs/2026-09-22-real-data-causal-estimation-design.md`
(worktree `real-data-causal`), "### Lane B — structural author with human
validation" items 1–5, §5, §6 (Lane B line), §7. Branch
`claude/lane-b-structural-author-scaffold`, base `31a4c5b6d` (origin/main).

No paid LLM call and no prod write was made. Every number below cites the
captured file:line in this directory. Every capture was regenerated on the
clean committed tree `edb96fbd2` (codex r2 HIGH 4): `disproofs.txt:1` says
`dirty_src_scripts_tests=no`, and every `score.json` / `authored.json` /
`attestations.json` / `dag.json` carries `meta.tree` = `{commit, dirty_src_scripts_tests}`
(`summary.md:6` and `review.md:6` repeat it), so a capture from a dirty tree
says so on its face.

## Cheapest disproofs run before building

Captured by `disproofs.py` → `disproofs.txt` (re-runnable from the worktree
root; `disproofs.txt:1` names the commit it ran at and whether `src/`, `scripts/`
or `tests/` were dirty at that moment — the committed capture says `no`).

| # | Assumption the deliverable rests on | Result (cited) |
|---|---|---|
| 1 | A DSPy signature with typed list / bool outputs can be driven by `dspy.utils.dummies.DummyLM` (so the parser and CLIs are testable without a paid LM) | `disproofs.txt:2` — dspy 3.1.0 parsed `[['f','T'],['f','Y'],['T','Y']]` as `list`, `False` as `bool` — survives |
| 2 | The graph builder's backdoor finder can be reused by the assembler | `disproofs.txt:3` — importing the node costs 13.1 s (the whole agent package); `disproofs.txt:4` — the extracted `src/ml/causal_role_dgp/backdoor.py` imports in 0.000 s → the node now delegates to it (pinned by `tests/unit/test_agents/test_causal_impact/test_graph_builder_backdoor_shared.py`) |
| 3 | `CitationResolver.verify_citation` can be exercised offline | `disproofs.txt:5` — `__init__(*, europe_pmc=None, crossref=None, umls=None)`: clients are injectable; the grader takes any object with `verify_citation` |
| 4 | Under the unit tree's dead-Supabase pin a `--review` cannot silently succeed | `disproofs.txt:6` — `repo.client=None`, `create_review -> None` in 0.0 s → the CLI treats `None` as failure (exit 3, files kept), pinned by `tests/unit/test_scripts/test_author_cohort_dag.py::test_review_under_the_dead_supabase_pin_fails_loudly_and_keeps_files` |
| 5 | The guide can be read from `docs/` at runtime inside the API container | `disproofs.txt:7` — `.dockerignore:76 == 'docs/'`: the docs tree is NOT in the image → sections 0–6 are embedded (`src/data/kg/_structural_author_guide.py`) and pinned byte-for-byte by `test_guide_sections_are_verbatim`; likewise the item-5 loader reads the approved DAG from the review row, not from `docs/layer4/generated/` |
| 6 | dspy keeps the docstring verbatim as instructions | `disproofs.txt:8` — 16545 vs 16546 chars, equal after `.strip()`: `inspect.cleandoc` drops the trailing newline only |
| 7 | The node edit stays under the module-size ratchet | `disproofs.txt:9` — 4237 lines (pin lowered from 4238 to 4237 in `tests/unit/test_tests_meta/test_module_size_ratchet.py`; the provenance helper lives in `src/ml/causal_role_dgp/extractor.py`) |
| 8 | (codex r2 MED 1) The assembler needs a fallback search because a full candidate set can fail the backdoor criterion while a proper subset is admissible | `nonmonotone_search.txt:3` — `unions checked: 19521; full-fails-but-subset-passes cases: 0` over every union of 2 classified fragments (≤3 authored edges each) and 3 fragments (≤2 edges) sharing latents by name (`nonmonotone_search.py`). Rebutted: in the fragment vocabulary an ancestor/confounder candidate always reaches Y, so a latent parent that also reaches T forms a chain only the candidate blocks, and the path conditioning opens runs through latents — a failing full set means no observed subset is admissible, which is what the assembler reports |

Red proof for the provenance field (T1): `red_t1_provenance.txt:2` —
`TypeError: ... unexpected keyword argument 'provenance'` at `31a4c5b6d`.

## Dry runs with the fake LM (`model_id=dummy`, `provenance=machine`)

### Scorer on the golden fixtures (spec item 3, gate = zero missed leaks)

`python -m scripts.measure_structural_author --lm fake --fake-source replay --cohort CSU_remibrutinib --out measure_fake_csu`
replays the committed CSU blind authored edges through the full pipeline
(parse → `extract_role` → grade → stamp → score):

- `measure_fake_csu/summary.md:10` — `PASS: gate missed_leaks == 0 — missed leaks 0 (rate 0.000 over 14 scored golden-leak features; 31 scored, 0 routed to review, n=31)`
- `measure_fake_csu/summary.md:11` — `exact role agreement 28/31 (0.903); leak-decision agreement 31/31 (1.000)` — identical to the committed record `docs/layer4/csu_golden_validation_review_record.json` (n 31, exact 28, missed 0).

`--cohort all` (91 briefs; the 60 non-CSU briefs get a stand-in confounder
fragment, clearly not an authored claim):

- `measure_fake_all91/summary.md:10` — `FAIL: gate missed_leaks == 0 — missed leaks 28 (rate 0.667 over 42 scored golden-leak features; ...)`, exit code 2: the gate has teeth on a stand-in author (28 = every leak-role feature of the PNH and BC cohorts; the rate is over the 42 golden leak features, not over all 91), and "the report is the deliverable" is the exit path.

### Cost estimate for the REAL benchmark (owner decision)

`measure_real_refused/` — `--lm real` without `--i-accept-cost` exits 3 after
writing the estimate (no LM call): `cost_estimate.json:5` prompt tokens
531,375 (measured from the real ChatAdapter prompt at 4 chars/token,
~23.4k chars per brief because the guide's sections 0–6 ride in every call),
output tokens 81,900 assumed (900/brief), `cost_estimate.json:11` ≈ USD 1.72
at ASSUMED 2.00/8.00 USD per Mtok. The rates are placeholders — pass the list
price of `DSPY_LM_MODEL` (`openai/gpt-5.6-terra` in `.env`) with
`--usd-per-mtok-in/--usd-per-mtok-out`.

No Lane E panel enters this benchmark, by construction: the 91 golden
briefs are literature-derived fixtures (label sets for ConcertAI CSU / PNH /
BC cohorts that exist as no frame on this platform), so the four voters have
nothing to run on and no panel can exist for them; the author gets the brief
alone, exactly what a real feature gets when its panel record is absent. The
panel is required on the cohort runs below (codex r2 MED 3, rebutted on this
ground and documented in the script).

Command for the real run (not executed):

    python -m scripts.measure_structural_author --lm real --i-accept-cost --resolver live \
        --cohort all --out docs/demos/results/<date>_structural_author_benchmark

### Cohort authoring CLI (spec item 4), fake LM

Run (b) shape, `python -m scripts.author_cohort_dag --manifest optum --treatment biologic_initiation --outcome initiated_biologic_180d --lm fake --diff-manifest-attestations --no-assumption --out-root author_fake`:

- `author_fake/optum_biologic_initiation_initiated_biologic_180d/manifest_diff.json:3-6` — `n_features 110, n_compared 110, edge_exact_agreement 93, role_agreement 93`; `disproofs.txt:10` — the 17 disagreements are exactly the manifest's 17 `_OPTUM_INSTRUMENT_FEATURES` (the fake author draws a confounder for everything). Each disagreement carries the author's reasoning and the manifest side's feature-specific grounding (the family bullet of `docs/layer4/optum_initiation_attestation_research.md` that names the feature, with its PMIDs and line).

Run (a) shape, `--manifest optum_mart --treatment treatment_dupixent --outcome persistent_at_180d_g28 --treatment-label "remibrutinib vs competitor biologic (CSU escalation therapy; rehearsed as Dupixent vs Xolair)" --lm fake`:

- `author_fake/optum_mart_treatment_dupixent_persistent_at_180d_g28/dag.json:9,13,25,26` — `lm "fake"`, 64 features, `is_dag true`, `adjustment_valid true`.
- `.../review.md` — the reviewer checklist item `escalation_decision_point` (the stated assumption), the per-feature table, `## Review items (0)` at line 86 (a stand-in author has nothing ambiguous to say).

Commands for the real runs (not executed; paid LLM + a prod `expert_reviews` write each):

    python -m scripts.author_cohort_dag --manifest optum_mart --treatment treatment_dupixent \
        --outcome persistent_at_180d_g28 \
        --treatment-label "remibrutinib vs competitor biologic (CSU escalation therapy)" \
        --outcome-label "persistence at 180 days (g28)" \
        --panel <Lane E panel.json for optum_mart/treatment_dupixent/persistent_at_180d_g28> \
        --lm real --i-accept-cost --resolver live --review

    python -m scripts.author_cohort_dag --manifest optum --treatment biologic_initiation \
        --outcome initiated_biologic_180d --lm real --i-accept-cost --resolver live \
        --allow-no-panel --no-assumption --diff-manifest-attestations

(run (b) has no Lane E panel — the `optum` manifest's causal panel is not
part of the program — hence `--allow-no-panel`, an explicit override; a real
run without `--panel` is refused otherwise.)

Cost per run at the same measured prompt size (`disproofs.txt:11`): (a) 64
briefs ≈ 0.70 × the 91-brief estimate; (b) 110 briefs ≈ 1.21 × it.

Brand mapping for the prior (codex r1 MED 3): the loader looks the approved
review up by the estimand key `lower(brand):treatment:outcome`, trying the
run's brand first and then the brandless key. Run (a) above is therefore
created WITHOUT `--brand` so that both a brandless and a branded Lane A
request find it; pass `--brand` only when the consuming requests carry that
exact brand.

## What is deliberately NOT in this directory

- No real-LM authored fragments, no benchmark on the real author (owner
  decision, spec §3 Lane B item 3).
- No `expert_reviews` row (owner decision, item 4).
- No Lane E panel: its branch (`claude/lane-e-feature-role-voters`) is not on
  main; the author consumes its `FeatureRoleRecord.to_dict()` shape through
  the typed adapter `PanelRecordView` (field names copied from the Lane E
  interface report) and the CLI takes `--panel panel.json`.
