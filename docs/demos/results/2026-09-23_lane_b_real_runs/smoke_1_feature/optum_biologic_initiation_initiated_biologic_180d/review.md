# Cohort DAG review: optum — biologic_initiation → initiated_biologic_180d

- treatment `biologic_initiation` = biologic_initiation
- outcome `initiated_biologic_180d` = initiated_biologic_180d
- author LM: `openai/gpt-5.6-terra`; prompt hash `1971d0d38fa1c895360eb48b3f5ab6bb53d8dff95fb9fea9e57eaa3b5c4bac7c`; guide hash `04009c46e34c2c457755f8ca322b25510ee8646c5caf32093e14b8ec7223889a`
- tree: commit 8cbc52ccae904bc16b53de1624d1428d0ff3441d (dirty src/scripts/tests: False)
- provenance: `machine` on every fragment (audit-only until this review is approved)
- features authored: 1; latents: none
- is a DAG: True; admissible observed adjustment set: True
- minimal adjustment set: ['age_at_index']
- full admissible set: ['age_at_index']

## Features

| feature | fragment role | cohort role | drift | ambiguous | review | panel role | leak | grades |
|---|---|---|---|---|---|---|---|---|
| age_at_index | confounder | confounder |  |  |  | None |  | T->Y:estimand, age_at_index->T:unsupported, age_at_index->Y:unsupported |

## Review items (0)

- none

## Edge rationale (non-estimand edges)

- age_at_index → T (unsupported): Age is a pre-treatment determinant of prescribing decisions because it is associated with comorbidity burden, treatment safety considerations, and patient/clinician preferences that affect whether advanced therapies such as biologics are initiated. — PMID:24472253
- age_at_index → Y (unsupported): Age can independently affect treatment uptake over follow-up through age-related differences in access, preferences, comorbidity, and medication use/adherence; therefore it can affect whether biologic initiation is observed by 180 days, apart from the index-treatment intervention. — PMID:24472253, DOI:10.1001/jama.2005.902

## Diff against the manifest's machine attestations

- compared 1/1; edge-exact agreement 1; role agreement 1; disagreements 0 (no threshold)
