# Lifecycle Change Doc — Template

> Plan v4 Gate N2 (codex-rescue MEDIUM-1). This template is the
> required artifact for any change to a `LIFECYCLE_STATE_*` constant
> (Python) or `lifecycle_state:` key (YAML/JSON/TOML) per
> `scripts/check_lifecycle_state.py --check-changes`.
>
> **Filename convention** (enforced by the scanner — must match exactly):
>
> ```
> docs/calibration/{slug}_lifecycle_change_{from_state}_to_{to_state}_{YYYYMMDD}.md
> ```
>
> * `{slug}` — namespaced by source type (N2 pass-2 M1 + new MED):
>   * `py_` prefix for Python `LIFECYCLE_STATE_*` constants — e.g.,
>     `py_t22`, `py_t23`, `py_t26a`, `py_t26b`, `py_t24`.
>   * `yaml_` prefix for YAML configs — `yaml_<filename-stem>`.
>   The bare-slug form (`t22`) is NOT accepted: it reintroduces the
>   cross-source collision risk the prefix was designed to eliminate.
> * `{from_state}` / `{to_state}` — lowercase enum values
>   (`development`, `advisory`, `calibrating`, `enforced`, `deprecated`).
> * `{YYYYMMDD}` — date the change is committed.
>
> Example: `docs/calibration/py_t22_lifecycle_change_advisory_to_calibrating_20260615.md`
>
> This template file (`example_lifecycle_change_template.md`) does NOT
> match the filename pattern intentionally — copy it to a new file with
> the correct name when authoring a real lifecycle change.

---

## Summary

| Field | Value |
| --- | --- |
| Gate | T2.2 — permutation-anchored AUC floor |
| Module | `src/agents/ml_foundation/model_trainer/nodes/evaluator.py` |
| Constant | `LIFECYCLE_STATE_T22` |
| From state | `advisory` |
| To state | `calibrating` |
| Effective date | 2026-06-15 |

---

## Required fields

The four fields below are REQUIRED for every transition INTO `enforced`.
The scanner's `enforced_doc_missing_fields` finding looks for the literal
strings `start_date:`, `end_date:`, `drift_summary:`, `signing_reviewer:`
in the doc body, so keep the colons.

```
start_date: 2026-06-15
end_date: 2026-09-15
drift_summary: |
  Calibration window 2026-03-15 to 2026-06-15 observed N=42 model_trainer
  runs across CSU + Optum. T2.2 ``permutation_anchored_auc_advisory_violated``
  fired 7 times (16.7% would-be reject rate); 6 of 7 fires concentrated on
  cohorts with n_train_pos < 25 (matches HBLP variance-inflation regime
  documented in plan v3 §3 Tier 1B step 2). After excluding the small-N
  subset the would-be-reject rate is 1/35 ≈ 2.9%, in band with the 1-5%
  domain-typical operating-point target. Recommendation: graduate to
  ``calibrating`` on the >=25 n_train_pos slice; keep T2.2 in advisory
  for small-N cohorts pending HBLP-conditioned threshold work.
signing_reviewer: Erik Nunez (analytics-lead)
```

---

## Risk assessment

* **What signal does this gate emit today?** ([cite the plan section + the
  emitting helper.])
* **What changes when we move to the next state?** (Behaviorally — not just
  a config flip. List the consumers that will start reading the new
  signal as a hard guardrail.)
* **What's the rollback plan?** (Which flag, file, or commit reverts the
  transition? Who has the rollback authority?)
* **What dashboards / alerts go live with this change?** (Link the
  Grafana / observability dashboard. If none, name the operator-facing
  surface — log line, JSON key, etc.)

---

## Test evidence

* Unit-test coverage for the new state's behavior (link tests):
  * `tests/unit/...`
* Integration / e2e test demonstrating the would-be-reject behavior on
  a representative cohort:
  * `tests/integration/...`
* Calibration-window data:
  * Path / artifact ID / dashboard URL.

---

## Reviewer checklist

Reviewer fills in BEFORE merging the lifecycle-change PR:

- [ ] Filename matches the canonical pattern.
- [ ] All four required fields are present (only checked for transitions
  INTO `enforced`).
- [ ] `from_state` matches the value at `--base-ref`; `to_state` matches
  the value at `HEAD`.
- [ ] At least one rollback path is documented.
- [ ] At least one observability surface is named.
- [ ] All cited tests are green at HEAD.

---

## Notes

Free-text rationale, links to plan sections, audit-trail context, etc.

---

> Generated from `docs/calibration/example_lifecycle_change_template.md`
> (Plan v4 Gate N2). Do not edit this template in-place — copy it to a
> new file with the canonical filename.
