# Structural Attestation Authoring Guide (Layer-4 deterministic role decider)

**Audience:** a domain author (clinical / epidemiology / pharmacology expert) with
**no code knowledge**. After reading this, you can attest any ML feature by drawing
a small causal diagram (a DAG) of how it relates to the treatment **T** and the
outcome **Y**. A deterministic extractor then reads your diagram and assigns the
feature a *causal role*; that role decides whether the feature is treated as
**leakage** (must not enter the model) or is **safe to keep**.

This guide is the authoritative reference. A reviewer should be able to attest a
feature using **only** this document — no source-reading required.

---

## 0. The one rule that matters most

> **Author the TRUE mechanism from the feature's meaning + the clinical/biological
> literature. Do NOT target a role.**

You are not trying to make a feature come out "instrument" or "confounder." You are
drawing the real-world causal story: *what causes what?* The role falls out of the
diagram mechanically. If you find yourself adding or deleting an arrow so that the
role "comes out right," stop — that is reverse-engineering the answer and it
invalidates the whole exercise. Draw the mechanism you believe is true, cite the
evidence, and let the extractor classify it.

If two equally-defensible causal stories give different roles, the feature is
**genuinely ambiguous** — author your best single story and flag it `ambiguous: true`.
Do not silently pick the story that gives a "nicer" role.

---

## 1. What you are drawing

A **DAG** (directed acyclic graph) is a set of **nodes** connected by **arrows**.
An arrow `A → B` means "A is a direct cause of B" (a change in A produces a change
in B, not the other way around). "Acyclic" means no arrow-loops: you can never
follow arrows from a node back to itself.

For every feature you author exactly one diagram with these nodes:

| Node label | Meaning |
|---|---|
| `"T"` | the **treatment** (here: remibrutinib / LOU064, a BTK inhibitor — the first remibrutinib dispense is the index/anchor) |
| `"Y"` | the **outcome** (here: `treatment_response_UAS7_reduction_180d` — UAS7 reduction at 180 days) |
| the **feature name** itself (e.g. `baseline_total_ige_iu_ml_preindex`) | the feature node you are classifying |
| `"U_<something>"` (optional) | a **latent** (unmeasured) common cause you need to make the story correct, e.g. `"U_disease_severity"`, `"U_atopy"` |

You then list the **edges** (arrows) as pairs, e.g.
`[["baseline_uas7", "T"], ["baseline_uas7", "Y"]]` means
`baseline_uas7 → T` and `baseline_uas7 → Y`.

### Edge-drawing conventions (read these carefully — they prevent the common errors)

1. **The measured feature is its own node.** Draw the feature node by its
   feature name. Do **not** replace it with the concept it proxies — if a feature
   is "prescriber's prior BTKi preference share," the node is that measured share,
   and the thing it proxies (the prescriber's true preference) is a *separate*
   cause feeding into it only if you genuinely need it (see convention 3).

2. **Minimize latent `U` nodes.** Only add a `U_<x>` node when the true mechanism
   *requires* an unmeasured common cause to be correct — most commonly when the
   feature and the outcome share a hidden driver (e.g. baseline disease severity
   drives both a baseline biomarker and the response). Do **not** sprinkle latent
   nodes "just in case": every `U` you add changes the role the extractor derives,
   and an unjustified `U` is a false attestation. If you add a `U`, you must be
   able to name it and cite why it is a common cause.

3. **Draw the arrow in the direction of causation, at the time the feature is
   measured.** A *baseline* (pre-index) biomarker that reflects disease severity
   is **caused by** severity (`U_severity → biomarker`) and severity also drives
   treatment choice and prognosis — it is not caused by the treatment. A
   *post-treatment* biomarker change is **caused by** the treatment
   (`T → biomarker_change`). The `knowable_at=` token in each feature's derivation
   tells you when it is measured: `index_date_minus_1` / `preindex` = before
   treatment; `index_date` = at the anchor; `index_date_plus_*` / `t_plus_*` =
   after treatment starts.

4. **Do not draw an edge you cannot defend from the feature's meaning + the
   literature.** Every arrow is a causal claim. If you are unsure whether
   `feature → Y` exists, that uncertainty usually *is* the role question (e.g.
   instrument-vs-confounder hinges on exactly whether `feature → Y` exists).
   Resolve it with evidence, not convenience.

5. **No cycles.** If your story implies `A → B` and `B → A`, you have mixed up two
   time points or two different quantities — split them into distinct nodes
   (e.g. baseline value vs post-treatment change).

---

## 2. The six roles — definitions you can draw to

Below, "T-path" means a directed path that starts at `T`; "Y-path" means one that
ends at `Y`. These definitions match the deterministic extractor exactly (the
extractor's priority order is given in §3).

### `instrument` — *affects treatment only, never the outcome except through treatment*
- **Edges:** `feature → T`, **AND** no directed path `feature → … → Y` once you
  remove `T` from the graph (exclusion restriction), **AND** the feature shares no
  common cause with `Y` (exogeneity — no `U` is an ancestor of both the feature and `Y`).
- **Plain words:** something that nudges *whether/which* treatment a patient gets
  but has **no other route** to the outcome. Classic examples: a prescriber's
  prescribing tendency, a payer formulary change, regional/calendar adoption — they
  shift treatment but do not themselves change the patient's disease response.
- **Watch out:** if the same thing *also* plausibly affects the outcome directly
  (e.g. a "high-volume specialist" who also gives better overall care), it is **not**
  a clean instrument — it becomes a confounder or ancestor. Draw the `feature → Y`
  edge if it really exists, and the role changes accordingly.

### `confounder` — *common cause of BOTH treatment and outcome*
- **Edges:** `feature → T` **AND** `feature → Y` (a direct parent of both).
- **Plain words:** a pre-treatment patient characteristic that influences *both*
  the decision to treat *and* the prognosis — baseline disease severity, prior
  therapy history, key comorbidities. This is the textbook backdoor path.
- **Watch out:** confounders are **pre-index** by construction. If the feature is
  measured after treatment starts, it cannot be a confounder.

### `ancestor` — *causes the outcome, but not through treatment, and is not downstream of treatment*
- **Edges:** a directed path `feature → … → Y` exists, **AND** the feature is **not**
  a descendant of `T`, **AND** the feature is **not** a parent of `T` (if it were a
  parent of T it would be an instrument or confounder).
- **Plain words:** a prognostic factor / risk factor for the outcome that does not
  influence the treatment decision and is not itself caused by treatment. Pure
  baseline risk markers (demographics, genetics, family history) that affect
  prognosis but not the prescribing choice.
- **Watch out:** ancestor vs confounder hinges on the `feature → T` edge. If the
  factor *also* drives treatment choice, it is a confounder, not an ancestor. If it
  truly does not touch the treatment decision, it is an ancestor.

### `mediator` — *on the causal path from treatment to outcome*
- **Edges:** the feature is a descendant of `T` (some path `T → … → feature`), **AND**
  there is a directed path `feature → … → Y`. I.e. `T → … → feature → … → Y`.
- **Plain words:** a post-treatment quantity through which the drug exerts its
  effect — a pharmacodynamic biomarker change that lies *on the mechanism* between
  taking the drug and the clinical response.
- **Watch out:** a mediator is a **leak role** (see §4). Even though it is "real
  biology," it is measured after treatment and carries outcome information, so it
  must not be a model input. Mediator vs descendant: a mediator still has its **own**
  path to Y; if every path from the feature to Y instead runs *through Y already*
  (the feature is downstream of the outcome), it is a descendant.

### `collider` — *common effect of treatment and outcome (where a T→feature path does NOT pass through Y)*
- **Edges:** the feature is a descendant of **both** `T` and `Y`, **AND** there is a
  `T → … → feature` path that does **not** go through `Y`. (Special case, the
  "M-structure": `T → feature ← U → Y`, where `U` is an independent common cause of
  the feature and Y that reaches Y by a route bypassing the feature — also a collider.)
- **Plain words:** something that both the treatment *and* the outcome (or an
  independent cause of the outcome) point into — e.g. a post-index event that is
  driven both by being on the drug *and* by how the disease is doing. Conditioning a
  model on a collider opens a spurious treatment–outcome association — it is harmful.
- **Watch out:** colliders are **post-index**. The discriminator vs descendant is
  whether treatment reaches the feature *without going through the outcome*. If the
  **only** way treatment reaches the feature is *through* the outcome, it is a
  descendant, not a collider (see next).

### `descendant` — *outcome-echo: every treatment→feature path goes through the outcome*
- **Edges:** `Y → feature` (a direct child of the outcome), or `T → Y → feature` —
  the key test is that if you removed `Y` from the graph, treatment could **no
  longer** reach the feature. The feature is purely downstream of the outcome.
- **Plain words:** a post-treatment quantity that is essentially a *re-measurement
  or consequence of the response itself* — a parallel patient-reported outcome at
  the same timepoint, a downstream complication that only happens because the
  disease did (or didn't) respond. It carries the answer.
- **Watch out:** descendant is a **leak role**. The line vs collider: a collider has
  an independent treatment route into the feature *not* through Y (so conditioning
  induces bias); a descendant's every route from T runs through Y (it is just an
  echo of the outcome). Both are leaks — see §4 — so for the *leak decision* the
  distinction does not change the verdict, but draw the honest mechanism anyway.

---

## 3. How the extractor reads your diagram (priority order — load-bearing)

The extractor applies these checks **in order** and returns the **first** match.
Order matters: a feature that could satisfy two definitions gets the higher one.

1. **Common descendant of T and Y?** If the feature is downstream of *both* T and Y:
   - if some `T → … → feature` path avoids `Y` → **collider**;
   - otherwise (every T-route goes through Y) → **descendant**.
2. **M-structure** `T → feature ← U → Y` (independent second parent `U` reaching Y by
   a path that bypasses the feature) → **collider**.
3. **Direct parent of both T and Y** → **confounder**.
4. **Descendant of T** (but not caught above): if it has a path to Y → **mediator**;
   else (a treatment dead-end, no path to Y) → **descendant**.
5. **Parent of T**: if it satisfies *exclusion* (no path to Y after T is removed)
   **and** *exogeneity* (no common ancestor with Y) → **instrument**; otherwise it
   falls through to the next check.
6. **Has a path to Y and is not downstream of T** → **ancestor**.
7. **None of the above** → the diagram is *unclassifiable*; the extractor raises an
   error and the feature is routed to **human review** (it is **not** silently kept).

> **Practical consequence of the order:** if you draw a latent common cause `U`
> feeding both a post-treatment feature and `Y`, the extractor will (correctly) read
> the M-structure and call it a **collider**, even if you think of the biomarker as a
> "mediator." That is a real modelling decision: do you believe the treatment acts
> *through* the biomarker (mediator: `T → feature → Y`), or that an independent
> driver moves both the biomarker and the response (collider: `T → feature ← U → Y`)?
> Draw the one you actually believe and cite it. For the **leak decision** both are
> leaks, so the verdict is unchanged either way.

---

## 4. Leak-vs-accept buckets — what the role *means* for the model

The whole point of the role is a binary safety decision: **may this feature enter
the model, or is it leakage?**

| Bucket | Roles | Decision |
|---|---|---|
| **LEAK** (flag — must NOT enter the model) | `mediator`, `collider`, `descendant` | reject / remediate (drop, or window/transform where valid) |
| **ACCEPT** (safe to keep) | `ancestor`, `confounder`, `instrument` | keep (with IV/causal caveats as appropriate) |
| **REVIEW** | *unclassifiable diagram* | route to a human; never auto-keep |

(Source of truth: `LEAK_ROLES` / `ACCEPT_ROLES` in `src/data/kg/ensemble_voter.py`.)

The intuition: anything measured **after** treatment that carries the drug's effect
or the outcome's signal (mediator / collider / descendant) leaks the answer into the
features and must be excluded. Anything that is a genuine **pre-treatment** cause —
of the outcome (ancestor), of both treatment and outcome (confounder), or of
treatment alone (instrument) — is legitimate and safe.

**The safety-critical error to avoid is a *missed leak*:** authoring a diagram that
makes a truly-leaky feature (mediator/collider/descendant) come out as
ancestor/confounder/instrument, so it is wrongly kept. This is why §0's rule is
absolute: do not target a role; draw the honest mechanism. A conservative error
(calling an accept-feature a leak, or routing to review) is safe; a missed leak is not.

---

## 5. Worked micro-examples

- **Baseline disease severity marker measured pre-index** (e.g. baseline UAS7): it
  drives *both* whether the patient is escalated to remibrutinib *and* the prognosis.
  Edges `feature → T`, `feature → Y`. → **confounder** (ACCEPT).
- **Prescriber's prior BTKi-preference share**: shifts which drug the patient gets,
  but the prescriber's tendency does not itself change *this* patient's disease
  biology. Edges: `feature → T` only, no `feature → Y`, no shared `U` with Y. →
  **instrument** (ACCEPT). *(If you believed high-preference prescribers also deliver
  systematically better care affecting outcome, you would add `feature → Y` and it
  would become a confounder — author what you actually believe.)*
- **Family history of atopy**: a prognostic risk factor for the urticaria course, but
  it does not enter the prescribing decision and is not caused by treatment. Edges:
  `feature → Y` only (or `feature → U_atopy → Y`), no `feature → T`. → **ancestor**
  (ACCEPT).
- **Post-treatment change in a pharmacodynamic biomarker on the drug's mechanism**
  (e.g. Δ basophil activation): the drug acts through it to produce the response.
  Edges `T → feature → Y`. → **mediator** (LEAK). *(If instead you believe an
  independent driver moves both the biomarker and response, draw `T → feature ← U → Y`
  → **collider**, still LEAK.)*
- **A second patient-reported outcome at the same 180-day timepoint** (e.g. UCT score,
  which moves because the disease responded): it is an echo of the response. Edges
  `Y → feature` (or `T → Y → feature`). → **descendant** (LEAK).
- **"Still on treatment / persistent at 180d" flag**: being on therapy at 180d is
  driven both by *starting* it (T) and by *whether it worked* (Y) — a common effect.
  If treatment reaches the flag without going through Y, → **collider** (LEAK).

---

## 6. Authoring checklist

For each feature, record:

- `feature_node` = the feature name (verbatim).
- `treatment_node` = `"T"`, `outcome_node` = `"Y"`.
- `edges` = the list of `[from, to]` arrows you drew.
- the **role you expect** (optional, for your own cross-check — the extractor is
  authoritative).
- a **cited rationale** (PMIDs / NCT IDs / URLs) supporting each non-obvious edge,
  drawn from the feature's meaning + the clinical/pharmacology literature.
- `ambiguous: true/false` — set `true` when a second equally-defensible mechanism
  would give a different role.

Then (done by the validation coordinator, not the author): the extractor is run on
your diagram, and the derived role is scored against the independent literature
label. **Authored once, scored once.** A disagreement is a legitimate finding and is
**never** patched by editing your edges to match the label.
