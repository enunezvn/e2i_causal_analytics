# `docs/api/` — what is in here and why

**Last Updated**: 2026-09-07

Most of this platform's HTTP surface documents itself: every route outside
CopilotKit carries an OpenAPI `summary=`, is served at `/api/docs`, and has its
response fields regenerated into `frontend/src/types/generated/api.ts` under the
CI `verify-types` gate. This directory holds only the two things that mechanism
cannot produce.

## Contents

| File | Tracked? | What it is |
|---|---|---|
| `chat.md` | yes | Hand-written reference for the chat + routing-classifier surface |
| `crystal_digests.md` | yes | Hand-written reference for crystal digests / executive insights |
| `openapi.json` | no — gitignored (`.gitignore:223`) | `make api-docs` output (`scripts/generate_api_docs.sh` → `python -m scripts.export_openapi`) |
| `index.html` | see note | `make api-docs` output (Redocly `build-docs` render of `openapi.json`) |

Verify the tracked set with `git ls-files docs/api`. (The repository-root
`README.md` describes this directory as "OpenAPI spec (auto-generated, not
tracked)" — that is true of the two generated artifacts, not of the two
hand-written references.)

> **Note:** `index.html` is a build artifact but is **not** currently matched by
> `.gitignore` (only `docs/api/openapi.json` is, at `.gitignore:223`). Running
> `make api-docs` therefore leaves an untracked `index.html` in `git status`.
> Adding it to `.gitignore` is a code/config change, tracked outside this doc.

## Why only these two files are hand-written

Both cover surfaces OpenAPI does not describe:

- **`chat.md`** — the CopilotKit / AG-UI routes are registered with
  `include_in_schema=False` (`add_copilotkit_routes()` in
  `src/api/routes/copilotkit.py`), so they never appear in `/api/docs`. The SSE
  frame contract, the AG-UI request channels (`state.filters`, `context`
  readables) and the classifier-mode semantics are protocol behaviour, not
  response schemas, so nothing generates them either.
- **`crystal_digests.md`** — the endpoints are in the schema, but the *meaning*
  of the fields (derivation rules, the lineage/provenance chain, the
  invalidation cascade, the SSE alert bridge's filter and backpressure
  semantics) is policy that no schema can carry.

## Decision (2026-09-07): no per-field `causal.md` / `segments.md`

Eleven merged-PR sweep rows in the 2026-09-07 documentation audit proposed
hand-written references for the causal and segments routes
(`POST /api/causal/discover-effects` and its `questions` / poll / `cancel`
siblings, `POST /api/causal/agent-analyze`, `GET /api/segments/datasets`,
`POST /api/segments/analyze`). **We decided not to write them.** Unlike the
CopilotKit routes, these are fully described by the generated surface:

- `grep -n include_in_schema src/api/routes/causal.py src/api/routes/segments.py | wc -l`
  → **0** — every one of these routes is in the OpenAPI schema and served at
  `/api/docs`. (The only `include_in_schema=False` routes in `src/api/` are
  `/api/health` and the CopilotKit catch-alls.)
- Every route carries a human `summary=` (e.g. *"Discover & rank the agent's
  VALIDATED causal effects (async submit -> poll)"*), so the rendered docs are
  not a bare field dump.
- Response fields (`dag_source`, `edge_provenance`, `library_agreement_score`,
  `definitions`, bootstrap stability, …) are regenerated into
  `frontend/src/types/generated/api.ts` by `make generate-types` and held there
  by the `verify-types` CI gate — a hand-written field table would be a second
  copy that can silently drift from the one CI checks.

The semantics that are *policy* rather than schema live where they can be read
in context instead: the job lifecycle (submit → poll → cancel, heartbeat
read-repair) and the 400 conditions in `docs/ARCHITECTURE.md` §3.4; the FCI /
latent-confounder policy on the in-app "How E2I Works" page; the shipped
endpoints in `CHANGELOG.md`.

**What would reverse this decision** (either fact is sufficient — re-run the
measure above):

1. Any causal or segments route becoming `include_in_schema=False`, or
   otherwise dropping out of `/api/docs`.
2. The OpenAPI `summary=` annotations being removed, leaving the rendered docs
   without prose.

If that happens, write one file per surface, verified against code the way
`chat.md` is.

## Conventions for the hand-written files

- **Cite symbols, not line numbers.** `` `Crystallizer` in
  `src/memory/crystallization/crystallizer.py` `` survives refactors; a
  `crystallizer.py:102` citation drifts silently and misleads the next reader.
  Where a number is unavoidable, regenerate it with `grep -n` at write time.
- **Every claim is verified against the tree at the stated date.** Each file
  carries a "Source of truth" list of the files it was checked against and the
  date it was checked.
- **`#N` is a GitHub issue; `PR #N` is a merged pull request.** They are
  different numbers for the same work and are not interchangeable.
