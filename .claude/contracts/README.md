# Agent Contract Corpus — Layout Guide

Single-canon layout (the legacy draft directories were deduplicated and removed
in #1348; unique draft content was promoted into the files below first).

**Flat files at this level** (current, maintained):

- `base-contract.md` — base agent contracts (state, config, errors, lifecycle)
- `integration-contracts.md` — platform integration contracts, incl. the
  Cross-Domain Producer/Consumer Contracts section (promoted from the legacy
  Master Contract Document in #1348)
- `data-contracts.md` — data-layer contracts
- `orchestrator-contracts.md` — the primary Orchestrator dispatch/response/
  aggregation spec
- `orchestrator-dispatch.yaml`, `agent-handoff.yaml`, `inter-agent.yaml` —
  dispatch/handoff/inter-agent YAML schemas and per-agent examples
- `tier0-contracts.md` — ML Foundation (Tier 0) pipeline contracts
- `tier1-orchestrator-contracts.md` — Orchestrator tier-1 contracts incl. the
  implementation-backed DSPy Hub role (promoted in #1348; where it disagrees
  with `orchestrator-contracts.md`, the latter wins)
- `tier1-tool-composer-contracts.md` — Tool Composer (Tier 1)
- `tier2-contracts.md` … `tier5-contracts.md` — per-tier agent contracts
  (tier2 includes the V4.2 Energy Score section promoted in #1348)
- `CONTRACT_VALIDATION.md` — corpus-level validation matrix

**`tier0/`** — cohort_constructor contracts (`cohort_constructor.md`,
`cohort_constructor_data.md`, `cohort_constructor_handoff.yaml`). The chat-vs-
pipeline routability callout and the implemented `CC_001`-`CC_007` error codes
live here and are authoritative for that agent.

The tier2–tier5 and tier1-tool-composer docs carry the 2026-07-29 doc-rot fixes
(#1343/#1344, PR #1347). When a contract doc disagrees with the implementation,
the implementation is the source of truth — fix the doc (see the per-agent
`src/agents/*/CONTRACT_VALIDATION.md` files for verified status).

This directory is version-controlled (gitignore exception added 2026-07-29) so
tracked docs never cite untracked canon.
