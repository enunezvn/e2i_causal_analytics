# Agent Contract Corpus — Layout Guide

**Canon** (current, maintained):

- Flat files at this level (`tier1-tool-composer-contracts.md`, `tier2-contracts.md` … `tier5-contracts.md`, `orchestrator-*.{md,yaml}`, `base-contract.md`, `integration-contracts.md`, `data-contracts.md`, `agent-handoff.yaml`, `inter-agent.yaml`, `tier0-contracts.md`)
- `tier0/` (cohort_constructor contracts)

The tier2–tier5 and tier1-tool-composer docs carry the 2026-07-29 doc-rot fixes (#1343/#1344, PR #1347). When a contract doc disagrees with the implementation, the implementation is the source of truth — fix the doc (see the per-agent `src/agents/*/CONTRACT_VALIDATION.md` files for verified status).

**Legacy drafts** (pending dedup, #1348 — do NOT cite as canon):

- `Base Structures/`
- `Master Contract Document/`
- `Orchestrator Contracts/`
- `Tier-Specific Contracts/` — mostly older, smaller drafts of the flat docs. Exceptions under review in #1348: `tier1-contracts.md` (distinct Orchestrator tier-1 content with no flat counterpart) and `tier0-contracts.md` (larger than the flat version).

This directory is version-controlled (gitignore exception added 2026-07-29) so tracked docs never cite untracked canon.
