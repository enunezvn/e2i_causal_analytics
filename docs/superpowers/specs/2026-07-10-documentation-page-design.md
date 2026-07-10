# Documentation Page ("Understanding E2I") — Design

**Date:** 2026-07-10
**Status:** Approved by user (design presented and accepted in-session)
**Scope:** One frontend PR

## Problem

The footer on every page of https://eznomics.site/ offers three links: Dashboard, System Status, and API Docs. There is no page that explains, for a person landing on the platform, what E2I is for, how its causal methodology works, how to use it well, and what impact to expect. The one prior attempt at this — `frontend/src/components/kpi/AgenticMethodology.tsx` — was removed from the KPI Dictionary page per user request because its content went stale (it hardcodes "28 Database Tables" and a 4-layer architecture that no longer match reality), and it now sits orphaned with zero consumers.

## Goal

Add a `/documentation` page — illustrated and interactive, in the spirit of a Claude artifact — that explains:

1. **Purpose** — why E2I exists (causation over correlation for pharma commercial analytics)
2. **Methodology** — how the causal pipeline and the agent system work
3. **Best practices** — how to use the platform well
4. **Expected impact** — what good outcomes look like, honestly framed

Reachable from a 4th footer link and a sidebar entry. The stale `AgenticMethodology.tsx` is superseded and deleted in the same PR.

## User decisions (binding)

| Decision | Choice |
|---|---|
| Placement | Footer 4th link **and** sidebar entry in "Data & Reference" |
| Audience | Layered — executive narrative up front, expandable technical depth |
| Structure | Scroll narrative with sticky section nav; each section has an interactive illustration |
| Orphan disposal | Supersede & delete `AgenticMethodology.tsx` in the same PR |
| Content sourcing | Approach C — static typed constants for structural facts; live stat chips (via existing hooks) for volatile counts, with silent graceful degradation |

## Non-goals

- No backend changes; no new API endpoints. The only network call is the existing `useKPIList` hook.
- No new npm dependencies. Interactivity uses what is already installed: framer-motion, Radix primitives (collapsible/tabs/tooltip), lucide-react, inline SVG.
- No fabricated performance/ROI numbers anywhere on the page (platform anti-mocking discipline). Illustrative content must be visually labeled "illustrative".
- No public/unauthenticated access. The page sits behind `ProtectedRoute` like every other page.

## Navigation & routing

### Footer (`frontend/src/components/layout/Footer.tsx`)

Add a 4th link inside the existing `<nav aria-label="Footer navigation">`, after "System Status" and before the external "API Docs" anchor (internal links grouped together):

```tsx
<Link
  to="/documentation"
  className="text-[var(--color-muted-foreground)] hover:text-[var(--color-foreground)] transition-colors"
>
  Documentation
</Link>
```

### Router (`frontend/src/router/routes.tsx`)

- Lazy import: `const Documentation = lazy(() => import('@/pages/Documentation'));`
- `routeConfig` entry, inserted between `/kpi-dictionary` and `/data-quality` so it renders adjacent to KPI Dictionary in the sidebar:

```tsx
{
  path: '/documentation',
  title: 'Documentation',
  description: 'Platform purpose, methodology, and best practices',
  icon: 'graduation-cap',
  section: 'data',
  showInNav: true,
},
```

- `RouteObject` wrapped in `<ProtectedRoute><LazyPage>` exactly like the other pages.

### Sidebar (`frontend/src/components/layout/Sidebar.tsx`)

Add a `'graduation-cap'` entry to the `NavIcon` `iconMap` (inline SVG, stroke style matching the existing icons). `'book-open'` is already taken by KPI Dictionary.

## Page structure

Route `/documentation`, page component `frontend/src/pages/Documentation.tsx`. A scroll narrative in four sections beneath the standard page header, with a sticky scroll-spy section nav (Purpose · Methodology · Best Practices · Expected Impact). Clicking a nav item smooth-scrolls to the section; scrolling updates the active highlight (IntersectionObserver). On small screens the nav collapses to a horizontal scrollable strip.

All styling uses the CSS-var token system (`--color-*` vars registered via Tailwind v4 `@theme inline`), so light and dark themes both work. Framer-motion entrance/expand animations are guarded by `useReducedMotion` — with reduced motion preferred, content renders statically.

### §1 Purpose — "Why E2I exists"

- Plain-language narrative: pharma commercial teams need to know what *causes* outcomes, not what merely correlates with them. E2I applies formal causal inference plus an agentic AI layer at **three linked levels**, each grounded in implemented capability (verified 2026-07-10 against the causal-path registry, predictive cohorts, and digital-twin intervention catalog):
  1. **HCP prescribing behavior** — which promotional levers actually change prescribing (rep detailing, speaker programs, sampling, peer influence, digital engagement, rep training); 8 simulatable intervention channels (`digital_twin/effect/provider.py` `INTERVENTION_CATALOG`); a dedicated HCP-adoption predictive cohort targeting intent-to-prescribe.
  2. **Patient journey outcomes** — what drives treatment initiation, 180-day persistence, and discontinuation (patient support programs, copay support, access); three patient-level predictive cohorts (`src/insights/predictive_cohort.py`).
  3. **Market & brand performance** — how upstream behaviors plus market dynamics (formulary status, competitor activity) aggregate into TRx/NRx/NBRx, market share, and ROI.
- **Interactive illustration A — `CausalScopeMap`:** a compact three-layer SVG diagram (HCP behavior → patient journey → brand outcomes) whose node labels are drawn verbatim from the causal registry's `_NODE_LABELS` (`src/insights/causal_context.py`) — every node shown is a modeled node, nothing invented. Hovering/tapping a layer highlights its nodes and shows a one-line description.
- **Stat chips row:** static structural chips (3 brands / 4 indications · 4 predictive cohorts · 8 intervention channels · 5 refutation tests) typed as constants; one **live** chip — KPI count from `useKPIList` — that renders only when the query succeeds. On error or while loading, the chip is simply absent: no spinner, no error banner, no placeholder number. (The 21-agent/6-tier fact moves to §2 where the tier stack shows it.)
- **Interactive illustration B — `CorrelationCausationToggle`:** an SVG scatter panel showing a convincing spurious correlation ("HCP calls correlate with TRx"), with a toggle that reveals the confounder (physician specialty) as a small DAG and shows how the adjusted effect differs. Data points are hand-authored illustrative coordinates (clearly labeled "illustrative example"), not real metrics.

### §2 Methodology — "How it works"

Two interactive components:

- **`CausalPipeline`** — a clickable 5-stage pipeline (Frame → Identify → Estimate → Refute → Act), visually modeled on the existing `QueryProcessingFlow` (stage cards, connecting arrows, per-stage color). Clicking a stage expands a detail panel with two layers:
  - *Plain language* (always visible when expanded): what the stage does and why it matters, ~2-3 sentences.
  - *"For analysts" collapsible* (Radix Collapsible): backdoor adjustment and DAG-based identification; EconML/CausalML cross-library validation; the five named refutation tests as implemented in `src/api/schemas/causal.py` (placebo treatment, random common cause, data subset, bootstrap, unobserved common cause — the last mapped from the E-value sensitivity analysis); proceed/review/block gates.
- **`AgentTierStack`** — the corrected successor to `AgenticMethodology`: a compact vertical stack of the 6 agent tiers; clicking a tier expands it to list its agents with one-line roles. Content is typed constants sourced from the current codebase/docs (21 agents, 6 tiers) — no counts that drift weekly.

### §3 Best Practices — "Using E2I well"

**`PracticeCards`** — paired Do/Don't cards grounded in real product behavior:

- Check the refutation gate before acting on an estimated effect (proceed/review/block).
- Treat "Informational" KPIs as context, not performance targets.
- Respect honest-null results — "no significant effect" is a finding, not a failure.
- Keep what-if simulation inputs inside observed data ranges.
- Select the intended brand before comparing metrics (models are per-brand).

Each card carries a role chip (Exec / Analyst); a filter row lets the reader show only their role's practices (default: all).

### §4 Expected Impact — "What good looks like"

**`ImpactPathways`** — four illustrated mechanism cards, each linking to the live page where the user sees *their own* numbers. **No fabricated ROI digits.**

1. Sharper targeting — CATE segments identify who responds above average → links to `/segment-analysis`
2. Better budget allocation — constrained optimization over channels → links to `/resource-optimization`
3. Cheaper experimentation — digital-twin pre-screening before field pilots → links to `/digital-twin`
4. Faster time-to-insight — natural-language chat over governed KPIs → links to the chat pane / dashboard

Any diagrammatic figure (e.g., a stylized gap→intervention→lift arc) carries a visible "illustrative" label and uses no plausible-real numerals.

## Code layout

```
frontend/src/pages/Documentation.tsx           page shell: header, SectionNav, 4 sections
frontend/src/pages/Documentation.test.tsx      vitest + RTL (pattern: KPIDictionary.test.tsx)
frontend/src/components/documentation/
  index.ts                                     barrel export
  content.ts                                   typed constants: SCOPE_LEVELS, PIPELINE_STAGES,
                                               AGENT_TIERS, PRACTICES, IMPACT_PATHWAYS, STAT_CHIPS
  SectionNav.tsx                               sticky scroll-spy nav
  CausalScopeMap.tsx                           §1 three-level capability map
  CorrelationCausationToggle.tsx               §1 illustration
  CausalPipeline.tsx                           §2 pipeline
  AgentTierStack.tsx                           §2 tier stack
  PracticeCards.tsx                            §3 do/don't cards
  ImpactPathways.tsx                           §4 pathway cards
```

Each component is presentational with typed props/constants; only `Documentation.tsx` touches the network (via `useKPIList`), keeping every visualization component testable without a QueryClient.

## Deletions & cleanup

- Delete `frontend/src/components/kpi/AgenticMethodology.tsx` (orphan; zero consumers; content confirmed stale; removal previously requested by user — see comment in `frontend/src/components/kpi/index.ts:13-14`).
- Update that `index.ts` comment to note the content now lives at `/documentation` (`src/pages/Documentation.tsx`).

## Error handling

- `useKPIList` failure or loading → live KPI chip not rendered; static chips unaffected. No error UI on a documentation page.
- Lazy chunk load failure is handled by the router's existing error boundary, same as all pages.
- IntersectionObserver unavailable (old browsers/jsdom) → nav still works as click-to-scroll; active-state highlighting degrades gracefully (guard `typeof IntersectionObserver !== 'undefined'`).

## Testing

- **`Documentation.test.tsx`** (vitest + RTL, `vi.mock('@/hooks/api/use-kpi')`, QueryClientProvider wrapper with `retry: false, gcTime: 0`):
  - all four sections and the section nav render;
  - scope map renders its three levels and highlights one on interaction;
  - pipeline stage expands on click and shows the "For analysts" collapsible;
  - tier stack expands a tier to reveal agents;
  - live KPI chip renders with mocked data, and is absent when the hook returns an error;
  - practice-card role filter narrows the card list.
- **`Footer.test.tsx`** (new — footer currently has no test): renders all 4 links; Documentation link points to `/documentation`; API Docs remains an external anchor.
- **Sidebar/routes:** confirm existing `Sidebar` tests still pass with the added nav route; snapshot-free assertions only.
- **e2e (Playwright):** a small `documentation.spec.ts` following `kpi-dictionary.spec.ts` structure (page loads, no errors, section nav visible, one interaction). `_smoke.spec.ts` is untouched (its 4 routes are fixed).
- **Type check:** `npx tsc -p tsconfig.app.json` (bare `tsc --noEmit` is a known false green in this repo).

## Verification plan

1. Local: targeted vitest for the new/changed tests; `tsc -p tsconfig.app.json`.
2. PR → CI green → merge (no squash) → push-to-main CI deploy (~12 min).
3. Live verification on https://eznomics.site/: footer link on several pages, sidebar entry, scroll nav behavior, all four interactives, both themes, and that the deleted component's absence breaks nothing (KPI Dictionary loads clean).

## Risks

- **Content staleness** (the failure mode that killed `AgenticMethodology`): mitigated by design — volatile counts are live-fetched or omitted; structural facts chosen because they change only with deliberate architecture changes; the spec records the sourcing rule so future edits follow it.
- **Bundle size:** page is lazy-loaded; components use existing deps only, so the main bundle is unaffected.
