# Gap Analysis — Restore Quick Win / Strategic Bet Framework (effort folded in) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:subagent-driven-development. Steps use `- [ ]`.

**Goal:** Revert the unrequested "Low/Medium/High Effort" relabel on /gap-analysis back to the Quick Win / Strategic Bet framework, with implementation effort folded in as a secondary attribute — while keeping the genuinely-useful #1032 fixes (latest-run-per-brand dedup, curated counts, All-Brands selector).

**Background / why this is not a literal revert (REASON-BEFORE-RULES):**
The list endpoint returns *all* `prioritized_opportunities` (`gaps.py:558`), and a Strategic Bet is NOT "high difficulty" — the prioritizer defines it as `implementation_difficulty=="high" AND expected_roi>2 AND cost>$50k`, top 5 (`prioritizer.py:367-373`); a Quick Win is `difficulty=="low" AND roi>1`, top 5. The pre-#1032 page badged every high-difficulty card "Strategic Bet", so cards outnumbered the curated headline count — that visual is the user's original "too many Strategic Bets" complaint. Therefore the primary badge must come from the opportunity's **true curated category**, not from its difficulty. Effort stays visible as a secondary sub-badge.

**Invariant (user-approved preview):** In the default ("All") view, the number of cards badged "Strategic Bet" equals the headline `strategic_bets_count` (same for Quick Wins). No phantom Strategic Bets.

**Architecture:** Backend tags each opportunity with `category ∈ {quick_win, strategic_bet, other}` by `gap_id` membership in the latest run's curated lists; frontend renders category as the primary badge (positive colors), effort as a folded-in secondary badge, and filters by category (positive dropdown). Counts/dedup/All-Brands untouched.

**Tech Stack:** FastAPI + Pydantic (backend), React + TypeScript + Vitest/RTL (frontend), pytest (backend tests).

---

## File Structure

- Modify: `src/api/routes/gaps.py` — add `category` to `PrioritizedOpportunity`; tag + curated-retention in `list_opportunities`.
- Modify: `tests/unit/test_api/test_routes/test_gaps.py` — category tagging + invariant tests.
- Modify: `frontend/src/types/gaps.ts` — add `category?` to `PrioritizedOpportunity`.
- Modify: `frontend/src/pages/GapAnalysis.tsx` — category badge + effort sub-badge, category dropdown, chart/table relabel.
- Modify: `frontend/src/pages/GapAnalysis.test.tsx` — framework/badge/dropdown/invariant tests.

---

## Task 1: Backend — tag opportunities with curated category

**Files:**
- Modify: `src/api/routes/gaps.py` (`PrioritizedOpportunity` ~192-200; `list_opportunities` ~541-578)
- Test: `tests/unit/test_api/test_routes/test_gaps.py`

- [ ] **Step 1 — Failing test.** In `test_gaps.py`, add a test that calls `list_opportunities` (via the existing test harness/fixture pattern already used in that file) for a brand whose latest run has: 3 prioritized opps where opp A is in `quick_wins`, opp B is in `strategic_bets`, opp C is in neither. Assert: response opp A `.category == "quick_win"`, B `.category == "strategic_bet"`, C `.category == "other"`; and that the count of returned opps with `category=="strategic_bet"` equals `strategic_bets_count` (same for quick_win). Reuse the file's existing fixture/builder style (do not invent a new harness).

- [ ] **Step 2 — Run, expect fail** (`AttributeError`/validation: `category` unknown). `pytest tests/unit/test_api/test_routes/test_gaps.py -q`.

- [ ] **Step 3 — Add the field.** In `PrioritizedOpportunity` (after `time_to_impact`):

```python
    # Curated category for the list view, assigned by membership in the latest
    # run's prioritizer quick_wins/strategic_bets lists (see list_opportunities).
    # NOT derived from implementation_difficulty — a high-difficulty opportunity
    # is only a "strategic_bet" if it also clears the ROI/cost thresholds.
    category: Optional[str] = Field(
        default=None, description="quick_win | strategic_bet | other (list view only)"
    )
```

- [ ] **Step 4 — Tag in `list_opportunities`.** Replace the per-analysis loop body so each appended opportunity is tagged, and curated opps are never dropped by `limit`:

```python
    for analysis in latest_analyses:
        quick_wins_count += len(analysis.quick_wins)
        strategic_bets_count += len(analysis.strategic_bets)

        qw_ids = {o.gap.gap_id for o in analysis.quick_wins}
        sb_ids = {o.gap.gap_id for o in analysis.strategic_bets}

        for opp in analysis.prioritized_opportunities:
            if min_roi and opp.roi_estimate.expected_roi < min_roi:
                continue
            if difficulty and opp.implementation_difficulty != difficulty:
                continue
            gid = opp.gap.gap_id
            cat = (
                "quick_win" if gid in qw_ids
                else "strategic_bet" if gid in sb_ids
                else "other"
            )
            all_opportunities.append(opp.model_copy(update={"category": cat}))
            total_value += opp.roi_estimate.estimated_revenue_impact

    # Sort by ROI; when truncating to `limit`, retain curated opportunities so the
    # per-card category badges stay consistent with the headline counts.
    all_opportunities.sort(key=lambda x: x.roi_estimate.expected_roi, reverse=True)
    if len(all_opportunities) > limit:
        curated = [o for o in all_opportunities if o.category != "other"]
        others = [o for o in all_opportunities if o.category == "other"]
        all_opportunities = (curated + others)[:limit]
        all_opportunities.sort(key=lambda x: x.roi_estimate.expected_roi, reverse=True)
```

(Keep the existing `quick_wins_count`/`strategic_bets_count` semantics — counts come from the curated lists, which are now guaranteed present in the returned set.)

- [ ] **Step 5 — Run tests, expect pass.** Also run the full file to ensure no regression: `pytest tests/unit/test_api/test_routes/test_gaps.py -q`.

- [ ] **Step 6 — Lint/format.** `ruff check src/api/routes/gaps.py tests/unit/test_api/test_routes/test_gaps.py` and `ruff format --check` (same files). Fix if needed.

- [ ] **Step 7 — Commit.** `git add -A && git commit` (message: `feat(gaps): tag opportunities with curated category for list view`).

---

## Task 2: Frontend — restore framework, fold effort in, fix dropdown

**Files:**
- Modify: `frontend/src/types/gaps.ts` (`PrioritizedOpportunity` ~162-175)
- Modify: `frontend/src/pages/GapAnalysis.tsx`
- Test: `frontend/src/pages/GapAnalysis.test.tsx`

- [ ] **Step 1 — Type.** In `types/gaps.ts`, add to `PrioritizedOpportunity`:

```ts
  /** Curated list-view category (set by the list endpoint). */
  category?: 'quick_win' | 'strategic_bet' | 'other';
```

- [ ] **Step 2 — Failing tests.** In `GapAnalysis.test.tsx`, with mocked `useOpportunities` returning opps carrying `category` plus `quick_wins_count`/`strategic_bets_count`, assert:
  - a card badged **"Strategic Bet"** renders (primary category badge), and the count of "Strategic Bet" badges in the default view equals `strategic_bets_count`;
  - each card shows a folded-in effort badge (e.g. text matching `/Effort:/i`);
  - the opportunity-type dropdown options are `All Opportunities`, `Quick Wins`, `Strategic Bets`, `Other` — and there is **no** "High Effort" option in that dropdown;
  - selecting `Strategic Bets` filters cards to category==strategic_bet.
  Reuse the existing test's mocking pattern for `@/hooks/api`.

- [ ] **Step 3 — Run, expect fail.** `cd frontend && npx vitest run src/pages/GapAnalysis.test.tsx` (use the repo's configured vitest flags if the suite OOMs: `--no-file-parallelism --pool=forks`).

- [ ] **Step 4 — Implement in `GapAnalysis.tsx`.**
  - Add category presentation constants + badge:

```tsx
const CATEGORY_LABELS: Record<string, string> = {
  quick_win: 'Quick Win',
  strategic_bet: 'Strategic Bet',
  other: 'Other',
};

const CATEGORY_COLORS: Record<string, string> = {
  quick_win: '#10b981',     // green — low effort, high ROI
  strategic_bet: '#8b5cf6', // purple — high impact, high effort
  other: '#6b7280',         // neutral gray
};

function getCategoryBadge(category?: string) {
  const key = category ?? 'other';
  const color = CATEGORY_COLORS[key] || '#6b7280';
  return (
    <Badge style={{ backgroundColor: `${color}20`, color, borderColor: color }} variant="outline">
      {CATEGORY_LABELS[key] || key}
    </Badge>
  );
}
```

  - Reframe the effort badge as a secondary, folded-in attribute. Replace `DIFFICULTY_LABELS` values with short forms and prefix "Effort:":

```tsx
const DIFFICULTY_LABELS: Record<string, string> = { low: 'Low', medium: 'Medium', high: 'High' };
// getDifficultyBadge → render label as `Effort: ${DIFFICULTY_LABELS[difficulty] ?? difficulty}`
```

  - Card header (~500-503): primary `getCategoryBadge(opp.category)` then the effort sub-badge:

```tsx
<h3 className="font-semibold">{opp.recommended_action}</h3>
{getCategoryBadge(opp.category)}
{getDifficultyBadge(opp.implementation_difficulty)}
```

  - Dropdown (~468-479): rename state `difficultyFilter`→`categoryFilter` (default `'all'`); options:

```tsx
<SelectContent>
  <SelectItem value="all">All Opportunities</SelectItem>
  <SelectItem value="quick_win">Quick Wins</SelectItem>
  <SelectItem value="strategic_bet">Strategic Bets</SelectItem>
  <SelectItem value="other">Other</SelectItem>
</SelectContent>
```

  - `filteredOpportunities` (~184-194): `const matchesCategory = categoryFilter === 'all' || (opp.category ?? 'other') === categoryFilter;` (replace the difficulty match; keep the search match).
  - Chart (~197-210, title ~569, desc ~571): group by `opp.category` using `CATEGORY_LABELS`/`CATEGORY_COLORS`; retitle to **"Average ROI by Opportunity Type"**, description **"Expected returns by opportunity type"**.
  - Table: header (~662) "Difficulty" → "Type"; cell (~685-687) → `getCategoryBadge(opp.category)` (effort may be shown as small muted text if desired, but Type is primary).

- [ ] **Step 5 — Run tests, expect pass.** Re-run the vitest command from Step 3.

- [ ] **Step 6 — Typecheck + lint.** `cd frontend && npx tsc -b && npm run lint` (or the repo's configured FE lint). Fix any TS18048/strict issues.

- [ ] **Step 7 — Commit.** `feat(gap-analysis): restore Quick Win/Strategic Bet framework with effort folded in`.

---

## Self-review checklist (controller, before finishing)
- No phantom Strategic Bets: in All view, strategic-bet badges == headline count (covered by BE + FE invariant tests).
- Effort folded IN (secondary badge), not removed.
- Dropdown selects opportunity TYPE with positive labels; no "High Effort" type option.
- Dedup (`_latest_completed_per_brand`), curated counts, All-Brands selector all untouched.
- `category` is additive/optional everywhere (no breakage if absent).
