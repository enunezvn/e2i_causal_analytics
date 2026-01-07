# KPI Dictionary Page Refinement Plan

**Goal**: Refine the KPI Dictionary page to use interactive KPI cards as the primary view and simplify the tab structure.

**Status**: Refinement of completed implementation based on user feedback.

---

## User Feedback Summary

After initial implementation, user requested:
1. **Use KPI Cards instead of static tables** - The `KPICardDetailed` component (lines 852-949) is preferred because it integrates:
   - Definition text
   - Formula box with calculator icon
   - **Threshold indicators** (Target/Warning/Critical with colored icons) ✨
   - Metadata grid (Tables, Frequency, Type, Causal Library)
2. **Remove Methodology tab** - Content is outdated

The card format shows all information in ONE unified view, unlike static tables.

---

## Current State (Completed)

The following components were created and are working:
- ✅ `StatusLegend.tsx` - Status indicator legend with thresholds (KEEP)
- ✅ `KPITable.tsx` - Static table component (REMOVE from page)
- ✅ `kpi-dictionary-content.ts` - KPI data for tables (REMOVE import)
- ✅ `KeyConcepts.tsx` - Causal inference assumptions (KEEP)
- ✅ `AgenticMethodology.tsx` - 6-tier architecture (REMOVE from page)

---

## Refinement Plan

### Phase 1: Update KPIDictionary Page - Simplify Tab Structure

**File to modify:**
- `frontend/src/pages/KPIDictionary.tsx` (~1,289 lines)

**What the user wants to KEEP:**
- Stats cards row (Total KPIs, Workstreams, Causal KPIs, System Status)
- Search input + filter indicator
- Workstream tabs (All, WS1 Data Quality, WS1 Model, WS2 Triggers, WS3 Business, Brand-Specific, Causal)
- `KPICardDetailed` grid showing cards with definition + formula + thresholds

**What to REMOVE from "Tables" tab (lines 1127-1181):**
- 4x `<KPITable>` sections (WS1-WS3 static tables)
- Statistical Methods table (`<table>` with STATISTICAL_CAUSAL_METHODS)
- The "Detailed KPI Cards (Searchable)" section header (cards become the main content)
- The border-top separator (no longer needed)

**Tab structure change:**
- BEFORE: 4 tabs (KPI Tables | Status Legend | Key Concepts | Methodology)
- AFTER: 3 tabs (KPI Cards | Status Legend | Key Concepts)

**Imports to REMOVE:**
- `import { KPITable } from '@/components/visualizations/dashboard/KPITable';` (line 43)
- `import { AgenticMethodology } from '@/components/kpi/AgenticMethodology';` (line 45)
- KPI content imports (lines 46-52):
  - `WS1_DATA_COVERAGE_KPIS`
  - `WS1_MODEL_PERFORMANCE_KPIS`
  - `WS2_TRIGGER_PERFORMANCE_KPIS`
  - `WS3_BUSINESS_IMPACT_KPIS`
  - `STATISTICAL_CAUSAL_METHODS`

**Icon imports to REMOVE:**
- `Table2` (line 34) - no longer used
- `Workflow` (line 37) - no longer used

**Estimated changes**: ~100 lines removed

---

### Phase 2: Cleanup Unused Components (Optional)

**Files to optionally delete:**
- `frontend/src/components/kpi/AgenticMethodology.tsx` - No longer used
- `frontend/src/components/visualizations/dashboard/KPITable.tsx` - No longer used
- `frontend/src/data/kpi-dictionary-content.ts` - No longer used (if tables removed)

**Files to update:**
- `frontend/src/components/kpi/index.ts` - Remove AgenticMethodology export
- `frontend/src/components/visualizations/dashboard/index.ts` - Remove KPITable export

**Note**: Can keep these files if they might be useful elsewhere, but remove imports from KPIDictionary.

---

### Phase 3: Verify Build & Test

**Commands:**
```bash
npm run build    # Verify no TypeScript errors
npm run lint     # Check for unused imports
```

**Manual verification:**
1. Navigate to `/kpi-dictionary`
2. Verify 3 tabs render (KPI Cards | Status Legend | Key Concepts)
3. Verify KPI cards show definition, formula, AND thresholds
4. Verify workstream filtering works
5. Verify search functionality works

---

## Final Tab Structure

| Tab | Content |
|-----|---------|
| **KPI Cards** | Stats cards + Searchable KPI cards with workstream tabs (All, WS1 Data Quality, WS1 Model, WS2 Triggers, WS3 Business, Brand-Specific, Causal) |
| **Status Legend** | Performance colors + Data Quality/Model/Trigger thresholds |
| **Key Concepts** | Causal inference assumptions, Data quality dimensions, Model performance considerations |

---

## Critical File

| File | Purpose |
|------|---------|
| `frontend/src/pages/KPIDictionary.tsx` | Main page to modify (~1,289 lines currently) |

---

## TODO

### Phase 1: Simplify KPIDictionary Page
- [ ] Remove unused imports (lines 34, 37, 43, 45, 46-52):
  - `Table2`, `Workflow` icons
  - `KPITable`, `AgenticMethodology` components
  - `WS1_DATA_COVERAGE_KPIS`, etc. data imports
- [ ] Rename "KPI Tables" tab to "KPI Cards" (line 1078)
- [ ] Remove "Methodology" tab trigger (lines 1088-1091)
- [ ] Remove static KPITable sections from content (lines 1127-1148)
- [ ] Remove Statistical Methods table (lines 1150-1181)
- [ ] Remove "Detailed KPI Cards" section header + border (lines 1183-1187)
- [ ] Remove Methodology TabsContent (lines 1282-1284)
- [ ] Move search/filter + card grid up as primary content

### Phase 2: Cleanup (Optional)
- [ ] Remove AgenticMethodology export from kpi/index.ts
- [ ] Optionally delete unused files (KPITable.tsx, AgenticMethodology.tsx, kpi-dictionary-content.ts)

### Phase 3: Verify
- [ ] Build passes (`npm run build`)
- [ ] Lint passes (`npm run lint`)
- [ ] Manual test: 3 tabs render (KPI Cards | Status Legend | Key Concepts)
- [ ] Manual test: Search works on cards
- [ ] Manual test: Workstream filtering works
- [ ] Manual test: Cards show definition + formula + thresholds
