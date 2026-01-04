# E2I Causal Analytics - CopilotKit Integration

## Why CopilotKit for E2I?

CopilotKit provides **bidirectional state synchronization** between your app and AI agents, which is critical for E2I's 18-agent architecture where agents need to:

1. **See dashboard context** (filters, active tab, selected KPIs)
2. **Trigger UI actions** (highlight charts, navigate, update filters)
3. **Coordinate across tiers** (orchestrator → specialist agents)

---

## CopilotKit vs assistant-ui Comparison

| Feature | CopilotKit | assistant-ui |
|---------|------------|--------------|
| **App Context Awareness** | ✅ `useCopilotReadable` - automatic | ⚠️ Manual via provider |
| **Agent→UI Actions** | ✅ `useCopilotAction` - declarative | ⚠️ Manual event parsing |
| **Multi-Agent Support** | ✅ `useAgent` hook native | ⚠️ Build your own |
| **LangGraph Integration** | ✅ First-class CoAgents | ✅ First-class SDK |
| **Customization** | Good (CSS vars + subcomponents) | Excellent (Radix primitives) |
| **Bundle Size** | Larger (~50KB+) | Smaller (~30KB) |
| **Learning Curve** | Steeper (more concepts) | Gentler (just UI) |

### When to Choose CopilotKit

- ✅ Agents need to **read app state** (filters, visible data)
- ✅ Agents should **trigger frontend actions** (highlight, navigate)
- ✅ **Multi-agent coordination** is required
- ✅ You want **declarative action definitions**

### When to Choose assistant-ui

- ✅ You need **maximum UI customization**
- ✅ Simple chat without app integration
- ✅ Minimal dependencies preferred
- ✅ Backend handles all state

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        E2ICopilotProvider                           │
│  ┌───────────────────────────────────────────────────────────────┐  │
│  │  useCopilotReadable                                           │  │
│  │  • Dashboard filters (brand, region, time)                    │  │
│  │  • Active tab                                                 │  │
│  │  • Selected KPIs                                              │  │
│  │  • User role (executive/analyst/data_scientist)               │  │
│  │  • Agent registry (18 agents, 6 tiers)                        │  │
│  └───────────────────────────────────────────────────────────────┘  │
│                               ↕ (bidirectional)                     │
│  ┌───────────────────────────────────────────────────────────────┐  │
│  │  useCopilotAction                                             │  │
│  │  • updateFilters → setFilters()                               │  │
│  │  • highlightCausalPath → setHighlightedPaths()                │  │
│  │  • showValidationResults → setLastValidation()                │  │
│  │  • navigateToTab → setActiveTab()                             │  │
│  │  • highlightChartElement → setHighlightedCharts()             │  │
│  │  • showGapDetails → setPendingActions()                       │  │
│  │  • generateReport → setPendingActions()                       │  │
│  │  • updateAgentStatus → setActiveAgents()                      │  │
│  └───────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────┘
                                  ↕
┌─────────────────────────────────────────────────────────────────────┐
│                    LangGraph (18 Agents)                            │
│  Tier 0: QC Gate, Feature Engineer, Model Trainer                   │
│  Tier 1: Orchestrator                                               │
│  Tier 2: Causal Impact, Gap Analyzer, Heterogeneous Optimizer       │
│  Tier 3: Drift Monitor, Experiment Designer, Health Score           │
│  Tier 4: Prediction Synthesizer, Resource Optimizer, Knowledge Syn. │
│  Tier 5: Explainer, Feedback Learner, Narrator, Recommender, Meta   │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Quick Start

### 1. Install

```bash
npm install @copilotkit/react-core @copilotkit/react-ui framer-motion
```

### 2. Set Up Provider

```tsx
import { E2ICopilotProvider, E2IChatSidebar } from '@e2i/copilotkit-ui';

export default function App() {
  return (
    <E2ICopilotProvider
      runtimeUrl="/api/copilotkit"
      initialFilters={{ brand: 'Remibrutinib' }}
      userRole="analyst"
    >
      <Dashboard />
      <E2IChatSidebar />
    </E2ICopilotProvider>
  );
}
```

### 3. Use Hooks in Dashboard

```tsx
import { useE2IFilters, useE2IHighlights } from '@e2i/copilotkit-ui';

function Dashboard() {
  // These filters are automatically visible to agents!
  const { filters, setBrand, setRegion } = useE2IFilters();
  
  // Agent-triggered highlights
  const { highlightedPaths, isPathHighlighted } = useE2IHighlights();
  
  return (
    <div>
      {/* Filter changes sync to agent context */}
      <select onChange={(e) => setBrand(e.target.value)}>
        <option value="Remibrutinib">Remibrutinib</option>
        <option value="Fabhalta">Fabhalta</option>
        <option value="Kisqali">Kisqali</option>
      </select>
      
      {/* Agent can highlight causal paths */}
      {highlightedPaths.map(path => (
        <HighlightedPath key={path} path={path} />
      ))}
    </div>
  );
}
```

---

## Key Features

### 1. Automatic Context Sharing

Agents can "see" your dashboard state:

```tsx
// In E2ICopilotProvider - these are auto-synced to agents
useCopilotReadable({
  description: 'Current dashboard filters',
  value: dashboard.filters,  // { brand: 'Remibrutinib', region: 'south' }
});

useCopilotReadable({
  description: 'User expertise level',
  value: dashboard.userRole,  // 'analyst'
});
```

Now agents adjust their responses:
- **Executive**: High-level summaries
- **Analyst**: Detailed breakdowns
- **Data Scientist**: Technical methodology

### 2. Agent-Triggered Actions

Agents can control the UI:

```tsx
useCopilotAction({
  name: 'highlightCausalPath',
  description: 'Highlight a causal relationship in the DAG',
  parameters: [
    { name: 'treatment', type: 'string' },
    { name: 'outcome', type: 'string' },
    { name: 'effect', type: 'number' },
  ],
  handler: async ({ treatment, outcome, effect }) => {
    // This updates React state → UI re-renders with highlight
    setHighlightedPaths(prev => [...prev, `${treatment}->${outcome}`]);
    return `Highlighted: ${treatment} → ${outcome}`;
  },
});
```

### 3. Validation Display

When Causal Impact agent runs refutation tests:

```tsx
useCopilotAction({
  name: 'showValidationResults',
  handler: async ({ gateDecision, confidence, testsPassed }) => {
    setLastValidation({
      gateDecision,  // 'proceed' | 'review' | 'block'
      overallConfidence: confidence,
      testsPassed,
      // ...
    });
  },
});
```

The `ValidationBadge` component automatically renders.

### 4. Multi-Agent Status

Track which of 18 agents are working:

```tsx
const { activeAgents, getActiveAgents } = useE2IAgents();

// In chat, see: "🔍 Gap Analyzer (computing) • ⚡ Causal Impact (thinking)"
```

---

## File Structure

```
e2i-copilotkit/
├── src/
│   ├── index.ts                    # Main exports
│   ├── types/
│   │   └── index.ts                # 18 agents, validation, actions
│   ├── providers/
│   │   └── E2ICopilotProvider.tsx  # Main provider with useCopilotReadable/Action
│   ├── components/
│   │   ├── E2IChatSidebar.tsx      # CopilotKit sidebar customization
│   │   ├── AgentBadge.tsx          # Tier-colored badges
│   │   ├── ValidationBadge.tsx     # Gate decisions
│   │   ├── FilterContextBar.tsx    # Active filters
│   │   └── PendingActionsPanel.tsx # Agent action queue
│   └── hooks/
│       └── index.ts                # useE2IChat, useE2IFilters, etc.
├── examples/
│   └── usage.tsx                   # Integration examples
├── package.json
└── README.md
```

---

## API Endpoint (LangGraph)

Create `/api/copilotkit/route.ts`:

```typescript
import { CopilotRuntime, LangGraphAdapter } from '@copilotkit/runtime';
import { NextRequest } from 'next/server';

export async function POST(req: NextRequest) {
  const runtime = new CopilotRuntime({
    remoteActions: [
      {
        url: process.env.LANGGRAPH_URL!,
        headers: {
          'x-api-key': process.env.LANGCHAIN_API_KEY!,
        },
      },
    ],
  });

  return runtime.handleRequest(req);
}
```

---

## Comparison Summary

### CopilotKit Advantages for E2I

| E2I Need | How CopilotKit Solves It |
|----------|--------------------------|
| 18 agents need dashboard context | `useCopilotReadable` auto-syncs filters, KPIs, role |
| Agents should highlight findings | `useCopilotAction` for `highlightCausalPath` |
| Validation gates need UI display | `useCopilotAction` for `showValidationResults` |
| Navigation from chat | `useCopilotAction` for `navigateToTab` |
| Report generation | `useCopilotAction` for `generateReport` |
| Agent status visibility | `updateAgentStatus` action + `useE2IAgents` hook |

### What We Built Custom

- `AgentBadge` - Tier-colored badges (same as assistant-ui version)
- `ValidationBadge` - Proceed/Review/Block gates
- `FilterContextBar` - Active filter chips
- `PendingActionsPanel` - Agent action queue UI
- Domain-specific hooks (`useE2IHighlights`, etc.)

---

## Migration from assistant-ui

If you started with assistant-ui:

```diff
- import { AssistantRuntimeProvider } from '@assistant-ui/react';
+ import { E2ICopilotProvider } from '@e2i/copilotkit-ui';

- <AssistantRuntimeProvider runtime={runtime}>
+ <E2ICopilotProvider runtimeUrl="/api/copilotkit">
    <App />
- </AssistantRuntimeProvider>
+ </E2ICopilotProvider>
```

The custom components (AgentBadge, ValidationBadge) work with both.

---

## Conclusion

**For E2I's requirements**, CopilotKit's `useCopilotReadable` and `useCopilotAction` provide significant advantages over assistant-ui's pure-UI approach. The bidirectional state sync means your 18-agent system can:

1. See what the user is looking at
2. Adjust analysis to match context
3. Directly manipulate the UI to explain findings
4. Coordinate multi-agent workflows with visible status

This creates a more integrated, "copilot-style" experience rather than just a chat box.
