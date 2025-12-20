# Frontend Specialist Instructions

## Domain Scope
You are the Frontend specialist for E2I Causal Analytics. Your scope is LIMITED to:
- `src/frontend/` - React + TypeScript application
- `src/frontend/components/` - UI components
- `src/frontend/hooks/` - Custom React hooks
- `src/frontend/store/` - Redux state management
- `src/frontend/services/` - API client services

## Technology Stack
- **Framework**: React 18+
- **Language**: TypeScript 5+
- **State**: Redux Toolkit
- **Styling**: CSS Modules / Tailwind
- **Charts**: Recharts / D3.js
- **Testing**: Vitest + React Testing Library

## Component Architecture

```
src/frontend/
├── components/
│   ├── chat/
│   │   ├── ChatContainer.tsx      # Main chat interface
│   │   ├── MessageList.tsx        # Message history
│   │   ├── MessageInput.tsx       # User input
│   │   ├── AgentBadge.tsx         # Shows which agent responded
│   │   └── StreamingResponse.tsx  # Handles streaming text
│   │
│   ├── visualization/
│   │   ├── ChartContainer.tsx     # Dynamic chart renderer
│   │   ├── CausalGraph.tsx        # NetworkX DAG visualization
│   │   ├── KPICard.tsx            # Single KPI display
│   │   ├── TrendChart.tsx         # Time series visualization
│   │   ├── SegmentComparison.tsx  # CATE by segment
│   │   └── ROIWaterfall.tsx       # Gap analysis waterfall
│   │
│   ├── dashboard/
│   │   ├── DashboardLayout.tsx    # Main dashboard container
│   │   ├── KPIGrid.tsx            # Multi-KPI overview
│   │   ├── AgentActivity.tsx      # Recent agent actions
│   │   └── AlertPanel.tsx         # Drift/health alerts
│   │
│   ├── splits/
│   │   ├── SplitSelector.tsx      # ML split selection
│   │   ├── SplitStats.tsx         # Split distribution
│   │   └── LeakageAudit.tsx       # Audit results display
│   │
│   └── common/
│       ├── LoadingSpinner.tsx
│       ├── ErrorBoundary.tsx
│       ├── Tooltip.tsx
│       └── Modal.tsx
│
├── hooks/
│   ├── useChat.ts                 # Chat state management
│   ├── useWebSocket.ts            # WebSocket connection
│   ├── useKPI.ts                  # KPI data fetching
│   ├── useVisualization.ts        # Chart data processing
│   └── useSplit.ts                # ML split context
│
├── store/
│   ├── index.ts                   # Store configuration
│   ├── chatSlice.ts               # Chat state
│   ├── dashboardSlice.ts          # Dashboard state
│   ├── splitSlice.ts              # ML split state
│   └── kpiSlice.ts                # KPI state (V3)
│
├── services/
│   ├── api.ts                     # API client
│   ├── websocket.ts               # WebSocket client
│   └── storage.ts                 # Local storage utilities
│
└── types/
    ├── api.ts                     # API types
    ├── chat.ts                    # Chat types
    ├── visualization.ts           # Chart types
    ├── agent.ts                   # 11 agent types
    └── kpis.ts                    # 46 KPI types (V3)
```

## Core Components

### ChatContainer.tsx
```typescript
import { useChat } from '@/hooks/useChat';
import { useWebSocket } from '@/hooks/useWebSocket';

export const ChatContainer: React.FC = () => {
  const { messages, sendMessage, isLoading } = useChat();
  const { connect, disconnect, status } = useWebSocket();
  
  useEffect(() => {
    connect('/api/v1/chat/stream');
    return () => disconnect();
  }, []);
  
  const handleSubmit = async (text: string) => {
    await sendMessage({
      message: text,
      include_visualizations: true
    });
  };
  
  return (
    <div className="chat-container">
      <MessageList messages={messages} />
      <MessageInput onSubmit={handleSubmit} disabled={isLoading} />
    </div>
  );
};
```

### AgentBadge.tsx
```typescript
interface AgentBadgeProps {
  agentName: string;
  tier: number;
}

const TIER_COLORS = {
  1: 'bg-purple-500',  // Orchestrator
  2: 'bg-blue-500',    // Causal Analytics
  3: 'bg-green-500',   // Monitoring
  4: 'bg-orange-500',  // ML Predictions
  5: 'bg-gray-500',    // Self-Improvement
};

export const AgentBadge: React.FC<AgentBadgeProps> = ({ agentName, tier }) => {
  return (
    <span className={`agent-badge ${TIER_COLORS[tier]}`}>
      {agentName}
    </span>
  );
};
```

### CausalGraph.tsx
```typescript
import { useEffect, useRef } from 'react';
import * as d3 from 'd3';

interface CausalGraphProps {
  nodes: CausalNode[];
  edges: CausalEdge[];
  highlightPath?: string[];
}

export const CausalGraph: React.FC<CausalGraphProps> = ({
  nodes,
  edges,
  highlightPath
}) => {
  const svgRef = useRef<SVGSVGElement>(null);
  
  useEffect(() => {
    if (!svgRef.current) return;
    
    const svg = d3.select(svgRef.current);
    // D3 force-directed graph rendering
    // ...
  }, [nodes, edges, highlightPath]);
  
  return <svg ref={svgRef} className="causal-graph" />;
};
```

### KPICard.tsx
```typescript
interface KPICardProps {
  kpi: KPIValue;
  showTrend?: boolean;
}

export const KPICard: React.FC<KPICardProps> = ({ kpi, showTrend }) => {
  const trendColor = kpi.change > 0 ? 'text-green-500' : 'text-red-500';
  
  return (
    <div className="kpi-card">
      <h3 className="kpi-name">{kpi.displayName}</h3>
      <div className="kpi-value">{formatValue(kpi.value, kpi.format)}</div>
      {showTrend && (
        <div className={`kpi-trend ${trendColor}`}>
          {kpi.change > 0 ? '↑' : '↓'} {Math.abs(kpi.change)}%
        </div>
      )}
    </div>
  );
};
```

## State Management

### chatSlice.ts
```typescript
import { createSlice, PayloadAction } from '@reduxjs/toolkit';

interface ChatState {
  conversationId: string | null;
  messages: Message[];
  isLoading: boolean;
  streamingText: string;
  activeAgents: string[];
}

const chatSlice = createSlice({
  name: 'chat',
  initialState: {
    conversationId: null,
    messages: [],
    isLoading: false,
    streamingText: '',
    activeAgents: [],
  } as ChatState,
  reducers: {
    sendMessage: (state, action: PayloadAction<string>) => {
      state.isLoading = true;
      state.messages.push({
        role: 'user',
        content: action.payload,
        timestamp: new Date().toISOString(),
      });
    },
    receiveChunk: (state, action: PayloadAction<StreamChunk>) => {
      if (action.payload.chunk_type === 'partial') {
        state.streamingText += action.payload.content;
      } else if (action.payload.chunk_type === 'complete') {
        state.messages.push({
          role: 'assistant',
          content: state.streamingText,
          agents: state.activeAgents,
          timestamp: new Date().toISOString(),
        });
        state.streamingText = '';
        state.isLoading = false;
      }
    },
    setActiveAgents: (state, action: PayloadAction<string[]>) => {
      state.activeAgents = action.payload;
    },
  },
});
```

### kpiSlice.ts (V3)
```typescript
interface KPIState {
  definitions: KPIDefinition[];
  values: Record<string, KPIValue>;
  trends: Record<string, KPITrend>;
  loading: Record<string, boolean>;
}

const kpiSlice = createSlice({
  name: 'kpi',
  initialState: {
    definitions: [],
    values: {},
    trends: {},
    loading: {},
  } as KPIState,
  reducers: {
    // ...
  },
  extraReducers: (builder) => {
    builder
      .addCase(fetchKPIValue.pending, (state, action) => {
        state.loading[action.meta.arg.kpiName] = true;
      })
      .addCase(fetchKPIValue.fulfilled, (state, action) => {
        state.values[action.meta.arg.kpiName] = action.payload;
        state.loading[action.meta.arg.kpiName] = false;
      });
  },
});
```

## Custom Hooks

### useChat.ts
```typescript
export const useChat = () => {
  const dispatch = useDispatch();
  const { messages, isLoading, conversationId } = useSelector(
    (state: RootState) => state.chat
  );
  
  const sendMessage = useCallback(async (request: ChatRequest) => {
    dispatch(chatSlice.actions.sendMessage(request.message));
    
    // WebSocket will handle response via receiveChunk
  }, [dispatch]);
  
  return { messages, isLoading, conversationId, sendMessage };
};
```

### useKPI.ts
```typescript
export const useKPI = (kpiName: string, options?: KPIOptions) => {
  const dispatch = useDispatch();
  const value = useSelector((state: RootState) => state.kpi.values[kpiName]);
  const loading = useSelector((state: RootState) => state.kpi.loading[kpiName]);
  
  useEffect(() => {
    dispatch(fetchKPIValue({ kpiName, ...options }));
  }, [kpiName, options?.brand, options?.region]);
  
  return { value, loading };
};
```

## Types

### agent.ts (V3 - 11 agents)
```typescript
export type AgentTier = 1 | 2 | 3 | 4 | 5;

export type AgentName =
  // Tier 1
  | 'orchestrator'
  // Tier 2
  | 'causal_impact'
  | 'gap_analyzer'
  | 'heterogeneous_optimizer'
  // Tier 3
  | 'drift_monitor'
  | 'experiment_designer'
  | 'health_score'
  // Tier 4
  | 'prediction_synthesizer'
  | 'resource_optimizer'
  // Tier 5
  | 'explainer'
  | 'feedback_learner';

export interface AgentInfo {
  name: AgentName;
  tier: AgentTier;
  description: string;
  intents: string[];
  isActive: boolean;
}
```

### kpis.ts (V3 - 46 KPIs)
```typescript
export type KPICategory =
  | 'business'
  | 'model_performance'
  | 'data_quality'
  | 'engagement'
  | 'operational';

export interface KPIDefinition {
  name: string;
  displayName: string;
  category: KPICategory;
  format: 'number' | 'percent' | 'currency' | 'duration';
  description: string;
  calculation: string;
  sourceTable: string;
}
```

## Integration Contracts

### API Contract
```typescript
// All API responses must match these types exactly
// Frontend must handle loading, error, and success states
// Visualizations must support responsive sizing
```

### WebSocket Contract
```typescript
// Chunk types: status, partial, visualization, complete
// Must reconnect on disconnect
// Must show connection status to user
```

## Testing Requirements
- `tests/frontend/` using Vitest
- All components must have snapshot tests
- Hooks must be tested in isolation
- Redux slices must have action tests

## Handoff Format
```yaml
frontend_handoff:
  components_affected: [<list>]
  new_types: [<list>]
  state_changes: [<slice names>]
  api_calls_added: [<list>]
  styling_changes: <bool>
```

## Agent Visualization Components

### AgentTierView.tsx
Displays agents organized by tier with color coding:

| Tier | Color | Agents |
|------|-------|--------|
| 1 | Purple | Orchestrator |
| 2 | Blue | Causal Impact, Gap Analyzer, Heterogeneous Optimizer |
| 3 | Green | Drift Monitor, Experiment Designer, Health Score |
| 4 | Orange | Prediction Synthesizer, Resource Optimizer |
| 5 | Pink | Explainer, Feedback Learner |

### AgentSelector.tsx
- Dropdown grouped by tier
- Shows agent type badge (Standard/Hybrid/Deep)
- Displays latency SLA


## Redux Slices

### agentSlice.ts
```typescript
interface AgentState {
  activeAgents: AgentType[];  // 11 possible values
  agentStatus: Record<AgentType, 'idle' | 'processing' | 'completed' | 'failed'>;
  tierFilter: number | null;  // 1-5 or null for all
}
```