/**
 * Mock Data for Memory API
 * ========================
 *
 * Provides realistic mock data for the Memory System API endpoints.
 * Used by MSW handlers for development and testing.
 */

import {
  MemoryType,
  RetrievalMethod,
  type MemorySearchResult,
  type MemorySearchResponse,
  type EpisodicMemoryResponse,
  type ProceduralFeedbackResponse,
  type SemanticPathResponse,
  type MemoryStatsResponse,
} from '@/types/memory';

// =============================================================================
// MOCK EPISODIC MEMORIES
// =============================================================================

export const mockEpisodicMemories: EpisodicMemoryResponse[] = [
  {
    id: 'em-001',
    content:
      'Patient John Smith (ID: patient-001) visited Dr. Emily Chen for routine diabetes checkup. HbA1c measured at 7.8%, indicating room for improvement. Recommended increasing Glucomax dosage.',
    event_type: 'consultation',
    session_id: 'session-2024-001',
    agent_name: 'clinical_agent',
    brand: 'Glucomax',
    region: 'Northeast',
    created_at: '2024-03-15T10:30:00Z',
    metadata: {
      importance: 0.85,
      verified: true,
    },
  },
  {
    id: 'em-002',
    content:
      'Causal discovery analysis completed. Identified strong causal link between HCP engagement activities and NRx volume increase (effect size: 0.34, p < 0.001).',
    event_type: 'discovery',
    session_id: 'session-2024-002',
    agent_name: 'causal_agent',
    brand: 'Glucomax',
    region: 'All',
    created_at: '2024-03-14T14:22:00Z',
    metadata: {
      importance: 0.92,
      algorithm: 'PC',
    },
  },
  {
    id: 'em-003',
    content:
      'User query: "What factors are driving NRx decline in the Midwest region?" Analysis initiated using hybrid retrieval across episodic and semantic memory.',
    event_type: 'query',
    session_id: 'session-2024-003',
    agent_name: 'orchestrator',
    region: 'Midwest',
    created_at: '2024-03-13T09:15:00Z',
    metadata: {
      importance: 0.7,
      response_time_ms: 234,
    },
  },
  {
    id: 'em-004',
    content:
      'Market access barrier identified in Northeast region for CardioPlus. Formulary restrictions at major PBMs affecting prescription volume.',
    event_type: 'insight',
    session_id: 'session-2024-004',
    agent_name: 'market_agent',
    brand: 'CardioPlus',
    region: 'Northeast',
    created_at: '2024-03-12T16:45:00Z',
    metadata: {
      importance: 0.88,
      action_required: true,
    },
  },
  {
    id: 'em-005',
    content:
      'Intervention simulation completed: Predicted 15% increase in patient adherence if reminder notifications are implemented. Confidence interval: [12%, 18%].',
    event_type: 'simulation',
    session_id: 'session-2024-005',
    agent_name: 'predictive_agent',
    brand: 'Glucomax',
    created_at: '2024-03-11T11:00:00Z',
    metadata: {
      importance: 0.81,
      simulation_runs: 1000,
    },
  },
];

// =============================================================================
// MOCK SEARCH RESULTS
// =============================================================================

export const mockSearchResults: MemorySearchResult[] = [
  {
    content:
      'HCP engagement program shows strong correlation with prescription volume increases.',
    source: 'episodic',
    source_id: 'em-002',
    score: 0.92,
    retrieval_method: 'hybrid',
    metadata: {
      timestamp: '2024-03-14T14:22:00Z',
      agent: 'causal_agent',
    },
  },
  {
    content:
      'Patient adherence rates improve with regular HCP follow-up appointments.',
    source: 'semantic',
    source_id: 'fact-001',
    score: 0.87,
    retrieval_method: 'dense',
    metadata: {
      category: 'clinical_insight',
      confidence: 0.89,
    },
  },
  {
    content:
      'Northeast region shows highest response to digital engagement initiatives.',
    source: 'episodic',
    source_id: 'em-001',
    score: 0.84,
    retrieval_method: 'sparse',
    metadata: {
      region: 'Northeast',
      date_range: '2024-Q1',
    },
  },
  {
    content:
      'Procedure: When analyzing KPI trends, first check for seasonality patterns, then examine regional variations.',
    source: 'procedural',
    source_id: 'proc-001',
    score: 0.79,
    retrieval_method: 'graph',
    metadata: {
      success_rate: 0.85,
      usage_count: 47,
    },
  },
  {
    content:
      'Glucomax market share increased 2.3% following the Q1 awareness campaign.',
    source: 'episodic',
    source_id: 'em-006',
    score: 0.76,
    retrieval_method: 'hybrid',
    metadata: {
      brand: 'Glucomax',
      verified: true,
    },
  },
];

// =============================================================================
// MOCK API RESPONSES
// =============================================================================

export function createMemorySearchResponse(
  query: string,
  method: RetrievalMethod = RetrievalMethod.HYBRID,
  k = 10
): MemorySearchResponse {
  // Simulate relevance scoring based on query terms
  const lowercaseQuery = query.toLowerCase();
  const filteredResults = mockSearchResults
    .filter(
      (result) =>
        result.content.toLowerCase().includes(lowercaseQuery) ||
        Object.values(result.metadata).some(
          (v) =>
            typeof v === 'string' && v.toLowerCase().includes(lowercaseQuery)
        )
    )
    .slice(0, k);

  // If no matches, return top results anyway for demo purposes
  const results = filteredResults.length > 0 ? filteredResults : mockSearchResults.slice(0, k);

  return {
    query,
    results: results.map((r) => ({
      ...r,
      retrieval_method: method,
    })),
    total_results: results.length,
    retrieval_method: method,
    search_latency_ms: Math.floor(Math.random() * 150) + 50,
    timestamp: new Date().toISOString(),
  };
}

export function createEpisodicMemoryResponse(
  id: string
): EpisodicMemoryResponse | null {
  return mockEpisodicMemories.find((m) => m.id === id) || null;
}

export function createNewEpisodicMemoryResponse(
  content: string,
  eventType: string,
  sessionId?: string,
  agentName?: string,
  brand?: string,
  region?: string
): EpisodicMemoryResponse {
  return {
    id: `em-${Date.now()}`,
    content,
    event_type: eventType,
    session_id: sessionId,
    agent_name: agentName,
    brand,
    region,
    created_at: new Date().toISOString(),
    metadata: {
      importance: 0.5,
    },
  };
}

export function createProceduralFeedbackResponse(
  procedureId: string,
  outcome: 'success' | 'partial' | 'failure',
  score: number
): ProceduralFeedbackResponse {
  return {
    procedure_id: procedureId,
    feedback_recorded: true,
    new_success_rate: outcome === 'success' ? score + 0.02 : score - 0.01,
    message: `Feedback recorded for procedure ${procedureId}. Outcome: ${outcome}`,
    timestamp: new Date().toISOString(),
  };
}

export function createSemanticPathResponse(): SemanticPathResponse {
  return {
    paths: [
      {
        nodes: ['HCP Engagement', 'NRx Volume', 'Revenue'],
        edges: ['CAUSES', 'IMPACTS'],
        confidence: 0.87,
        effect_size: 0.34,
      },
      {
        nodes: ['Patient Adherence', 'Treatment Outcome', 'NRx Volume'],
        edges: ['INFLUENCES', 'CAUSES'],
        confidence: 0.82,
        effect_size: 0.28,
      },
    ],
    total_paths: 2,
    max_depth_searched: 3,
    query_latency_ms: Math.floor(Math.random() * 200) + 100,
    timestamp: new Date().toISOString(),
  };
}

export function createMemoryStatsResponse(): MemoryStatsResponse {
  return {
    episodic: {
      total_memories: mockEpisodicMemories.length + 1247,
      recent_24h: 23,
    },
    procedural: {
      total_procedures: 156,
      average_success_rate: 0.847,
    },
    semantic: {
      total_entities: 3421,
      total_relationships: 8976,
    },
    last_updated: new Date().toISOString(),
  };
}

// =============================================================================
// HELPER FUNCTIONS
// =============================================================================

export function filterMemoriesByType(
  memoryTypes: MemoryType[]
): MemorySearchResult[] {
  if (memoryTypes.includes(MemoryType.ALL)) {
    return mockSearchResults;
  }

  return mockSearchResults.filter((result) => {
    if (memoryTypes.includes(MemoryType.EPISODIC) && result.source === 'episodic') {
      return true;
    }
    if (memoryTypes.includes(MemoryType.PROCEDURAL) && result.source === 'procedural') {
      return true;
    }
    if (memoryTypes.includes(MemoryType.SEMANTIC) && result.source === 'semantic') {
      return true;
    }
    return false;
  });
}
