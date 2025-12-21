/**
 * Mock Data for Graph API
 * =======================
 *
 * Provides realistic mock data for the Knowledge Graph API endpoints.
 * Used by MSW handlers for development and testing.
 */

import {
  EntityType,
  RelationshipType,
  type GraphNode,
  type GraphRelationship,
  type GraphPath,
  type ListNodesResponse,
  type ListRelationshipsResponse,
  type GraphStatsResponse,
  type GraphHealthResponse,
  type SearchGraphResponse,
  type TraverseResponse,
  type CausalChainResponse,
  type CypherQueryResponse,
  type AddEpisodeResponse,
  type NodeNetworkResponse,
} from '@/types/graph';

// =============================================================================
// MOCK NODES
// =============================================================================

export const mockNodes: GraphNode[] = [
  {
    id: 'patient-001',
    type: EntityType.PATIENT,
    name: 'John Smith',
    properties: {
      age: 45,
      diagnosis: 'Type 2 Diabetes',
      risk_score: 0.72,
      region: 'Northeast',
    },
    created_at: '2024-01-15T10:30:00Z',
    updated_at: '2024-03-20T14:45:00Z',
  },
  {
    id: 'patient-002',
    type: EntityType.PATIENT,
    name: 'Sarah Johnson',
    properties: {
      age: 62,
      diagnosis: 'Hypertension',
      risk_score: 0.58,
      region: 'Midwest',
    },
    created_at: '2024-02-01T09:00:00Z',
    updated_at: '2024-03-18T11:20:00Z',
  },
  {
    id: 'hcp-001',
    type: EntityType.HCP,
    name: 'Dr. Emily Chen',
    properties: {
      specialty: 'Endocrinology',
      npi: '1234567890',
      prescribing_volume: 'high',
      region: 'Northeast',
    },
    created_at: '2023-06-01T00:00:00Z',
    updated_at: '2024-03-15T16:00:00Z',
  },
  {
    id: 'hcp-002',
    type: EntityType.HCP,
    name: 'Dr. Michael Brown',
    properties: {
      specialty: 'Cardiology',
      npi: '0987654321',
      prescribing_volume: 'medium',
      region: 'Midwest',
    },
    created_at: '2023-08-15T00:00:00Z',
    updated_at: '2024-03-10T09:30:00Z',
  },
  {
    id: 'brand-001',
    type: EntityType.BRAND,
    name: 'Glucomax',
    properties: {
      category: 'Diabetes',
      launch_date: '2020-01-01',
      market_share: 0.23,
    },
    created_at: '2020-01-01T00:00:00Z',
    updated_at: '2024-03-01T00:00:00Z',
  },
  {
    id: 'brand-002',
    type: EntityType.BRAND,
    name: 'CardioPlus',
    properties: {
      category: 'Cardiovascular',
      launch_date: '2019-06-15',
      market_share: 0.18,
    },
    created_at: '2019-06-15T00:00:00Z',
    updated_at: '2024-02-28T00:00:00Z',
  },
  {
    id: 'kpi-001',
    type: EntityType.KPI,
    name: 'NRx Volume',
    properties: {
      metric_type: 'prescriptions',
      aggregation: 'sum',
      target: 10000,
      current: 8500,
    },
    created_at: '2024-01-01T00:00:00Z',
    updated_at: '2024-03-20T00:00:00Z',
  },
  {
    id: 'kpi-002',
    type: EntityType.KPI,
    name: 'Patient Adherence Rate',
    properties: {
      metric_type: 'percentage',
      aggregation: 'average',
      target: 0.85,
      current: 0.78,
    },
    created_at: '2024-01-01T00:00:00Z',
    updated_at: '2024-03-20T00:00:00Z',
  },
  {
    id: 'region-001',
    type: EntityType.REGION,
    name: 'Northeast',
    properties: {
      states: ['NY', 'NJ', 'CT', 'MA', 'PA'],
      population: 55000000,
    },
    created_at: '2023-01-01T00:00:00Z',
    updated_at: '2024-01-01T00:00:00Z',
  },
  {
    id: 'trigger-001',
    type: EntityType.TRIGGER,
    name: 'HbA1c > 7.5%',
    properties: {
      condition: 'HbA1c > 7.5',
      action: 'Recommend intensification',
      priority: 'high',
    },
    created_at: '2024-01-15T00:00:00Z',
    updated_at: '2024-02-01T00:00:00Z',
  },
  {
    id: 'causal-path-001',
    type: EntityType.CAUSAL_PATH,
    name: 'HCP Engagement -> NRx',
    properties: {
      effect_size: 0.34,
      confidence: 0.89,
      p_value: 0.001,
    },
    created_at: '2024-02-01T00:00:00Z',
    updated_at: '2024-03-15T00:00:00Z',
  },
  {
    id: 'agent-001',
    type: EntityType.AGENT,
    name: 'Causal Discovery Agent',
    properties: {
      status: 'active',
      last_run: '2024-03-20T10:00:00Z',
      discoveries: 156,
    },
    created_at: '2024-01-01T00:00:00Z',
    updated_at: '2024-03-20T10:00:00Z',
  },
];

// =============================================================================
// MOCK RELATIONSHIPS
// =============================================================================

export const mockRelationships: GraphRelationship[] = [
  {
    id: 'rel-001',
    type: RelationshipType.TREATED_BY,
    source_id: 'patient-001',
    target_id: 'hcp-001',
    properties: { start_date: '2023-06-15' },
    confidence: 1.0,
    created_at: '2023-06-15T00:00:00Z',
  },
  {
    id: 'rel-002',
    type: RelationshipType.TREATED_BY,
    source_id: 'patient-002',
    target_id: 'hcp-002',
    properties: { start_date: '2023-09-01' },
    confidence: 1.0,
    created_at: '2023-09-01T00:00:00Z',
  },
  {
    id: 'rel-003',
    type: RelationshipType.PRESCRIBES,
    source_id: 'hcp-001',
    target_id: 'brand-001',
    properties: { frequency: 'weekly', avg_quantity: 30 },
    confidence: 0.95,
    created_at: '2023-07-01T00:00:00Z',
  },
  {
    id: 'rel-004',
    type: RelationshipType.PRESCRIBES,
    source_id: 'hcp-002',
    target_id: 'brand-002',
    properties: { frequency: 'monthly', avg_quantity: 60 },
    confidence: 0.88,
    created_at: '2023-10-01T00:00:00Z',
  },
  {
    id: 'rel-005',
    type: RelationshipType.CAUSES,
    source_id: 'trigger-001',
    target_id: 'kpi-001',
    properties: { effect_size: 0.45, lag_days: 14 },
    confidence: 0.82,
    created_at: '2024-02-15T00:00:00Z',
  },
  {
    id: 'rel-006',
    type: RelationshipType.IMPACTS,
    source_id: 'brand-001',
    target_id: 'kpi-002',
    properties: { direction: 'positive', magnitude: 'medium' },
    confidence: 0.76,
    created_at: '2024-01-20T00:00:00Z',
  },
  {
    id: 'rel-007',
    type: RelationshipType.LOCATED_IN,
    source_id: 'hcp-001',
    target_id: 'region-001',
    properties: { primary: true },
    confidence: 1.0,
    created_at: '2023-06-01T00:00:00Z',
  },
  {
    id: 'rel-008',
    type: RelationshipType.DISCOVERED,
    source_id: 'agent-001',
    target_id: 'causal-path-001',
    properties: { algorithm: 'PC', iterations: 1000 },
    confidence: 0.89,
    created_at: '2024-02-01T00:00:00Z',
  },
  {
    id: 'rel-009',
    type: RelationshipType.INFLUENCES,
    source_id: 'hcp-001',
    target_id: 'kpi-001',
    properties: { influence_type: 'direct', weight: 0.67 },
    confidence: 0.84,
    created_at: '2024-02-10T00:00:00Z',
  },
  {
    id: 'rel-010',
    type: RelationshipType.RECEIVED,
    source_id: 'patient-001',
    target_id: 'brand-001',
    properties: { prescription_date: '2024-03-01', refills: 3 },
    confidence: 1.0,
    created_at: '2024-03-01T00:00:00Z',
  },
];

// =============================================================================
// MOCK PATHS
// =============================================================================

export const mockPaths: GraphPath[] = [
  {
    nodes: [mockNodes[0], mockNodes[2], mockNodes[4]],
    relationships: [mockRelationships[0], mockRelationships[2]],
    total_confidence: 0.95,
    path_length: 2,
  },
  {
    nodes: [mockNodes[9], mockNodes[6]],
    relationships: [mockRelationships[4]],
    total_confidence: 0.82,
    path_length: 1,
  },
];

// =============================================================================
// MOCK API RESPONSES
// =============================================================================

export function createListNodesResponse(
  nodes: GraphNode[] = mockNodes,
  limit = 50,
  offset = 0
): ListNodesResponse {
  const paginatedNodes = nodes.slice(offset, offset + limit);
  return {
    nodes: paginatedNodes,
    total_count: nodes.length,
    limit,
    offset,
    has_more: offset + limit < nodes.length,
    query_latency_ms: Math.floor(Math.random() * 50) + 10,
    timestamp: new Date().toISOString(),
  };
}

export function createListRelationshipsResponse(
  relationships: GraphRelationship[] = mockRelationships,
  limit = 50,
  offset = 0
): ListRelationshipsResponse {
  const paginatedRelationships = relationships.slice(offset, offset + limit);
  return {
    relationships: paginatedRelationships,
    total_count: relationships.length,
    limit,
    offset,
    has_more: offset + limit < relationships.length,
    query_latency_ms: Math.floor(Math.random() * 50) + 10,
    timestamp: new Date().toISOString(),
  };
}

export function createGraphStatsResponse(): GraphStatsResponse {
  const nodesByType: Record<string, number> = {};
  const relationshipsByType: Record<string, number> = {};

  mockNodes.forEach((node) => {
    nodesByType[node.type] = (nodesByType[node.type] || 0) + 1;
  });

  mockRelationships.forEach((rel) => {
    relationshipsByType[rel.type] = (relationshipsByType[rel.type] || 0) + 1;
  });

  return {
    total_nodes: mockNodes.length,
    total_relationships: mockRelationships.length,
    nodes_by_type: nodesByType,
    relationships_by_type: relationshipsByType,
    total_episodes: 42,
    total_communities: 5,
    last_updated: new Date().toISOString(),
    timestamp: new Date().toISOString(),
  };
}

export function createGraphHealthResponse(): GraphHealthResponse {
  return {
    status: 'healthy',
    graphiti: 'connected',
    falkordb: 'connected',
    websocket_connections: 3,
    timestamp: new Date().toISOString(),
  };
}

export function createSearchGraphResponse(query: string): SearchGraphResponse {
  const lowercaseQuery = query.toLowerCase();
  const matchingNodes = mockNodes.filter(
    (node) =>
      node.name.toLowerCase().includes(lowercaseQuery) ||
      node.type.toLowerCase().includes(lowercaseQuery)
  );

  return {
    results: matchingNodes.map((node) => ({
      id: node.id,
      name: node.name,
      type: node.type,
      score: Math.random() * 0.3 + 0.7, // Random score between 0.7 and 1.0
      properties: node.properties,
    })),
    total_results: matchingNodes.length,
    query,
    query_latency_ms: Math.floor(Math.random() * 100) + 50,
    timestamp: new Date().toISOString(),
  };
}

export function createTraverseResponse(startNodeId: string): TraverseResponse {
  const startNode = mockNodes.find((n) => n.id === startNodeId);
  const connectedRelationships = mockRelationships.filter(
    (r) => r.source_id === startNodeId || r.target_id === startNodeId
  );
  const connectedNodeIds = new Set(
    connectedRelationships.flatMap((r) => [r.source_id, r.target_id])
  );
  const connectedNodes = mockNodes.filter((n) => connectedNodeIds.has(n.id));

  return {
    subgraph: {
      nodes: connectedNodes,
      relationships: connectedRelationships,
    },
    nodes: connectedNodes,
    relationships: connectedRelationships,
    paths:
      startNode && connectedNodes.length > 1
        ? [
            {
              nodes: connectedNodes.slice(0, 3),
              relationships: connectedRelationships.slice(0, 2),
              total_confidence: 0.85,
              path_length: Math.min(2, connectedNodes.length - 1),
            },
          ]
        : [],
    max_depth_reached: 2,
    query_latency_ms: Math.floor(Math.random() * 150) + 50,
    timestamp: new Date().toISOString(),
  };
}

export function createCausalChainResponse(): CausalChainResponse {
  return {
    chains: mockPaths,
    total_chains: mockPaths.length,
    strongest_chain: mockPaths[0],
    aggregate_effect: 0.42,
    query_latency_ms: Math.floor(Math.random() * 200) + 100,
    timestamp: new Date().toISOString(),
  };
}

export function createCypherQueryResponse(): CypherQueryResponse {
  return {
    results: mockNodes.slice(0, 5).map((node) => ({
      n: {
        id: node.id,
        name: node.name,
        type: node.type,
        ...node.properties,
      },
    })),
    columns: ['n'],
    row_count: 5,
    read_only: true,
    query_latency_ms: Math.floor(Math.random() * 100) + 20,
    timestamp: new Date().toISOString(),
  };
}

export function createAddEpisodeResponse(): AddEpisodeResponse {
  return {
    episode_id: `episode-${Date.now()}`,
    extracted_entities: [
      { type: 'Patient', name: 'Sample Patient', confidence: 0.92 },
      { type: 'Brand', name: 'Sample Drug', confidence: 0.88 },
    ],
    extracted_relationships: [
      {
        type: 'RECEIVED',
        source_id: 'patient-new',
        target_id: 'brand-new',
        confidence: 0.85,
      },
    ],
    content_summary: 'Patient received medication prescription.',
    processing_latency_ms: Math.floor(Math.random() * 500) + 200,
    timestamp: new Date().toISOString(),
  };
}

export function createNodeNetworkResponse(nodeId: string): NodeNetworkResponse {
  const node = mockNodes.find((n) => n.id === nodeId);
  const connectedRelationships = mockRelationships.filter(
    (r) => r.source_id === nodeId || r.target_id === nodeId
  );

  const connectedNodes: Record<
    string,
    Array<{ id: string; properties: Record<string, unknown> }>
  > = {};

  connectedRelationships.forEach((rel) => {
    const connectedId = rel.source_id === nodeId ? rel.target_id : rel.source_id;
    const connectedNode = mockNodes.find((n) => n.id === connectedId);
    if (connectedNode) {
      const type = connectedNode.type;
      if (!connectedNodes[type]) {
        connectedNodes[type] = [];
      }
      connectedNodes[type].push({
        id: connectedNode.id,
        properties: connectedNode.properties,
      });
    }
  });

  return {
    node_id: nodeId,
    node_type: node?.type || EntityType.PATIENT,
    connected_nodes: connectedNodes,
    total_connections: connectedRelationships.length,
    max_depth: 1,
    query_latency_ms: Math.floor(Math.random() * 80) + 20,
    timestamp: new Date().toISOString(),
  };
}
