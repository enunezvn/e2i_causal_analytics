/**
 * MSW Request Handlers
 * ====================
 *
 * Defines mock API handlers for development and testing.
 * These handlers intercept HTTP requests and return mock data.
 */

import { http, HttpResponse, delay } from 'msw';
import { env } from '@/config/env';

// Graph mock data
import {
  mockNodes,
  mockRelationships,
  createListNodesResponse,
  createListRelationshipsResponse,
  createGraphStatsResponse,
  createGraphHealthResponse,
  createSearchGraphResponse,
  createTraverseResponse,
  createCausalChainResponse,
  createCypherQueryResponse,
  createAddEpisodeResponse,
  createNodeNetworkResponse,
} from './data/graph';

// Memory mock data
import {
  createMemorySearchResponse,
  createEpisodicMemoryResponse,
  createNewEpisodicMemoryResponse,
  createProceduralFeedbackResponse,
  createSemanticPathResponse,
  createMemoryStatsResponse,
  mockEpisodicMemories,
} from './data/memory';

import { RetrievalMethod } from '@/types/memory';

// Base URL for API requests
const baseUrl = env.apiUrl;

/**
 * Simulate network delay for more realistic behavior
 */
const simulateDelay = () => delay(Math.random() * 200 + 50);

/**
 * Graph API Handlers
 */
const graphHandlers = [
  // GET /graph/nodes - List nodes
  http.get(`${baseUrl}/graph/nodes`, async ({ request }) => {
    await simulateDelay();

    const url = new URL(request.url);
    const entityTypes = url.searchParams.get('entity_types');
    const search = url.searchParams.get('search');
    const limit = parseInt(url.searchParams.get('limit') || '50', 10);
    const offset = parseInt(url.searchParams.get('offset') || '0', 10);

    let filteredNodes = [...mockNodes];

    // Filter by entity types
    if (entityTypes) {
      const types = entityTypes.split(',');
      filteredNodes = filteredNodes.filter((node) =>
        types.includes(node.type)
      );
    }

    // Filter by search term
    if (search) {
      const searchLower = search.toLowerCase();
      filteredNodes = filteredNodes.filter(
        (node) =>
          node.name.toLowerCase().includes(searchLower) ||
          node.type.toLowerCase().includes(searchLower)
      );
    }

    return HttpResponse.json(
      createListNodesResponse(filteredNodes, limit, offset)
    );
  }),

  // GET /graph/nodes/:id - Get single node
  http.get(`${baseUrl}/graph/nodes/:id`, async ({ params }) => {
    await simulateDelay();

    const { id } = params;
    const node = mockNodes.find((n) => n.id === id);

    if (!node) {
      return HttpResponse.json(
        { detail: `Node with id '${id}' not found` },
        { status: 404 }
      );
    }

    return HttpResponse.json({
      ...node,
      timestamp: new Date().toISOString(),
    });
  }),

  // GET /graph/nodes/:id/network - Get node network
  http.get(`${baseUrl}/graph/nodes/:id/network`, async ({ params }) => {
    await simulateDelay();

    const { id } = params;
    const node = mockNodes.find((n) => n.id === id);

    if (!node) {
      return HttpResponse.json(
        { detail: `Node with id '${id}' not found` },
        { status: 404 }
      );
    }

    return HttpResponse.json(createNodeNetworkResponse(id as string));
  }),

  // GET /graph/relationships - List relationships
  http.get(`${baseUrl}/graph/relationships`, async ({ request }) => {
    await simulateDelay();

    const url = new URL(request.url);
    const relationshipTypes = url.searchParams.get('relationship_types');
    const sourceId = url.searchParams.get('source_id');
    const targetId = url.searchParams.get('target_id');
    const minConfidence = parseFloat(
      url.searchParams.get('min_confidence') || '0'
    );
    const limit = parseInt(url.searchParams.get('limit') || '50', 10);
    const offset = parseInt(url.searchParams.get('offset') || '0', 10);

    let filteredRels = [...mockRelationships];

    // Filter by relationship types
    if (relationshipTypes) {
      const types = relationshipTypes.split(',');
      filteredRels = filteredRels.filter((rel) => types.includes(rel.type));
    }

    // Filter by source/target
    if (sourceId) {
      filteredRels = filteredRels.filter((rel) => rel.source_id === sourceId);
    }
    if (targetId) {
      filteredRels = filteredRels.filter((rel) => rel.target_id === targetId);
    }

    // Filter by confidence
    if (minConfidence > 0) {
      filteredRels = filteredRels.filter(
        (rel) => (rel.confidence || 0) >= minConfidence
      );
    }

    return HttpResponse.json(
      createListRelationshipsResponse(filteredRels, limit, offset)
    );
  }),

  // POST /graph/traverse - Traverse graph
  http.post(`${baseUrl}/graph/traverse`, async ({ request }) => {
    await simulateDelay();

    const body = (await request.json()) as { start_node_id: string };
    const { start_node_id } = body;

    const node = mockNodes.find((n) => n.id === start_node_id);
    if (!node) {
      return HttpResponse.json(
        { detail: `Start node '${start_node_id}' not found` },
        { status: 404 }
      );
    }

    return HttpResponse.json(createTraverseResponse(start_node_id));
  }),

  // POST /graph/causal-chains - Query causal chains
  http.post(`${baseUrl}/graph/causal-chains`, async () => {
    await simulateDelay();
    return HttpResponse.json(createCausalChainResponse());
  }),

  // POST /graph/query - Execute Cypher query
  http.post(`${baseUrl}/graph/query`, async () => {
    await simulateDelay();
    return HttpResponse.json(createCypherQueryResponse());
  }),

  // POST /graph/episodes - Add episode
  http.post(`${baseUrl}/graph/episodes`, async () => {
    await simulateDelay();
    return HttpResponse.json(createAddEpisodeResponse(), { status: 201 });
  }),

  // POST /graph/search - Natural language search
  http.post(`${baseUrl}/graph/search`, async ({ request }) => {
    await simulateDelay();

    const body = (await request.json()) as { query: string };
    const { query } = body;

    return HttpResponse.json(createSearchGraphResponse(query));
  }),

  // GET /graph/stats - Graph statistics
  http.get(`${baseUrl}/graph/stats`, async () => {
    await simulateDelay();
    return HttpResponse.json(createGraphStatsResponse());
  }),

  // GET /graph/health - Graph health check
  http.get(`${baseUrl}/graph/health`, async () => {
    await simulateDelay();
    return HttpResponse.json(createGraphHealthResponse());
  }),
];

/**
 * Memory API Handlers
 */
const memoryHandlers = [
  // POST /memory/search - Hybrid memory search
  http.post(`${baseUrl}/memory/search`, async ({ request }) => {
    await simulateDelay();

    const body = (await request.json()) as {
      query: string;
      k?: number;
      retrieval_method?: RetrievalMethod;
    };
    const { query, k = 10, retrieval_method = RetrievalMethod.HYBRID } = body;

    return HttpResponse.json(createMemorySearchResponse(query, retrieval_method, k));
  }),

  // GET /memory/episodic - List episodic memories
  http.get(`${baseUrl}/memory/episodic`, async ({ request }) => {
    await simulateDelay();

    const url = new URL(request.url);
    const limit = parseInt(url.searchParams.get('limit') || '20', 10);
    const offset = parseInt(url.searchParams.get('offset') || '0', 10);

    const paginatedMemories = mockEpisodicMemories.slice(offset, offset + limit);

    return HttpResponse.json({
      memories: paginatedMemories,
      total: mockEpisodicMemories.length,
      limit,
      offset,
      has_more: offset + limit < mockEpisodicMemories.length,
      timestamp: new Date().toISOString(),
    });
  }),

  // GET /memory/episodic/:id - Get single episodic memory
  http.get(`${baseUrl}/memory/episodic/:id`, async ({ params }) => {
    await simulateDelay();

    const { id } = params;
    const memory = createEpisodicMemoryResponse(id as string);

    if (!memory) {
      return HttpResponse.json(
        { detail: `Episodic memory with id '${id}' not found` },
        { status: 404 }
      );
    }

    return HttpResponse.json(memory);
  }),

  // POST /memory/episodic - Create episodic memory
  http.post(`${baseUrl}/memory/episodic`, async ({ request }) => {
    await simulateDelay();

    const body = (await request.json()) as {
      content: string;
      event_type: string;
      session_id?: string;
      agent_name?: string;
      brand?: string;
      region?: string;
    };

    return HttpResponse.json(
      createNewEpisodicMemoryResponse(
        body.content,
        body.event_type,
        body.session_id,
        body.agent_name,
        body.brand,
        body.region
      ),
      { status: 201 }
    );
  }),

  // POST /memory/procedural/feedback - Record procedural feedback
  http.post(`${baseUrl}/memory/procedural/feedback`, async ({ request }) => {
    await simulateDelay();

    const body = (await request.json()) as {
      procedure_id: string;
      outcome: 'success' | 'partial' | 'failure';
      score: number;
    };

    return HttpResponse.json(
      createProceduralFeedbackResponse(body.procedure_id, body.outcome, body.score)
    );
  }),

  // POST /memory/semantic/paths - Query semantic paths
  http.post(`${baseUrl}/memory/semantic/paths`, async () => {
    await simulateDelay();
    return HttpResponse.json(createSemanticPathResponse());
  }),

  // GET /memory/stats - Memory statistics
  http.get(`${baseUrl}/memory/stats`, async () => {
    await simulateDelay();
    return HttpResponse.json(createMemoryStatsResponse());
  }),
];

/**
 * Health Check Handlers
 */
const healthHandlers = [
  // GET /health - Overall health
  http.get(`${baseUrl}/health`, async () => {
    await simulateDelay();
    return HttpResponse.json({
      status: 'healthy',
      version: '0.1.0',
      services: {
        api: 'healthy',
        database: 'healthy',
        redis: 'healthy',
        falkordb: 'healthy',
      },
      timestamp: new Date().toISOString(),
    });
  }),

  // GET /healthz - Kubernetes liveness
  http.get(`${baseUrl}/healthz`, async () => {
    return HttpResponse.json({ status: 'ok' });
  }),

  // GET /ready - Kubernetes readiness
  http.get(`${baseUrl}/ready`, async () => {
    return HttpResponse.json({ status: 'ready' });
  }),
];

/**
 * All handlers combined
 */
export const handlers = [...graphHandlers, ...memoryHandlers, ...healthHandlers];
