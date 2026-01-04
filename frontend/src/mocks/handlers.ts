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
 * Monitoring API Handlers
 */
const monitoringHandlers = [
  // POST /monitoring/drift/detect - Trigger drift detection
  http.post(`${baseUrl}/monitoring/drift/detect`, async ({ request }) => {
    await simulateDelay();
    const body = (await request.json()) as { model_id: string };
    return HttpResponse.json({
      task_id: `drift_${Date.now()}`,
      model_id: body.model_id,
      status: 'completed',
      overall_drift_score: 0.15,
      features_checked: 12,
      features_with_drift: ['days_since_last_visit', 'total_prescriptions'],
      results: [
        {
          feature: 'days_since_last_visit',
          drift_type: 'data',
          test_statistic: 2.45,
          p_value: 0.014,
          drift_detected: true,
          severity: 'medium',
          baseline_period: '2024-01-01 to 2024-06-30',
          current_period: '2024-07-01 to 2024-12-31',
        },
      ],
      drift_summary: 'Minor drift detected in 2 features',
      recommended_actions: ['Monitor closely', 'Consider feature engineering'],
      detection_latency_ms: 245,
      timestamp: new Date().toISOString(),
    });
  }),

  // GET /monitoring/drift/status/:taskId - Get drift detection status
  http.get(`${baseUrl}/monitoring/drift/status/:taskId`, async ({ params }) => {
    await simulateDelay();
    const { taskId } = params;
    return HttpResponse.json({
      task_id: taskId,
      status: 'completed',
      ready: true,
      result: {
        model_id: 'propensity_v2.1.0',
        overall_drift_score: 0.15,
        drift_summary: 'Minor drift detected',
      },
    });
  }),

  // GET /monitoring/drift/latest/:modelId - Get latest drift status
  http.get(`${baseUrl}/monitoring/drift/latest/:modelId`, async ({ params }) => {
    await simulateDelay();
    const { modelId } = params;
    return HttpResponse.json({
      task_id: `drift_${Date.now()}`,
      model_id: modelId,
      status: 'completed',
      overall_drift_score: 0.15,
      features_checked: 12,
      features_with_drift: ['days_since_last_visit', 'total_prescriptions'],
      results: [
        {
          feature: 'days_since_last_visit',
          drift_type: 'data',
          test_statistic: 2.45,
          p_value: 0.014,
          drift_detected: true,
          severity: 'medium',
          baseline_period: '2024-01-01 to 2024-06-30',
          current_period: '2024-07-01 to 2024-12-31',
        },
      ],
      drift_summary: 'Minor drift detected in 2 features',
      recommended_actions: ['Monitor closely', 'Consider feature engineering'],
      detection_latency_ms: 245,
      timestamp: new Date().toISOString(),
    });
  }),

  // GET /monitoring/drift/history/:modelId - Get drift history
  http.get(`${baseUrl}/monitoring/drift/history/:modelId`, async ({ params }) => {
    await simulateDelay();
    const { modelId } = params;
    return HttpResponse.json({
      model_id: modelId,
      total_records: 5,
      records: [
        {
          id: 'drift_001',
          model_version: `${modelId}_v2.1.0`,
          feature_name: 'days_since_last_visit',
          drift_type: 'data',
          drift_score: 0.15,
          severity: 'medium',
          detected_at: new Date(Date.now() - 86400000).toISOString(),
          baseline_start: '2024-01-01',
          baseline_end: '2024-06-30',
          current_start: '2024-07-01',
          current_end: '2024-12-31',
        },
      ],
    });
  }),

  // GET /monitoring/alerts - List alerts
  http.get(`${baseUrl}/monitoring/alerts`, async () => {
    await simulateDelay();
    return HttpResponse.json({
      total_count: 3,
      active_count: 2,
      alerts: [
        {
          id: 'alert_001',
          model_version: 'propensity_v2.1.0',
          alert_type: 'drift',
          severity: 'high',
          title: 'High drift detected in propensity model',
          description: 'Feature days_since_last_visit shows significant drift',
          status: 'active',
          triggered_at: new Date(Date.now() - 3600000).toISOString(),
        },
        {
          id: 'alert_002',
          model_version: 'churn_v1.2.0',
          alert_type: 'performance',
          severity: 'medium',
          title: 'Performance degradation in churn model',
          description: 'Accuracy dropped by 3% over the last week',
          status: 'acknowledged',
          triggered_at: new Date(Date.now() - 7200000).toISOString(),
          acknowledged_at: new Date(Date.now() - 3600000).toISOString(),
          acknowledged_by: 'user_123',
        },
      ],
    });
  }),

  // GET /monitoring/alerts/:alertId - Get specific alert
  http.get(`${baseUrl}/monitoring/alerts/:alertId`, async ({ params }) => {
    await simulateDelay();
    const { alertId } = params;
    return HttpResponse.json({
      id: alertId,
      model_version: 'propensity_v2.1.0',
      alert_type: 'drift',
      severity: 'high',
      title: 'High drift detected in propensity model',
      description: 'Feature days_since_last_visit shows significant drift',
      status: 'active',
      triggered_at: new Date(Date.now() - 3600000).toISOString(),
    });
  }),

  // POST /monitoring/alerts/:alertId/action - Update alert
  http.post(`${baseUrl}/monitoring/alerts/:alertId/action`, async ({ params }) => {
    await simulateDelay();
    const { alertId } = params;
    return HttpResponse.json({
      id: alertId,
      model_version: 'propensity_v2.1.0',
      alert_type: 'drift',
      severity: 'high',
      title: 'High drift detected in propensity model',
      description: 'Feature days_since_last_visit shows significant drift',
      status: 'acknowledged',
      triggered_at: new Date(Date.now() - 3600000).toISOString(),
      acknowledged_at: new Date().toISOString(),
      acknowledged_by: 'user_123',
    });
  }),

  // GET /monitoring/health/:modelId - Get model health
  http.get(`${baseUrl}/monitoring/health/:modelId`, async ({ params }) => {
    await simulateDelay();
    const { modelId } = params;
    return HttpResponse.json({
      model_id: modelId,
      overall_health: 'healthy',
      last_check: new Date().toISOString(),
      drift_score: 0.12,
      active_alerts: 0,
      last_retrained: new Date(Date.now() - 30 * 86400000).toISOString(),
      performance_trend: 'stable',
      recommendations: [],
    });
  }),

  // GET /monitoring/runs - List monitoring runs
  http.get(`${baseUrl}/monitoring/runs`, async () => {
    await simulateDelay();
    return HttpResponse.json({
      total_runs: 10,
      runs: [
        {
          id: 'run_001',
          model_version: 'propensity_v2.1.0',
          run_type: 'scheduled',
          started_at: new Date(Date.now() - 3600000).toISOString(),
          completed_at: new Date(Date.now() - 3500000).toISOString(),
          features_checked: 15,
          drift_detected_count: 2,
          alerts_generated: 1,
          duration_ms: 100000,
        },
      ],
    });
  }),

  // POST /monitoring/performance/record - Record performance
  http.post(`${baseUrl}/monitoring/performance/record`, async ({ request }) => {
    await simulateDelay();
    const body = (await request.json()) as { model_id: string };
    return HttpResponse.json({
      model_id: body.model_id,
      recorded: true,
      metrics: {
        accuracy: 0.89,
        precision: 0.87,
        recall: 0.91,
        f1_score: 0.89,
        auc: 0.94,
      },
      recorded_at: new Date().toISOString(),
    });
  }),

  // GET /monitoring/performance/:modelId/trend - Get performance trend
  http.get(`${baseUrl}/monitoring/performance/:modelId/trend`, async ({ params }) => {
    await simulateDelay();
    const { modelId } = params;
    return HttpResponse.json({
      model_id: modelId,
      metric_name: 'accuracy',
      current_value: 0.89,
      baseline_value: 0.91,
      change_percent: -2.2,
      trend: 'stable',
      is_significant: false,
      alert_threshold_breached: false,
      history: [
        { metric_name: 'accuracy', metric_value: 0.91, recorded_at: new Date(Date.now() - 7 * 86400000).toISOString() },
        { metric_name: 'accuracy', metric_value: 0.90, recorded_at: new Date(Date.now() - 5 * 86400000).toISOString() },
        { metric_name: 'accuracy', metric_value: 0.89, recorded_at: new Date().toISOString() },
      ],
    });
  }),

  // GET /monitoring/performance/:modelId/alerts - Get performance alerts
  http.get(`${baseUrl}/monitoring/performance/:modelId/alerts`, async ({ params }) => {
    await simulateDelay();
    const { modelId } = params;
    return HttpResponse.json({
      model_id: modelId,
      alert_count: 1,
      alerts: [
        {
          metric: 'accuracy',
          threshold: 0.90,
          current: 0.87,
          severity: 'medium',
          message: 'Accuracy below threshold',
        },
      ],
    });
  }),

  // GET /monitoring/performance/:modelId/compare/:otherModelId - Compare models
  http.get(`${baseUrl}/monitoring/performance/:modelId/compare/:otherModelId`, async ({ params }) => {
    await simulateDelay();
    const { modelId, otherModelId } = params;
    return HttpResponse.json({
      model_a: modelId,
      model_b: otherModelId,
      metric_name: 'accuracy',
      model_a_value: 0.89,
      model_b_value: 0.87,
      difference: 0.02,
      better_model: modelId,
      statistical_significance: true,
      p_value: 0.03,
    });
  }),

  // POST /monitoring/sweep/production - Trigger production sweep
  http.post(`${baseUrl}/monitoring/sweep/production`, async () => {
    await simulateDelay();
    return HttpResponse.json({
      task_id: `sweep_${Date.now()}`,
      status: 'queued',
      models_queued: 5,
      started_at: new Date().toISOString(),
    });
  }),

  // POST /monitoring/retraining/evaluate/:modelId - Evaluate retraining need
  http.post(`${baseUrl}/monitoring/retraining/evaluate/:modelId`, async ({ params }) => {
    await simulateDelay();
    const { modelId } = params;
    return HttpResponse.json({
      model_id: modelId,
      should_retrain: true,
      confidence: 0.85,
      reasons: ['High drift score', 'Performance degradation'],
      recommended_action: 'Schedule retraining',
      evaluated_at: new Date().toISOString(),
    });
  }),

  // POST /monitoring/retraining/trigger/:modelId - Trigger retraining
  http.post(`${baseUrl}/monitoring/retraining/trigger/:modelId`, async ({ params }) => {
    await simulateDelay();
    const { modelId } = params;
    return HttpResponse.json({
      job_id: `retrain_${Date.now()}`,
      model_id: modelId,
      status: 'queued',
      created_at: new Date().toISOString(),
      triggered_by: 'ui_user',
    });
  }),

  // GET /monitoring/retraining/status/:jobId - Get retraining status
  http.get(`${baseUrl}/monitoring/retraining/status/:jobId`, async ({ params }) => {
    await simulateDelay();
    const { jobId } = params;
    return HttpResponse.json({
      job_id: jobId,
      model_id: 'propensity_v2.1.0',
      status: 'running',
      progress: 45,
      created_at: new Date(Date.now() - 600000).toISOString(),
      started_at: new Date(Date.now() - 300000).toISOString(),
    });
  }),

  // POST /monitoring/retraining/:jobId/complete - Complete retraining
  http.post(`${baseUrl}/monitoring/retraining/:jobId/complete`, async ({ params }) => {
    await simulateDelay();
    const { jobId } = params;
    return HttpResponse.json({
      job_id: jobId,
      model_id: 'propensity_v2.1.0',
      status: 'completed',
      completed_at: new Date().toISOString(),
      performance_after: 0.92,
      success: true,
    });
  }),

  // POST /monitoring/retraining/:jobId/rollback - Rollback retraining
  http.post(`${baseUrl}/monitoring/retraining/:jobId/rollback`, async ({ params }) => {
    await simulateDelay();
    const { jobId } = params;
    return HttpResponse.json({
      job_id: jobId,
      model_id: 'propensity_v2.1.0',
      status: 'rolled_back',
      rolled_back_at: new Date().toISOString(),
      reason: 'Performance degradation',
    });
  }),

  // POST /monitoring/retraining/sweep - Trigger retraining sweep
  http.post(`${baseUrl}/monitoring/retraining/sweep`, async () => {
    await simulateDelay();
    return HttpResponse.json({
      task_id: `retrain_sweep_${Date.now()}`,
      status: 'queued',
      models_queued: 3,
      started_at: new Date().toISOString(),
    });
  }),
];

/**
 * KPI API Handlers
 */
const kpiHandlers = [
  // GET /kpis - List KPIs
  http.get(`${baseUrl}/kpis`, async () => {
    await simulateDelay();
    return HttpResponse.json({
      kpis: [
        {
          id: 'WS1-DQ-001',
          name: 'Data Completeness',
          definition: 'Percentage of complete records in the dataset',
          formula: 'COUNT(complete) / COUNT(*) * 100',
          calculation_type: 'direct',
          workstream: 'ws1_data_quality',
          tables: ['hcp_data'],
          columns: ['*'],
          threshold: { target: 95, warning: 90, critical: 80 },
          unit: '%',
          frequency: 'daily',
          primary_causal_library: 'none',
        },
        {
          id: 'WS1-MP-001',
          name: 'Model Accuracy',
          definition: 'Overall model prediction accuracy',
          formula: 'TP + TN / Total',
          calculation_type: 'derived',
          workstream: 'ws1_model_performance',
          tables: ['predictions'],
          columns: ['prediction', 'actual'],
          threshold: { target: 90, warning: 85, critical: 75 },
          unit: '%',
          frequency: 'daily',
          primary_causal_library: 'none',
        },
        {
          id: 'WS2-TR-001',
          name: 'Trigger Rate',
          definition: 'Rate of marketing triggers generated',
          formula: 'COUNT(triggers) / COUNT(visits) * 100',
          calculation_type: 'derived',
          workstream: 'ws2_triggers',
          tables: ['triggers', 'visits'],
          columns: ['trigger_id', 'visit_id'],
          threshold: { target: 25, warning: 20, critical: 10 },
          unit: '%',
          frequency: 'weekly',
          primary_causal_library: 'dowhy',
        },
      ],
      total: 3,
      workstream: null,
      causal_library: null,
    });
  }),

  // GET /kpis/workstreams - List workstreams
  http.get(`${baseUrl}/kpis/workstreams`, async () => {
    await simulateDelay();
    return HttpResponse.json({
      workstreams: [
        { id: 'ws1_data_quality', name: 'Data Quality', kpi_count: 8, description: 'Data quality metrics' },
        { id: 'ws1_model_performance', name: 'Model Performance', kpi_count: 12, description: 'ML model performance metrics' },
        { id: 'ws2_triggers', name: 'Triggers', kpi_count: 6, description: 'Marketing trigger metrics' },
        { id: 'ws3_business', name: 'Business', kpi_count: 15, description: 'Business outcome metrics' },
        { id: 'causal_metrics', name: 'Causal Metrics', kpi_count: 5, description: 'Causal inference metrics' },
      ],
      total: 5,
    });
  }),

  // GET /kpis/health - KPI system health
  http.get(`${baseUrl}/kpis/health`, async () => {
    await simulateDelay();
    return HttpResponse.json({
      status: 'healthy',
      registry_loaded: true,
      total_kpis: 46,
      cache_enabled: true,
      cache_size: 120,
      database_connected: true,
      workstreams_available: ['ws1_data_quality', 'ws1_model_performance', 'ws2_triggers', 'ws3_business', 'causal_metrics'],
      last_calculation: new Date(Date.now() - 60000).toISOString(),
    });
  }),

  // GET /kpis/:kpiId - Get KPI value
  http.get(`${baseUrl}/kpis/:kpiId`, async ({ params }) => {
    await simulateDelay();
    const { kpiId } = params;
    return HttpResponse.json({
      kpi_id: kpiId,
      value: 92.5,
      status: 'good',
      calculated_at: new Date().toISOString(),
      cached: true,
      cache_expires_at: new Date(Date.now() + 300000).toISOString(),
      metadata: { sample_size: 1000, calculation_method: 'direct' },
    });
  }),

  // GET /kpis/:kpiId/metadata - Get KPI metadata
  http.get(`${baseUrl}/kpis/:kpiId/metadata`, async ({ params }) => {
    await simulateDelay();
    const { kpiId } = params;
    return HttpResponse.json({
      id: kpiId,
      name: 'Data Completeness',
      definition: 'Percentage of complete records',
      formula: 'COUNT(complete) / COUNT(*) * 100',
      calculation_type: 'direct',
      workstream: 'ws1_data_quality',
      tables: ['hcp_data'],
      columns: ['*'],
      threshold: { target: 95, warning: 90, critical: 80 },
      unit: '%',
      frequency: 'daily',
      primary_causal_library: 'none',
    });
  }),

  // POST /kpis/calculate - Calculate KPI
  http.post(`${baseUrl}/kpis/calculate`, async ({ request }) => {
    await simulateDelay();
    const body = (await request.json()) as { kpi_id: string };
    return HttpResponse.json({
      kpi_id: body.kpi_id,
      value: 93.2,
      status: 'good',
      calculated_at: new Date().toISOString(),
      cached: false,
      metadata: { sample_size: 1250 },
    });
  }),

  // POST /kpis/batch - Batch calculate KPIs
  http.post(`${baseUrl}/kpis/batch`, async ({ request }) => {
    await simulateDelay();
    const body = (await request.json()) as { kpi_ids?: string[]; workstream?: string };
    const kpiIds = body.kpi_ids || ['WS1-DQ-001', 'WS1-DQ-002'];
    return HttpResponse.json({
      workstream: body.workstream,
      results: kpiIds.map((id) => ({
        kpi_id: id,
        value: Math.random() * 30 + 70,
        status: 'good',
        calculated_at: new Date().toISOString(),
        cached: false,
        metadata: {},
      })),
      calculated_at: new Date().toISOString(),
      total_kpis: kpiIds.length,
      successful: kpiIds.length,
      failed: 0,
    });
  }),

  // POST /kpis/invalidate - Invalidate cache
  http.post(`${baseUrl}/kpis/invalidate`, async () => {
    await simulateDelay();
    return HttpResponse.json({
      invalidated_count: 10,
      message: 'Cache invalidated successfully',
    });
  }),
];

/**
 * Predictions API Handlers
 */
const predictionsHandlers = [
  // POST /models/predict/:modelName - Single prediction
  http.post(`${baseUrl}/models/predict/:modelName`, async ({ params }) => {
    await simulateDelay();
    const { modelName } = params;
    return HttpResponse.json({
      model_name: modelName,
      prediction: 0.78,
      confidence: 0.92,
      probabilities: { churned: 0.78, retained: 0.22 },
      feature_importance: { days_since_visit: 0.35, total_rx: 0.28, specialty: 0.15 },
      latency_ms: 45,
      model_version: '2.1.0',
      timestamp: new Date().toISOString(),
    });
  }),

  // POST /models/predict/:modelName/batch - Batch predictions
  http.post(`${baseUrl}/models/predict/:modelName/batch`, async ({ params, request }) => {
    await simulateDelay();
    const { modelName } = params;
    const body = (await request.json()) as { instances: unknown[] };
    return HttpResponse.json({
      model_name: modelName,
      predictions: body.instances.map((_) => ({
        model_name: modelName,
        prediction: Math.random(),
        confidence: 0.85 + Math.random() * 0.1,
        latency_ms: 20 + Math.random() * 30,
        timestamp: new Date().toISOString(),
      })),
      total_count: body.instances.length,
      success_count: body.instances.length,
      failed_count: 0,
      total_latency_ms: 150,
      timestamp: new Date().toISOString(),
    });
  }),

  // GET /models/:modelName/health - Model health
  http.get(`${baseUrl}/models/:modelName/health`, async ({ params }) => {
    await simulateDelay();
    const { modelName } = params;
    return HttpResponse.json({
      model_name: modelName,
      status: 'healthy',
      endpoint: `http://localhost:3000/models/${modelName}`,
      last_check: new Date().toISOString(),
    });
  }),

  // GET /models/:modelName/info - Model info
  http.get(`${baseUrl}/models/:modelName/info`, async ({ params }) => {
    await simulateDelay();
    const { modelName } = params;
    return HttpResponse.json({
      name: modelName,
      version: '2.1.0',
      type: 'classification',
      description: `${modelName} prediction model for HCP targeting`,
      input_schema: {
        hcp_id: 'string',
        territory: 'string',
        specialty: 'string',
        days_since_visit: 'number',
      },
      metrics: { accuracy: 0.89, precision: 0.87, recall: 0.91, f1: 0.89, auc: 0.94 },
      trained_at: new Date(Date.now() - 30 * 86400000).toISOString(),
    });
  }),

  // GET /models/status - All models status
  http.get(`${baseUrl}/models/status`, async ({ request }) => {
    await simulateDelay();
    const url = new URL(request.url);
    const modelsParam = url.searchParams.get('models');
    const filterModels = modelsParam ? modelsParam.split(',') : null;

    const allModels = [
      { model_name: 'churn_model', status: 'healthy', endpoint: 'http://localhost:3000/models/churn', last_check: new Date().toISOString() },
      { model_name: 'conversion_model', status: 'healthy', endpoint: 'http://localhost:3000/models/conversion', last_check: new Date().toISOString() },
      { model_name: 'causal_model', status: 'healthy', endpoint: 'http://localhost:3000/models/causal', last_check: new Date().toISOString() },
    ];

    const models = filterModels
      ? allModels.filter((m) => filterModels.includes(m.model_name))
      : allModels;

    return HttpResponse.json({
      total_models: models.length,
      healthy_count: models.filter((m) => m.status === 'healthy').length,
      unhealthy_count: models.filter((m) => m.status !== 'healthy').length,
      models,
      timestamp: new Date().toISOString(),
    });
  }),
];

/**
 * Digital Twin API Handlers
 */
const digitalTwinHandlers = [
  // POST /digital-twin/simulate - Run simulation
  // Returns SimulationResponse matching types/digital-twin.ts
  http.post(`${baseUrl}/digital-twin/simulate`, async ({ request }) => {
    await delay(1500); // Longer delay to simulate processing

    const body = (await request.json()) as {
      intervention_type: string;
      brand: string;
      sample_size: number;
      duration_days: number;
    };

    // Generate realistic mock simulation results
    const ateEstimate = 25 + Math.random() * 30; // 25-55 effect
    const trxLift = 8 + Math.random() * 7; // 8-15% TRx lift
    const nrxLift = 5 + Math.random() * 5; // 5-10% NRx lift
    const marketShare = 0.5 + Math.random() * 1; // 0.5-1.5% market share change
    const roiEstimate = 2.5 + Math.random() * 2; // 2.5-4.5x ROI

    // Generate projection data points
    const startDate = new Date();
    const projections = Array.from({ length: Math.ceil(body.duration_days / 7) }, (_, i) => {
      const date = new Date(startDate);
      date.setDate(date.getDate() + i * 7);
      const baseValue = 1000 + i * 50;
      const effect = ateEstimate * (1 - Math.exp(-0.3 * (i + 1)));
      return {
        date: date.toISOString().split('T')[0],
        with_intervention: Math.round(baseValue + effect),
        without_intervention: Math.round(baseValue),
        lower_bound: Math.round(baseValue + effect * 0.7),
        upper_bound: Math.round(baseValue + effect * 1.3),
      };
    });

    // Determine recommendation type based on ROI
    const recommendationType = roiEstimate > 3 ? 'deploy' : roiEstimate > 2 ? 'refine' : 'analyze';
    const confidenceLevel = roiEstimate > 3.5 ? 'high' : roiEstimate > 2.5 ? 'medium' : 'low';

    return HttpResponse.json({
      simulation_id: `sim_${Date.now()}`,
      created_at: new Date().toISOString(),
      request: {
        intervention_type: body.intervention_type,
        brand: body.brand,
        sample_size: body.sample_size,
        duration_days: body.duration_days,
      },
      outcomes: {
        ate: {
          lower: ateEstimate * 0.8,
          estimate: ateEstimate,
          upper: ateEstimate * 1.2,
        },
        trx_lift: {
          lower: trxLift * 0.7,
          estimate: trxLift,
          upper: trxLift * 1.3,
        },
        nrx_lift: {
          lower: nrxLift * 0.6,
          estimate: nrxLift,
          upper: nrxLift * 1.4,
        },
        market_share_change: {
          lower: marketShare * 0.5,
          estimate: marketShare,
          upper: marketShare * 1.5,
        },
        roi: {
          lower: roiEstimate * 0.7,
          estimate: roiEstimate,
          upper: roiEstimate * 1.4,
        },
        nnt: Math.round(20 + Math.random() * 30),
        cate_by_segment: {
          'High-Value HCPs': {
            lower: ateEstimate * 1.1,
            estimate: ateEstimate * 1.4,
            upper: ateEstimate * 1.7,
          },
          'Medium-Value HCPs': {
            lower: ateEstimate * 0.7,
            estimate: ateEstimate * 0.9,
            upper: ateEstimate * 1.1,
          },
          'Low-Value HCPs': {
            lower: ateEstimate * 0.3,
            estimate: ateEstimate * 0.5,
            upper: ateEstimate * 0.7,
          },
        },
      },
      fidelity: {
        overall_score: 0.82 + Math.random() * 0.1,
        data_coverage: 0.88 + Math.random() * 0.08,
        calibration: 0.79 + Math.random() * 0.12,
        temporal_alignment: 0.85 + Math.random() * 0.1,
        feature_completeness: 0.90 + Math.random() * 0.08,
        confidence_level: confidenceLevel,
        warnings: [
          'Limited historical data for this segment in Q4',
          'Model trained on 18 months of data',
        ],
      },
      sensitivity: [
        {
          parameter: 'Sample Size',
          base_value: body.sample_size,
          low_value: body.sample_size * 0.5,
          high_value: body.sample_size * 1.5,
          ate_at_low: ateEstimate * 0.85,
          ate_at_high: ateEstimate * 1.1,
          sensitivity_score: 0.65,
        },
        {
          parameter: 'Duration',
          base_value: body.duration_days,
          low_value: body.duration_days * 0.5,
          high_value: body.duration_days * 1.5,
          ate_at_low: ateEstimate * 0.7,
          ate_at_high: ateEstimate * 1.15,
          sensitivity_score: 0.72,
        },
        {
          parameter: 'Engagement Quality',
          base_value: 0.8,
          low_value: 0.5,
          high_value: 1.0,
          ate_at_low: ateEstimate * 0.6,
          ate_at_high: ateEstimate * 1.25,
          sensitivity_score: 0.85,
        },
        {
          parameter: 'Market Conditions',
          base_value: 1.0,
          low_value: 0.8,
          high_value: 1.2,
          ate_at_low: ateEstimate * 0.9,
          ate_at_high: ateEstimate * 1.08,
          sensitivity_score: 0.35,
        },
        {
          parameter: 'Competitor Activity',
          base_value: 0.5,
          low_value: 0.2,
          high_value: 0.8,
          ate_at_low: ateEstimate * 1.15,
          ate_at_high: ateEstimate * 0.85,
          sensitivity_score: 0.55,
        },
      ],
      recommendation: {
        type: recommendationType,
        confidence: confidenceLevel,
        rationale: `Based on the simulation, the ${body.intervention_type.replace(/_/g, ' ')} for ${body.brand} shows a projected ${trxLift.toFixed(1)}% lift in TRx volume with an ROI of ${roiEstimate.toFixed(1)}x. The model has ${confidenceLevel} confidence in these predictions.`,
        evidence: [
          'High-value HCPs show 40% stronger response than average',
          'Effect builds over first 6-8 weeks then stabilizes',
          `ROI projection of ${roiEstimate.toFixed(1)}x based on 85% historical simulation accuracy`,
          `Expected ${Math.round(ateEstimate)} additional TRx per 100 treated HCPs`,
        ],
        risk_factors: [
          'Seasonal variation in Q4 may impact actual results by ±15%',
          'Assumes consistent engagement quality across all territories',
          'Competitor launches could reduce effect by up to 20%',
        ],
        suggested_refinements: recommendationType === 'refine' ? {
          sample_size: Math.round(body.sample_size * 1.2),
          duration_days: Math.round(body.duration_days * 1.25),
          focus_segment: 'High-Value HCPs',
        } : undefined,
        expected_value: Math.round(body.sample_size * 150 * roiEstimate),
      },
      projections,
      execution_time_ms: 1200 + Math.round(Math.random() * 800),
    });
  }),

  // GET /digital-twin/simulations - List simulations
  http.get(`${baseUrl}/digital-twin/simulations`, async () => {
    await simulateDelay();
    return HttpResponse.json({
      simulations: [
        {
          simulation_id: 'sim_001',
          intervention_type: 'hcp_engagement',
          brand: 'Remibrutinib',
          status: 'completed',
          predicted_ate: 0.092,
          created_at: new Date(Date.now() - 86400000).toISOString(),
        },
        {
          simulation_id: 'sim_002',
          intervention_type: 'sampling_campaign',
          brand: 'Fabhalta',
          status: 'completed',
          predicted_ate: 0.078,
          created_at: new Date(Date.now() - 172800000).toISOString(),
        },
      ],
      total: 2,
    });
  }),

  // GET /digital-twin/simulations/:id - Get simulation details
  http.get(`${baseUrl}/digital-twin/simulations/:id`, async ({ params }) => {
    await simulateDelay();
    const { id } = params;
    return HttpResponse.json({
      simulation_id: id,
      intervention_type: 'hcp_engagement',
      brand: 'Remibrutinib',
      sample_size: 1000,
      duration_days: 90,
      status: 'completed',
      predicted_ate: 0.092,
      confidence_interval: { lower: 0.065, upper: 0.119 },
      p_value: 0.002,
      created_at: new Date(Date.now() - 86400000).toISOString(),
      completed_at: new Date(Date.now() - 86300000).toISOString(),
    });
  }),

  // GET /digital-twin/fidelity - Get fidelity metrics
  http.get(`${baseUrl}/digital-twin/fidelity`, async () => {
    await simulateDelay();
    return HttpResponse.json({
      overall_score: 0.87,
      data_coverage: 0.92,
      model_confidence: 0.84,
      historical_accuracy: 0.89,
      last_calibration: new Date(Date.now() - 7 * 86400000).toISOString(),
      next_calibration: new Date(Date.now() + 7 * 86400000).toISOString(),
    });
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
export const handlers = [
  ...graphHandlers,
  ...memoryHandlers,
  ...monitoringHandlers,
  ...kpiHandlers,
  ...predictionsHandlers,
  ...digitalTwinHandlers,
  ...healthHandlers,
];
