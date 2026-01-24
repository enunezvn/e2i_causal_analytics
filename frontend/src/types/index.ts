/**
 * E2I Causal Analytics - API Types
 * =================================
 *
 * Centralized exports for all API types and interfaces.
 *
 * Usage:
 *   import { GraphNode, EntityType, MemorySearchRequest } from '@/types'
 *   import type { ExplainResponse, RAGSearchRequest } from '@/types'
 *
 * @module types
 */

// Common API types
export * from './api';

// Knowledge Graph types
export * from './graph';

// Memory System types
export * from './memory';

// Cognitive Workflow types
export * from './cognitive';

// Model Interpretability types
export * from './explain';

// Hybrid RAG types
export * from './rag';

// Digital Twin types
export * from './digital-twin';

// Gap Analysis types
export * from './gaps';

// Segment Analysis types
export * from './segments';

// Resource Optimization types
export * from './resources';

// Causal Inference types
export * from './causal';

// A/B Testing & Experiments types
export * from './experiments';

// Feedback Learning types
export * from './feedback';

// Health Score types
export * from './health-score';

// Audit Chain types
export * from './audit';

// KPI types (Phase 3 - Type Safety)
export * from './kpi';

// Monitoring types (Phase 3 - Type Safety)
export * from './monitoring';

// Predictions types (Phase 3 - Type Safety)
export * from './predictions';

// Auto-generated types from OpenAPI spec (Phase 4 - Type Generation)
// Run `npm run generate:types` to regenerate from backend API
// Note: Use namespace import to avoid conflicts with hand-crafted types
// import type { Generated } from '@/types' then access Generated.components['schemas']['...']
export * as Generated from './generated';
