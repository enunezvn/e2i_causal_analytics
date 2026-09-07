/**
 * Agents API Client
 * =================
 *
 * Thin, validated readers for the agent-orchestration endpoints behind the
 * Agent Orchestration dashboard:
 *
 *   - `GET /agents/status`          — the live agent roster (id/name/tier/status)
 *   - `GET /agents/activity`        — recent agent actions (audit_chain_entries;
 *                                     the automated health poller is excluded
 *                                     server-side)
 *   - `GET /analytics/tier-metrics` — per-tier agent performance
 *
 * Every call goes through `getValidated`, so auth headers, correlation IDs and
 * Zod response validation apply exactly as they did when these fetches were
 * inline `useQuery` bodies in `pages/AgentOrchestration.tsx` (#1941). The Zod
 * schemas stay in `lib/api-schemas.ts` — the canonical wire contract — and are
 * only referenced from here, so no field can be dropped by the move.
 *
 * `getAgentTierMetrics` lives in this module rather than in `api/analytics.ts`
 * because its payload is per-*tier agent* performance consumed only by the
 * agent dashboard; it just happens to be served by the analytics router.
 *
 * @module api/agents
 */

import { getValidated } from '@/lib/api-client';
import {
  AgentStatusResponseSchema,
  AgentActivityResponseSchema,
  TierMetricsResponseSchema,
} from '@/lib/api-schemas';
import type {
  AgentStatusResponseValidated,
  AgentActivityResponseValidated,
  TierMetricsResponseValidated,
} from '@/lib/api-schemas';

// =============================================================================
// API FUNCTIONS
// =============================================================================

/**
 * Fetch the live agent roster.
 *
 * @returns Validated agent status (agents array plus `total` / `timestamp`)
 */
export async function getAgentStatus(): Promise<AgentStatusResponseValidated> {
  return getValidated(AgentStatusResponseSchema, '/agents/status');
}

/**
 * Fetch the recent agent activity feed (newest first).
 *
 * An empty `activities` list is an honest "no recent activity" — the endpoint
 * never fabricates rows.
 *
 * @param hours - Look-back window in hours (default 24)
 * @param limit - Maximum rows to return (default 50)
 * @returns Validated activity feed
 */
export async function getAgentActivity(
  hours: number = 24,
  limit: number = 50
): Promise<AgentActivityResponseValidated> {
  return getValidated(AgentActivityResponseSchema, '/agents/activity', {
    hours,
    limit,
  });
}

/**
 * Fetch per-tier agent performance (avg response time, tasks completed).
 *
 * `avg_response_time_ms` / `success_rate` are null when unmeasured — the page
 * renders an em dash rather than a fabricated value.
 *
 * @param hours - Look-back window in hours (default 24)
 * @returns Validated per-tier metrics
 */
export async function getAgentTierMetrics(
  hours: number = 24
): Promise<TierMetricsResponseValidated> {
  return getValidated(TierMetricsResponseSchema, '/analytics/tier-metrics', {
    hours,
  });
}
