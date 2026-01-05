/**
 * Memory Architecture Page
 * ========================
 *
 * Visualizes the E2I tri-memory cognitive architecture:
 * - Working Memory (Redis) - Short-term context
 * - Episodic Memory (Supabase) - Historical interactions
 * - Semantic Memory (FalkorDB) - Knowledge graph
 * - Procedural Memory - Learned procedures
 *
 * @module pages/MemoryArchitecture
 */

import { useState } from 'react';
import { Brain, Database, Network, Cog, Clock, Activity, RefreshCw, AlertCircle } from 'lucide-react';
import { useMemoryStats, useEpisodicMemories } from '@/hooks/api/use-memory';
import { QueryProcessingFlow } from '@/components/visualizations/QueryProcessingFlow';
import type { EpisodicMemoryResponse } from '@/types/memory';

// =============================================================================
// TYPES
// =============================================================================

interface MemoryCardProps {
  title: string;
  icon: React.ReactNode;
  status: 'healthy' | 'degraded' | 'error' | 'unknown';
  children: React.ReactNode;
  className?: string;
}

interface StatItemProps {
  label: string;
  value: string | number;
  subtext?: string;
}

// =============================================================================
// COMPONENTS
// =============================================================================

function StatusBadge({ status }: { status: 'healthy' | 'degraded' | 'error' | 'unknown' }) {
  const styles = {
    healthy: 'bg-green-100 text-green-800 dark:bg-green-900/30 dark:text-green-400',
    degraded: 'bg-yellow-100 text-yellow-800 dark:bg-yellow-900/30 dark:text-yellow-400',
    error: 'bg-red-100 text-red-800 dark:bg-red-900/30 dark:text-red-400',
    unknown: 'bg-gray-100 text-gray-800 dark:bg-gray-900/30 dark:text-gray-400',
  };

  return (
    <span className={`px-2 py-1 rounded-full text-xs font-medium ${styles[status]}`}>
      {status.charAt(0).toUpperCase() + status.slice(1)}
    </span>
  );
}

function StatItem({ label, value, subtext }: StatItemProps) {
  return (
    <div className="flex flex-col">
      <span className="text-sm text-[var(--color-text-secondary)]">{label}</span>
      <span className="text-2xl font-bold text-[var(--color-text-primary)]">{value}</span>
      {subtext && <span className="text-xs text-[var(--color-text-tertiary)]">{subtext}</span>}
    </div>
  );
}

function MemoryCard({ title, icon, status, children, className = '' }: MemoryCardProps) {
  return (
    <div className={`bg-[var(--color-card)] rounded-lg border border-[var(--color-border)] p-6 ${className}`}>
      <div className="flex items-center justify-between mb-4">
        <div className="flex items-center gap-3">
          <div className="p-2 rounded-lg bg-[var(--color-primary)]/10 text-[var(--color-primary)]">
            {icon}
          </div>
          <h3 className="text-lg font-semibold text-[var(--color-text-primary)]">{title}</h3>
        </div>
        <StatusBadge status={status} />
      </div>
      {children}
    </div>
  );
}

function ArchitectureDiagram() {
  return (
    <div className="bg-[var(--color-card)] rounded-lg border border-[var(--color-border)] p-6 mb-6">
      <h3 className="text-lg font-semibold text-[var(--color-text-primary)] mb-4">
        Cognitive Memory Architecture
      </h3>
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        {/* Working Memory */}
        <div className="flex flex-col items-center p-4 bg-blue-50 dark:bg-blue-900/20 rounded-lg border border-blue-200 dark:border-blue-800">
          <Clock className="h-8 w-8 text-blue-600 dark:text-blue-400 mb-2" />
          <span className="font-medium text-blue-900 dark:text-blue-300">Working Memory</span>
          <span className="text-xs text-blue-700 dark:text-blue-400 mt-1">Redis Cache</span>
          <span className="inline-flex items-center px-2 py-0.5 rounded-full text-xs font-medium bg-blue-100 dark:bg-blue-800 text-blue-800 dark:text-blue-200 mt-2">
            &lt;50ms latency
          </span>
          <span className="text-xs text-blue-600 dark:text-blue-500 mt-2 text-center">
            Short-term context, active session state
          </span>
        </div>

        {/* Episodic Memory */}
        <div className="flex flex-col items-center p-4 bg-purple-50 dark:bg-purple-900/20 rounded-lg border border-purple-200 dark:border-purple-800">
          <Database className="h-8 w-8 text-purple-600 dark:text-purple-400 mb-2" />
          <span className="font-medium text-purple-900 dark:text-purple-300">Episodic Memory</span>
          <span className="text-xs text-purple-700 dark:text-purple-400 mt-1">Supabase + pgvector</span>
          <span className="inline-flex items-center px-2 py-0.5 rounded-full text-xs font-medium bg-purple-100 dark:bg-purple-800 text-purple-800 dark:text-purple-200 mt-2">
            &lt;200ms latency
          </span>
          <span className="text-xs text-purple-600 dark:text-purple-500 mt-2 text-center">
            Historical interactions, past experiences
          </span>
        </div>

        {/* Semantic Memory */}
        <div className="flex flex-col items-center p-4 bg-emerald-50 dark:bg-emerald-900/20 rounded-lg border border-emerald-200 dark:border-emerald-800">
          <Network className="h-8 w-8 text-emerald-600 dark:text-emerald-400 mb-2" />
          <span className="font-medium text-emerald-900 dark:text-emerald-300">Semantic Memory</span>
          <span className="text-xs text-emerald-700 dark:text-emerald-400 mt-1">FalkorDB Graph</span>
          <span className="inline-flex items-center px-2 py-0.5 rounded-full text-xs font-medium bg-emerald-100 dark:bg-emerald-800 text-emerald-800 dark:text-emerald-200 mt-2">
            &lt;500ms latency
          </span>
          <span className="text-xs text-emerald-600 dark:text-emerald-500 mt-2 text-center">
            Knowledge graph, entity relationships
          </span>
        </div>
      </div>

      {/* Procedural Memory at bottom */}
      <div className="mt-4 flex justify-center">
        <div className="flex flex-col items-center p-4 bg-orange-50 dark:bg-orange-900/20 rounded-lg border border-orange-200 dark:border-orange-800 w-full md:w-1/2">
          <Cog className="h-8 w-8 text-orange-600 dark:text-orange-400 mb-2" />
          <span className="font-medium text-orange-900 dark:text-orange-300">Procedural Memory</span>
          <span className="text-xs text-orange-700 dark:text-orange-400 mt-1">Supabase + pgvector</span>
          <span className="inline-flex items-center px-2 py-0.5 rounded-full text-xs font-medium bg-orange-100 dark:bg-orange-800 text-orange-800 dark:text-orange-200 mt-2">
            &lt;200ms latency
          </span>
          <span className="text-xs text-orange-600 dark:text-orange-500 mt-2 text-center">
            Optimized procedures, behavioral patterns
          </span>
        </div>
      </div>

      {/* Connection arrows description */}
      <div className="mt-6 text-center text-sm text-[var(--color-text-secondary)]">
        Memory systems work together: Working Memory coordinates active context,
        Episodic Memory recalls past interactions, Semantic Memory provides domain knowledge,
        and Procedural Memory applies learned optimizations.
      </div>
    </div>
  );
}

function EpisodicMemoryList({ memories }: { memories: EpisodicMemoryResponse[] }) {
  if (memories.length === 0) {
    return (
      <div className="text-center py-4 text-[var(--color-text-secondary)]">
        No recent episodic memories
      </div>
    );
  }

  return (
    <div className="space-y-3 mt-4">
      <h4 className="text-sm font-medium text-[var(--color-text-secondary)]">Recent Memories</h4>
      {memories.slice(0, 5).map((memory) => (
        <div
          key={memory.id}
          className="p-3 bg-[var(--color-background)] rounded border border-[var(--color-border)]"
        >
          <div className="flex items-center justify-between mb-1">
            <span className="text-xs font-medium text-[var(--color-primary)]">{memory.event_type}</span>
            <span className="text-xs text-[var(--color-text-tertiary)]">
              {new Date(memory.created_at).toLocaleString()}
            </span>
          </div>
          <p className="text-sm text-[var(--color-text-primary)] line-clamp-2">{memory.content}</p>
          {memory.agent_name && (
            <div className="mt-2 flex items-center gap-2">
              <span className="text-xs text-[var(--color-text-secondary)]">Agent:</span>
              <span className="text-xs text-[var(--color-text-primary)]">{memory.agent_name}</span>
            </div>
          )}
        </div>
      ))}
    </div>
  );
}

// =============================================================================
// MAIN PAGE
// =============================================================================

export default function MemoryArchitecture() {
  const [_refreshKey, setRefreshKey] = useState(0);

  const { data: statsData, isLoading: statsLoading, error: statsError } = useMemoryStats();
  const { data: episodicData, isLoading: episodicLoading } = useEpisodicMemories({ limit: 5 });

  const stats = statsData;
  const episodicMemories = episodicData || [];

  const handleRefresh = () => {
    setRefreshKey((prev) => prev + 1);
  };

  // Determine overall system status
  const getOverallStatus = (): 'healthy' | 'degraded' | 'error' | 'unknown' => {
    if (statsError) return 'error';
    if (statsLoading) return 'unknown';
    if (!stats) return 'unknown';
    return 'healthy';
  };

  // Individual memory status (simulated based on data availability)
  const getMemoryStatus = (hasData: boolean): 'healthy' | 'degraded' | 'error' | 'unknown' => {
    if (statsLoading) return 'unknown';
    if (!stats) return 'unknown';
    return hasData ? 'healthy' : 'degraded';
  };

  return (
    <div className="p-6 space-y-6">
      {/* Header */}
      <div className="flex flex-col md:flex-row md:items-center md:justify-between gap-4">
        <div>
          <h1 className="text-2xl font-bold text-[var(--color-text-primary)] flex items-center gap-3">
            <Brain className="h-7 w-7 text-[var(--color-primary)]" />
            Memory Architecture
          </h1>
          <p className="text-[var(--color-text-secondary)] mt-1">
            E2I Tri-Memory Cognitive System - Working, Episodic, Semantic & Procedural Memory
          </p>
        </div>
        <div className="flex items-center gap-3">
          <StatusBadge status={getOverallStatus()} />
          <button
            onClick={handleRefresh}
            className="flex items-center gap-2 px-4 py-2 bg-[var(--color-primary)] text-white rounded-lg hover:bg-[var(--color-primary-hover)] transition-colors"
          >
            <RefreshCw className="h-4 w-4" />
            Refresh
          </button>
        </div>
      </div>

      {/* Error state */}
      {statsError && (
        <div className="bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 rounded-lg p-4 flex items-center gap-3">
          <AlertCircle className="h-5 w-5 text-red-600 dark:text-red-400" />
          <div>
            <p className="font-medium text-red-800 dark:text-red-300">Failed to load memory statistics</p>
            <p className="text-sm text-red-600 dark:text-red-400">Please check the API connection and try again.</p>
          </div>
        </div>
      )}

      {/* Architecture Diagram */}
      <ArchitectureDiagram />

      {/* Query Processing Flow */}
      <QueryProcessingFlow animate={true} animationSpeed={1000} />

      {/* Stats Cards Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        {/* Working Memory */}
        <MemoryCard
          title="Working Memory"
          icon={<Clock className="h-5 w-5" />}
          status={getMemoryStatus(true)}
        >
          <div className="space-y-4">
            <StatItem
              label="Backend"
              value="Redis"
              subtext="In-memory cache"
            />
            <StatItem
              label="Target Latency"
              value="<50ms"
              subtext="P95 response time"
            />
            <StatItem
              label="TTL"
              value="24h"
              subtext="Session context expiry"
            />
            <div className="pt-2 border-t border-[var(--color-border)]">
              <p className="text-xs text-[var(--color-text-secondary)]">
                Stores active conversation context, current filters, and session state.
              </p>
            </div>
          </div>
        </MemoryCard>

        {/* Episodic Memory */}
        <MemoryCard
          title="Episodic Memory"
          icon={<Database className="h-5 w-5" />}
          status={getMemoryStatus(!!stats?.episodic)}
        >
          <div className="space-y-4">
            <StatItem
              label="Backend"
              value="Supabase"
              subtext="pgvector embeddings"
            />
            <StatItem
              label="Target Latency"
              value="<200ms"
              subtext="P95 response time"
            />
            <StatItem
              label="Total Memories"
              value={statsLoading ? '...' : (stats?.episodic?.total_memories || 0).toLocaleString()}
              subtext="Historical interactions"
            />
            <div className="pt-2 border-t border-[var(--color-border)]">
              <p className="text-xs text-[var(--color-text-secondary)]">
                Past queries, responses, and user interactions stored in Supabase.
              </p>
            </div>
          </div>
        </MemoryCard>

        {/* Semantic Memory */}
        <MemoryCard
          title="Semantic Memory"
          icon={<Network className="h-5 w-5" />}
          status={getMemoryStatus(!!stats?.semantic)}
        >
          <div className="space-y-4">
            <StatItem
              label="Backend"
              value="FalkorDB"
              subtext="Graph database"
            />
            <StatItem
              label="Target Latency"
              value="<500ms"
              subtext="P95 response time"
            />
            <StatItem
              label="Entities"
              value={statsLoading ? '...' : (stats?.semantic?.total_entities || 0).toLocaleString()}
              subtext="Knowledge nodes"
            />
            <div className="pt-2 border-t border-[var(--color-border)]">
              <p className="text-xs text-[var(--color-text-secondary)]">
                Domain knowledge graph powered by FalkorDB for causal reasoning.
              </p>
            </div>
          </div>
        </MemoryCard>

        {/* Procedural Memory */}
        <MemoryCard
          title="Procedural Memory"
          icon={<Cog className="h-5 w-5" />}
          status={getMemoryStatus(!!stats?.procedural)}
        >
          <div className="space-y-4">
            <StatItem
              label="Backend"
              value="Supabase"
              subtext="pgvector embeddings"
            />
            <StatItem
              label="Target Latency"
              value="<200ms"
              subtext="P95 response time"
            />
            <StatItem
              label="Procedures"
              value={statsLoading ? '...' : (stats?.procedural?.total_procedures || 0).toLocaleString()}
              subtext="Learned patterns"
            />
            <div className="pt-2 border-t border-[var(--color-border)]">
              <p className="text-xs text-[var(--color-text-secondary)]">
                Optimized prompts, tool sequences, and behavioral patterns.
              </p>
            </div>
          </div>
        </MemoryCard>
      </div>

      {/* Recent Episodic Memories Section */}
      <div className="bg-[var(--color-card)] rounded-lg border border-[var(--color-border)] p-6">
        <div className="flex items-center justify-between mb-4">
          <div className="flex items-center gap-3">
            <Activity className="h-5 w-5 text-[var(--color-primary)]" />
            <h3 className="text-lg font-semibold text-[var(--color-text-primary)]">
              Recent Episodic Memories
            </h3>
          </div>
          <span className="text-sm text-[var(--color-text-secondary)]">
            Last 5 interactions
          </span>
        </div>

        {episodicLoading ? (
          <div className="flex items-center justify-center py-8">
            <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-[var(--color-primary)]" />
          </div>
        ) : (
          <EpisodicMemoryList memories={episodicMemories} />
        )}
      </div>

      {/* Memory System Info */}
      <div className="bg-[var(--color-card)] rounded-lg border border-[var(--color-border)] p-6">
        <h3 className="text-lg font-semibold text-[var(--color-text-primary)] mb-4">
          About the Memory System
        </h3>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6 text-sm text-[var(--color-text-secondary)]">
          <div>
            <h4 className="font-medium text-[var(--color-text-primary)] mb-2">Memory Integration</h4>
            <p>
              The E2I cognitive architecture uses four complementary memory systems that work together
              to provide contextual, historical, and domain-aware responses. Each memory type serves
              a specific purpose in the agent workflow.
            </p>
          </div>
          <div>
            <h4 className="font-medium text-[var(--color-text-primary)] mb-2">Retrieval Methods</h4>
            <ul className="list-disc list-inside space-y-1">
              <li><strong>Vector Search</strong> - Semantic similarity matching</li>
              <li><strong>Keyword Search</strong> - Exact term matching</li>
              <li><strong>Hybrid Search</strong> - Combined vector + keyword</li>
              <li><strong>Graph Traversal</strong> - Relationship-based retrieval</li>
            </ul>
          </div>
        </div>
        {stats?.last_updated && (
          <div className="mt-4 pt-4 border-t border-[var(--color-border)]">
            <p className="text-xs text-[var(--color-text-tertiary)]">
              Last updated: {new Date(stats.last_updated).toLocaleString()}
            </p>
          </div>
        )}
      </div>
    </div>
  );
}
