/**
 * Agent Progress Renderer Component
 * ==================================
 *
 * Displays real-time progress from the LangGraph agent using CoAgent state sync.
 * Uses useCoAgentStateRender to render progress indicators in the chat.
 *
 * This component should be included in the chat UI to show:
 * - Current processing step (node)
 * - Progress percentage bar
 * - List of progress steps
 * - Tools currently executing
 *
 * @module components/chat/AgentProgressRenderer
 */

import { useCoAgentStateRender } from '@copilotkit/react-core';
import { Loader2, CheckCircle2, AlertCircle, Wrench, Brain } from 'lucide-react';
import { cn } from '@/lib/utils';
import { Progress } from '@/components/ui/progress';
import { Badge } from '@/components/ui/badge';
import type { E2IAgentState } from '@/providers/E2ICopilotProvider';

// =============================================================================
// TYPES
// =============================================================================

export interface AgentProgressRendererProps {
  /** Additional CSS classes */
  className?: string;
}

// =============================================================================
// HELPER FUNCTIONS
// =============================================================================

function getNodeIcon(nodeName: string | undefined) {
  switch (nodeName) {
    case 'chat':
      return Brain;
    case 'tools':
      return Wrench;
    case 'synthesize':
      return CheckCircle2;
    default:
      return Loader2;
  }
}

function getNodeLabel(nodeName: string | undefined) {
  switch (nodeName) {
    case 'chat':
      return 'Processing Query';
    case 'tools':
      return 'Executing Tools';
    case 'synthesize':
      return 'Synthesizing Results';
    case 'idle':
      return 'Ready';
    default:
      return 'Working...';
  }
}

function getStatusColor(status: string) {
  switch (status) {
    case 'processing':
      return 'text-blue-500';
    case 'waiting':
      return 'text-amber-500';
    case 'complete':
      return 'text-emerald-500';
    case 'error':
      return 'text-rose-500';
    default:
      return 'text-slate-500';
  }
}

// =============================================================================
// PROGRESS DISPLAY COMPONENT
// =============================================================================

interface ProgressDisplayProps {
  state: E2IAgentState;
  nodeName: string | undefined;
  status: 'inProgress' | 'complete' | 'error';
}

function ProgressDisplay({ state, nodeName, status }: ProgressDisplayProps) {
  const NodeIcon = getNodeIcon(nodeName);
  const nodeLabel = getNodeLabel(nodeName);
  const statusColor = getStatusColor(state.agent_status);

  // Don't render if complete and no progress to show
  if (status === 'complete' && (!state.progress_steps || state.progress_steps.length === 0)) {
    return null;
  }

  // Don't render if idle with no progress
  if (state.agent_status === 'idle' && state.progress_percent === 0) {
    return null;
  }

  return (
    <div className="rounded-lg border bg-card p-4 space-y-3 animate-in fade-in-0 slide-in-from-bottom-2">
      {/* Header with node info */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          <NodeIcon
            className={cn(
              'h-4 w-4',
              statusColor,
              status === 'inProgress' && state.agent_status === 'processing' && 'animate-spin'
            )}
          />
          <span className="font-medium text-sm">{nodeLabel}</span>
        </div>
        <Badge
          variant="outline"
          className={cn('text-xs', statusColor)}
        >
          {state.progress_percent}%
        </Badge>
      </div>

      {/* Progress bar */}
      <Progress
        value={state.progress_percent}
        className="h-2"
      />

      {/* Progress steps */}
      {state.progress_steps && state.progress_steps.length > 0 && (
        <ul className="space-y-1 text-xs text-muted-foreground">
          {state.progress_steps.map((step, i) => (
            <li
              key={i}
              className={cn(
                'flex items-center gap-1.5',
                i === state.progress_steps.length - 1 && status === 'inProgress'
                  ? 'text-foreground font-medium'
                  : 'text-muted-foreground'
              )}
            >
              {i === state.progress_steps.length - 1 && status === 'inProgress' ? (
                <Loader2 className="h-3 w-3 animate-spin text-blue-500" />
              ) : (
                <CheckCircle2 className="h-3 w-3 text-emerald-500" />
              )}
              {step}
            </li>
          ))}
        </ul>
      )}

      {/* Tools executing */}
      {state.tools_executing && state.tools_executing.length > 0 && (
        <div className="flex flex-wrap gap-1.5">
          {state.tools_executing.map((tool, i) => (
            <Badge key={i} variant="secondary" className="text-[10px] px-1.5 py-0">
              <Wrench className="h-2.5 w-2.5 mr-1" />
              {tool}
            </Badge>
          ))}
        </div>
      )}

      {/* Error message */}
      {state.error_message && (
        <div className="flex items-center gap-2 text-rose-500 text-xs">
          <AlertCircle className="h-3.5 w-3.5" />
          {state.error_message}
        </div>
      )}
    </div>
  );
}

// =============================================================================
// MAIN COMPONENT
// =============================================================================

/**
 * AgentProgressRenderer uses CopilotKit's useCoAgentStateRender hook to display
 * real-time progress from the LangGraph agent directly in the chat interface.
 *
 * This component renders nothing itself - it registers a render function with
 * CopilotKit that displays progress during agent execution.
 *
 * @example
 * ```tsx
 * // Include in chat sidebar or chat popup
 * <AgentProgressRenderer />
 * ```
 */
export function AgentProgressRenderer({ className }: AgentProgressRendererProps) {
  useCoAgentStateRender<E2IAgentState>({
    name: 'default', // Must match LangGraphAgent name
    render: ({ state, status, nodeName }) => {
      return (
        <div className={className}>
          <ProgressDisplay
            state={state}
            nodeName={nodeName}
            status={status}
          />
        </div>
      );
    },
  });

  // This component renders nothing directly - it registers a render function
  return null;
}

export default AgentProgressRenderer;
