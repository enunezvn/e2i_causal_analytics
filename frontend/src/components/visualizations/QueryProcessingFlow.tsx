/**
 * QueryProcessingFlow Component
 * =============================
 *
 * Visualizes the E2I query processing pipeline:
 * Query → Intent Classification → Memory Selection → Retrieval → Response
 *
 * Features:
 * - Animated flow indicators
 * - Stage-by-stage breakdown
 * - Latency targets per stage
 *
 * @module components/visualizations/QueryProcessingFlow
 */

import { useState, useEffect } from 'react';
import {
  MessageSquare,
  Brain,
  Database,
  Search,
  MessageCircle,
  ArrowRight,
  Clock,
  CheckCircle2,
  Loader2,
} from 'lucide-react';

// =============================================================================
// TYPES
// =============================================================================

interface ProcessingStage {
  id: string;
  name: string;
  description: string;
  icon: React.ReactNode;
  latency: string;
  color: string;
}

export interface QueryProcessingFlowProps {
  /** Whether to animate the flow */
  animate?: boolean;
  /** Animation speed in ms per stage */
  animationSpeed?: number;
  /** Additional CSS classes */
  className?: string;
}

// =============================================================================
// CONSTANTS
// =============================================================================

const PROCESSING_STAGES: ProcessingStage[] = [
  {
    id: 'query',
    name: 'Query Input',
    description: 'Natural language query from user',
    icon: <MessageSquare className="h-5 w-5" />,
    latency: '<10ms',
    color: 'blue',
  },
  {
    id: 'intent',
    name: 'Intent Classification',
    description: 'NLP extracts intent, entities, and context',
    icon: <Brain className="h-5 w-5" />,
    latency: '<100ms',
    color: 'purple',
  },
  {
    id: 'memory',
    name: 'Memory Selection',
    description: 'Routes to appropriate memory systems',
    icon: <Database className="h-5 w-5" />,
    latency: '<50ms',
    color: 'emerald',
  },
  {
    id: 'retrieval',
    name: 'Retrieval',
    description: 'Fetches relevant context from memories',
    icon: <Search className="h-5 w-5" />,
    latency: '<500ms',
    color: 'orange',
  },
  {
    id: 'response',
    name: 'Response Generation',
    description: 'Agent generates contextual response',
    icon: <MessageCircle className="h-5 w-5" />,
    latency: '<2s',
    color: 'pink',
  },
];

// =============================================================================
// COLOR HELPERS
// =============================================================================

const getColorClasses = (color: string, isActive: boolean) => {
  const colors: Record<string, { bg: string; border: string; text: string; icon: string }> = {
    blue: {
      bg: isActive ? 'bg-blue-100 dark:bg-blue-900/40' : 'bg-blue-50 dark:bg-blue-900/20',
      border: isActive ? 'border-blue-400 dark:border-blue-500' : 'border-blue-200 dark:border-blue-800',
      text: 'text-blue-900 dark:text-blue-300',
      icon: 'text-blue-600 dark:text-blue-400',
    },
    purple: {
      bg: isActive ? 'bg-purple-100 dark:bg-purple-900/40' : 'bg-purple-50 dark:bg-purple-900/20',
      border: isActive ? 'border-purple-400 dark:border-purple-500' : 'border-purple-200 dark:border-purple-800',
      text: 'text-purple-900 dark:text-purple-300',
      icon: 'text-purple-600 dark:text-purple-400',
    },
    emerald: {
      bg: isActive ? 'bg-emerald-100 dark:bg-emerald-900/40' : 'bg-emerald-50 dark:bg-emerald-900/20',
      border: isActive ? 'border-emerald-400 dark:border-emerald-500' : 'border-emerald-200 dark:border-emerald-800',
      text: 'text-emerald-900 dark:text-emerald-300',
      icon: 'text-emerald-600 dark:text-emerald-400',
    },
    orange: {
      bg: isActive ? 'bg-orange-100 dark:bg-orange-900/40' : 'bg-orange-50 dark:bg-orange-900/20',
      border: isActive ? 'border-orange-400 dark:border-orange-500' : 'border-orange-200 dark:border-orange-800',
      text: 'text-orange-900 dark:text-orange-300',
      icon: 'text-orange-600 dark:text-orange-400',
    },
    pink: {
      bg: isActive ? 'bg-pink-100 dark:bg-pink-900/40' : 'bg-pink-50 dark:bg-pink-900/20',
      border: isActive ? 'border-pink-400 dark:border-pink-500' : 'border-pink-200 dark:border-pink-800',
      text: 'text-pink-900 dark:text-pink-300',
      icon: 'text-pink-600 dark:text-pink-400',
    },
  };

  return colors[color] || colors.blue;
};

// =============================================================================
// COMPONENT
// =============================================================================

export function QueryProcessingFlow({
  animate = true,
  animationSpeed = 800,
  className = '',
}: QueryProcessingFlowProps) {
  const [activeStage, setActiveStage] = useState<number>(-1);
  const [isAnimating, setIsAnimating] = useState(false);

  // Animation loop
  useEffect(() => {
    if (!animate) {
      setActiveStage(-1);
      return;
    }

    setIsAnimating(true);
    let currentStage = 0;

    const interval = setInterval(() => {
      setActiveStage(currentStage);
      currentStage = (currentStage + 1) % (PROCESSING_STAGES.length + 1);

      if (currentStage === 0) {
        // Brief pause at end before restarting
        setTimeout(() => setActiveStage(-1), animationSpeed / 2);
      }
    }, animationSpeed);

    return () => {
      clearInterval(interval);
      setIsAnimating(false);
    };
  }, [animate, animationSpeed]);

  return (
    <div className={`bg-[var(--color-card)] rounded-lg border border-[var(--color-border)] p-6 ${className}`}>
      <div className="flex items-center justify-between mb-6">
        <div className="flex items-center gap-3">
          <div className="p-2 rounded-lg bg-[var(--color-primary)]/10 text-[var(--color-primary)]">
            <Clock className="h-5 w-5" />
          </div>
          <div>
            <h3 className="text-lg font-semibold text-[var(--color-text-primary)]">
              Query Processing Flow
            </h3>
            <p className="text-sm text-[var(--color-text-secondary)]">
              End-to-end latency target: &lt;3s
            </p>
          </div>
        </div>
        {isAnimating && (
          <div className="flex items-center gap-2 text-sm text-[var(--color-text-secondary)]">
            <Loader2 className="h-4 w-4 animate-spin" />
            <span>Simulating flow...</span>
          </div>
        )}
      </div>

      {/* Flow Diagram */}
      <div className="flex flex-col lg:flex-row items-stretch lg:items-center gap-2 lg:gap-0">
        {PROCESSING_STAGES.map((stage, index) => {
          const isActive = activeStage === index;
          const isCompleted = activeStage > index;
          const colorClasses = getColorClasses(stage.color, isActive);

          return (
            <div key={stage.id} className="flex flex-col lg:flex-row items-stretch lg:items-center flex-1">
              {/* Stage Card */}
              <div
                className={`
                  flex-1 p-4 rounded-lg border-2 transition-all duration-300
                  ${colorClasses.bg} ${colorClasses.border}
                  ${isActive ? 'scale-105 shadow-lg' : 'scale-100'}
                `}
              >
                <div className="flex items-center gap-3 mb-2">
                  <div className={`${colorClasses.icon}`}>
                    {isCompleted ? (
                      <CheckCircle2 className="h-5 w-5 text-green-500" />
                    ) : isActive ? (
                      <Loader2 className="h-5 w-5 animate-spin" />
                    ) : (
                      stage.icon
                    )}
                  </div>
                  <span className={`font-medium text-sm ${colorClasses.text}`}>
                    {stage.name}
                  </span>
                </div>
                <p className="text-xs text-[var(--color-text-secondary)] mb-2 line-clamp-2">
                  {stage.description}
                </p>
                <div className="flex items-center gap-1 text-xs">
                  <Clock className="h-3 w-3 text-[var(--color-text-tertiary)]" />
                  <span className="text-[var(--color-text-tertiary)]">{stage.latency}</span>
                </div>
              </div>

              {/* Arrow between stages */}
              {index < PROCESSING_STAGES.length - 1 && (
                <div className="flex items-center justify-center lg:px-2 py-2 lg:py-0">
                  <ArrowRight
                    className={`
                      h-5 w-5 transition-all duration-300 rotate-90 lg:rotate-0
                      ${isCompleted || isActive
                        ? 'text-[var(--color-primary)] scale-110'
                        : 'text-[var(--color-text-tertiary)]'
                      }
                    `}
                  />
                </div>
              )}
            </div>
          );
        })}
      </div>

      {/* Memory System Breakdown */}
      <div className="mt-6 pt-4 border-t border-[var(--color-border)]">
        <h4 className="text-sm font-medium text-[var(--color-text-secondary)] mb-3">
          Memory System Latency Breakdown
        </h4>
        <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
          <div className="p-3 rounded-lg bg-blue-50 dark:bg-blue-900/20 border border-blue-200 dark:border-blue-800">
            <span className="text-xs text-blue-700 dark:text-blue-400">Working Memory</span>
            <p className="text-sm font-bold text-blue-900 dark:text-blue-300">&lt;50ms</p>
          </div>
          <div className="p-3 rounded-lg bg-purple-50 dark:bg-purple-900/20 border border-purple-200 dark:border-purple-800">
            <span className="text-xs text-purple-700 dark:text-purple-400">Episodic Memory</span>
            <p className="text-sm font-bold text-purple-900 dark:text-purple-300">&lt;200ms</p>
          </div>
          <div className="p-3 rounded-lg bg-emerald-50 dark:bg-emerald-900/20 border border-emerald-200 dark:border-emerald-800">
            <span className="text-xs text-emerald-700 dark:text-emerald-400">Semantic Memory</span>
            <p className="text-sm font-bold text-emerald-900 dark:text-emerald-300">&lt;500ms</p>
          </div>
          <div className="p-3 rounded-lg bg-orange-50 dark:bg-orange-900/20 border border-orange-200 dark:border-orange-800">
            <span className="text-xs text-orange-700 dark:text-orange-400">Procedural Memory</span>
            <p className="text-sm font-bold text-orange-900 dark:text-orange-300">&lt;200ms</p>
          </div>
        </div>
      </div>
    </div>
  );
}

export default QueryProcessingFlow;
