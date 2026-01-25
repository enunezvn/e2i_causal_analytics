/**
 * Custom Assistant Message Component
 * ===================================
 *
 * Custom implementation of CopilotKit's AssistantMessage that fixes
 * the multiple action buttons issue during streaming.
 *
 * Root cause: Default CopilotKit AssistantMessage only checks `isLoading`
 * to hide the toolbar, but during streaming `isGenerating` is true while
 * `isLoading` is false. This causes the toolbar to render on each streaming
 * emit, creating multiple stacked action button rows.
 *
 * Fix: Hide toolbar when either `isLoading` OR `isGenerating` is true.
 *
 * Features:
 * - Routing rationale display (expandable "Why this agent?" section)
 * - Confidence score visualization
 * - Agent dispatch information
 *
 * @module components/chat/CustomAssistantMessage
 */

import React, { useState } from 'react';
import {
  AssistantMessageProps,
  Markdown,
  useChatContext,
} from '@copilotkit/react-ui';
import {
  ChevronDown,
  ChevronRight,
  Bot,
  Gauge,
  Network,
  Info,
} from 'lucide-react';

// =============================================================================
// ROUTING INFO TYPES
// =============================================================================

/**
 * Routing information extracted from agent responses
 */
interface RoutingInfo {
  /** Agent that handled the query */
  routedAgent?: string;
  /** List of agents that were dispatched */
  agentsDispatched?: string[];
  /** Confidence score (0-1) */
  confidence?: number;
  /** Explanation for why this agent was selected */
  rationale?: string;
  /** Detected intent */
  intent?: string;
  /** Intent classification confidence */
  intentConfidence?: number;
  /** Execution time in milliseconds */
  executionTimeMs?: number;
}

/**
 * Extract routing information from message content if available.
 * Some responses may include routing metadata in a structured format.
 */
function extractRoutingInfo(content: string): RoutingInfo | null {
  // Check if content contains routing markers
  // These patterns match common response structures from the orchestrator
  const routingPatterns = [
    /routed\s+(?:to|by)\s+([A-Za-z_]+)/i,
    /using\s+(?:the\s+)?([A-Za-z_]+)\s+agent/i,
    /(?:causal[_\s]?impact|gap[_\s]?analyzer|experiment[_\s]?designer|drift[_\s]?monitor)/i,
  ];

  // Try to detect agent from content
  let detectedAgent: string | undefined;
  for (const pattern of routingPatterns) {
    const match = content.match(pattern);
    if (match) {
      detectedAgent = match[1] || match[0];
      break;
    }
  }

  // Look for confidence indicators
  const confidenceMatch = content.match(/confidence[:\s]+(\d+(?:\.\d+)?)/i);
  const confidence = confidenceMatch ? parseFloat(confidenceMatch[1]) : undefined;

  // If we found any routing info, return it
  if (detectedAgent || confidence) {
    return {
      routedAgent: detectedAgent?.replace(/[_\s]+/g, ' ').trim(),
      confidence: confidence ? (confidence > 1 ? confidence / 100 : confidence) : undefined,
    };
  }

  return null;
}

// =============================================================================
// ROUTING INFO DISPLAY COMPONENT
// =============================================================================

interface RoutingInfoDisplayProps {
  info: RoutingInfo;
  isExpanded: boolean;
  onToggle: () => void;
}

/**
 * Displays routing rationale in an expandable section.
 * Shows which agent handled the query and why.
 */
function RoutingInfoDisplay({ info, isExpanded, onToggle }: RoutingInfoDisplayProps) {
  const ChevronIcon = isExpanded ? ChevronDown : ChevronRight;

  // Format confidence as percentage
  const confidencePercent = info.confidence
    ? Math.round(info.confidence * 100)
    : null;

  // Determine confidence color
  const getConfidenceColor = (conf: number) => {
    if (conf >= 80) return 'text-emerald-600 bg-emerald-50';
    if (conf >= 60) return 'text-amber-600 bg-amber-50';
    return 'text-rose-600 bg-rose-50';
  };

  return (
    <div className="mt-2 border-t border-border/50 pt-2">
      <button
        onClick={onToggle}
        className="flex items-center gap-1.5 text-xs text-muted-foreground hover:text-foreground transition-colors"
        aria-expanded={isExpanded}
      >
        <ChevronIcon className="h-3 w-3" />
        <Bot className="h-3 w-3" />
        <span>Why this agent?</span>
        {confidencePercent !== null && (
          <span
            className={`ml-1.5 px-1.5 py-0.5 rounded text-[10px] font-medium ${getConfidenceColor(confidencePercent)}`}
          >
            {confidencePercent}%
          </span>
        )}
      </button>

      {isExpanded && (
        <div className="mt-2 pl-4 space-y-2 text-xs animate-in fade-in-0 slide-in-from-top-1">
          {/* Routed Agent */}
          {info.routedAgent && (
            <div className="flex items-start gap-2">
              <Network className="h-3.5 w-3.5 text-blue-500 mt-0.5 flex-shrink-0" />
              <div>
                <span className="text-muted-foreground">Routed to: </span>
                <span className="font-medium capitalize">{info.routedAgent}</span>
              </div>
            </div>
          )}

          {/* Agents Dispatched */}
          {info.agentsDispatched && info.agentsDispatched.length > 0 && (
            <div className="flex items-start gap-2">
              <Network className="h-3.5 w-3.5 text-indigo-500 mt-0.5 flex-shrink-0" />
              <div>
                <span className="text-muted-foreground">Agents involved: </span>
                <span className="font-medium">
                  {info.agentsDispatched.map(a => a.replace(/_/g, ' ')).join(', ')}
                </span>
              </div>
            </div>
          )}

          {/* Confidence Score */}
          {confidencePercent !== null && (
            <div className="flex items-start gap-2">
              <Gauge className="h-3.5 w-3.5 text-emerald-500 mt-0.5 flex-shrink-0" />
              <div>
                <span className="text-muted-foreground">Confidence: </span>
                <span className="font-medium">{confidencePercent}%</span>
              </div>
            </div>
          )}

          {/* Intent */}
          {info.intent && (
            <div className="flex items-start gap-2">
              <Info className="h-3.5 w-3.5 text-purple-500 mt-0.5 flex-shrink-0" />
              <div>
                <span className="text-muted-foreground">Detected intent: </span>
                <span className="font-medium capitalize">{info.intent.replace(/_/g, ' ')}</span>
                {info.intentConfidence && (
                  <span className="text-muted-foreground ml-1">
                    ({Math.round(info.intentConfidence * 100)}%)
                  </span>
                )}
              </div>
            </div>
          )}

          {/* Rationale */}
          {info.rationale && (
            <div className="mt-1.5 p-2 bg-muted/50 rounded-md text-muted-foreground">
              {info.rationale}
            </div>
          )}

          {/* Execution Time */}
          {info.executionTimeMs && (
            <div className="text-muted-foreground">
              Processed in {(info.executionTimeMs / 1000).toFixed(2)}s
            </div>
          )}
        </div>
      )}
    </div>
  );
}

// =============================================================================
// MAIN COMPONENT
// =============================================================================

/**
 * CustomAssistantMessage wraps the CopilotKit assistant message with proper
 * toolbar visibility control during streaming.
 *
 * The toolbar (Regenerate, Copy, Thumbs up, Thumbs down) is only shown when:
 * - There is actual message content
 * - The LLM is NOT loading (thinking)
 * - The LLM is NOT generating (streaming)
 */
export function CustomAssistantMessage(props: AssistantMessageProps) {
  const { icons, labels } = useChatContext();
  const {
    message,
    isLoading,
    isGenerating,
    onRegenerate,
    onCopy,
    onThumbsUp,
    onThumbsDown,
    isCurrentMessage,
    feedback,
    markdownTagRenderers,
  } = props;

  const [copied, setCopied] = useState(false);
  const [routingExpanded, setRoutingExpanded] = useState(false);

  const content = message?.content || '';
  const subComponent = message?.generativeUI?.();

  /**
   * Detect if content is a raw JSON tool result that shouldn't be shown to users.
   * The backend emits multiple message types:
   * 1. Raw JSON tool results: {"success": true, "query_type": "agent_analysis", ...}
   * 2. Human-readable formatted responses
   *
   * Raw JSON tool results should be completely hidden (not just toolbar-less).
   */
  const isRawJsonToolResult = React.useMemo(() => {
    if (!content) return false;
    const trimmed = content.trim();
    // Check if it starts with { and looks like a JSON object
    if (!trimmed.startsWith('{')) return false;
    try {
      const parsed = JSON.parse(trimmed);
      // Tool results typically have these fields - expanded detection
      return (
        typeof parsed === 'object' &&
        parsed !== null &&
        (
          'success' in parsed ||
          'fallback' in parsed ||
          'confidence' in parsed ||
          'query_type' in parsed ||
          'agent_filter' in parsed ||
          ('data' in parsed && 'count' in parsed)
        )
      );
    } catch {
      return false;
    }
  }, [content]);

  // KEY FIX: Only show toolbar when:
  // 1. NOT loading (thinking)
  // 2. NOT generating (streaming)
  // 3. NOT a raw JSON tool result (intermediate message)
  const showToolbar = content && !isLoading && !isGenerating && !isRawJsonToolResult;

  // Extract routing info from content for observability display
  const routingInfo = React.useMemo(() => {
    if (!content || isLoading || isGenerating) return null;
    return extractRoutingInfo(content);
  }, [content, isLoading, isGenerating]);

  // Show routing info only for completed messages with valid content
  const showRoutingInfo = routingInfo && showToolbar;

  // Don't render raw JSON tool results at all - they're internal implementation details
  if (isRawJsonToolResult) {
    return null;
  }

  const handleCopy = () => {
    if (content) {
      navigator.clipboard.writeText(content);
      setCopied(true);
      if (onCopy) {
        onCopy(content);
      }
      setTimeout(() => setCopied(false), 2000);
    }
  };

  const handleRegenerate = () => {
    if (onRegenerate) {
      onRegenerate();
    }
  };

  const handleThumbsUp = () => {
    if (onThumbsUp && message) {
      onThumbsUp(message);
    }
  };

  const handleThumbsDown = () => {
    if (onThumbsDown && message) {
      onThumbsDown(message);
    }
  };

  const LoadingIcon = () => <span>{icons.activityIcon}</span>;

  return (
    <>
      {content && (
        <div className="copilotKitMessage copilotKitAssistantMessage">
          {/* Use CopilotKit's Markdown component for proper rendering */}
          <Markdown content={content} components={markdownTagRenderers} />

          {/* Routing info display - expandable "Why this agent?" section */}
          {showRoutingInfo && (
            <RoutingInfoDisplay
              info={routingInfo}
              isExpanded={routingExpanded}
              onToggle={() => setRoutingExpanded(!routingExpanded)}
            />
          )}

          {/* Action toolbar - ONLY show when streaming is complete */}
          {showToolbar && (
            <div
              className={`copilotKitMessageControls ${isCurrentMessage ? 'currentMessage' : ''}`}
            >
              <button
                className="copilotKitMessageControlButton"
                onClick={handleRegenerate}
                aria-label={labels.regenerateResponse}
                title={labels.regenerateResponse}
              >
                {icons.regenerateIcon}
              </button>

              <button
                className="copilotKitMessageControlButton"
                onClick={handleCopy}
                aria-label={labels.copyToClipboard}
                title={labels.copyToClipboard}
              >
                {copied ? (
                  <span style={{ fontSize: '10px', fontWeight: 'bold' }}>✓</span>
                ) : (
                  icons.copyIcon
                )}
              </button>

              {onThumbsUp && (
                <button
                  className={`copilotKitMessageControlButton ${
                    feedback === 'thumbsUp' ? 'active' : ''
                  }`}
                  onClick={handleThumbsUp}
                  aria-label={labels.thumbsUp}
                  title={labels.thumbsUp}
                >
                  {icons.thumbsUpIcon}
                </button>
              )}

              {onThumbsDown && (
                <button
                  className={`copilotKitMessageControlButton ${
                    feedback === 'thumbsDown' ? 'active' : ''
                  }`}
                  onClick={handleThumbsDown}
                  aria-label={labels.thumbsDown}
                  title={labels.thumbsDown}
                >
                  {icons.thumbsDownIcon}
                </button>
              )}
            </div>
          )}
        </div>
      )}

      {/* Render any generative UI sub-component */}
      {subComponent && <div style={{ marginBottom: '0.5rem' }}>{subComponent}</div>}

      {/* Loading indicator */}
      {isLoading && <LoadingIcon />}
    </>
  );
}

export default CustomAssistantMessage;
