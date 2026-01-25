/**
 * E2I Chat Sidebar Component
 * ==========================
 *
 * Sliding sidebar chat interface using CopilotKit.
 * Provides natural language interaction with E2I agents.
 *
 * Features:
 * - Collapsible sidebar panel
 * - Agent status indicators
 * - Message history
 * - Keyboard shortcut (Cmd/Ctrl + /)
 *
 * @module components/chat/E2IChatSidebar
 */

import * as React from 'react';
import { CopilotChat } from '@copilotkit/react-ui';
import { motion, AnimatePresence } from 'framer-motion';
import {
  MessageSquare,
  X,
  Bot,
  Sparkles,
  Copy,
  Check,
} from 'lucide-react';
import { cn } from '@/lib/utils';
import { Button } from '@/components/ui/button';
import { useE2ICopilot, useCopilotEnabled } from '@/providers/E2ICopilotProvider';
import { AgentStatusPanel } from './AgentStatusPanel';
import { AgentProgressRenderer } from './AgentProgressRenderer';
import { useChatFeedback, FeedbackRating } from '@/hooks/use-chat-feedback';
import { CustomAssistantMessage } from './CustomAssistantMessage';

// =============================================================================
// TYPES
// =============================================================================

export interface E2IChatSidebarProps {
  /** Default open state */
  defaultOpen?: boolean;
  /** Position of the sidebar */
  position?: 'left' | 'right';
  /** Width of the sidebar */
  width?: string;
  /** Show agent status panel */
  showAgentStatus?: boolean;
  /** Additional CSS classes */
  className?: string;
}

// =============================================================================
// COMPONENT
// =============================================================================

/**
 * E2IChatSidebar provides a sliding chat panel for interacting with E2I agents.
 *
 * @example
 * ```tsx
 * <E2IChatSidebar defaultOpen={false} position="right" />
 * ```
 */
export function E2IChatSidebar({
  defaultOpen = false,
  position = 'right',
  width = '400px',
  showAgentStatus = true,
  className,
}: E2IChatSidebarProps) {
  const copilotEnabled = useCopilotEnabled();
  const { chatOpen, setChatOpen, agents, filters } = useE2ICopilot();
  const [showAgents, setShowAgents] = React.useState(false);
  const [traceIdCopied, setTraceIdCopied] = React.useState(false);
  const { submitFeedback } = useChatFeedback();

  // Generate a stable session ID for feedback tracking and support tickets
  const sessionIdRef = React.useRef<string>(
    `session_${Date.now()}_${Math.random().toString(36).substring(2, 9)}`
  );

  // Copy trace ID to clipboard for support ticket correlation
  const copyTraceId = React.useCallback(() => {
    navigator.clipboard.writeText(sessionIdRef.current).then(() => {
      setTraceIdCopied(true);
      setTimeout(() => setTraceIdCopied(false), 2000);
    }).catch((err) => {
      console.error('[E2IChatSidebar] Failed to copy trace ID:', err);
    });
  }, []);

  // Shortened trace ID for display (show last 12 chars)
  const shortTraceId = React.useMemo(() => {
    const id = sessionIdRef.current;
    return id.length > 16 ? `...${id.slice(-12)}` : id;
  }, []);

  // Feedback handlers for CopilotKit thumbs up/down buttons
  // DEBUG: Log when handlers are created
  console.log('[E2IChatSidebar] Creating feedback handlers, submitFeedback available:', !!submitFeedback);

  const handleThumbsUp = React.useCallback(
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    (message: any) => {
      // DEBUG: Immediate log to verify callback is invoked
      console.log('[E2IChatSidebar] handleThumbsUp CALLED with message:', message);

      try {
        const messageId = message.id ? parseInt(message.id, 10) || Date.now() : Date.now();
        const content = typeof message.content === 'string'
          ? message.content
          : JSON.stringify(message.content || '');

        // Fire and forget - don't await to avoid blocking the UI
        submitFeedback({
          messageId,
          sessionId: sessionIdRef.current,
          rating: 'thumbs_up' as FeedbackRating,
          responsePreview: content.substring(0, 500),
          agentName: 'copilotkit',
        }).then(() => {
          console.log('[E2IChatSidebar] Thumbs up feedback submitted for message:', messageId);
        }).catch((error) => {
          console.error('[E2IChatSidebar] Failed to submit thumbs up feedback:', error);
        });
      } catch (error) {
        console.error('[E2IChatSidebar] Error in handleThumbsUp:', error);
      }
    },
    [submitFeedback]
  );

  const handleThumbsDown = React.useCallback(
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    (message: any) => {
      // DEBUG: Immediate log to verify callback is invoked
      console.log('[E2IChatSidebar] handleThumbsDown CALLED with message:', message);

      try {
        const messageId = message.id ? parseInt(message.id, 10) || Date.now() : Date.now();
        const content = typeof message.content === 'string'
          ? message.content
          : JSON.stringify(message.content || '');

        // Fire and forget - don't await to avoid blocking the UI
        submitFeedback({
          messageId,
          sessionId: sessionIdRef.current,
          rating: 'thumbs_down' as FeedbackRating,
          responsePreview: content.substring(0, 500),
          agentName: 'copilotkit',
        }).then(() => {
          console.log('[E2IChatSidebar] Thumbs down feedback submitted for message:', messageId);
        }).catch((error) => {
          console.error('[E2IChatSidebar] Failed to submit thumbs down feedback:', error);
        });
      } catch (error) {
        console.error('[E2IChatSidebar] Error in handleThumbsDown:', error);
      }
    },
    [submitFeedback]
  );

  // Initialize with defaultOpen
  React.useEffect(() => {
    if (copilotEnabled) {
      setChatOpen(defaultOpen);
    }
  }, [defaultOpen, setChatOpen, copilotEnabled]);

  // Keyboard shortcut: Cmd/Ctrl + /
  React.useEffect(() => {
    if (!copilotEnabled) return;

    const handleKeyDown = (e: KeyboardEvent) => {
      if ((e.metaKey || e.ctrlKey) && e.key === '/') {
        e.preventDefault();
        setChatOpen((prev) => !prev);
      }
    };

    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [setChatOpen, copilotEnabled]);

  // If CopilotKit is not enabled, don't render the sidebar
  if (!copilotEnabled) {
    return null;
  }

  // Count active agents
  const activeAgentCount = agents.filter((a) => a.status === 'active' || a.status === 'processing').length;

  return (
    <>
      {/* Toggle Button */}
      <AnimatePresence>
        {!chatOpen && (
          <motion.div
            initial={{ opacity: 0, x: position === 'right' ? 20 : -20 }}
            animate={{ opacity: 1, x: 0 }}
            exit={{ opacity: 0, x: position === 'right' ? 20 : -20 }}
            className={cn(
              'fixed z-50 bottom-6',
              position === 'right' ? 'right-6' : 'left-6'
            )}
          >
            <Button
              onClick={() => setChatOpen(true)}
              size="lg"
              className="rounded-full shadow-lg h-14 w-14 bg-gradient-to-r from-blue-600 to-indigo-600 hover:from-blue-700 hover:to-indigo-700"
            >
              <MessageSquare className="h-6 w-6" />
            </Button>
            {activeAgentCount > 0 && (
              <span className="absolute -top-1 -right-1 bg-emerald-500 text-white text-xs rounded-full h-5 w-5 flex items-center justify-center">
                {activeAgentCount}
              </span>
            )}
          </motion.div>
        )}
      </AnimatePresence>

      {/* Sidebar Panel */}
      <AnimatePresence>
        {chatOpen && (
          <motion.div
            initial={{ x: position === 'right' ? '100%' : '-100%' }}
            animate={{ x: 0 }}
            exit={{ x: position === 'right' ? '100%' : '-100%' }}
            transition={{ type: 'spring', damping: 25, stiffness: 200 }}
            className={cn(
              'fixed inset-y-0 z-50 flex flex-col bg-background border-l shadow-xl',
              position === 'right' ? 'right-0' : 'left-0 border-l-0 border-r',
              className
            )}
            style={{ width }}
          >
            {/* Header */}
            <div className="flex items-center justify-between p-4 border-b bg-muted/50">
              <div className="flex items-center gap-3">
                <div className="p-2 rounded-lg bg-gradient-to-r from-blue-600 to-indigo-600">
                  <Bot className="h-5 w-5 text-white" />
                </div>
                <div>
                  <h2 className="font-semibold">E2I Assistant</h2>
                  <p className="text-xs text-muted-foreground">
                    {filters.brand} | {activeAgentCount} agents active
                  </p>
                </div>
              </div>

              <div className="flex items-center gap-1">
                {showAgentStatus && (
                  <Button
                    variant="ghost"
                    size="icon"
                    onClick={() => setShowAgents(!showAgents)}
                    className={cn(showAgents && 'bg-muted')}
                  >
                    <Sparkles className="h-4 w-4" />
                  </Button>
                )}
                <Button
                  variant="ghost"
                  size="icon"
                  onClick={() => setChatOpen(false)}
                >
                  <X className="h-4 w-4" />
                </Button>
              </div>
            </div>

            {/* Agent Status Panel (Collapsible) */}
            <AnimatePresence>
              {showAgents && (
                <motion.div
                  initial={{ height: 0, opacity: 0 }}
                  animate={{ height: 'auto', opacity: 1 }}
                  exit={{ height: 0, opacity: 0 }}
                  className="overflow-hidden border-b"
                >
                  <AgentStatusPanel agents={agents} compact />
                </motion.div>
              )}
            </AnimatePresence>

            {/* Chat Area - min-h-0 is critical for flexbox scroll containers */}
            <div className="flex-1 overflow-y-auto min-h-0">
              {/* CoAgent progress renderer - displays real-time progress from LangGraph */}
              <AgentProgressRenderer className="px-3 pt-2" />

              <CopilotChat
                AssistantMessage={CustomAssistantMessage}
                onThumbsUp={handleThumbsUp}
                onThumbsDown={handleThumbsDown}
                instructions={`You are helping an analyst work with the E2I Causal Analytics platform.

Current context:
- Brand filter: ${filters.brand}
- Date range: ${filters.dateRange.start} to ${filters.dateRange.end}
${filters.territory ? `- Territory: ${filters.territory}` : ''}
${filters.hcpSegment ? `- HCP Segment: ${filters.hcpSegment}` : ''}

Active agents: ${agents.filter(a => a.status === 'active').map(a => a.name).join(', ')}

Available actions:
- navigateTo: Navigate to any page
- setBrandFilter: Change brand filter
- setDateRange: Set analytics date range
- highlightCausalPaths: Highlight paths on visualizations
- setDetailLevel: Adjust response complexity

Be concise and helpful. Focus on pharmaceutical commercial analytics (TRx, NRx, market share, etc.).`}
                labels={{
                  initial: 'How can I help you explore E2I analytics?',
                  placeholder: 'Ask about KPIs, agents, or insights...',
                }}
                className="h-full"
              />
            </div>

            {/* Footer with Trace ID for Support */}
            <div className="p-3 border-t bg-muted/30 text-xs text-muted-foreground">
              <div className="flex items-center justify-between mb-1.5">
                <span>Press ⌘/ to toggle</span>
                <span>{filters.brand}</span>
              </div>
              {/* Trace ID for support ticket correlation */}
              <div className="flex items-center justify-between pt-1.5 border-t border-border/50">
                <span className="text-[10px] text-muted-foreground/70">
                  Trace ID: <code className="font-mono">{shortTraceId}</code>
                </span>
                <button
                  onClick={copyTraceId}
                  className="flex items-center gap-1 px-1.5 py-0.5 rounded text-[10px] hover:bg-muted transition-colors"
                  title="Copy full trace ID for support"
                  aria-label="Copy trace ID"
                >
                  {traceIdCopied ? (
                    <>
                      <Check className="h-3 w-3 text-emerald-500" />
                      <span className="text-emerald-500">Copied!</span>
                    </>
                  ) : (
                    <>
                      <Copy className="h-3 w-3" />
                      <span>Copy</span>
                    </>
                  )}
                </button>
              </div>
            </div>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Backdrop */}
      <AnimatePresence>
        {chatOpen && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            onClick={() => setChatOpen(false)}
            className="fixed inset-0 z-40 bg-black/20 backdrop-blur-sm"
          />
        )}
      </AnimatePresence>
    </>
  );
}

export default E2IChatSidebar;
