/**
 * E2I Chat Sidebar Component
 * ==========================
 *
 * Sliding sidebar chat interface using CopilotKit.
 * Provides natural language interaction with E2I agents.
 *
 * Features:
 * - Collapsible sidebar panel, non-modal: no backdrop, the page behind stays
 *   readable and interactive while the chat is open
 * - Drag-to-resize width (min 320px up to full page width; double-click resets)
 * - Agent status indicators
 * - Message history
 * - Keyboard shortcut (Cmd/Ctrl + /)
 *
 * @module components/chat/E2IChatSidebar
 */

import * as React from 'react';
import { useLocation } from 'react-router-dom';
import { CopilotContext, useCopilotChatInternal } from '@copilotkit/react-core';
import { CopilotChat } from '@copilotkit/react-ui';
import { useQuery } from '@tanstack/react-query';
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
import { logger } from '@/lib/logger';
import { getValidated, post } from '@/lib/api-client';
import { AgentStatusResponseSchema } from '@/lib/api-schemas';
import { Button } from '@/components/ui/button';
import { useE2ICopilot, useCopilotEnabled, type AgentInfo } from '@/providers/E2ICopilotProvider';
import { useResizablePanel } from '@/hooks/use-resizable-panel';
import { useUIStore } from '@/stores/ui-store';
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
  /** Default width of the sidebar (px value; the user can drag-resize from there) */
  width?: string;
  /** Show agent status panel */
  showAgentStatus?: boolean;
  /** Additional CSS classes */
  className?: string;
}

// =============================================================================
// SUGGESTIONS
// =============================================================================

type ChatSuggestion = { title: string; message: string };

/**
 * Follow-up pills rendered above the input, only once a conversation exists.
 * Two tiers: after each completed assistant turn, ConversationSuggestions
 * fetches conversation-adaptive pills from POST /chat/suggestions (one
 * fast-tier LLM call over the recent transcript, no orchestrator); the
 * static context-aware set below is the FALLBACK — shown until the first
 * generation lands and whenever generation fails. A suggestions array
 * bypasses CopilotKit's suggestion engine and is passed through reactively,
 * so pills swap live as new generations arrive — but the bypass also skips
 * the engine's message-count gating, so the empty-until-first-user-message
 * gate is reimplemented here: an empty pane must not presume what the user
 * needs before they have asked anything.
 *
 * CopilotKit's own LLM suggestions (suggestions="auto") are deliberately NOT
 * used: the engine clones the default agent and forces a `copilotkitSuggest`
 * tool call via forwardedProps.toolChoice, which our LangGraph runtime
 * ignores (copilotkit.py binds tools with its own tool_choice="auto") —
 * every exchange would burn a full orchestrator run and never yield a pill.
 * The /chat/suggestions endpoint is the cheap replacement.
 *
 * Keep fallback pill topics inside what the bound backend tools can actually
 * answer (KPIs, causal paths, agents — see E2I_CHATBOT_TOOLS in
 * chatbot_tools.py); the chart pills route through the renderKpiTrend
 * generative-UI action so users discover inline visuals.
 */
const PAGE_SUGGESTIONS: Record<string, ChatSuggestion> = {
  '/': {
    title: 'Executive summary',
    message:
      'Give me an executive summary of current brand performance and the top actions to take.',
  },
  '/causal-analysis': {
    title: 'Strongest causal paths',
    message:
      'What are the strongest causal paths driving TRx, and how confident are we in them?',
  },
  '/knowledge-graph': {
    title: 'Paths to share growth',
    message: 'What causal paths lead to market share growth?',
  },
  '/time-series': {
    title: 'Compare brand TRx',
    message: 'Compare TRx across Remibrutinib, Fabhalta, and Kisqali in a table.',
  },
  '/segment-analysis': {
    title: 'Segment responders',
    message: 'Which patient segments respond best to interventions?',
  },
  '/agent-orchestration': {
    title: 'Agent activity',
    message: 'Which agents are active right now and what are they working on?',
  },
};

const DEFAULT_PAGE_SUGGESTION: ChatSuggestion = {
  title: 'Top causal drivers',
  message: 'What are the top causal drivers of TRx right now?',
};

function buildChatSuggestions(pathname: string, brand: string): ChatSuggestion[] {
  return [
    { title: '📈 Chart the TRx trend', message: 'Chart the TRx trend' },
    {
      title: `📊 ${brand} market share`,
      message: `Chart the market share trend for ${brand}`,
    },
    PAGE_SUGGESTIONS[pathname] ?? DEFAULT_PAGE_SUGGESTION,
    {
      title: 'Biggest KPI movers',
      message:
        'Which KPIs moved the most recently? Summarize the biggest movers in a table.',
    },
  ];
}

/**
 * Conversation probe: reports whether the user has sent at least one message
 * (the pill gate) and fetches conversation-adaptive pills from
 * POST /chat/suggestions after each completed assistant turn. Runs as a
 * child of the (copilot-enabled) pane because the chat hooks THROW outside
 * the CopilotKit provider, and this sidebar's top-level hooks also run with
 * copilot disabled. Conversation state lives on the agent (verified live:
 * the legacy CopilotMessagesContext stays empty on this architecture, and
 * `useCopilotChat().visibleMessages` is typed but no longer returned by the
 * 1.51.2 implementation — always undefined). useCopilotChatInternal is
 * exported from the package index and returns the agent's live AG-UI
 * messages plus isLoading (Boolean(agent.isRunning) — verified present in
 * the installed bundle, unlike visibleMessages); it only reads config and
 * subscribes, registering nothing, so mounting it alongside CopilotChat is
 * side-effect-free. The gate is keyed to USER messages so the
 * `labels.initial` greeting can't open it. Renders nothing.
 */
function ConversationSuggestions({
  pathname,
  brand,
  onUpdate,
}: {
  pathname: string;
  brand: string;
  onUpdate: (state: { hasUserMessage: boolean; adaptive: ChatSuggestion[] | null }) => void;
}) {
  const { messages, isLoading } = useCopilotChatInternal();
  const [adaptive, setAdaptive] = React.useState<ChatSuggestion[] | null>(null);
  // One fetch per completed turn (keyed on transcript length — also absorbs
  // StrictMode double-effects), and stale responses are dropped whenever a
  // newer request has been issued.
  const lastFetchKeyRef = React.useRef<string | null>(null);
  const requestSeqRef = React.useRef(0);

  // User/assistant text turns only — tool/system/developer messages and
  // non-string content never reach the suggestion endpoint.
  const transcript = React.useMemo(() => {
    const turns: Array<{ role: 'user' | 'assistant'; content: string }> = [];
    for (const m of messages ?? []) {
      if (
        (m.role === 'user' || m.role === 'assistant') &&
        typeof m.content === 'string' &&
        m.content.trim()
      ) {
        turns.push({
          role: m.role === 'user' ? 'user' : 'assistant',
          content: m.content.slice(0, 1500),
        });
      }
    }
    return turns;
  }, [messages]);

  const hasUserMessage = transcript.some((t) => t.role === 'user');

  React.useEffect(() => {
    if (isLoading || !hasUserMessage) return;
    const last = transcript[transcript.length - 1];
    if (!last || last.role !== 'assistant') return;
    const key = String(transcript.length);
    if (lastFetchKeyRef.current === key) return;
    lastFetchKeyRef.current = key;
    const seq = ++requestSeqRef.current;
    post<{ suggestions: ChatSuggestion[] }>('/chat/suggestions', {
      messages: transcript.slice(-12),
      page: pathname,
      brand,
    })
      .then((res) => {
        if (seq !== requestSeqRef.current) return;
        const pills = (res.suggestions ?? [])
          .filter((s) => typeof s?.title === 'string' && s.title && typeof s?.message === 'string' && s.message)
          .slice(0, 4);
        setAdaptive(pills.length > 0 ? pills : null);
      })
      .catch(() => {
        // Generation failed (e.g. 502) — fall back to the static
        // context-aware pills; never crash or blank the pane.
        if (seq === requestSeqRef.current) setAdaptive(null);
      });
  }, [isLoading, hasUserMessage, transcript, pathname, brand]);

  React.useEffect(() => {
    onUpdate({ hasUserMessage, adaptive });
  }, [hasUserMessage, adaptive, onUpdate]);

  return null;
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
  const { pathname } = useLocation();

  // Suggestion pills: none until a conversation exists (pills are follow-ups,
  // not openers), then the conversation-adaptive set generated by the backend,
  // with the static context-aware set as fallback until the first generation
  // lands or when generation fails. State is reported by the
  // ConversationSuggestions child, which can read conversation state.
  const [pillState, setPillState] = React.useState<{
    hasUserMessage: boolean;
    adaptive: ChatSuggestion[] | null;
  }>({ hasUserMessage: false, adaptive: null });
  const chatSuggestions = React.useMemo(() => {
    if (!pillState.hasUserMessage) return [];
    return pillState.adaptive ?? buildChatSuggestions(pathname, filters.brand);
  }, [pillState, pathname, filters.brand]);
  const [showAgents, setShowAgents] = React.useState(false);
  const [traceIdCopied, setTraceIdCopied] = React.useState(false);
  const { submitFeedback } = useChatFeedback();

  // Drag-to-resize: the persisted width lives in the UI store (survives
  // reloads); the `width` prop is only the default. Clamped between 320px and
  // the full window width; double-click on the handle resets to the default.
  const chatPanelWidth = useUIStore((s) => s.chatPanelWidth);
  const setChatPanelWidth = useUIStore((s) => s.setChatPanelWidth);
  const {
    width: panelWidth,
    isDragging,
    handleProps,
  } = useResizablePanel({
    defaultWidth: Number.parseInt(width, 10) || 400,
    minWidth: 320,
    edge: position === 'right' ? 'left' : 'right',
    persistedWidth: chatPanelWidth,
    onWidthChange: setChatPanelWidth,
    ariaLabel: 'Resize chat panel',
  });

  // Live agent status from the SAME real source the Agent Orchestration page
  // uses: GET /agents/status derives status from audit_chain_entries (an agent
  // is ACTIVE only if it recorded an action within the last ~15 min). This
  // replaces the provider's static registry, whose `activeAgents` map was never
  // populated — so the panel/badge previously read a permanent "0 active". The
  // registry is the graceful fallback while the query is loading/unavailable.
  const { data: agentStatus } = useQuery({
    queryKey: ['agent-status'],
    queryFn: () => getValidated(AgentStatusResponseSchema, '/agents/status'),
    refetchInterval: 30000,
    retry: false,
    enabled: copilotEnabled,
  });

  const liveAgents: AgentInfo[] = React.useMemo(
    () =>
      (agentStatus?.agents ?? agents).map((a) => ({
        id: a.id,
        name: a.name,
        tier: a.tier,
        status: a.status,
        capabilities: a.capabilities ?? [],
      })),
    [agentStatus, agents]
  );

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

  // Feedback handlers for CopilotKit thumbs up/down buttons.
  // (Removed a per-render console.log here that ran on every render and leaked
  //  to the production console — #18 console hygiene.)
  //
  // The rated message is identified server-side by (threadId, response prefix):
  // message persistence stores the CopilotKit threadId as the session key, and
  // the AG-UI message uuid the client holds is unrelated to the DB row id.
  // Read the raw context (not useCopilotContext(), which THROWS outside the
  // provider — this sidebar also renders with copilot disabled).
  const { threadId } = React.useContext(CopilotContext);

  const rateMessage = React.useCallback(
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    (message: any, rating: FeedbackRating) => {
      logger.debug(`[E2IChatSidebar] ${rating} CALLED with message:`, message);

      try {
        const messageUuid = typeof message.id === 'string' ? message.id : undefined;
        const content = typeof message.content === 'string'
          ? message.content
          : JSON.stringify(message.content || '');

        // Fire and forget - don't await to avoid blocking the UI.
        // No agentName: the server derives attribution from the matched
        // message row (the client is not the authority on who responded).
        // responseText enables exact full-content resolution; the 500-char
        // preview is what gets stored on the feedback row.
        submitFeedback({
          messageKey: messageUuid ?? String(Date.now()),
          messageUuid,
          sessionId: threadId,
          rating,
          responsePreview: content.substring(0, 500),
          responseText: content.substring(0, 20000),
        }).then((result) => {
          if (result.success) {
            logger.debug(`[E2IChatSidebar] ${rating} feedback submitted for message:`, messageUuid);
          } else {
            console.error(`[E2IChatSidebar] ${rating} feedback rejected:`, result.error);
          }
        }).catch((error) => {
          console.error(`[E2IChatSidebar] Failed to submit ${rating} feedback:`, error);
        });
      } catch (error) {
        console.error(`[E2IChatSidebar] Error in ${rating} handler:`, error);
      }
    },
    [submitFeedback, threadId]
  );

  const handleThumbsUp = React.useCallback(
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    (message: any) => rateMessage(message, 'thumbs_up'),
    [rateMessage]
  );

  const handleThumbsDown = React.useCallback(
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    (message: any) => rateMessage(message, 'thumbs_down'),
    [rateMessage]
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

  // Count active agents from the live roster
  const activeAgentCount = liveAgents.filter((a) => a.status === 'active' || a.status === 'processing').length;

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
              // Opaque theme background via the CSS var directly: `bg-background`
              // is a NO-OP in this Tailwind v4 setup (no @theme mapping for
              // --color-background), which left the panel transparent — the page
              // showed through the "semi-transparent" pane.
              'fixed inset-y-0 z-50 flex flex-col bg-[var(--color-background)] border-l shadow-xl',
              position === 'right' ? 'right-0' : 'left-0 border-l-0 border-r',
              className
            )}
            // min() keeps a previously saved wide panel inside the viewport if
            // the window has since shrunk
            style={{ width: `min(${panelWidth}px, 100vw)` }}
          >
            {/* Resize Handle (inner edge) */}
            <div
              {...handleProps}
              title="Drag to resize — double-click to reset"
              className={cn(
                'absolute inset-y-0 z-10 w-2 cursor-col-resize touch-none',
                'transition-colors hover:bg-blue-500/30 focus-visible:bg-blue-500/40 focus-visible:outline-none',
                isDragging && 'bg-blue-500/40',
                position === 'right' ? 'left-0' : 'right-0'
              )}
            />

            {/* Header */}
            <div className="flex items-center justify-between p-4 border-b bg-muted/50">
              <div className="flex items-center gap-3">
                <div className="p-2 rounded-lg bg-gradient-to-r from-blue-600 to-indigo-600">
                  <Bot className="h-5 w-5 text-white" />
                </div>
                <div>
                  <h2 className="font-semibold">E2I Assistant</h2>
                  <p className="text-xs text-muted-foreground">
                    Strategic analytics
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
                  <AgentStatusPanel agents={liveAgents} compact />
                </motion.div>
              )}
            </AnimatePresence>

            {/* Chat Area - a flex column so CopilotChat (flex-1) owns all
                remaining height: messages scroll internally and the input sits
                at the pane bottom. overflow-hidden (not auto) — an outer
                scrollbar here would scroll the input away with the messages. */}
            <div className="flex min-h-0 flex-1 flex-col overflow-hidden">
              {/* CoAgent progress renderer - displays real-time progress from LangGraph */}
              <AgentProgressRenderer className="shrink-0 px-3 pt-2" />

              <ConversationSuggestions
                pathname={pathname}
                brand={filters.brand}
                onUpdate={setPillState}
              />
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

Active agents: ${liveAgents.filter(a => a.status === 'active').map(a => a.name).join(', ')}

Available actions:
- navigateTo: Navigate to any page
- setBrandFilter: Change brand filter
- setDateRange: Set analytics date range
- highlightCausalPaths: Highlight paths on visualizations
- setDetailLevel: Adjust response complexity

Always fetch real data with the available tools before answering, cite the actual metric values and their source, and deliver rich, evidence-based strategic insight — what the numbers mean for the business and the concrete next action — not just the figure. Focus on pharmaceutical commercial analytics (TRx, NRx, market share, causal drivers).

Visual answers: whenever the answer involves a KPI's evolution over time, a trend, or a period comparison, call renderKpiTrend so the chart renders inline alongside your text (kpiId: trx, nrx, nbrx, trx_share, conversion_rate, roi, or a registry code like WS3-BI-005; nbrx and trx_share need a brand). Use markdown tables for multi-row numeric comparisons instead of prose lists of figures.`}
                labels={{
                  initial: 'How can I help you explore E2I analytics?',
                  placeholder: 'Ask about KPIs, agents, or insights...',
                }}
                suggestions={chatSuggestions}
                className="min-h-0 flex-1"
              />
            </div>

            {/* Footer with Trace ID for Support */}
            <div className="p-3 border-t bg-muted/30 text-xs text-muted-foreground">
              <div className="flex items-center justify-between mb-1.5">
                <span>Press ⌘/ to toggle</span>
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

      {/* No backdrop: the pane is a non-modal docked panel — the page behind
          it must stay readable AND interactive (scroll, hover, filter) so the
          analyst can reference on-screen data while chatting. Close via the
          header X, the FAB, or Cmd/Ctrl+/. */}
    </>
  );
}

export default E2IChatSidebar;
