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
import {
  useE2ICopilot,
  useCopilotEnabled,
  type AgentInfo,
  type E2IFilters,
} from '@/providers/E2ICopilotProvider';
import { useResizablePanel } from '@/hooks/use-resizable-panel';
import { useUIStore } from '@/stores/ui-store';
import { AgentStatusPanel } from './AgentStatusPanel';
import { AgentProgressRenderer } from './AgentProgressRenderer';
import { ChatRunFailureNotice } from './ChatRunFailureNotice';
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
  /**
   * Default width of the sidebar (px value; the user can drag-resize from there).
   * See the default below for why it is 480px rather than a rounder 400px.
   */
  width?: string;
  /** Show agent status panel */
  showAgentStatus?: boolean;
  /** Additional CSS classes */
  className?: string;
}

// =============================================================================
// SUGGESTIONS
// =============================================================================

export type ChatSuggestion = { title: string; message: string };

/**
 * Suggestion pills rendered above the input. Three tiers:
 *
 * 1. OPENER (empty conversation): when the pane is open before the user has
 *    asked anything, ConversationSuggestions asks POST /chat/suggestions
 *    (messages: []) for opener questions grounded in what the page is
 *    showing — pages publish a compact data summary via usePageChatContext,
 *    forwarded as page_context. Debounced so a page still loading its data
 *    gets to publish its richer summary before the call goes out. (This
 *    intentionally reverses the 2026-07-09 empty-until-first-user-message
 *    gate — decided 2026-07-10: openers grounded in live page content are
 *    the point. Do not re-add the gate.)
 * 2. PER-TURN (conversation exists): after each completed assistant turn,
 *    the same endpoint generates conversation-adaptive follow-ups from the
 *    recent transcript (one fast-tier LLM call, no orchestrator).
 * 3. STATIC FALLBACK + TOP-UP: the route+brand template set below shows
 *    instantly while a generation is in flight and whenever generation fails
 *    (502) — never a blank pill row, never invented output. Since 2026-09-05
 *    the backend post-filters generated pills against the assistant's
 *    capability catalog and may return fewer than four; topUpChatSuggestions
 *    fills the row back up from the same static set.
 *
 * A suggestions array bypasses CopilotKit's suggestion engine and is passed
 * through reactively, so pills swap live as new generations arrive.
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
 * chatbot_tools.py and the capability catalog in
 * src/services/chat_capability_catalog.py); the chart pills route through the
 * renderKpiTrend / renderChart generative-UI actions so users discover inline
 * visuals. The page summary a page publishes via usePageChatContext reaches
 * BOTH the pill endpoint (page_context) and the agent (readable #5 in
 * E2ICopilotProvider), so pills may refer to on-screen values.
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

// Exported for tests (#1749): the static pills are the guaranteed floor of
// the suggestion surface, so they must respect the dashboard's brand
// selection — and never render the 'All' sentinel as if it were a brand.
export function buildChatSuggestions(
  pathname: string,
  brand: E2IFilters['brand']
): ChatSuggestion[] {
  return [
    { title: '📈 Chart the TRx trend', message: 'Chart the TRx trend' },
    brand === 'All'
      ? {
          title: '📊 Compare brand market share',
          message:
            'Compare the market share trend for Remibrutinib, Fabhalta, and Kisqali in a table.',
        }
      : {
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
 * Top the adaptive pills back up to four with the static route+brand set.
 *
 * The backend post-filters generated pills against the assistant's
 * capability catalog (2026-09-05) and may return fewer than four; the static
 * pills are the guaranteed floor, so they fill the gap. Adaptive pills come
 * first, duplicates (case-insensitive title OR message) are skipped, never
 * more than four.
 * Exported for tests.
 */
export function topUpChatSuggestions(
  adaptive: ChatSuggestion[] | null,
  pathname: string,
  brand: E2IFilters['brand']
): ChatSuggestion[] {
  const statics = buildChatSuggestions(pathname, brand);
  if (!adaptive || adaptive.length === 0) return statics;
  const out = adaptive.slice(0, 4);
  const norm = (s: string) => s.trim().toLowerCase();
  const seenTitles = new Set(out.map((p) => norm(p.title)));
  const seenMessages = new Set(out.map((p) => norm(p.message)));
  for (const pill of statics) {
    if (out.length >= 4) break;
    if (seenTitles.has(norm(pill.title)) || seenMessages.has(norm(pill.message))) continue;
    seenTitles.add(norm(pill.title));
    seenMessages.add(norm(pill.message));
    out.push(pill);
  }
  return out;
}

/**
 * Conversation probe: fetches suggestion pills from POST /chat/suggestions —
 * page-grounded openers while the conversation is empty (messages: [] plus
 * the page's published page_context), then conversation-adaptive follow-ups
 * after each completed assistant turn. Runs as a
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
 * side-effect-free. The opener/per-turn mode switch is keyed to USER
 * messages so the `labels.initial` greeting can't flip it. Renders nothing.
 */
function ConversationSuggestions({
  pathname,
  brand,
  pageContext,
  onUpdate,
}: {
  pathname: string;
  brand: E2IFilters['brand'];
  /** Compact on-screen data summary published by the current page (or null). */
  pageContext: string | null;
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

  const fetchPills = React.useCallback(
    // brand is omitted when 'All' (#1749): the sentinel is not a brand
    // constraint, and sending it alongside a page_context that names the
    // real selection fed the suggestions LLM contradictory signals.
    (body: {
      messages: Array<{ role: 'user' | 'assistant'; content: string }>;
      page: string;
      brand?: string;
      page_context?: string;
    }) => {
      const seq = ++requestSeqRef.current;
      post<{ suggestions: ChatSuggestion[] }>('/chat/suggestions', body)
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
    },
    []
  );

  // Opener mode: empty conversation → ask for openers grounded in the page's
  // published data summary. Debounced 800ms so a page that is still loading
  // publishes its richer summary before the call goes out; keyed so the same
  // (route, brand, context) is only fetched once — a context UPGRADE (data
  // finished loading) legitimately refetches, the pills improve.
  const openerKeyRef = React.useRef<string | null>(null);
  React.useEffect(() => {
    if (hasUserMessage) return;
    const key = `${pathname}|${brand}|${pageContext ?? ''}`;
    if (openerKeyRef.current === key) return;
    const timer = setTimeout(() => {
      openerKeyRef.current = key;
      fetchPills({
        messages: [],
        page: pathname,
        ...(brand === 'All' ? {} : { brand }),
        ...(pageContext ? { page_context: pageContext.slice(0, 4000) } : {}),
      });
    }, 800);
    return () => clearTimeout(timer);
  }, [hasUserMessage, pathname, brand, pageContext, fetchPills]);

  // Per-turn mode: refresh follow-ups after each completed assistant turn.
  React.useEffect(() => {
    if (isLoading || !hasUserMessage) return;
    const last = transcript[transcript.length - 1];
    if (!last || last.role !== 'assistant') return;
    const key = String(transcript.length);
    if (lastFetchKeyRef.current === key) return;
    lastFetchKeyRef.current = key;
    fetchPills({
      messages: transcript.slice(-12),
      page: pathname,
      ...(brand === 'All' ? {} : { brand }),
    });
  }, [isLoading, hasUserMessage, transcript, pathname, brand, fetchPills]);

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
  // 480px, not 400px, so a markdown table fits without horizontal scrolling.
  // The panel loses ~68px to chrome before markdown gets any width (the
  // messages list's own scrollbar, .copilotKitMessagesContainer's 24px side
  // padding, and the assistant bubble's 12px right padding), so the usable
  // column is much narrower than the panel. Measured against the 5-column
  // region x metric table the chat actually emits: 400px left 332px of column
  // against 377px of table (the ROI column sat off-screen), 460px was the exact
  // break-even with zero slack, and 480px gives 412px — enough for the table to
  // render at its natural 408px width. Overriding this only changes where the
  // drag starts; the panel is still resizable from 320px and the width the user
  // drags to persists in the UI store.
  width = '480px',
  showAgentStatus = true,
  className,
}: E2IChatSidebarProps) {
  const copilotEnabled = useCopilotEnabled();
  const { chatOpen, setChatOpen, agents, filters, pageChatContext } = useE2ICopilot();
  const { pathname } = useLocation();

  // Suggestion pills: the LLM-generated set (page-grounded openers before
  // the first user message, conversation-adaptive follow-ups after), with
  // the static route+brand template set shown instantly as fallback until a
  // generation lands or when generation fails. State is reported by the
  // ConversationSuggestions child, which can read conversation state.
  const [pillState, setPillState] = React.useState<{
    hasUserMessage: boolean;
    adaptive: ChatSuggestion[] | null;
  }>({ hasUserMessage: false, adaptive: null });
  const chatSuggestions = React.useMemo(() => {
    return topUpChatSuggestions(pillState.adaptive, pathname, filters.brand);
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
    defaultWidth: Number.parseInt(width, 10) || 480,
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
                pageContext={pageChatContext}
                onUpdate={setPillState}
              />
              <CopilotChat
                AssistantMessage={CustomAssistantMessage}
                onThumbsUp={handleThumbsUp}
                onThumbsDown={handleThumbsDown}
                instructions={`You are helping an analyst work with the E2I Causal Analytics platform.

Current context:
- Brand filter: ${filters.brand === 'All' ? 'All brands (no brand filter)' : filters.brand}
- Region filter: ${filters.region === 'All US' ? 'All US regions (no region filter)' : filters.region}
- Date range: ${filters.dateRange.start} to ${filters.dateRange.end}
${filters.territory ? `- Territory: ${filters.territory}` : ''}
${filters.hcpSegment ? `- HCP Segment: ${filters.hcpSegment}` : ''}

Active agents: ${liveAgents.filter(a => a.status === 'active').map(a => a.name).join(', ')}

Available actions:
- navigateTo: Navigate to any page
- setBrandFilter: Change brand filter
- setRegionFilter: Change region filter (US-Census regions)
- setDateRange: Set analytics date range
- highlightCausalPaths: Highlight paths on visualizations
- setDetailLevel: Adjust response complexity

Always fetch real data with the available tools before answering, cite the actual metric values and their source, and deliver rich, evidence-based strategic insight — what the numbers mean for the business and the concrete next action — not just the figure. Focus on pharmaceutical commercial analytics (TRx, NRx, market share, causal drivers).

Visual answers: whenever the answer involves a KPI — its value, its evolution over time, a period comparison, or several KPIs side by side — render a chart inline alongside your text. There are two chart actions and the choice is mechanical:

- renderKpiTrend — ONLY for a trend over time of the Rx-volume and commercial KPIs it names (trx, nrx, nbrx, trx_share, conversion_rate, roi, or their registry codes). Prefer it for those, it renders more cheaply.
- renderChart — EVERY other case. Any other registry KPI (data-quality, model-performance, trigger, brand, causal), any KPI you want at its current value rather than over time, several KPIs compared side by side, or any chart shape other than a line.

If a request does not clearly fit renderKpiTrend's list, use renderChart — it covers the whole registry and never needs you to know in advance whether a KPI has a monthly series: one with history is charted over time, a point-in-time KPI is charted at its current value, and two or more KPIs are compared side by side. Name KPIs for renderChart however reads best — a registry code (WS1-MP-001, CM-001), a short key (roc_auc, trigger_precision), or a display name ("Cross-source Match Rate"); all resolve. Omit chartType unless the user asks for a specific shape.

Both actions take the same scope arguments: nbrx and trx_share are tracked per brand only, so pass brand for those. For trx/nrx/nbrx the trend can be split by patient axis — compareBy 'severity' or 'lot' renders ONE chart with a line per severity tier / line of therapy, so for cross-segment comparisons make a single call with compareBy, never one call per tier; segment ('low'/'medium'/'high') or therapyLine ('0'-'3') charts one tier. Other KPIs have no per-tier series. Use markdown tables for multi-row numeric comparisons instead of prose lists of figures.`}
                labels={{
                  initial: 'How can I help you explore E2I analytics?',
                  placeholder: 'Ask about KPIs, agents, or insights...',
                }}
                suggestions={chatSuggestions}
                className="min-h-0 flex-1"
              />

              {/* Failed/aborted run notice (#1340 UI-D1): a run that dies
                  client-side (net::ERR_ABORTED) is swallowed by CopilotKit —
                  without this the user's message just sits there with no
                  answer and no error. Sits directly under the input, next to
                  the turn it refers to. Needs the CopilotKit provider (chat
                  hooks throw outside it), same as ConversationSuggestions. */}
              <ChatRunFailureNotice className="shrink-0 mx-3 mb-2" />
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
