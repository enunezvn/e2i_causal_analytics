/**
 * Chat Run Failure Notice
 * =======================
 *
 * Surfaces aborted/failed copilot agent runs to the user (#1340 UI-D1, the
 * od9uob3 "silent dead turn"): when the POST /api/copilotkit run dies
 * client-side (net::ERR_ABORTED right after response headers), CopilotKit
 * 1.51.2 swallows the failure — sendMessage wraps runAgent in a try/catch
 * that only console.errors — so the user message stays in the chat with no
 * progress card, no answer, and no error message.
 *
 * Two independent detection signals, either one shows the notice:
 *
 * 1. Run lifecycle: subscribe to the AG-UI agent's onRunFailed /
 *    onRunErrorEvent. Covers failures that leave a partial assistant
 *    fragment as well as ones that leave nothing.
 * 2. Dead-turn heuristic: the transcript ends with a user message and no run
 *    is in flight, sustained past a short grace period. Covers any
 *    termination path that never emits a failure event (the captured
 *    od9uob3 signature).
 *
 * Retry re-sends through the normal sendMessage path: the dangling user
 * message (and any partial fragment the failed run left after it) is removed
 * first, then the same content is sent again — one user bubble, and the
 * follow-up run resends the full, by-then-settled history, consistent with
 * the by-design full-history-resend model. Completed turns are never touched.
 *
 * Safety invariants:
 * - The failed turn is identified by id AT FAILURE-DETECTION TIME
 *   (failedUserId). The notice only shows — and retry only acts — while
 *   that exact id is still the transcript's last user message; if the
 *   transcript has moved on, the stale notice hides rather than offering a
 *   retry that could delete turns the failure does not own.
 * - Retry is single-flight: a ref guard rejects re-entrant clicks, the
 *   button is disabled while a retry is in progress, and the transcript is
 *   revalidated against the live message store immediately before any
 *   delete/send.
 *
 * Must be mounted inside the CopilotKit provider (useCopilotChatInternal
 * throws outside it) — same constraint and same mount point as
 * ConversationSuggestions in E2IChatSidebar.
 *
 * @module components/chat/ChatRunFailureNotice
 */

import * as React from 'react';
import { useCopilotChatInternal } from '@copilotkit/react-core';
import { AlertCircle, RotateCcw } from 'lucide-react';
import { cn } from '@/lib/utils';
import { Button } from '@/components/ui/button';

/**
 * How long a "transcript ends with a user message and nothing is running"
 * state must persist before it is treated as a dead turn. Absorbs the normal
 * gap between message append and run start on a healthy send.
 */
export const DEAD_TURN_GRACE_MS = 1500;

// Minimal structural view of the AG-UI agent's subscription surface — the
// full AgentSubscriber type lives in @ag-ui/client, which is a transitive
// (undeclared) dependency; depend only on the slice we use.
interface RunLifecycleSubscriber {
  onRunInitialized?: () => void;
  onRunFailed?: (params: { error?: Error }) => void;
  onRunErrorEvent?: (params: { event?: { message?: string } }) => void;
}

interface SubscribableAgent {
  subscribe?: (subscriber: RunLifecycleSubscriber) => { unsubscribe: () => void };
}

export interface ChatRunFailureNoticeProps {
  /** Additional CSS classes */
  className?: string;
}

type MessageLike = { id: string; role: string; content?: unknown };

/** Index of the last user message, or -1. */
function lastUserIndexOf(list: readonly MessageLike[]): number {
  for (let i = list.length - 1; i >= 0; i--) {
    if (list[i].role === 'user') return i;
  }
  return -1;
}

function lastUserIdOf(list: readonly MessageLike[]): string | null {
  const index = lastUserIndexOf(list);
  return index === -1 ? null : list[index].id;
}

export function ChatRunFailureNotice({ className }: ChatRunFailureNoticeProps) {
  const { messages, isLoading, sendMessage, deleteMessage, agent } =
    useCopilotChatInternal();

  const [runError, setRunError] = React.useState<string | null>(null);
  const [deadTurn, setDeadTurn] = React.useState(false);
  const [dismissedKey, setDismissedKey] = React.useState<string | null>(null);
  // Id of the user message that anchored the failed turn, captured at
  // failure-detection time — the ONLY turn retry is allowed to touch.
  const [failedUserId, setFailedUserId] = React.useState<string | null>(null);
  const [isRetrying, setIsRetrying] = React.useState(false);
  const retryingRef = React.useRef(false);

  const list = React.useMemo(() => messages ?? [], [messages]);
  // Live view of the transcript for event callbacks and for revalidation
  // inside retry (the store can move between render and click).
  const messagesRef = React.useRef<readonly MessageLike[]>(list);
  messagesRef.current = list;

  // Signal 1: run lifecycle events from the AG-UI agent.
  React.useEffect(() => {
    const subscribable = agent as unknown as SubscribableAgent | undefined;
    if (!subscribable?.subscribe) return;
    const subscription = subscribable.subscribe({
      onRunInitialized: () => {
        // A new run is starting (fresh send or our retry) — stale notices
        // would misattribute themselves to it.
        setRunError(null);
        setDeadTurn(false);
        setDismissedKey(null);
        setFailedUserId(null);
      },
      onRunFailed: ({ error }) => {
        setRunError(error?.message || 'The request was interrupted.');
        setFailedUserId(lastUserIdOf(messagesRef.current));
      },
      onRunErrorEvent: ({ event }) => {
        setRunError(event?.message || 'The agent reported an error.');
        setFailedUserId(lastUserIdOf(messagesRef.current));
      },
    });
    return () => subscription.unsubscribe();
  }, [agent]);

  // Signal 2: dead-turn heuristic — the transcript ends with a user message
  // while nothing is running, sustained past the grace period.
  const last = list.length > 0 ? list[list.length - 1] : undefined;
  const danglingUserId = last && last.role === 'user' ? last.id : null;

  React.useEffect(() => {
    if (!danglingUserId || isLoading) {
      setDeadTurn(false);
      return;
    }
    const timer = setTimeout(() => {
      setDeadTurn(true);
      setFailedUserId(danglingUserId);
    }, DEAD_TURN_GRACE_MS);
    return () => clearTimeout(timer);
  }, [danglingUserId, isLoading]);

  const retry = React.useCallback(async () => {
    // Single-flight: reject re-entrant clicks (double-click lands before the
    // first retry's async send resolves).
    if (retryingRef.current) return;
    retryingRef.current = true;
    setIsRetrying(true);
    try {
      // Revalidate against the LIVE store, not the render snapshot: the
      // failed turn must still be the transcript's last user message,
      // otherwise abort without touching anything — turns the failure does
      // not own are never deleted.
      const current = messagesRef.current;
      const lastUserIndex = lastUserIndexOf(current);
      if (lastUserIndex === -1) return;
      const failed = current[lastUserIndex];
      if (failedUserId === null || failed.id !== failedUserId) return;
      const content = typeof failed.content === 'string' ? failed.content : '';
      if (!content) return;

      // Deterministic delete set, snapshotted before mutating: the failed
      // user message plus anything the failed run left after it (partial
      // assistant text, tool fragments). Completed prior turns untouched.
      const toDelete = current.slice(lastUserIndex).map((m) => m.id);
      for (let i = toDelete.length - 1; i >= 0; i--) {
        deleteMessage(toDelete[i]);
      }
      setRunError(null);
      setDeadTurn(false);
      setDismissedKey(null);
      setFailedUserId(null);
      // Re-send re-appends the one user bubble and triggers a run that
      // resends the full remaining (settled) history.
      await sendMessage({ id: failed.id, role: 'user', content });
    } finally {
      retryingRef.current = false;
      setIsRetrying(false);
    }
  }, [failedUserId, deleteMessage, sendMessage]);

  // The notice is only actionable — and only shown — while the recorded
  // failed turn is still the transcript's dangling tail. If the transcript
  // moved on (normally onRunInitialized already cleared us), the stale
  // notice hides instead of offering a retry against someone else's turns.
  const currentLastUserId = lastUserIdOf(list);
  const failureMatchesTail =
    failedUserId !== null && failedUserId === currentLastUserId;
  const noticeKey = `${failedUserId ?? 'none'}|${runError ?? ''}`;
  const visible =
    !isLoading &&
    (runError !== null || deadTurn) &&
    failureMatchesTail &&
    dismissedKey !== noticeKey;

  if (!visible) {
    return null;
  }

  return (
    <div
      role="alert"
      className={cn(
        'rounded-lg border border-rose-500/40 bg-rose-500/10 p-3 text-sm space-y-2',
        className
      )}
    >
      <div className="flex items-start gap-2">
        <AlertCircle className="h-4 w-4 mt-0.5 shrink-0 text-rose-500" />
        <div className="flex-1 space-y-0.5">
          <p className="font-medium">The assistant didn&apos;t respond</p>
          <p className="text-xs text-muted-foreground">
            {runError
              ? `The request was interrupted (${runError}).`
              : 'The request was interrupted before a response arrived.'}{' '}
            Your message is still in the conversation.
          </p>
        </div>
      </div>
      <div className="flex items-center gap-2 pl-6">
        <Button
          size="sm"
          variant="outline"
          onClick={retry}
          disabled={isRetrying}
          className="h-7 px-2 text-xs"
        >
          <RotateCcw className="h-3 w-3 mr-1" />
          Retry
        </Button>
        <Button
          size="sm"
          variant="ghost"
          onClick={() => setDismissedKey(noticeKey)}
          className="h-7 px-2 text-xs text-muted-foreground"
        >
          Dismiss
        </Button>
      </div>
    </div>
  );
}

export default ChatRunFailureNotice;
