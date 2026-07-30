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

export function ChatRunFailureNotice({ className }: ChatRunFailureNoticeProps) {
  const { messages, isLoading, sendMessage, deleteMessage, agent } =
    useCopilotChatInternal();

  const [runError, setRunError] = React.useState<string | null>(null);
  const [deadTurn, setDeadTurn] = React.useState(false);
  const [dismissedKey, setDismissedKey] = React.useState<string | null>(null);

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
      },
      onRunFailed: ({ error }) => {
        setRunError(error?.message || 'The request was interrupted.');
      },
      onRunErrorEvent: ({ event }) => {
        setRunError(event?.message || 'The agent reported an error.');
      },
    });
    return () => subscription.unsubscribe();
  }, [agent]);

  // Signal 2: dead-turn heuristic — the transcript ends with a user message
  // while nothing is running, sustained past the grace period.
  const list = React.useMemo(() => messages ?? [], [messages]);
  const last = list.length > 0 ? list[list.length - 1] : undefined;
  const danglingUserId = last && last.role === 'user' ? last.id : null;

  React.useEffect(() => {
    if (!danglingUserId || isLoading) {
      setDeadTurn(false);
      return;
    }
    const timer = setTimeout(() => setDeadTurn(true), DEAD_TURN_GRACE_MS);
    return () => clearTimeout(timer);
  }, [danglingUserId, isLoading]);

  const retry = React.useCallback(async () => {
    // Locate the dangling user turn: the last user message plus anything the
    // failed run left after it (partial assistant text, tool fragments).
    let lastUserIndex = -1;
    for (let i = list.length - 1; i >= 0; i--) {
      if (list[i].role === 'user') {
        lastUserIndex = i;
        break;
      }
    }
    if (lastUserIndex === -1) return;
    const failed = list[lastUserIndex];
    const content = typeof failed.content === 'string' ? failed.content : '';
    if (!content) return;

    // Remove the failed turn before re-sending so the user bubble is not
    // duplicated; sendMessage then re-appends it and triggers a run that
    // resends the full remaining (settled) history.
    for (let i = list.length - 1; i >= lastUserIndex; i--) {
      deleteMessage(list[i].id);
    }
    setRunError(null);
    setDeadTurn(false);
    setDismissedKey(null);
    await sendMessage({ id: failed.id, role: 'user', content });
  }, [list, deleteMessage, sendMessage]);

  const noticeKey = `${danglingUserId ?? 'none'}|${runError ?? ''}`;
  const visible =
    !isLoading && (runError !== null || deadTurn) && dismissedKey !== noticeKey;

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
        <Button size="sm" variant="outline" onClick={retry} className="h-7 px-2 text-xs">
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
