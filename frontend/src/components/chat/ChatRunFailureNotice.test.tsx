/**
 * ChatRunFailureNotice Component Tests
 * ====================================
 *
 * Covers issue #1340 UI-D1: an aborted/failed agent run (the od9uob3
 * "silent dead turn" — POST /api/copilotkit killed client-side with no
 * progress card, no answer, no error message) must surface an error bubble
 * with a retry affordance instead of silently vanishing. Retry re-sends the
 * dangling user message through the normal sendMessage path (full-history
 * resend model) without duplicating the user bubble.
 */

import { describe, it, expect, vi, beforeEach, afterEach, type Mock } from 'vitest';
import { render, screen, fireEvent, act } from '@testing-library/react';

type ChatMessage = { id: string; role: string; content?: unknown };

type Subscriber = {
  onRunInitialized?: (params: Record<string, unknown>) => void;
  onRunFailed?: (params: { error?: Error }) => void;
  onRunErrorEvent?: (params: { event?: { message?: string } }) => void;
};

// Plain stand-in agent (no auto-mocks: subscription bookkeeping must be real).
function makeFakeAgent() {
  const subscribers: Subscriber[] = [];
  return {
    subscribe(subscriber: Subscriber) {
      subscribers.push(subscriber);
      return {
        unsubscribe: () => {
          const index = subscribers.indexOf(subscriber);
          if (index >= 0) subscribers.splice(index, 1);
        },
      };
    },
    emit(name: keyof Subscriber, params: Record<string, unknown>) {
      for (const subscriber of [...subscribers]) {
        subscriber[name]?.(params as never);
      }
    },
    subscriberCount() {
      return subscribers.length;
    },
  };
}

const harness = vi.hoisted(() => ({
  chat: null as null | {
    messages: { id: string; role: string; content?: unknown }[];
    isLoading: boolean;
    sendMessage: (message: unknown) => Promise<void>;
    deleteMessage: (messageId: string) => void;
    agent: unknown;
  },
}));

vi.mock('@copilotkit/react-core', () => ({
  useCopilotChatInternal: () => harness.chat,
}));

import { ChatRunFailureNotice, DEAD_TURN_GRACE_MS } from './ChatRunFailureNotice';

describe('ChatRunFailureNotice (UI-D1, #1340)', () => {
  let agent: ReturnType<typeof makeFakeAgent>;
  let sendMessage: Mock<(message: unknown) => Promise<void>>;
  let deleteMessage: Mock<(messageId: string) => void>;

  function setChat(messages: ChatMessage[], isLoading = false) {
    harness.chat = { messages, isLoading, sendMessage, deleteMessage, agent };
  }

  beforeEach(() => {
    vi.useFakeTimers();
    agent = makeFakeAgent();
    sendMessage = vi.fn<(message: unknown) => Promise<void>>(() => Promise.resolve());
    deleteMessage = vi.fn<(messageId: string) => void>();
  });

  afterEach(() => {
    vi.useRealTimers();
  });

  it('surfaces an error bubble with a retry affordance when the run fails', () => {
    setChat([
      { id: 'u1', role: 'user', content: 'Compare TRx across brands' },
    ]);
    render(<ChatRunFailureNotice />);

    expect(screen.queryByRole('alert')).not.toBeInTheDocument();

    act(() => {
      agent.emit('onRunFailed', { error: new Error('net::ERR_ABORTED') });
    });

    expect(screen.getByRole('alert')).toBeInTheDocument();
    expect(screen.getByRole('button', { name: /retry/i })).toBeInTheDocument();
    expect(screen.getByText(/net::ERR_ABORTED/)).toBeInTheDocument();
  });

  it('surfaces a silent dead turn (no run-failed event) after the grace period', () => {
    setChat([
      { id: 'u1', role: 'user', content: 'hello' },
      { id: 'a1', role: 'assistant', content: 'hi' },
      { id: 'u2', role: 'user', content: 'Compare TRx by quarter' },
    ]);
    render(<ChatRunFailureNotice />);

    expect(screen.queryByRole('alert')).not.toBeInTheDocument();

    act(() => {
      vi.advanceTimersByTime(DEAD_TURN_GRACE_MS + 100);
    });

    expect(screen.getByRole('alert')).toBeInTheDocument();
    expect(screen.getByRole('button', { name: /retry/i })).toBeInTheDocument();
  });

  it('shows nothing while a run is in flight', () => {
    setChat(
      [{ id: 'u1', role: 'user', content: 'Compare TRx by quarter' }],
      true
    );
    render(<ChatRunFailureNotice />);

    act(() => {
      vi.advanceTimersByTime(DEAD_TURN_GRACE_MS * 3);
    });

    expect(screen.queryByRole('alert')).not.toBeInTheDocument();
  });

  it('shows nothing when the last turn completed normally', () => {
    setChat([
      { id: 'u1', role: 'user', content: 'hello' },
      { id: 'a1', role: 'assistant', content: 'full answer' },
    ]);
    render(<ChatRunFailureNotice />);

    act(() => {
      vi.advanceTimersByTime(DEAD_TURN_GRACE_MS * 3);
    });

    expect(screen.queryByRole('alert')).not.toBeInTheDocument();
  });

  it('retry re-sends the dangling user message without duplicating the bubble', async () => {
    setChat([
      { id: 'u1', role: 'user', content: 'hello' },
      { id: 'a1', role: 'assistant', content: 'hi' },
      { id: 'u2', role: 'user', content: 'Compare TRx by quarter' },
    ]);
    render(<ChatRunFailureNotice />);

    act(() => {
      agent.emit('onRunFailed', { error: new Error('net::ERR_ABORTED') });
    });

    await act(async () => {
      fireEvent.click(screen.getByRole('button', { name: /retry/i }));
    });

    // The dangling user message is removed before the re-send: one user
    // bubble, and the follow-up run resends the full (settled) history.
    expect(deleteMessage).toHaveBeenCalledTimes(1);
    expect(deleteMessage).toHaveBeenCalledWith('u2');
    expect(sendMessage).toHaveBeenCalledTimes(1);
    expect(sendMessage).toHaveBeenCalledWith(
      expect.objectContaining({ role: 'user', content: 'Compare TRx by quarter' })
    );
    // Completed turns are never touched.
    expect(deleteMessage).not.toHaveBeenCalledWith('u1');
    expect(deleteMessage).not.toHaveBeenCalledWith('a1');
    expect(screen.queryByRole('alert')).not.toBeInTheDocument();
  });

  it('retry also clears a partial assistant fragment left by an interrupted stream', async () => {
    setChat([
      { id: 'u1', role: 'user', content: 'Compare TRx by quarter' },
      { id: 'a1', role: 'assistant', content: 'Partial ans' },
    ]);
    render(<ChatRunFailureNotice />);

    act(() => {
      agent.emit('onRunFailed', { error: new Error('stream interrupted') });
    });

    await act(async () => {
      fireEvent.click(screen.getByRole('button', { name: /retry/i }));
    });

    expect(deleteMessage).toHaveBeenCalledWith('a1');
    expect(deleteMessage).toHaveBeenCalledWith('u1');
    expect(sendMessage).toHaveBeenCalledWith(
      expect.objectContaining({ role: 'user', content: 'Compare TRx by quarter' })
    );
  });

  it('clears the notice when a new run starts', () => {
    setChat([{ id: 'u1', role: 'user', content: 'Compare TRx by quarter' }]);
    render(<ChatRunFailureNotice />);

    act(() => {
      agent.emit('onRunFailed', { error: new Error('net::ERR_ABORTED') });
    });
    expect(screen.getByRole('alert')).toBeInTheDocument();

    act(() => {
      agent.emit('onRunInitialized', {});
    });
    expect(screen.queryByRole('alert')).not.toBeInTheDocument();
  });

  it('can be dismissed', () => {
    setChat([{ id: 'u1', role: 'user', content: 'Compare TRx by quarter' }]);
    render(<ChatRunFailureNotice />);

    act(() => {
      agent.emit('onRunFailed', { error: new Error('net::ERR_ABORTED') });
    });
    expect(screen.getByRole('alert')).toBeInTheDocument();

    fireEvent.click(screen.getByRole('button', { name: /dismiss/i }));
    expect(screen.queryByRole('alert')).not.toBeInTheDocument();
  });

  it('double-clicking Retry sends exactly one re-send with a deterministic delete set', async () => {
    setChat([
      { id: 'u1', role: 'user', content: 'hello' },
      { id: 'a1', role: 'assistant', content: 'hi' },
      { id: 'u2', role: 'user', content: 'Compare TRx by quarter' },
    ]);
    render(<ChatRunFailureNotice />);

    act(() => {
      agent.emit('onRunFailed', { error: new Error('net::ERR_ABORTED') });
    });

    const retryButton = screen.getByRole('button', { name: /retry/i });
    await act(async () => {
      // Two clicks land before the first retry's async send resolves.
      fireEvent.click(retryButton);
      fireEvent.click(retryButton);
    });

    expect(sendMessage).toHaveBeenCalledTimes(1);
    expect(deleteMessage).toHaveBeenCalledTimes(1);
    expect(deleteMessage).toHaveBeenCalledWith('u2');
  });

  it('hides the notice when the recorded failure no longer matches the transcript tail', () => {
    // Failure captured while the dangling turn was u1...
    setChat([{ id: 'u1', role: 'user', content: 'first question' }]);
    const { rerender } = render(<ChatRunFailureNotice />);
    act(() => {
      agent.emit('onRunFailed', { error: new Error('net::ERR_ABORTED') });
    });
    expect(screen.getByRole('alert')).toBeInTheDocument();

    // ...but the transcript has since moved on to completed exchanges the
    // notice's run events never saw (state resync). Offering retry here
    // would delete a valid completed exchange — the stale notice must go.
    setChat([
      { id: 'u1', role: 'user', content: 'first question' },
      { id: 'a1', role: 'assistant', content: 'answer one' },
      { id: 'u2', role: 'user', content: 'second question' },
      { id: 'a2', role: 'assistant', content: 'answer two' },
    ]);
    rerender(<ChatRunFailureNotice />);

    expect(screen.queryByRole('alert')).not.toBeInTheDocument();
  });

  it('hides the notice when a completed reply to the failed turn arrives without run events', () => {
    // Failure recorded for user turn u1 while it was dangling...
    setChat([{ id: 'u1', role: 'user', content: 'Compare TRx by quarter' }]);
    const { rerender } = render(<ChatRunFailureNotice />);
    act(() => {
      agent.emit('onRunFailed', { error: new Error('net::ERR_ABORTED') });
    });
    expect(screen.getByRole('alert')).toBeInTheDocument();

    // ...then an assistant reply to that SAME turn arrives via event-less
    // recovery (resync) — no onRunInitialized ever fires. u1 is still the
    // last user message, but the turn is now answered: the notice must hide
    // rather than offer a retry that would delete the completed reply.
    setChat([
      { id: 'u1', role: 'user', content: 'Compare TRx by quarter' },
      { id: 'a1', role: 'assistant', content: 'Here is the full comparison.' },
    ]);
    rerender(<ChatRunFailureNotice />);

    expect(screen.queryByRole('alert')).not.toBeInTheDocument();
  });

  it('retry aborts with zero mutations when a reply to the failed turn arrived in the store', async () => {
    // Same scenario, but the reply lands via in-place store mutation with no
    // re-render, so the stale notice is still on screen when the click hits:
    // the pre-mutation revalidation must abort — completed turns are never
    // deleted, even when they answer the failed turn itself.
    const messages: ChatMessage[] = [
      { id: 'u1', role: 'user', content: 'Compare TRx by quarter' },
    ];
    setChat(messages);
    render(<ChatRunFailureNotice />);
    act(() => {
      agent.emit('onRunFailed', { error: new Error('net::ERR_ABORTED') });
    });
    const retryButton = screen.getByRole('button', { name: /retry/i });

    messages.push({
      id: 'a1',
      role: 'assistant',
      content: 'Here is the full comparison.',
    });

    await act(async () => {
      fireEvent.click(retryButton);
    });

    expect(deleteMessage).not.toHaveBeenCalled();
    expect(sendMessage).not.toHaveBeenCalled();
  });

  it('retry aborts cleanly when the failed turn no longer matches the current transcript', async () => {
    const messages: ChatMessage[] = [
      { id: 'u1', role: 'user', content: 'hello' },
      { id: 'a1', role: 'assistant', content: 'hi' },
      { id: 'u2', role: 'user', content: 'Compare TRx by quarter' },
    ];
    setChat(messages);
    render(<ChatRunFailureNotice />);

    act(() => {
      agent.emit('onRunFailed', { error: new Error('net::ERR_ABORTED') });
    });
    const retryButton = screen.getByRole('button', { name: /retry/i });

    // The message store mutates in place (no re-render) before the click
    // lands: the failed turn u2 is gone, replaced by a different exchange.
    messages.splice(
      2,
      1,
      { id: 'u9', role: 'user', content: 'different question' },
      { id: 'a9', role: 'assistant', content: 'completed answer' }
    );

    await act(async () => {
      fireEvent.click(retryButton);
    });

    // Clean abort: never delete or re-send turns the failure does not own.
    expect(deleteMessage).not.toHaveBeenCalled();
    expect(sendMessage).not.toHaveBeenCalled();
  });

  it('unsubscribes from the agent on unmount', () => {
    setChat([{ id: 'u1', role: 'user', content: 'hello' }]);
    const { unmount } = render(<ChatRunFailureNotice />);
    expect(agent.subscriberCount()).toBeGreaterThan(0);
    unmount();
    expect(agent.subscriberCount()).toBe(0);
  });
});
