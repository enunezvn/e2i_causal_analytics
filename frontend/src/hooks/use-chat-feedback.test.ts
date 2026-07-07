/**
 * useChatFeedback — payload contract tests.
 *
 * Regression guard for the message-identification fix: the live CopilotKit
 * stream only knows its AG-UI uuid, so the hook must NOT send a fabricated
 * numeric message_id (the old sidebar derived one via parseInt(uuid), which
 * could attach feedback to a row from a DIFFERENT session). Instead it sends
 * session_id + response_preview (+ message_uuid for tracing) and lets the
 * server resolve the DB row; an explicit dbMessageId is passed through only
 * when the surface genuinely knows it.
 */

import { describe, it, expect, vi, beforeEach } from 'vitest';
import { renderHook, act } from '@testing-library/react';

vi.mock('@/lib/api-client', () => ({
  post: vi.fn(),
  ApiError: class ApiError extends Error {},
}));

import { post } from '@/lib/api-client';
import { useChatFeedback } from './use-chat-feedback';

const postMock = post as ReturnType<typeof vi.fn>;

describe('useChatFeedback', () => {
  beforeEach(() => {
    vi.clearAllMocks();
    postMock.mockResolvedValue({ success: true, feedback_id: 1 });
  });

  it('live-chat submission sends session + content + uuid, and NO fabricated message_id or agent label', async () => {
    const { result } = renderHook(() => useChatFeedback());

    await act(async () => {
      await result.current.submitFeedback({
        messageKey: 'ag-ui-uuid-1',
        messageUuid: 'ag-ui-uuid-1',
        sessionId: 'thread-uuid',
        rating: 'thumbs_up',
        responsePreview: 'The TRx performance...',
        responseText: 'The TRx performance... full rated response text',
      });
    });

    const [url, body] = postMock.mock.calls[0];
    expect(url).toBe('/copilotkit/feedback');
    expect(body.message_id).toBeUndefined();
    expect(body.message_uuid).toBe('ag-ui-uuid-1');
    expect(body.session_id).toBe('thread-uuid');
    expect(body.response_preview).toBe('The TRx performance...');
    // Full text lets the server resolve by exact content match (two responses
    // can share a 500-char prefix).
    expect(body.response_text).toBe('The TRx performance... full rated response text');
    // Attribution is server-derived from the matched row — the sidebar no
    // longer dictates agent_name (it used to hardcode 'copilotkit').
    expect(body.agent_name).toBeUndefined();
  });

  it('persisted-history submission passes the known DB id through', async () => {
    const { result } = renderHook(() => useChatFeedback());

    await act(async () => {
      await result.current.submitFeedback({
        messageKey: '123',
        dbMessageId: 123,
        sessionId: 'session-1',
        rating: 'thumbs_down',
      });
    });

    expect(postMock.mock.calls[0][1].message_id).toBe(123);
  });

  it('keys local rated-state by messageKey (string uuid)', async () => {
    const { result } = renderHook(() => useChatFeedback());

    await act(async () => {
      await result.current.submitFeedback({
        messageKey: 'ag-ui-uuid-2',
        sessionId: 'thread-uuid',
        rating: 'thumbs_up',
        responsePreview: 'x',
      });
    });

    expect(result.current.hasRated('ag-ui-uuid-2')).toBe(true);
    expect(result.current.getRating('ag-ui-uuid-2')).toBe('thumbs_up');
    expect(result.current.hasRated('other')).toBe(false);
  });

  it('does not record a rating when the server rejects the resolution', async () => {
    postMock.mockResolvedValue({ success: false, error: 'No persisted assistant message' });
    const { result } = renderHook(() => useChatFeedback());

    await act(async () => {
      const res = await result.current.submitFeedback({
        messageKey: 'k1',
        sessionId: 'thread-uuid',
        rating: 'thumbs_up',
        responsePreview: 'y',
      });
      expect(res.success).toBe(false);
    });

    expect(result.current.hasRated('k1')).toBe(false);
    expect(result.current.state.error).toContain('No persisted assistant message');
  });
});
