/**
 * Chat Feedback Hook
 * ==================
 *
 * Provides functionality to submit thumbs up/down feedback for chatbot messages.
 * Works with the /api/copilotkit/feedback endpoint.
 *
 * Since CopilotKit doesn't have built-in feedback support yet (Issue #1150),
 * this hook provides a custom solution for collecting user feedback.
 *
 * Updated: Uses apiClient instead of raw fetch() to include auth headers
 * (Phase 1 System Evaluation - Data Flow issue fix)
 *
 * @module hooks/use-chat-feedback
 */

import { useState, useCallback } from 'react';
import { post, ApiError } from '@/lib/api-client';

// =============================================================================
// TYPES
// =============================================================================

export type FeedbackRating = 'thumbs_up' | 'thumbs_down';

export interface FeedbackSubmission {
  /**
   * Client-side key for local rated-state (any stable string — the CopilotKit
   * message uuid for live chat, or String(dbMessageId) for persisted history).
   */
  messageKey: string;
  /**
   * The chatbot_messages.id DB key, when the surface genuinely knows it
   * (persisted-history views). Live CopilotKit messages do NOT know it — omit
   * and the backend resolves the row by sessionId + responsePreview instead.
   * (Never derive this from the CopilotKit uuid: parseInt on a uuid yields a
   * leading-digit fragment that can collide with a real row from a different
   * session.)
   */
  dbMessageId?: number;
  /** CopilotKit AG-UI message uuid, stored server-side for tracing. */
  messageUuid?: string;
  /** CopilotKit threadId — the session key the backend persists messages under. */
  sessionId: string;
  rating: FeedbackRating;
  comment?: string;
  queryText?: string;
  responsePreview?: string;
  agentName?: string;
  toolsUsed?: string[];
}

export interface FeedbackResult {
  success: boolean;
  /** Backend canonical snake_case (copilotkit.py FeedbackResponse.feedback_id). */
  feedback_id?: number;
  message?: string;
  error?: string;
}

export interface FeedbackState {
  /** Map of messageKey -> rating */
  ratings: Record<string, FeedbackRating>;
  /** Whether a submission is in progress */
  isSubmitting: boolean;
  /** Last error message */
  error: string | null;
}

export interface UseChatFeedbackReturn {
  /** Current feedback state */
  state: FeedbackState;
  /** Submit feedback for a message */
  submitFeedback: (feedback: FeedbackSubmission) => Promise<FeedbackResult>;
  /** Get the rating for a specific message */
  getRating: (messageKey: string) => FeedbackRating | undefined;
  /** Check if a message has been rated */
  hasRated: (messageKey: string) => boolean;
  /** Clear all ratings (e.g., on session change) */
  clearRatings: () => void;
}

// =============================================================================
// HOOK
// =============================================================================

/**
 * Hook for managing chat feedback submissions.
 *
 * @example
 * ```tsx
 * function ChatMessage({ message, threadId }) {
 *   const { submitFeedback, getRating, hasRated } = useChatFeedback();
 *   const content = message.content;
 *
 *   const handleThumbsUp = async () => {
 *     await submitFeedback({
 *       messageKey: message.id,          // CopilotKit uuid — local state key
 *       messageUuid: message.id,
 *       sessionId: threadId,             // useCopilotContext().threadId
 *       rating: 'thumbs_up',
 *       responsePreview: content.substring(0, 500), // server resolves DB row
 *     });
 *   };
 *
 *   return (
 *     <div>
 *       <p>{content}</p>
 *       {!hasRated(message.id) && (
 *         <button onClick={handleThumbsUp}>👍</button>
 *       )}
 *       {getRating(message.id) === 'thumbs_up' && <span>Thanks!</span>}
 *     </div>
 *   );
 * }
 * ```
 */
export function useChatFeedback(): UseChatFeedbackReturn {
  const [state, setState] = useState<FeedbackState>({
    ratings: {},
    isSubmitting: false,
    error: null,
  });

  const submitFeedback = useCallback(
    async (feedback: FeedbackSubmission): Promise<FeedbackResult> => {
      setState((prev) => ({
        ...prev,
        isSubmitting: true,
        error: null,
      }));

      try {
        // Use apiClient post() instead of raw fetch() to include auth headers
        // (Phase 1 System Evaluation - Data Flow fix)
        const result = await post<FeedbackResult>('/copilotkit/feedback', {
          // Omitted (undefined) when only the CopilotKit uuid is known — the
          // backend then resolves the DB row by session + response prefix.
          message_id: feedback.dbMessageId,
          message_uuid: feedback.messageUuid,
          session_id: feedback.sessionId,
          rating: feedback.rating,
          comment: feedback.comment,
          query_text: feedback.queryText,
          response_preview: feedback.responsePreview,
          agent_name: feedback.agentName,
          tools_used: feedback.toolsUsed,
        });

        if (result.success) {
          // Update local state with the new rating
          setState((prev) => ({
            ...prev,
            ratings: {
              ...prev.ratings,
              [feedback.messageKey]: feedback.rating,
            },
            isSubmitting: false,
          }));
        } else {
          setState((prev) => ({
            ...prev,
            isSubmitting: false,
            error: result.error || 'Failed to submit feedback',
          }));
        }

        return result;
      } catch (error) {
        // Handle ApiError from api-client
        const errorMessage =
          error instanceof ApiError
            ? error.message
            : error instanceof Error
              ? error.message
              : 'Network error';
        setState((prev) => ({
          ...prev,
          isSubmitting: false,
          error: errorMessage,
        }));
        return {
          success: false,
          error: errorMessage,
        };
      }
    },
    []
  );

  const getRating = useCallback(
    (messageKey: string): FeedbackRating | undefined => {
      return state.ratings[messageKey];
    },
    [state.ratings]
  );

  const hasRated = useCallback(
    (messageKey: string): boolean => {
      return messageKey in state.ratings;
    },
    [state.ratings]
  );

  const clearRatings = useCallback(() => {
    setState({
      ratings: {},
      isSubmitting: false,
      error: null,
    });
  }, []);

  return {
    state,
    submitFeedback,
    getRating,
    hasRated,
    clearRatings,
  };
}

export default useChatFeedback;
