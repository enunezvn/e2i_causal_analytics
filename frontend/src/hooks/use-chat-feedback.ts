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
 * @module hooks/use-chat-feedback
 */

import { useState, useCallback } from 'react';

// =============================================================================
// TYPES
// =============================================================================

export type FeedbackRating = 'thumbs_up' | 'thumbs_down';

export interface FeedbackSubmission {
  messageId: number;
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
  feedbackId?: number;
  message?: string;
  error?: string;
}

export interface FeedbackState {
  /** Map of messageId -> rating */
  ratings: Record<number, FeedbackRating>;
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
  getRating: (messageId: number) => FeedbackRating | undefined;
  /** Check if a message has been rated */
  hasRated: (messageId: number) => boolean;
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
 * function ChatMessage({ messageId, sessionId, content }) {
 *   const { submitFeedback, getRating, hasRated } = useChatFeedback();
 *
 *   const handleThumbsUp = async () => {
 *     await submitFeedback({
 *       messageId,
 *       sessionId,
 *       rating: 'thumbs_up',
 *       responsePreview: content.substring(0, 500),
 *     });
 *   };
 *
 *   return (
 *     <div>
 *       <p>{content}</p>
 *       {!hasRated(messageId) && (
 *         <button onClick={handleThumbsUp}>👍</button>
 *       )}
 *       {getRating(messageId) === 'thumbs_up' && <span>Thanks!</span>}
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
        const response = await fetch('/api/copilotkit/feedback', {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({
            message_id: feedback.messageId,
            session_id: feedback.sessionId,
            rating: feedback.rating,
            comment: feedback.comment,
            query_text: feedback.queryText,
            response_preview: feedback.responsePreview,
            agent_name: feedback.agentName,
            tools_used: feedback.toolsUsed,
          }),
        });

        const result: FeedbackResult = await response.json();

        if (result.success) {
          // Update local state with the new rating
          setState((prev) => ({
            ...prev,
            ratings: {
              ...prev.ratings,
              [feedback.messageId]: feedback.rating,
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
        const errorMessage =
          error instanceof Error ? error.message : 'Network error';
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
    (messageId: number): FeedbackRating | undefined => {
      return state.ratings[messageId];
    },
    [state.ratings]
  );

  const hasRated = useCallback(
    (messageId: number): boolean => {
      return messageId in state.ratings;
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
