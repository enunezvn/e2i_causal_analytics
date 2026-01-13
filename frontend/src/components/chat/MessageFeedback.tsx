/**
 * Message Feedback Component
 * ==========================
 *
 * Provides thumbs up/down feedback buttons for chat messages.
 * Integrates with the useChatFeedback hook and /api/copilotkit/feedback endpoint.
 *
 * @module components/chat/MessageFeedback
 */

import * as React from 'react';
import { ThumbsUp, ThumbsDown, Loader2, Check } from 'lucide-react';
import { cn } from '@/lib/utils';
import { Button } from '@/components/ui/button';
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from '@/components/ui/tooltip';
import { useChatFeedback, FeedbackRating } from '@/hooks/use-chat-feedback';

// =============================================================================
// TYPES
// =============================================================================

export interface MessageFeedbackProps {
  /** Database message ID */
  messageId: number;
  /** Session ID for the conversation */
  sessionId: string;
  /** The user query that led to this response (for context) */
  queryText?: string;
  /** The response content (preview will be extracted) */
  responseContent?: string;
  /** Agent that generated the response */
  agentName?: string;
  /** Tools used to generate the response */
  toolsUsed?: string[];
  /** Size variant */
  size?: 'sm' | 'md';
  /** Additional CSS classes */
  className?: string;
  /** Callback when feedback is submitted */
  onFeedbackSubmit?: (rating: FeedbackRating, success: boolean) => void;
}

// =============================================================================
// COMPONENT
// =============================================================================

/**
 * MessageFeedback provides thumbs up/down buttons for rating assistant responses.
 *
 * @example
 * ```tsx
 * <MessageFeedback
 *   messageId={123}
 *   sessionId="user-uuid~timestamp"
 *   queryText="What is TRx performance?"
 *   responseContent="The TRx performance for Remibrutinib..."
 *   agentName="tool_composer"
 *   onFeedbackSubmit={(rating, success) => console.log(rating, success)}
 * />
 * ```
 */
export function MessageFeedback({
  messageId,
  sessionId,
  queryText,
  responseContent,
  agentName,
  toolsUsed,
  size = 'sm',
  className,
  onFeedbackSubmit,
}: MessageFeedbackProps) {
  const { submitFeedback, getRating, hasRated, state } = useChatFeedback();
  const currentRating = getRating(messageId);
  const isRated = hasRated(messageId);

  const handleFeedback = async (rating: FeedbackRating) => {
    const result = await submitFeedback({
      messageId,
      sessionId,
      rating,
      queryText,
      responsePreview: responseContent?.substring(0, 500),
      agentName,
      toolsUsed,
    });

    onFeedbackSubmit?.(rating, result.success);
  };

  const buttonSize = size === 'sm' ? 'h-7 w-7' : 'h-8 w-8';
  const iconSize = size === 'sm' ? 'h-3.5 w-3.5' : 'h-4 w-4';

  // If already rated, show confirmation
  if (isRated) {
    return (
      <div
        className={cn(
          'flex items-center gap-1 text-xs text-muted-foreground',
          className
        )}
      >
        <Check className="h-3 w-3 text-emerald-500" />
        <span>
          {currentRating === 'thumbs_up' ? 'Thanks!' : 'Feedback noted'}
        </span>
      </div>
    );
  }

  return (
    <div className={cn('flex items-center gap-1', className)}>
      <Tooltip>
        <TooltipTrigger asChild>
          <Button
            variant="ghost"
            size="icon"
            className={cn(buttonSize, 'hover:text-emerald-600 hover:bg-emerald-50')}
            onClick={() => handleFeedback('thumbs_up')}
            disabled={state.isSubmitting}
          >
            {state.isSubmitting ? (
              <Loader2 className={cn(iconSize, 'animate-spin')} />
            ) : (
              <ThumbsUp className={iconSize} />
            )}
          </Button>
        </TooltipTrigger>
        <TooltipContent>
          <p>Helpful response</p>
        </TooltipContent>
      </Tooltip>

      <Tooltip>
        <TooltipTrigger asChild>
          <Button
            variant="ghost"
            size="icon"
            className={cn(buttonSize, 'hover:text-red-600 hover:bg-red-50')}
            onClick={() => handleFeedback('thumbs_down')}
            disabled={state.isSubmitting}
          >
            <ThumbsDown className={iconSize} />
          </Button>
        </TooltipTrigger>
        <TooltipContent>
          <p>Not helpful</p>
        </TooltipContent>
      </Tooltip>
    </div>
  );
}

export default MessageFeedback;
