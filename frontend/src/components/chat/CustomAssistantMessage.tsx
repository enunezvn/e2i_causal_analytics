/**
 * Custom Assistant Message Component
 * ===================================
 *
 * Custom implementation of CopilotKit's AssistantMessage that fixes
 * the multiple action buttons issue during streaming.
 *
 * Root cause: Default CopilotKit AssistantMessage only checks `isLoading`
 * to hide the toolbar, but during streaming `isGenerating` is true while
 * `isLoading` is false. This causes the toolbar to render on each streaming
 * emit, creating multiple stacked action button rows.
 *
 * Fix: Hide toolbar when either `isLoading` OR `isGenerating` is true.
 *
 * @module components/chat/CustomAssistantMessage
 */

import { useState } from 'react';
import {
  AssistantMessageProps,
  Markdown,
  useChatContext,
} from '@copilotkit/react-ui';

/**
 * CustomAssistantMessage wraps the CopilotKit assistant message with proper
 * toolbar visibility control during streaming.
 *
 * The toolbar (Regenerate, Copy, Thumbs up, Thumbs down) is only shown when:
 * - There is actual message content
 * - The LLM is NOT loading (thinking)
 * - The LLM is NOT generating (streaming)
 */
export function CustomAssistantMessage(props: AssistantMessageProps) {
  const { icons, labels } = useChatContext();
  const {
    message,
    isLoading,
    isGenerating,
    onRegenerate,
    onCopy,
    onThumbsUp,
    onThumbsDown,
    isCurrentMessage,
    feedback,
    markdownTagRenderers,
  } = props;

  const [copied, setCopied] = useState(false);

  const content = message?.content || '';
  const subComponent = message?.generativeUI?.();

  // KEY FIX: Only show toolbar when NOT loading AND NOT generating (streaming complete)
  // The default CopilotKit AssistantMessage only checks `!isLoading`, missing `!isGenerating`
  const showToolbar = content && !isLoading && !isGenerating;

  const handleCopy = () => {
    if (content) {
      navigator.clipboard.writeText(content);
      setCopied(true);
      if (onCopy) {
        onCopy(content);
      }
      setTimeout(() => setCopied(false), 2000);
    }
  };

  const handleRegenerate = () => {
    if (onRegenerate) {
      onRegenerate();
    }
  };

  const handleThumbsUp = () => {
    if (onThumbsUp && message) {
      onThumbsUp(message);
    }
  };

  const handleThumbsDown = () => {
    if (onThumbsDown && message) {
      onThumbsDown(message);
    }
  };

  const LoadingIcon = () => <span>{icons.activityIcon}</span>;

  return (
    <>
      {content && (
        <div className="copilotKitMessage copilotKitAssistantMessage">
          {/* Use CopilotKit's Markdown component for proper rendering */}
          <Markdown content={content} components={markdownTagRenderers} />

          {/* Action toolbar - ONLY show when streaming is complete */}
          {showToolbar && (
            <div
              className={`copilotKitMessageControls ${isCurrentMessage ? 'currentMessage' : ''}`}
            >
              <button
                className="copilotKitMessageControlButton"
                onClick={handleRegenerate}
                aria-label={labels.regenerateResponse}
                title={labels.regenerateResponse}
              >
                {icons.regenerateIcon}
              </button>

              <button
                className="copilotKitMessageControlButton"
                onClick={handleCopy}
                aria-label={labels.copyToClipboard}
                title={labels.copyToClipboard}
              >
                {copied ? (
                  <span style={{ fontSize: '10px', fontWeight: 'bold' }}>✓</span>
                ) : (
                  icons.copyIcon
                )}
              </button>

              {onThumbsUp && (
                <button
                  className={`copilotKitMessageControlButton ${
                    feedback === 'thumbsUp' ? 'active' : ''
                  }`}
                  onClick={handleThumbsUp}
                  aria-label={labels.thumbsUp}
                  title={labels.thumbsUp}
                >
                  {icons.thumbsUpIcon}
                </button>
              )}

              {onThumbsDown && (
                <button
                  className={`copilotKitMessageControlButton ${
                    feedback === 'thumbsDown' ? 'active' : ''
                  }`}
                  onClick={handleThumbsDown}
                  aria-label={labels.thumbsDown}
                  title={labels.thumbsDown}
                >
                  {icons.thumbsDownIcon}
                </button>
              )}
            </div>
          )}
        </div>
      )}

      {/* Render any generative UI sub-component */}
      {subComponent && <div style={{ marginBottom: '0.5rem' }}>{subComponent}</div>}

      {/* Loading indicator */}
      {isLoading && <LoadingIcon />}
    </>
  );
}

export default CustomAssistantMessage;
