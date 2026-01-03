/**
 * E2I Chat Popup Component
 * ========================
 *
 * Floating popup chat interface using CopilotKit.
 * Alternative to sidebar for quick interactions.
 *
 * Features:
 * - Keyboard shortcut (Cmd/Ctrl + /)
 * - Floating modal design
 * - Compact message history
 * - Quick dismiss on backdrop click
 *
 * @module components/chat/E2IChatPopup
 */

import * as React from 'react';
import { CopilotPopup } from '@copilotkit/react-ui';
import { motion, AnimatePresence } from 'framer-motion';
import { MessageSquare, X, Bot, Keyboard } from 'lucide-react';
import { cn } from '@/lib/utils';
import { Button } from '@/components/ui/button';
import { useE2ICopilot, useCopilotEnabled } from '@/providers/E2ICopilotProvider';

// =============================================================================
// TYPES
// =============================================================================

export interface E2IChatPopupProps {
  /** Position of the popup trigger button */
  position?: 'bottom-right' | 'bottom-left' | 'top-right' | 'top-left';
  /** Additional CSS classes */
  className?: string;
  /** Show keyboard shortcut hint */
  showShortcutHint?: boolean;
}

// =============================================================================
// COMPONENT
// =============================================================================

/**
 * E2IChatPopup provides a floating chat popup for quick AI interactions.
 *
 * @example
 * ```tsx
 * <E2IChatPopup position="bottom-right" showShortcutHint />
 * ```
 */
export function E2IChatPopup({
  position = 'bottom-right',
  className,
  showShortcutHint = true,
}: E2IChatPopupProps) {
  const copilotEnabled = useCopilotEnabled();
  const { chatOpen, setChatOpen, filters, agents } = useE2ICopilot();

  // Keyboard shortcut: Cmd/Ctrl + /
  React.useEffect(() => {
    if (!copilotEnabled) return;

    const handleKeyDown = (e: KeyboardEvent) => {
      if ((e.metaKey || e.ctrlKey) && e.key === '/') {
        e.preventDefault();
        setChatOpen((prev) => !prev);
      }
      // Escape to close
      if (e.key === 'Escape' && chatOpen) {
        setChatOpen(false);
      }
    };

    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [setChatOpen, chatOpen, copilotEnabled]);

  if (!copilotEnabled) {
    return null;
  }

  const positionClasses = {
    'bottom-right': 'bottom-6 right-6',
    'bottom-left': 'bottom-6 left-6',
    'top-right': 'top-6 right-6',
    'top-left': 'top-6 left-6',
  };

  const activeAgentCount = agents.filter(
    (a) => a.status === 'active' || a.status === 'processing'
  ).length;

  return (
    <div className={cn('fixed z-50', positionClasses[position], className)}>
      {/* Trigger Button */}
      <AnimatePresence>
        {!chatOpen && (
          <motion.div
            initial={{ opacity: 0, scale: 0.8 }}
            animate={{ opacity: 1, scale: 1 }}
            exit={{ opacity: 0, scale: 0.8 }}
            className="relative"
          >
            <Button
              onClick={() => setChatOpen(true)}
              size="lg"
              className="rounded-full shadow-lg h-14 w-14 bg-gradient-to-r from-blue-600 to-indigo-600 hover:from-blue-700 hover:to-indigo-700"
            >
              <MessageSquare className="h-6 w-6" />
            </Button>

            {/* Active agent indicator */}
            {activeAgentCount > 0 && (
              <span className="absolute -top-1 -right-1 bg-emerald-500 text-white text-xs rounded-full h-5 w-5 flex items-center justify-center animate-pulse">
                {activeAgentCount}
              </span>
            )}

            {/* Keyboard shortcut hint */}
            {showShortcutHint && (
              <motion.div
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.5 }}
                className="absolute -top-10 left-1/2 -translate-x-1/2 whitespace-nowrap"
              >
                <div className="flex items-center gap-1 text-xs bg-background/90 backdrop-blur px-2 py-1 rounded-md shadow-sm border">
                  <Keyboard className="h-3 w-3" />
                  <span className="text-muted-foreground">⌘/</span>
                </div>
              </motion.div>
            )}
          </motion.div>
        )}
      </AnimatePresence>

      {/* Popup */}
      <AnimatePresence>
        {chatOpen && (
          <>
            {/* Backdrop */}
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              onClick={() => setChatOpen(false)}
              className="fixed inset-0 z-40 bg-black/20 backdrop-blur-sm"
            />

            {/* Popup Container */}
            <motion.div
              initial={{ opacity: 0, scale: 0.95, y: 20 }}
              animate={{ opacity: 1, scale: 1, y: 0 }}
              exit={{ opacity: 0, scale: 0.95, y: 20 }}
              transition={{ type: 'spring', damping: 25, stiffness: 300 }}
              className={cn(
                'fixed z-50 w-[400px] h-[500px] rounded-xl shadow-2xl bg-background border overflow-hidden',
                positionClasses[position]
              )}
            >
              {/* Header */}
              <div className="flex items-center justify-between p-3 border-b bg-muted/50">
                <div className="flex items-center gap-2">
                  <div className="p-1.5 rounded-lg bg-gradient-to-r from-blue-600 to-indigo-600">
                    <Bot className="h-4 w-4 text-white" />
                  </div>
                  <div>
                    <h3 className="font-semibold text-sm">E2I Quick Chat</h3>
                    <p className="text-[10px] text-muted-foreground">
                      {filters.brand} | {activeAgentCount} active
                    </p>
                  </div>
                </div>
                <Button
                  variant="ghost"
                  size="icon"
                  onClick={() => setChatOpen(false)}
                  className="h-7 w-7"
                >
                  <X className="h-4 w-4" />
                </Button>
              </div>

              {/* Chat Area */}
              <div className="h-[calc(100%-56px)]">
                <CopilotPopup
                  defaultOpen={true}
                  clickOutsideToClose={false}
                  instructions={`You are helping an analyst with quick questions about E2I analytics.
Current brand: ${filters.brand}
Active agents: ${activeAgentCount}

Be brief and helpful. Focus on pharmaceutical commercial analytics.`}
                  labels={{
                    initial: 'Quick question?',
                    placeholder: 'Ask anything...',
                  }}
                  className="h-full"
                />
              </div>
            </motion.div>
          </>
        )}
      </AnimatePresence>
    </div>
  );
}

export default E2IChatPopup;
