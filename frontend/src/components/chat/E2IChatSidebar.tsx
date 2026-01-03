/**
 * E2I Chat Sidebar Component
 * ==========================
 *
 * Sliding sidebar chat interface using CopilotKit.
 * Provides natural language interaction with E2I agents.
 *
 * Features:
 * - Collapsible sidebar panel
 * - Agent status indicators
 * - Message history
 * - Keyboard shortcut (Cmd/Ctrl + /)
 *
 * @module components/chat/E2IChatSidebar
 */

import * as React from 'react';
import { CopilotSidebar } from '@copilotkit/react-ui';
import { motion, AnimatePresence } from 'framer-motion';
import {
  MessageSquare,
  X,
  ChevronLeft,
  ChevronRight,
  Bot,
  Sparkles,
} from 'lucide-react';
import { cn } from '@/lib/utils';
import { Button } from '@/components/ui/button';
import { useE2ICopilot, useCopilotEnabled } from '@/providers/E2ICopilotProvider';
import { AgentStatusPanel } from './AgentStatusPanel';

// =============================================================================
// TYPES
// =============================================================================

export interface E2IChatSidebarProps {
  /** Default open state */
  defaultOpen?: boolean;
  /** Position of the sidebar */
  position?: 'left' | 'right';
  /** Width of the sidebar */
  width?: string;
  /** Show agent status panel */
  showAgentStatus?: boolean;
  /** Additional CSS classes */
  className?: string;
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
  width = '400px',
  showAgentStatus = true,
  className,
}: E2IChatSidebarProps) {
  const copilotEnabled = useCopilotEnabled();
  const { chatOpen, setChatOpen, agents, filters } = useE2ICopilot();
  const [showAgents, setShowAgents] = React.useState(false);

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

  // Count active agents
  const activeAgentCount = agents.filter((a) => a.status === 'active' || a.status === 'processing').length;

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
              'fixed inset-y-0 z-50 flex flex-col bg-background border-l shadow-xl',
              position === 'right' ? 'right-0' : 'left-0 border-l-0 border-r',
              className
            )}
            style={{ width }}
          >
            {/* Header */}
            <div className="flex items-center justify-between p-4 border-b bg-muted/50">
              <div className="flex items-center gap-3">
                <div className="p-2 rounded-lg bg-gradient-to-r from-blue-600 to-indigo-600">
                  <Bot className="h-5 w-5 text-white" />
                </div>
                <div>
                  <h2 className="font-semibold">E2I Assistant</h2>
                  <p className="text-xs text-muted-foreground">
                    {filters.brand} | {activeAgentCount} agents active
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
                  <AgentStatusPanel agents={agents} compact />
                </motion.div>
              )}
            </AnimatePresence>

            {/* Chat Area */}
            <div className="flex-1 overflow-hidden">
              <CopilotSidebar
                defaultOpen={true}
                clickOutsideToClose={false}
                instructions={`You are helping an analyst work with the E2I Causal Analytics platform.

Current context:
- Brand filter: ${filters.brand}
- Date range: ${filters.dateRange.start} to ${filters.dateRange.end}
${filters.territory ? `- Territory: ${filters.territory}` : ''}
${filters.hcpSegment ? `- HCP Segment: ${filters.hcpSegment}` : ''}

Active agents: ${agents.filter(a => a.status === 'active').map(a => a.name).join(', ')}

Available actions:
- navigateTo: Navigate to any page
- setBrandFilter: Change brand filter
- setDateRange: Set analytics date range
- highlightCausalPaths: Highlight paths on visualizations
- setDetailLevel: Adjust response complexity

Be concise and helpful. Focus on pharmaceutical commercial analytics (TRx, NRx, market share, etc.).`}
                labels={{
                  initial: 'How can I help you explore E2I analytics?',
                  placeholder: 'Ask about KPIs, agents, or insights...',
                }}
                className="h-full"
              />
            </div>

            {/* Footer */}
            <div className="p-3 border-t bg-muted/30 text-xs text-muted-foreground">
              <div className="flex items-center justify-between">
                <span>Press ⌘/ to toggle</span>
                <span>{filters.brand}</span>
              </div>
            </div>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Backdrop */}
      <AnimatePresence>
        {chatOpen && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            onClick={() => setChatOpen(false)}
            className="fixed inset-0 z-40 bg-black/20 backdrop-blur-sm"
          />
        )}
      </AnimatePresence>
    </>
  );
}

export default E2IChatSidebar;
