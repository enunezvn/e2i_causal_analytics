/**
 * Layout Component
 * ================
 *
 * Main layout wrapper for the E2I Causal Analytics dashboard.
 * Combines Header, Sidebar, Footer, and main content area.
 *
 * Usage:
 *   import { Layout } from '@/components/layout/Layout'
 *   <Layout>
 *     <PageContent />
 *   </Layout>
 */

import type { ReactNode } from 'react';
import { useSidebarState } from '@/stores/ui-store';
import { Header } from './Header';
import { Sidebar } from './Sidebar';
import { Footer } from './Footer';
import { E2IChatSidebar } from '@/components/chat';
import { ErrorBoundary, ChatErrorBoundary } from '@/components/ui/error-boundary';

/**
 * Layout props interface
 */
interface LayoutProps {
  /** Content to render in the main area */
  children: ReactNode;
  /** Additional CSS classes for the layout wrapper */
  className?: string;
  /** Whether to hide the header */
  hideHeader?: boolean;
  /** Whether to hide the sidebar */
  hideSidebar?: boolean;
  /** Whether to hide the footer */
  hideFooter?: boolean;
}

/**
 * Layout component
 *
 * Provides the main application layout structure with:
 * - Fixed sidebar navigation (collapsible)
 * - Sticky header with page title and actions
 * - Scrollable main content area
 * - Footer with links and status
 *
 * The layout is responsive:
 * - On mobile: sidebar is hidden by default, toggle via header menu
 * - On desktop: sidebar is visible, can be collapsed for more space
 */
export function Layout({
  children,
  className = '',
  hideHeader = false,
  hideSidebar = false,
  hideFooter = false,
}: LayoutProps) {
  const { isCollapsed } = useSidebarState();

  // Calculate main content margin based on sidebar state
  // On desktop (lg+): margin-left matches sidebar width
  // On mobile: no margin (sidebar is overlay)
  const sidebarWidth = isCollapsed ? 'lg:ml-16' : 'lg:ml-64';

  return (
    <div className={`min-h-screen bg-[var(--color-background)] ${className}`}>
      {/* Sidebar - fixed on left */}
      {!hideSidebar && <Sidebar />}

      {/* Main content wrapper - takes remaining space */}
      <div
        className={`
          flex flex-col min-h-screen
          transition-[margin] duration-300 ease-in-out
          ${!hideSidebar ? sidebarWidth : ''}
        `}
      >
        {/* Header - sticky at top */}
        {!hideHeader && <Header />}

        {/* Main content area - flexible height, scrollable */}
        <main
          className="flex-1 p-4 lg:p-6"
          role="main"
          id="main-content"
          tabIndex={-1}
        >
          {/* Skip link target for accessibility */}
          <a href="#main-content" className="sr-only focus:not-sr-only">
            Skip to main content
          </a>
          <ErrorBoundary sectionName="Page content">
            {children}
          </ErrorBoundary>
        </main>

        {/* Footer - at bottom */}
        {!hideFooter && <Footer />}
      </div>

      {/* E2I Chat Sidebar - AI Assistant (wrapped in error boundary) */}
      <ChatErrorBoundary>
        <E2IChatSidebar position="right" showAgentStatus />
      </ChatErrorBoundary>
    </div>
  );
}

export default Layout;
