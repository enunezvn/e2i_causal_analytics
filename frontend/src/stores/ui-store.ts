/**
 * UI Store
 * ========
 *
 * Zustand store for managing global UI state including:
 * - Sidebar visibility and state
 * - Theme preferences
 * - Loading states
 * - Notification/toast management
 * - Modal states
 *
 * Usage:
 *   import { useUIStore } from '@/stores/ui-store'
 *   const { sidebarOpen, toggleSidebar } = useUIStore()
 */

import { create } from 'zustand';
import { devtools, persist } from 'zustand/middleware';
import { useShallow } from 'zustand/react/shallow';

/**
 * Theme options for the application
 */
export type Theme = 'light' | 'dark' | 'system';

/**
 * Notification severity levels
 */
export type NotificationSeverity = 'info' | 'success' | 'warning' | 'error';

/**
 * Notification item structure
 */
export interface Notification {
  id: string;
  title: string;
  message?: string;
  severity: NotificationSeverity;
  duration?: number;
  dismissible?: boolean;
  createdAt: number;
}

/**
 * Modal configuration
 */
export interface ModalConfig {
  id: string;
  isOpen: boolean;
  data?: Record<string, unknown>;
}

/**
 * UI store state interface
 */
export interface UIState {
  // Sidebar
  sidebarOpen: boolean;
  sidebarCollapsed: boolean;

  // Theme
  theme: Theme;

  // Loading
  globalLoading: boolean;
  loadingMessage: string | null;

  // Notifications
  notifications: Notification[];

  // Modals
  modals: Record<string, ModalConfig>;

  // Mobile
  isMobileMenuOpen: boolean;

  // Breadcrumbs
  breadcrumbs: Array<{ label: string; href?: string }>;
}

/**
 * UI store actions interface
 */
export interface UIActions {
  // Sidebar actions
  toggleSidebar: () => void;
  setSidebarOpen: (open: boolean) => void;
  toggleSidebarCollapsed: () => void;
  setSidebarCollapsed: (collapsed: boolean) => void;

  // Theme actions
  setTheme: (theme: Theme) => void;

  // Loading actions
  setGlobalLoading: (loading: boolean, message?: string | null) => void;

  // Notification actions
  addNotification: (
    notification: Omit<Notification, 'id' | 'createdAt'>
  ) => string;
  removeNotification: (id: string) => void;
  clearNotifications: () => void;

  // Modal actions
  openModal: (id: string, data?: Record<string, unknown>) => void;
  closeModal: (id: string) => void;
  toggleModal: (id: string) => void;
  isModalOpen: (id: string) => boolean;
  getModalData: (id: string) => Record<string, unknown> | undefined;

  // Mobile actions
  toggleMobileMenu: () => void;
  setMobileMenuOpen: (open: boolean) => void;

  // Breadcrumb actions
  setBreadcrumbs: (breadcrumbs: Array<{ label: string; href?: string }>) => void;
  clearBreadcrumbs: () => void;

  // Reset
  reset: () => void;
}

/**
 * Combined UI store type
 */
export type UIStore = UIState & UIActions;

/**
 * Initial state for the UI store
 */
const initialState: UIState = {
  sidebarOpen: true,
  sidebarCollapsed: false,
  theme: 'system',
  globalLoading: false,
  loadingMessage: null,
  notifications: [],
  modals: {},
  isMobileMenuOpen: false,
  breadcrumbs: [],
};

/**
 * Generate a unique notification ID
 */
function generateNotificationId(): string {
  return `notification-${Date.now()}-${Math.random().toString(36).substring(2, 9)}`;
}

/**
 * UI Store
 *
 * Global store for UI-related state management.
 * Persists theme and sidebar preferences to localStorage.
 */
export const useUIStore = create<UIStore>()(
  devtools(
    persist(
      (set, get) => ({
        // Initial state
        ...initialState,

        // Sidebar actions
        toggleSidebar: () => set((state) => ({ sidebarOpen: !state.sidebarOpen })),
        setSidebarOpen: (open) => set({ sidebarOpen: open }),
        toggleSidebarCollapsed: () =>
          set((state) => ({ sidebarCollapsed: !state.sidebarCollapsed })),
        setSidebarCollapsed: (collapsed) => set({ sidebarCollapsed: collapsed }),

        // Theme actions
        setTheme: (theme) => set({ theme }),

        // Loading actions
        setGlobalLoading: (loading, message = null) =>
          set({ globalLoading: loading, loadingMessage: message }),

        // Notification actions
        addNotification: (notification) => {
          const id = generateNotificationId();
          const newNotification: Notification = {
            ...notification,
            id,
            createdAt: Date.now(),
            dismissible: notification.dismissible ?? true,
            duration: notification.duration ?? 5000,
          };

          set((state) => ({
            notifications: [...state.notifications, newNotification],
          }));

          // Auto-remove notification after duration (if not 0)
          if (newNotification.duration && newNotification.duration > 0) {
            setTimeout(() => {
              get().removeNotification(id);
            }, newNotification.duration);
          }

          return id;
        },

        removeNotification: (id) =>
          set((state) => ({
            notifications: state.notifications.filter((n) => n.id !== id),
          })),

        clearNotifications: () => set({ notifications: [] }),

        // Modal actions
        openModal: (id, data) =>
          set((state) => ({
            modals: {
              ...state.modals,
              [id]: { id, isOpen: true, data },
            },
          })),

        closeModal: (id) =>
          set((state) => ({
            modals: {
              ...state.modals,
              [id]: { ...state.modals[id], id, isOpen: false },
            },
          })),

        toggleModal: (id) =>
          set((state) => {
            const current = state.modals[id];
            return {
              modals: {
                ...state.modals,
                [id]: { id, isOpen: !current?.isOpen, data: current?.data },
              },
            };
          }),

        isModalOpen: (id) => {
          const state = get();
          return state.modals[id]?.isOpen ?? false;
        },

        getModalData: (id) => {
          const state = get();
          return state.modals[id]?.data;
        },

        // Mobile actions
        toggleMobileMenu: () =>
          set((state) => ({ isMobileMenuOpen: !state.isMobileMenuOpen })),
        setMobileMenuOpen: (open) => set({ isMobileMenuOpen: open }),

        // Breadcrumb actions
        setBreadcrumbs: (breadcrumbs) => set({ breadcrumbs }),
        clearBreadcrumbs: () => set({ breadcrumbs: [] }),

        // Reset
        reset: () => set(initialState),
      }),
      {
        name: 'e2i-ui-store',
        // Only persist specific fields
        partialize: (state) => ({
          theme: state.theme,
          sidebarOpen: state.sidebarOpen,
          sidebarCollapsed: state.sidebarCollapsed,
        }),
      }
    ),
    { name: 'UIStore' }
  )
);

/**
 * Selector hooks for common UI state slices
 * Using useShallow to prevent infinite re-renders
 */
export const useSidebarState = () =>
  useUIStore(
    useShallow((state) => ({
      isOpen: state.sidebarOpen,
      isCollapsed: state.sidebarCollapsed,
      toggle: state.toggleSidebar,
      setOpen: state.setSidebarOpen,
      toggleCollapsed: state.toggleSidebarCollapsed,
      setCollapsed: state.setSidebarCollapsed,
    }))
  );

export const useTheme = () =>
  useUIStore(
    useShallow((state) => ({
      theme: state.theme,
      setTheme: state.setTheme,
    }))
  );

export const useGlobalLoading = () =>
  useUIStore(
    useShallow((state) => ({
      isLoading: state.globalLoading,
      message: state.loadingMessage,
      setLoading: state.setGlobalLoading,
    }))
  );

export const useNotifications = () =>
  useUIStore(
    useShallow((state) => ({
      notifications: state.notifications,
      add: state.addNotification,
      remove: state.removeNotification,
      clear: state.clearNotifications,
    }))
  );

export const useBreadcrumbs = () =>
  useUIStore(
    useShallow((state) => ({
      breadcrumbs: state.breadcrumbs,
      set: state.setBreadcrumbs,
      clear: state.clearBreadcrumbs,
    }))
  );

export default useUIStore;
