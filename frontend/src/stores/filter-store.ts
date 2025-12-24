/**
 * Filter Store
 * ============
 *
 * Zustand store for managing filter state across dashboard components.
 * Handles date ranges, search queries, category filters, and pagination.
 *
 * Usage:
 *   import { useFilterStore } from '@/stores/filter-store'
 *   const { dateRange, setDateRange } = useFilterStore()
 */

import { create } from 'zustand';
import { devtools, persist } from 'zustand/middleware';

/**
 * Predefined date range presets
 */
export type DateRangePreset =
  | 'today'
  | 'yesterday'
  | 'last7days'
  | 'last30days'
  | 'last90days'
  | 'thisMonth'
  | 'lastMonth'
  | 'thisYear'
  | 'custom';

/**
 * Date range configuration
 */
export interface DateRange {
  start: Date | null;
  end: Date | null;
  preset: DateRangePreset;
}

/**
 * Sort direction options
 */
export type SortDirection = 'asc' | 'desc';

/**
 * Sort configuration
 */
export interface SortConfig {
  field: string;
  direction: SortDirection;
}

/**
 * Pagination configuration
 */
export interface PaginationConfig {
  page: number;
  pageSize: number;
  totalItems: number;
  totalPages: number;
}

/**
 * Generic filter value types
 */
export type FilterValue = string | number | boolean | string[] | null;

/**
 * Active filters map
 */
export type ActiveFilters = Record<string, FilterValue>;

/**
 * Graph-specific filters
 */
export interface GraphFilters {
  nodeTypes: string[];
  relationshipTypes: string[];
  minConnections: number;
  maxDepth: number;
  searchQuery: string;
}

/**
 * Metrics-specific filters
 */
export interface MetricsFilters {
  models: string[];
  metrics: string[];
  aggregation: 'hourly' | 'daily' | 'weekly' | 'monthly';
}

/**
 * Filter store state interface
 */
export interface FilterState {
  // Global search
  globalSearch: string;

  // Date range
  dateRange: DateRange;

  // Sorting
  sort: SortConfig;

  // Pagination
  pagination: PaginationConfig;

  // Active filters (generic key-value pairs)
  activeFilters: ActiveFilters;

  // Domain-specific filters
  graphFilters: GraphFilters;
  metricsFilters: MetricsFilters;

  // Filter presets
  savedPresets: Record<string, ActiveFilters>;
}

/**
 * Filter store actions interface
 */
export interface FilterActions {
  // Global search actions
  setGlobalSearch: (query: string) => void;
  clearGlobalSearch: () => void;

  // Date range actions
  setDateRange: (range: Partial<DateRange>) => void;
  setDateRangePreset: (preset: DateRangePreset) => void;
  clearDateRange: () => void;

  // Sort actions
  setSort: (field: string, direction?: SortDirection) => void;
  toggleSortDirection: () => void;
  clearSort: () => void;

  // Pagination actions
  setPage: (page: number) => void;
  setPageSize: (size: number) => void;
  setTotalItems: (total: number) => void;
  nextPage: () => void;
  previousPage: () => void;
  resetPagination: () => void;

  // Generic filter actions
  setFilter: (key: string, value: FilterValue) => void;
  removeFilter: (key: string) => void;
  setFilters: (filters: ActiveFilters) => void;
  clearFilters: () => void;
  hasActiveFilters: () => boolean;

  // Graph filter actions
  setGraphFilters: (filters: Partial<GraphFilters>) => void;
  resetGraphFilters: () => void;

  // Metrics filter actions
  setMetricsFilters: (filters: Partial<MetricsFilters>) => void;
  resetMetricsFilters: () => void;

  // Preset actions
  savePreset: (name: string) => void;
  loadPreset: (name: string) => void;
  deletePreset: (name: string) => void;

  // Reset
  reset: () => void;
}

/**
 * Combined filter store type
 */
export type FilterStore = FilterState & FilterActions;

/**
 * Default date range
 */
const defaultDateRange: DateRange = {
  start: null,
  end: null,
  preset: 'last30days',
};

/**
 * Default sort configuration
 */
const defaultSort: SortConfig = {
  field: 'createdAt',
  direction: 'desc',
};

/**
 * Default pagination configuration
 */
const defaultPagination: PaginationConfig = {
  page: 1,
  pageSize: 20,
  totalItems: 0,
  totalPages: 0,
};

/**
 * Default graph filters
 */
const defaultGraphFilters: GraphFilters = {
  nodeTypes: [],
  relationshipTypes: [],
  minConnections: 0,
  maxDepth: 3,
  searchQuery: '',
};

/**
 * Default metrics filters
 */
const defaultMetricsFilters: MetricsFilters = {
  models: [],
  metrics: [],
  aggregation: 'daily',
};

/**
 * Initial state for the filter store
 */
const initialState: FilterState = {
  globalSearch: '',
  dateRange: defaultDateRange,
  sort: defaultSort,
  pagination: defaultPagination,
  activeFilters: {},
  graphFilters: defaultGraphFilters,
  metricsFilters: defaultMetricsFilters,
  savedPresets: {},
};

/**
 * Calculate date range from preset
 */
function getDateRangeFromPreset(preset: DateRangePreset): { start: Date | null; end: Date | null } {
  const now = new Date();
  const today = new Date(now.getFullYear(), now.getMonth(), now.getDate());
  const endOfDay = new Date(today.getTime() + 24 * 60 * 60 * 1000 - 1);

  switch (preset) {
    case 'today':
      return { start: today, end: endOfDay };
    case 'yesterday': {
      const yesterday = new Date(today.getTime() - 24 * 60 * 60 * 1000);
      const endOfYesterday = new Date(today.getTime() - 1);
      return { start: yesterday, end: endOfYesterday };
    }
    case 'last7days': {
      const start = new Date(today.getTime() - 7 * 24 * 60 * 60 * 1000);
      return { start, end: endOfDay };
    }
    case 'last30days': {
      const start = new Date(today.getTime() - 30 * 24 * 60 * 60 * 1000);
      return { start, end: endOfDay };
    }
    case 'last90days': {
      const start = new Date(today.getTime() - 90 * 24 * 60 * 60 * 1000);
      return { start, end: endOfDay };
    }
    case 'thisMonth': {
      const start = new Date(now.getFullYear(), now.getMonth(), 1);
      return { start, end: endOfDay };
    }
    case 'lastMonth': {
      const start = new Date(now.getFullYear(), now.getMonth() - 1, 1);
      const end = new Date(now.getFullYear(), now.getMonth(), 0, 23, 59, 59, 999);
      return { start, end };
    }
    case 'thisYear': {
      const start = new Date(now.getFullYear(), 0, 1);
      return { start, end: endOfDay };
    }
    case 'custom':
    default:
      return { start: null, end: null };
  }
}

/**
 * Filter Store
 *
 * Global store for filter state management across dashboards.
 * Persists saved presets and some preferences to localStorage.
 */
export const useFilterStore = create<FilterStore>()(
  devtools(
    persist(
      (set, get) => ({
        // Initial state
        ...initialState,

        // Global search actions
        setGlobalSearch: (query) => set({ globalSearch: query }),
        clearGlobalSearch: () => set({ globalSearch: '' }),

        // Date range actions
        setDateRange: (range) =>
          set((state) => ({
            dateRange: { ...state.dateRange, ...range },
          })),

        setDateRangePreset: (preset) => {
          const { start, end } = getDateRangeFromPreset(preset);
          set({ dateRange: { start, end, preset } });
        },

        clearDateRange: () => set({ dateRange: defaultDateRange }),

        // Sort actions
        setSort: (field, direction) =>
          set((state) => ({
            sort: {
              field,
              direction: direction ?? state.sort.direction,
            },
          })),

        toggleSortDirection: () =>
          set((state) => ({
            sort: {
              ...state.sort,
              direction: state.sort.direction === 'asc' ? 'desc' : 'asc',
            },
          })),

        clearSort: () => set({ sort: defaultSort }),

        // Pagination actions
        setPage: (page) =>
          set((state) => ({
            pagination: { ...state.pagination, page },
          })),

        setPageSize: (pageSize) =>
          set((state) => ({
            pagination: {
              ...state.pagination,
              pageSize,
              page: 1, // Reset to first page when changing page size
              totalPages: Math.ceil(state.pagination.totalItems / pageSize),
            },
          })),

        setTotalItems: (totalItems) =>
          set((state) => ({
            pagination: {
              ...state.pagination,
              totalItems,
              totalPages: Math.ceil(totalItems / state.pagination.pageSize),
            },
          })),

        nextPage: () =>
          set((state) => ({
            pagination: {
              ...state.pagination,
              page: Math.min(state.pagination.page + 1, state.pagination.totalPages),
            },
          })),

        previousPage: () =>
          set((state) => ({
            pagination: {
              ...state.pagination,
              page: Math.max(state.pagination.page - 1, 1),
            },
          })),

        resetPagination: () =>
          set((state) => ({
            pagination: { ...defaultPagination, pageSize: state.pagination.pageSize },
          })),

        // Generic filter actions
        setFilter: (key, value) =>
          set((state) => ({
            activeFilters: { ...state.activeFilters, [key]: value },
            pagination: { ...state.pagination, page: 1 }, // Reset to first page
          })),

        removeFilter: (key) =>
          set((state) => {
            const { [key]: _removed, ...rest } = state.activeFilters;
            void _removed; // Explicitly mark as intentionally unused
            return {
              activeFilters: rest,
              pagination: { ...state.pagination, page: 1 },
            };
          }),

        setFilters: (filters) =>
          set({
            activeFilters: filters,
            pagination: { ...defaultPagination },
          }),

        clearFilters: () =>
          set({
            activeFilters: {},
            pagination: { ...defaultPagination },
          }),

        hasActiveFilters: () => {
          const state = get();
          return (
            Object.keys(state.activeFilters).length > 0 ||
            state.globalSearch !== '' ||
            state.dateRange.preset !== 'last30days'
          );
        },

        // Graph filter actions
        setGraphFilters: (filters) =>
          set((state) => ({
            graphFilters: { ...state.graphFilters, ...filters },
          })),

        resetGraphFilters: () => set({ graphFilters: defaultGraphFilters }),

        // Metrics filter actions
        setMetricsFilters: (filters) =>
          set((state) => ({
            metricsFilters: { ...state.metricsFilters, ...filters },
          })),

        resetMetricsFilters: () => set({ metricsFilters: defaultMetricsFilters }),

        // Preset actions
        savePreset: (name) =>
          set((state) => ({
            savedPresets: {
              ...state.savedPresets,
              [name]: { ...state.activeFilters },
            },
          })),

        loadPreset: (name) =>
          set((state) => {
            const preset = state.savedPresets[name];
            if (preset) {
              return {
                activeFilters: { ...preset },
                pagination: { ...defaultPagination },
              };
            }
            return state;
          }),

        deletePreset: (name) =>
          set((state) => {
            const { [name]: _removed, ...rest } = state.savedPresets;
            void _removed; // Explicitly mark as intentionally unused
            return { savedPresets: rest };
          }),

        // Reset
        reset: () => set(initialState),
      }),
      {
        name: 'e2i-filter-store',
        // Only persist saved presets and page size preference
        partialize: (state) => ({
          savedPresets: state.savedPresets,
          pagination: { pageSize: state.pagination.pageSize },
        }),
      }
    ),
    { name: 'FilterStore' }
  )
);

/**
 * Selector hooks for common filter state slices
 */
export const useGlobalSearch = () =>
  useFilterStore((state) => ({
    query: state.globalSearch,
    setQuery: state.setGlobalSearch,
    clear: state.clearGlobalSearch,
  }));

export const useDateRangeFilter = () =>
  useFilterStore((state) => ({
    dateRange: state.dateRange,
    setDateRange: state.setDateRange,
    setPreset: state.setDateRangePreset,
    clear: state.clearDateRange,
  }));

export const useSortFilter = () =>
  useFilterStore((state) => ({
    sort: state.sort,
    setSort: state.setSort,
    toggleDirection: state.toggleSortDirection,
    clear: state.clearSort,
  }));

export const usePaginationFilter = () =>
  useFilterStore((state) => ({
    pagination: state.pagination,
    setPage: state.setPage,
    setPageSize: state.setPageSize,
    setTotalItems: state.setTotalItems,
    nextPage: state.nextPage,
    previousPage: state.previousPage,
    reset: state.resetPagination,
  }));

export const useGraphFilters = () =>
  useFilterStore((state) => ({
    filters: state.graphFilters,
    setFilters: state.setGraphFilters,
    reset: state.resetGraphFilters,
  }));

export const useMetricsFilters = () =>
  useFilterStore((state) => ({
    filters: state.metricsFilters,
    setFilters: state.setMetricsFilters,
    reset: state.resetMetricsFilters,
  }));

export default useFilterStore;
