/**
 * Providers Index
 * ===============
 *
 * Centralized exports for all React context providers.
 *
 * @module providers
 */

export {
  E2ICopilotProvider,
  CopilotKitWrapper,
  useE2ICopilot,
  useCopilotEnabled,
} from './E2ICopilotProvider';
export type {
  E2ICopilotProviderProps,
  CopilotKitWrapperProps,
  E2ICopilotContextValue,
  E2IFilters,
  AgentInfo,
  UserPreferences,
} from './E2ICopilotProvider';
