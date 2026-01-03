/**
 * CopilotKit Mock
 * ===============
 * Mock implementation of @copilotkit/react-core for testing.
 */

import * as React from 'react';

// Mock CopilotKit component
export const CopilotKit: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  return React.createElement(React.Fragment, null, children);
};

// Mock hooks
export const useCopilotReadable = () => undefined;
export const useCopilotAction = () => undefined;
export const useCopilotChat = () => ({
  messages: [],
  isLoading: false,
  sendMessage: () => Promise.resolve(),
  resetMessages: () => {},
});

// Mock context
export const CopilotContext = React.createContext<null>(null);
export const useCopilotContext = () => null;

export default {
  CopilotKit,
  useCopilotReadable,
  useCopilotAction,
  useCopilotChat,
  CopilotContext,
  useCopilotContext,
};
