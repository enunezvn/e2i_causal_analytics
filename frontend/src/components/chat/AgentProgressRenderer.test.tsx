/**
 * AgentProgressRenderer Component Tests
 * =====================================
 *
 * Covers issue #1340 UI-D2: a completed progress card must pin its terminal
 * header instead of flipping between "Working..." and "Processing Query"
 * across re-renders (nodeName is unstable after the run ends — the library
 * recomputes it from live agent state, so the same 100%/"Response complete"
 * card rendered "Working... 100%" to a user scrolling up).
 */

import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen } from '@testing-library/react';
import type { ReactElement } from 'react';

// Capture the render callback that AgentProgressRenderer registers with
// CopilotKit so tests can invoke it with controlled state/status/nodeName —
// exactly what useCoAgentStateRender does at runtime.
type StateRenderArgs = {
  state: {
    agent_status: 'idle' | 'processing' | 'waiting' | 'complete' | 'error';
    progress_percent: number;
    progress_steps: string[];
    tools_executing: string[];
    error_message?: string;
    current_node?: string;
  };
  status: 'inProgress' | 'complete' | 'error';
  nodeName: string | undefined;
};

const harness = vi.hoisted(() => ({
  capturedRender: null as null | ((args: StateRenderArgs) => ReactElement),
}));

vi.mock('@copilotkit/react-core', () => ({
  useCoAgentStateRender: (options: { render: (args: StateRenderArgs) => ReactElement }) => {
    harness.capturedRender = options.render;
  },
}));

import { AgentProgressRenderer } from './AgentProgressRenderer';

function renderProgressCard(args: StateRenderArgs) {
  render(<AgentProgressRenderer />);
  if (!harness.capturedRender) {
    throw new Error('AgentProgressRenderer did not register a state render');
  }
  return render(harness.capturedRender(args));
}

const completedState: StateRenderArgs['state'] = {
  agent_status: 'complete',
  progress_percent: 100,
  progress_steps: ['Processing your query...', 'Response complete'],
  tools_executing: [],
};

describe('AgentProgressRenderer terminal-state pinning (UI-D2, #1340)', () => {
  beforeEach(() => {
    harness.capturedRender = null;
  });

  it('pins a completed card to its terminal header when nodeName is undefined', () => {
    renderProgressCard({
      state: completedState,
      status: 'complete',
      nodeName: undefined,
    });

    // The stale header: default nodeName label leaked into a finished card.
    expect(screen.queryByText('Working...')).not.toBeInTheDocument();
    expect(screen.getByText('Complete')).toBeInTheDocument();
    expect(screen.getByText('100%')).toBeInTheDocument();
  });

  it('pins a completed card to the same terminal header when nodeName is still set', () => {
    // Re-renders flip nodeName between '' and the last LangGraph node ('chat');
    // the terminal header must be identical either way.
    renderProgressCard({
      state: completedState,
      status: 'complete',
      nodeName: 'chat',
    });

    expect(screen.queryByText('Processing Query')).not.toBeInTheDocument();
    expect(screen.queryByText('Working...')).not.toBeInTheDocument();
    expect(screen.getByText('Complete')).toBeInTheDocument();
  });

  it('pins the terminal header when the agent state is complete even if the render status lags', () => {
    renderProgressCard({
      state: completedState,
      status: 'inProgress',
      nodeName: 'chat',
    });

    expect(screen.getByText('Complete')).toBeInTheDocument();
    expect(screen.queryByText('Processing Query')).not.toBeInTheDocument();
  });

  it('pins an errored card to an Error header', () => {
    renderProgressCard({
      state: {
        ...completedState,
        agent_status: 'error',
        error_message: 'Run failed',
      },
      status: 'error',
      nodeName: undefined,
    });

    expect(screen.queryByText('Working...')).not.toBeInTheDocument();
    expect(screen.getByText('Error')).toBeInTheDocument();
    expect(screen.getByText('Run failed')).toBeInTheDocument();
  });

  it('keeps the live node label while the run is in flight', () => {
    renderProgressCard({
      state: {
        agent_status: 'processing',
        progress_percent: 25,
        progress_steps: ['Processing your query...'],
        tools_executing: [],
      },
      status: 'inProgress',
      nodeName: 'chat',
    });

    expect(screen.getByText('Processing Query')).toBeInTheDocument();
    expect(screen.queryByText('Complete')).not.toBeInTheDocument();
  });

  it('still shows Working... for an unknown node while in flight', () => {
    renderProgressCard({
      state: {
        agent_status: 'processing',
        progress_percent: 10,
        progress_steps: ['Starting...'],
        tools_executing: [],
      },
      status: 'inProgress',
      nodeName: undefined,
    });

    expect(screen.getByText('Working...')).toBeInTheDocument();
  });
});
