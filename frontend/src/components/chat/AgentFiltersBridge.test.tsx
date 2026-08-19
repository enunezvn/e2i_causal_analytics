/**
 * AgentFiltersBridge Tests
 * ========================
 *
 * 2026-08-19 review: the dashboard's brand filter had NO channel to the chat
 * backend — the CopilotChat `instructions` prop and `useCopilotReadable`
 * never leave the browser for agent runs (the wire carries only
 * {threadId, state, messages, actions}). Measured: filter set to
 * Remibrutinib at 09:44:45, chat still asked "which brand?" at 09:51.
 *
 * AgentFiltersBridge closes the gap through the ONE channel that reaches the
 * graph: CoAgent shared state. It must (1) register the "default" agent with
 * the filters in initialState, (2) push filter changes into agent state via
 * setState, (3) merge — never clobber — other keys living in agent state
 * (progress fields, the `copilotkit` actions channel), and (4) not re-push
 * when filters are unchanged (a push loop would spam agent state).
 */

import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render } from '@testing-library/react';

type AgentState = Record<string, unknown>;
type CoAgentOptions = { name: string; initialState?: AgentState };
type SetStateArg = AgentState | ((prev: AgentState | undefined) => AgentState);

const harness = vi.hoisted(() => ({
  options: null as null | CoAgentOptions,
  setStateCalls: [] as SetStateArg[],
}));

vi.mock('@copilotkit/react-core', () => ({
  useCoAgent: (options: CoAgentOptions) => {
    harness.options = options;
    return {
      name: options.name,
      state: options.initialState,
      setState: (arg: SetStateArg) => {
        harness.setStateCalls.push(arg);
      },
      running: false,
      start: vi.fn(),
      stop: vi.fn(),
      run: vi.fn(),
    };
  },
}));

import { AgentFiltersBridge } from './AgentFiltersBridge';
import type { E2IFilters } from '../../providers/E2ICopilotProvider';

const FILTERS: E2IFilters = {
  brand: 'Remibrutinib',
  territory: 'west',
  dateRange: { start: '2026-05-21', end: '2026-08-19' },
  hcpSegment: null,
};

/** Resolve a recorded setState call against a previous agent state. */
function resolve(call: SetStateArg, prev: AgentState | undefined): AgentState {
  return typeof call === 'function' ? call(prev) : call;
}

describe('AgentFiltersBridge', () => {
  beforeEach(() => {
    harness.options = null;
    harness.setStateCalls = [];
  });

  it('registers the "default" agent with filters in initialState', () => {
    render(<AgentFiltersBridge filters={FILTERS} />);
    expect(harness.options?.name).toBe('default');
    expect(harness.options?.initialState).toMatchObject({ filters: FILTERS });
  });

  it('pushes the filters into agent state on mount', () => {
    render(<AgentFiltersBridge filters={FILTERS} />);
    expect(harness.setStateCalls.length).toBeGreaterThan(0);
    const state = resolve(harness.setStateCalls[0], undefined);
    expect(state.filters).toEqual(FILTERS);
  });

  it('pushes again when filters change', () => {
    const { rerender } = render(<AgentFiltersBridge filters={FILTERS} />);
    const callsAfterMount = harness.setStateCalls.length;
    const changed: E2IFilters = { ...FILTERS, brand: 'Fabhalta' };
    rerender(<AgentFiltersBridge filters={changed} />);
    expect(harness.setStateCalls.length).toBeGreaterThan(callsAfterMount);
    const state = resolve(harness.setStateCalls[harness.setStateCalls.length - 1], undefined);
    expect((state.filters as E2IFilters).brand).toBe('Fabhalta');
  });

  it('does not re-push for unchanged filters across rerenders', () => {
    const { rerender } = render(<AgentFiltersBridge filters={FILTERS} />);
    const callsAfterMount = harness.setStateCalls.length;
    rerender(<AgentFiltersBridge filters={{ ...FILTERS }} />);
    expect(harness.setStateCalls.length).toBe(callsAfterMount);
  });

  it('merges with existing agent state instead of clobbering it', () => {
    render(<AgentFiltersBridge filters={FILTERS} />);
    const call = harness.setStateCalls[0];
    const prev: AgentState = {
      copilotkit: { actions: [{ name: 'renderKpiTrend' }] },
      progress_percent: 50,
    };
    const next = resolve(call, prev);
    expect(next.copilotkit).toEqual(prev.copilotkit);
    expect(next.progress_percent).toBe(50);
    expect(next.filters).toEqual(FILTERS);
  });
});
