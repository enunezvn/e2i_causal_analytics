/**
 * AgentFiltersBridge
 * ==================
 *
 * Pushes the dashboard's active filters into the "default" agent's CoAgent
 * shared state — the ONLY channel that reaches the backend chat graph for
 * agent runs (2026-08-19 review: the CopilotChat `instructions` prop and
 * `useCopilotReadable` never leave the browser; the wire carries only
 * {threadId, state, messages, actions}, and the backend hard-codes
 * `context: []`). The backend declares a matching `filters` state channel
 * (E2IAgentState) and folds it into the chat/synthesis prompts so an
 * ambiguous question resolves brand/period from the UI instead of asking
 * "which brand?" with the filter already set.
 */

import { useEffect, useRef } from 'react';
import { useCoAgent } from '@copilotkit/react-core';
import type { E2IFilters } from '../../providers/E2ICopilotProvider';

export interface AgentFiltersBridgeProps {
  filters: E2IFilters;
}

/** Agent state shape shared with the backend graph (subset we own). */
interface FiltersAgentState extends Record<string, unknown> {
  filters?: E2IFilters;
}

export function AgentFiltersBridge({ filters }: AgentFiltersBridgeProps): null {
  const { setState } = useCoAgent<FiltersAgentState>({
    name: 'default',
    initialState: { filters },
  });

  // setState's identity follows the live agent state, so it cannot appear in
  // the effect deps without looping (push -> state change -> new setState ->
  // push ...). Key the effect on the serialized filters instead and read the
  // latest setState through a ref.
  const setStateRef = useRef(setState);
  setStateRef.current = setState;
  const filtersKey = JSON.stringify(filters);

  useEffect(() => {
    // Merge, never clobber: agent state also carries the `copilotkit`
    // actions channel and the backend's progress fields.
    setStateRef.current((prev) => ({ ...(prev ?? {}), filters }));
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [filtersKey]);

  return null;
}
