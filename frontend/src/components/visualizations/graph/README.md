# Knowledge Graph visualization

Cytoscape.js-based rendering for the causal knowledge graph. Every graph in the
app — the Knowledge Graph page (`pages/KnowledgeGraph.tsx`), `CytoscapeGraph`
here, and `ActiveCausalChains` (`components/insights/`) — is instantiated through
the single `useCytoscape` hook (`frontend/src/hooks/use-cytoscape.ts`). The live
Cytoscape instance is held in a React ref and is intentionally **not** on the
global scope; components that need it use the `onReady?.(cy)` callback.

## Debugging: the `window.cy` handle (`e2i-graph-debug`)

For ad-hoc console inspection of the rendered graph, `useCytoscape` can publish
the live instance to `window.cy`. This is the convenient way to eyeball the
causal edges (and their ATE values) directly, without replaying the
`/api/graph/relationships` fetch by hand.

### When it's on

| Build | Behavior |
|-------|----------|
| **Dev** (`vite dev`, `import.meta.env.DEV === true`) | On automatically. |
| **Production** (deployed build, e.g. eznomics.site) | **Off by default.** Opt in per-browser. |

A deployed build is a production `vite build`, so `import.meta.env.DEV` is
`false` and a pure dev gate would be stripped from the bundle — useless for
inspecting the live site. The flag closes that gap: it's a `localStorage`
opt-in that also works on production builds. **Normal users never get the
global.**

### Enable on a deployed build

In the browser console on the Knowledge Graph page:

```js
localStorage.setItem('e2i-graph-debug', '1');
location.reload();   // the handle is published when the graph re-mounts
```

Disable / restore the default:

```js
localStorage.removeItem('e2i-graph-debug');
location.reload();
```

The gate (`graphDebugEnabled()` in `use-cytoscape.ts`) fails closed — if
`localStorage` throws (privacy mode / disabled storage) the handle stays off.

### Inspect causal edges

```js
// All CAUSES edges with their brand + effect size
window.cy.edges()
  .map((e) => e.data())
  .filter((e) => e.type === 'CAUSES')
  .map((e) => ({
    source: e.source,
    target: e.target,
    brand: e.properties?.brand,       // array, e.g. ["Fabhalta"]
    ate: e.properties?.ate_estimate,
  }));
```

Edge `data()` shape: top-level `id` / `source` / `target` / `type` /
`confidence`, plus a nested `properties` object carrying `brand` (an array) and
`ate_estimate`. Brand-distinct causal axes to sanity-check (persistence
outcome `var:persistent_180d`):

| Brand | Source variable | ATE |
|-------|-----------------|-----|
| Fabhalta | `var:complement_inhibitor_status` | −0.129 |
| Kisqali | `var:disease_stage` | −0.105 |
| Remibrutinib | `var:urticaria_severity_uas7` | +0.141 |

### Notes

- **Last graph to mount wins** the single `window.cy` handle (the usual
  Cytoscape console convention). If more than one graph is on screen, `window.cy`
  points at whichever initialized most recently.
- **No cross-graph clobber:** on teardown a graph only clears `window.cy` if it
  still points at that graph's instance, so unmounting one graph can't wipe
  another's live handle.
- The flag is read only at graph initialize/teardown — set it, then reload.
