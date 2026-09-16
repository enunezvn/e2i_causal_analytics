/**
 * DagDiff tests — the structural delta between two versions of one estimand.
 *
 * The panel is read by an operator deciding whether a re-run changed the DAG
 * enough to need a fresh sign-off, so an empty group must not render an empty
 * heading and an unchanged delta must SAY it is unchanged rather than render
 * nothing at all.
 */
import { render, screen } from '@testing-library/react';
import { describe, expect, it } from 'vitest';
import { DagDiff } from './DagDiff';

describe('DagDiff', () => {
  it('lists added and removed nodes, edges and adjustment sets', () => {
    render(
      <DagDiff
        changes={{
          nodes_added: ['W'],
          nodes_removed: [],
          edges_added: [['W', 'T']],
          edges_removed: [],
          adjustment_sets_added: [['W', 'Z']],
          adjustment_sets_removed: [['W']],
          is_changed: true,
        }}
      />
    );
    expect(screen.getByText('+ W')).toBeInTheDocument();
    expect(screen.getByText('+ W → T')).toBeInTheDocument();
    expect(screen.getByText('+ {W, Z}')).toBeInTheDocument();
    expect(screen.getByText('− {W}')).toBeInTheDocument();
  });

  it('says so when nothing changed', () => {
    render(
      <DagDiff
        changes={{
          nodes_added: [],
          nodes_removed: [],
          edges_added: [],
          edges_removed: [],
          adjustment_sets_added: [],
          adjustment_sets_removed: [],
          is_changed: false,
        }}
      />
    );
    expect(
      screen.getByText('No structural change from the previous version.')
    ).toBeInTheDocument();
  });

  it('omits empty groups', () => {
    render(
      <DagDiff
        changes={{
          nodes_added: ['W'],
          nodes_removed: [],
          edges_added: [],
          edges_removed: [],
          adjustment_sets_added: [],
          adjustment_sets_removed: [],
          is_changed: true,
        }}
      />
    );
    expect(screen.getByText('Nodes')).toBeInTheDocument();
    expect(screen.queryByText('Edges')).toBeNull();
    expect(screen.queryByText('Adjustment sets')).toBeNull();
  });
});
