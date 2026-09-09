import { describe, it, expect } from 'vitest';
// The "Open review" deep link is a router <Link>: render under the router-wrapped helper.
import { renderWithAllProviders, screen } from '@/test/utils';
import { ReviewStatusPanel } from './ReviewStatusPanel';

describe('ReviewStatusPanel', () => {
  it('renders nothing when the run carried no review state and no DAG record', () => {
    const { container } = renderWithAllProviders(<ReviewStatusPanel />);
    expect(container).toBeEmptyDOMElement();
  });

  it('shows a pending structure with a deep link into the queue', () => {
    renderWithAllProviders(
      <ReviewStatusPanel decision="pending_review" reviewId="rev-9" discoveredDagId={null} />
    );
    expect(screen.getByText('Pending expert review')).toBeInTheDocument();
    const link = screen.getByRole('link', { name: /open review/i });
    expect(link).toHaveAttribute('href', '/expert-reviews?review=rev-9');
  });

  it('shows the rejection and the halt message from the run warnings', () => {
    renderWithAllProviders(
      <ReviewStatusPanel
        decision="rejected"
        reviewId="rev-rejected"
        warnings={[
          'Estimate withheld: a domain expert REJECTED this DAG structure by Dr. No (review rev-rejected): collider.',
          'Refutation gate BLOCKED — the estimate did not survive robustness checks.',
        ]}
      />
    );
    expect(screen.getByText('Structure rejected')).toBeInTheDocument();
    expect(screen.getByText(/by Dr\. No/)).toBeInTheDocument();
    expect(screen.queryByText(/Refutation gate BLOCKED/)).not.toBeInTheDocument();
  });

  it('shows the durable discovery record id', () => {
    renderWithAllProviders(<ReviewStatusPanel discoveredDagId="8a61b3db-6aad-4b01-96e4-bbea0af861b4" />);
    expect(screen.getByText(/Durable discovery record/)).toBeInTheDocument();
    expect(screen.getByText('8a61b3db-6aad-4b01-96e4-bbea0af861b4')).toBeInTheDocument();
  });

  it('never invents a label for an unknown decision', () => {
    renderWithAllProviders(<ReviewStatusPanel decision="something_new" />);
    expect(screen.getByText('something_new')).toBeInTheDocument();
  });
});
