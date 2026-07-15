import { describe, it, expect } from 'vitest';
import { render, screen } from '@testing-library/react';

import { ClinicalContextPanel } from './ClinicalContextPanel';
import type { ClinicalContext } from '@/types/causal';

const FULL: ClinicalContext = {
  brand: 'Kisqali',
  drug_name: 'ribociclib',
  disease: 'Malignant neoplasm of breast',
  our_outcome: 'persistent_180d',
  mapped_endpoint: 'Treatment persistence / duration of therapy',
  mechanism: { mechanism_of_action: 'CDK4/6 inhibitor', source: 'chembl' },
  pivotal_endpoints: {
    endpoints: [
      { measure: 'Overall Survival (OS)', time_frame: 'Up to 5 years', nct_id: 'NCT01958021' },
      { measure: 'Progression-Free Survival (PFS)', time_frame: null, nct_id: 'NCT01958021' },
    ],
    source: 'clinicaltrials.gov',
  },
  real_world_evidence: {
    pmid: '35642282',
    title: 'CDK4/6 inhibitor treatment use in women with advanced breast cancer.',
    journal: 'J Oncol Pharm Pract',
    pubdate: '2023 Jul',
    doi: '10.1177/10781552221102884',
    url: 'https://pubmed.ncbi.nlm.nih.gov/35642282/',
    source: 'pubmed',
  },
  approved_indications: {
    indications: ['HR+/HER2- advanced breast cancer in combination with an aromatase inhibitor'],
    limitations_of_use: 'Not indicated for patients with early-stage breast cancer.',
    boxed_warning: 'QT interval prolongation has been observed. Monitor ECG prior to treatment.',
    source: 'openfda',
  },
  competitor_landscape: {
    competitors: ['Verzenio', 'Ibrance'],
    count: 2,
    source: 'curated',
  },
  honesty_label:
    'Effect estimate = a SYNTHETIC patient cohort. Clinical context below is REAL and cited.',
};

describe('ClinicalContextPanel', () => {
  it('renders drug, MoA, mapped endpoint, pivotal endpoints, and a linked citation', () => {
    render(<ClinicalContextPanel context={FULL} />);
    expect(screen.getByText(/ribociclib/i)).toBeInTheDocument();
    // Exact match targets only the MoA span; the RWE title (which also starts with
    // "CDK4/6 inhibitor") is longer and renders verbatim, so it cannot collide here.
    expect(screen.getByText('CDK4/6 inhibitor')).toBeInTheDocument();
    expect(screen.getByText(/Treatment persistence/i)).toBeInTheDocument();
    expect(screen.getByText(/Overall Survival/)).toBeInTheDocument();
    const link = screen.getByRole('link', { name: /35642282|breast cancer|CDK4/i });
    expect(link).toHaveAttribute('href', 'https://pubmed.ncbi.nlm.nih.gov/35642282/');
  });

  it('always shows the synthetic/real honesty label', () => {
    render(<ClinicalContextPanel context={FULL} />);
    expect(screen.getByText(/SYNTHETIC/)).toBeInTheDocument();
    expect(screen.getByText(/REAL and cited/i)).toBeInTheDocument();
  });

  it('marks a static_fallback source honestly and omits a missing citation', () => {
    const degraded: ClinicalContext = {
      ...FULL,
      mechanism: { mechanism_of_action: 'complement Factor B inhibitor', source: 'static_fallback' },
      pivotal_endpoints: {
        endpoints: [{ measure: 'Transfusion avoidance', time_frame: null, nct_id: null }],
        source: 'static_fallback',
      },
      real_world_evidence: null,
    };
    render(<ClinicalContextPanel context={degraded} />);
    expect(screen.getAllByText(/curated|fallback/i).length).toBeGreaterThan(0);
    // Curated-fallback endpoints carry no nct_id, so no CT.gov deep-link is rendered,
    // and with RWE null there is no citation link either.
    expect(screen.queryByRole('link')).toBeNull();
  });

  it('renders the Regulatory/Label section with live FDA label chip when source=openfda', () => {
    render(<ClinicalContextPanel context={FULL} />);
    expect(screen.getByText(/HR\+\/HER2-/)).toBeInTheDocument();
    expect(screen.getByText(/live FDA label/i)).toBeInTheDocument();
    expect(screen.getByText(/Limitations of use/i)).toBeInTheDocument();
  });

  it('does NOT surface the boxed warning (this panel grounds the commercial signal, not safety labeling)', () => {
    render(<ClinicalContextPanel context={FULL} />);
    expect(screen.queryByText(/BOXED WARNING/i)).toBeNull();
    expect(screen.queryByText(/QT interval prolongation/i)).toBeNull();
  });

  it('labels static_fallback approved_indications as curated fallback, not live FDA', () => {
    const fallback: ClinicalContext = {
      ...FULL,
      approved_indications: {
        indications: ['HR+/HER2- advanced breast cancer'],
        limitations_of_use: null,
        boxed_warning: null,
        source: 'static_fallback',
      },
    };
    render(<ClinicalContextPanel context={fallback} />);
    // "live FDA label" chip must NOT appear for static_fallback
    expect(screen.queryByText(/live FDA label/i)).toBeNull();
    // The curated fallback chip must appear (at least one)
    expect(screen.getAllByText(/curated fallback/i).length).toBeGreaterThan(0);
  });

  it('renders competitor landscape chips and count', () => {
    render(<ClinicalContextPanel context={FULL} />);
    expect(screen.getByText(/2 rivals/i)).toBeInTheDocument();
    expect(screen.getByText('Verzenio')).toBeInTheDocument();
    expect(screen.getByText('Ibrance')).toBeInTheDocument();
    // Source chip: curated (not FDA-sourced)
    expect(screen.getByText(/^curated$/i)).toBeInTheDocument();
  });

  it('omits the competitor landscape section when count is 0', () => {
    const noRivals: ClinicalContext = {
      ...FULL,
      competitor_landscape: { competitors: [], count: 0, source: 'curated' },
    };
    render(<ClinicalContextPanel context={noRivals} />);
    expect(screen.queryByText(/rivals/i)).toBeNull();
  });

  it('omits both new sections when approved_indications and competitor_landscape are absent', () => {
    const bare: ClinicalContext = {
      ...FULL,
      approved_indications: null,
      competitor_landscape: null,
    };
    render(<ClinicalContextPanel context={bare} />);
    expect(screen.queryByText(/live FDA label/i)).toBeNull();
    expect(screen.queryByText(/rivals/i)).toBeNull();
  });

  describe('trial endpoints as grounding (not a parameter dump)', () => {
    it('surfaces endpoint measures as clinical ground truth for the outcome definition', () => {
      render(<ClinicalContextPanel context={FULL} />);
      // Exact match targets the section heading only; the panel's intro sentence also
      // contains the words "real trial endpoints" as prose, so a regex would collide.
      expect(screen.getByText('Real trial endpoints')).toBeInTheDocument();
      expect(screen.getByText(/Overall Survival/)).toBeInTheDocument();
      expect(screen.getByText(/clinical ground truth our synthetic outcome stands in for/i))
        .toBeInTheDocument();
    });

    it('does NOT surface per-endpoint parameters (time frame, NCT deep-link, analysis scenario)', () => {
      const withParams: ClinicalContext = {
        ...FULL,
        real_world_evidence: null,
        seminal_real_world_evidence: undefined,
        pivotal_endpoints: {
          endpoints: [
            {
              measure:
                'Change From Baseline in Weekly Urticaria Score (UAS7) at Week 12 (Scenario 1 With UAS7 as Primary Efficacy Endpoint)',
              time_frame: 'Baseline, Week 12',
              nct_id: 'NCT05030311',
            },
          ],
          source: 'clinicaltrials.gov',
        },
      };
      render(<ClinicalContextPanel context={withParams} />);
      // The raw parameters the user asked us to stop surfacing must be gone.
      expect(screen.queryByText(/Time frame:/i)).toBeNull();
      expect(screen.queryByRole('link', { name: /NCT05030311/i })).toBeNull();
      expect(screen.queryByLabelText(/about these trial endpoints/i)).toBeNull();
      expect(screen.queryByText(/weeks after trial baseline/i)).toBeNull();
      expect(screen.queryByText(/pre-specified analysis scenario/i)).toBeNull();
      // The measure title itself is still shown (as grounding).
      expect(screen.getByText(/UAS7/)).toBeInTheDocument();
    });

    it('caps the endpoint list and discloses how many more exist', () => {
      const many: ClinicalContext = {
        ...FULL,
        pivotal_endpoints: {
          endpoints: Array.from({ length: 8 }, (_, i) => ({
            measure: `Endpoint measure ${i + 1}`,
            time_frame: null,
            nct_id: null,
          })),
          source: 'clinicaltrials.gov',
        },
      };
      render(<ClinicalContextPanel context={many} />);
      expect(screen.getByText('Endpoint measure 5')).toBeInTheDocument();
      expect(screen.queryByText('Endpoint measure 6')).toBeNull();
      expect(screen.getByText(/\+ 3 more registered\s+trial endpoints/i)).toBeInTheDocument();
    });

    it('keeps the grounding line honest for a curated fallback (no CT.gov claim)', () => {
      const fallback: ClinicalContext = {
        ...FULL,
        pivotal_endpoints: {
          endpoints: [
            {
              measure: 'Change from baseline in UAS7 (Urticaria Activity Score over 7 days)',
              time_frame: null,
              nct_id: null,
            },
          ],
          source: 'static_fallback',
        },
      };
      render(<ClinicalContextPanel context={fallback} />);
      expect(screen.getByText(/curated reference/i)).toBeInTheDocument();
      expect(screen.queryByText(/actually measured/i)).toBeNull();
    });

    it('omits the endpoint section entirely when there are no endpoints', () => {
      const noEndpoints: ClinicalContext = {
        ...FULL,
        pivotal_endpoints: { endpoints: [], source: 'static_fallback' },
      };
      render(<ClinicalContextPanel context={noEndpoints} />);
      // Exact match: the intro sentence (always shown) also mentions "real trial
      // endpoints" as prose, so only the exact section heading should be absent.
      expect(screen.queryByText('Real trial endpoints')).toBeNull();
    });
  });

  it('renders a curated brand-specific seminal RWE and demotes the live one to "Additional"', () => {
    const withSeminal: ClinicalContext = {
      ...FULL,
      seminal_real_world_evidence: {
        pmid: '36135090',
        title: 'Real-World Clinical Outcomes of Ribociclib in Premenopausal HR+/HER2- BC',
        journal: 'Current Oncology',
        pubdate: '2022',
        doi: '10.3390/curroncol29090521',
        url: 'https://pubmed.ncbi.nlm.nih.gov/36135090/',
        source: 'curated',
      },
    };
    render(<ClinicalContextPanel context={withSeminal} />);
    // The seminal block names the brand-specific paper and is labelled brand-specific.
    expect(screen.getByText(/Seminal real-world evidence/i)).toBeInTheDocument();
    expect(screen.getByText(/curated · brand-specific/i)).toBeInTheDocument();
    const seminalLink = screen.getByRole('link', { name: /Ribociclib in Premenopausal/i });
    expect(seminalLink).toHaveAttribute('href', 'https://pubmed.ncbi.nlm.nih.gov/36135090/');
    // The live relevance citation is demoted to "Additional" so it does not read as
    // the brand's own evidence (the reported competitor-paper confusion).
    expect(screen.getByText(/Additional real-world evidence/i)).toBeInTheDocument();
  });
});
