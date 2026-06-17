import { Page, Locator } from '@playwright/test'
import { BasePage } from './base.page'
import { gotoAndWaitForHeading } from '../fixtures/page-harness'

/**
 * Page Object Model for the Causal Analysis page (`/causal-analysis`).
 *
 * Agent-driven causal inference dashboard (DoWhy / EconML / CausalML). The page
 * leverages the causal_impact agent to build the DAG and estimate the
 * treatment->outcome effect. Honest states this POM exposes:
 *  - healthy banner: "Causal Engine Healthy" (from GET /api/causal/health)
 *  - degraded banner: "Service Issue" when status != 'healthy'
 *  - overview KPI cards driven by real health/estimator data
 *  - empty: EmptyState "No analysis run yet" (before a run)
 */
export class CausalAnalysisPage extends BasePage {
  readonly url = '/causal-analysis'
  readonly pageTitle = /Causal Analysis|E2I|Causal Analytics/i

  constructor(page: Page) {
    super(page)
  }

  async goto(): Promise<void> {
    await gotoAndWaitForHeading(this.page, this.url, /Causal Analysis/i)
  }

  get pageHeader(): Locator {
    return this.page.getByRole('heading', { name: /Causal Analysis/i }).first()
  }

  get pageDescription(): Locator {
    return this.page.getByText(/Agent-driven causal inference/i).first()
  }

  get healthyBanner(): Locator {
    return this.page.getByText('Causal Engine Healthy', { exact: true }).first()
  }

  get serviceIssueBanner(): Locator {
    return this.page.getByText('Service Issue', { exact: true }).first()
  }

  get librariesCard(): Locator {
    return this.page.getByText('Libraries', { exact: true }).first()
  }

  get estimatorsCard(): Locator {
    return this.page.getByText('Estimators', { exact: true }).first()
  }

  // Honest empty state on the default (Analysis) tab before a run.
  get emptyState(): Locator {
    return this.page.getByText('No analysis run yet', { exact: true }).first()
  }

  get runAnalysisButton(): Locator {
    return this.page.getByRole('button', { name: /Run Analysis/i }).first()
  }

  get analysisTab(): Locator {
    return this.page.getByRole('tab', { name: /Analysis/i }).first()
  }
}
