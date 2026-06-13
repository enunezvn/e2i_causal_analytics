import { Page, Locator } from '@playwright/test'
import { BasePage } from './base.page'
import { gotoAndWaitForHeading } from '../fixtures/page-harness'

/**
 * Page Object Model for the Causal Analysis page (`/causal-analysis`).
 *
 * Multi-library causal inference dashboard (DoWhy / EconML / CausalML) with
 * hierarchical CATE estimation. Honest states this POM exposes:
 *  - healthy banner: "Causal Engine Healthy" (from GET /api/causal/health)
 *  - degraded banner: "Service Issue" when status != 'healthy'
 *  - overview KPI cards driven by real health/estimator data
 *  - empty: EmptyState "No hierarchical CATE analysis available" (before a run)
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
    return this.page.getByText(/Multi-library causal inference/i).first()
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

  // Honest empty state on the default (Hierarchical CATE) tab before a run.
  get emptyState(): Locator {
    return this.page.getByText('No hierarchical CATE analysis available', { exact: true }).first()
  }

  get runAnalysisButton(): Locator {
    return this.page.getByRole('button', { name: /Run Analysis/i }).first()
  }

  get hierarchicalTab(): Locator {
    return this.page.getByRole('tab', { name: /Hierarchical CATE/i }).first()
  }
}
