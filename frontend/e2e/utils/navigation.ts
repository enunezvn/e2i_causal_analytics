import { Page } from '@playwright/test'
import { ROUTES, TIMEOUTS } from '../fixtures/test-data'

/**
 * Navigation helper utilities for E2E tests.
 */

/**
 * Navigate to a page and wait for it to load.
 */
export async function navigateTo(page: Page, route: string): Promise<void> {
  await page.goto(route)
  await page.waitForLoadState('networkidle')
}

/**
 * Navigate using the sidebar navigation.
 */
export async function navigateViaSidebar(page: Page, linkText: string): Promise<void> {
  const sidebar = page.locator('[data-testid="sidebar"], nav, aside').first()
  const link = sidebar.getByText(linkText, { exact: false })
  await link.click()
  await page.waitForLoadState('networkidle')
}

/**
 * Navigate to home page.
 */
export async function goToHome(page: Page): Promise<void> {
  await navigateTo(page, ROUTES.HOME)
}

/**
 * Navigate to causal discovery page.
 */
export async function goToCausalDiscovery(page: Page): Promise<void> {
  await navigateTo(page, ROUTES.CAUSAL_DISCOVERY)
}

/**
 * Navigate to knowledge graph page.
 */
export async function goToKnowledgeGraph(page: Page): Promise<void> {
  await navigateTo(page, ROUTES.KNOWLEDGE_GRAPH)
}

/**
 * Navigate to model performance page.
 */
export async function goToModelPerformance(page: Page): Promise<void> {
  await navigateTo(page, ROUTES.MODEL_PERFORMANCE)
}

/**
 * Navigate to feature importance page.
 */
export async function goToFeatureImportance(page: Page): Promise<void> {
  await navigateTo(page, ROUTES.FEATURE_IMPORTANCE)
}

/**
 * Navigate to time series page.
 */
export async function goToTimeSeries(page: Page): Promise<void> {
  await navigateTo(page, ROUTES.TIME_SERIES)
}

/**
 * Navigate to intervention impact page.
 */
export async function goToInterventionImpact(page: Page): Promise<void> {
  await navigateTo(page, ROUTES.INTERVENTION_IMPACT)
}

/**
 * Navigate to predictive analytics page.
 */
export async function goToPredictiveAnalytics(page: Page): Promise<void> {
  await navigateTo(page, ROUTES.PREDICTIVE_ANALYTICS)
}

/**
 * Navigate to data quality page.
 */
export async function goToDataQuality(page: Page): Promise<void> {
  await navigateTo(page, ROUTES.DATA_QUALITY)
}

/**
 * Navigate to system health page.
 */
export async function goToSystemHealth(page: Page): Promise<void> {
  await navigateTo(page, ROUTES.SYSTEM_HEALTH)
}

/**
 * Navigate to monitoring page.
 */
export async function goToMonitoring(page: Page): Promise<void> {
  await navigateTo(page, ROUTES.MONITORING)
}

/**
 * Navigate to agent orchestration page.
 */
export async function goToAgentOrchestration(page: Page): Promise<void> {
  await navigateTo(page, ROUTES.AGENT_ORCHESTRATION)
}

/**
 * Navigate to KPI dictionary page.
 */
export async function goToKPIDictionary(page: Page): Promise<void> {
  await navigateTo(page, ROUTES.KPI_DICTIONARY)
}

/**
 * Navigate to memory architecture page.
 */
export async function goToMemoryArchitecture(page: Page): Promise<void> {
  await navigateTo(page, ROUTES.MEMORY_ARCHITECTURE)
}

/**
 * Navigate to digital twin page.
 */
export async function goToDigitalTwin(page: Page): Promise<void> {
  await navigateTo(page, ROUTES.DIGITAL_TWIN)
}

/**
 * Wait for page to be ready.
 */
export async function waitForPageReady(page: Page, timeout = TIMEOUTS.PAGE_LOAD): Promise<void> {
  await page.waitForLoadState('domcontentloaded', { timeout })
  await page.waitForLoadState('networkidle', { timeout })
}

/**
 * Refresh the current page and wait for load.
 */
export async function refreshPage(page: Page): Promise<void> {
  await page.reload()
  await waitForPageReady(page)
}

/**
 * Go back in browser history.
 */
export async function goBack(page: Page): Promise<void> {
  await page.goBack()
  await waitForPageReady(page)
}

/**
 * Check if current URL matches expected route.
 */
export async function isOnRoute(page: Page, route: string): Promise<boolean> {
  const url = page.url()
  return url.endsWith(route) || url.includes(route)
}
