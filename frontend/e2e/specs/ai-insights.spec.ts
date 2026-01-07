/**
 * AI Agent Insights Page E2E Tests
 * =================================
 *
 * Tests for the AI Agent Insights page functionality.
 */

import { test, expect } from '@playwright/test';

test.describe('AI Agent Insights Page', () => {
  test.beforeEach(async ({ page }) => {
    // Navigate to AI Insights page
    await page.goto('/ai-insights');
    // Wait for page to load
    await page.waitForLoadState('networkidle');
  });

  test.describe('Page Load', () => {
    test('should load successfully', async ({ page }) => {
      await expect(page).toHaveURL(/ai-insights/);
    });

    test('should display page title', async ({ page }) => {
      await expect(page.getByRole('heading', { name: /AI Agent Insights/i })).toBeVisible();
    });

    test('should display page description', async ({ page }) => {
      await expect(page.getByText(/GPT-powered executive summaries/i)).toBeVisible();
    });

    test('should show active agents badge', async ({ page }) => {
      await expect(page.getByText(/Agents Active/i)).toBeVisible();
    });
  });

  test.describe('Executive AI Brief', () => {
    test('should display Executive AI Brief section', async ({ page }) => {
      await expect(page.getByText(/Executive AI Brief/i)).toBeVisible();
    });
  });

  test.describe('Priority Actions', () => {
    test('should display Priority Actions section', async ({ page }) => {
      await expect(page.getByText(/Priority Actions/i)).toBeVisible();
    });
  });

  test.describe('Predictive Alerts', () => {
    test('should display Predictive Alerts section', async ({ page }) => {
      await expect(page.getByText('Predictive Alerts', { exact: true })).toBeVisible();
    });
  });

  test.describe('Active Causal Chains', () => {
    test('should display Active Causal Chains section', async ({ page }) => {
      await expect(page.getByText(/Active Causal Chains/i)).toBeVisible();
    });

    test('should have zoom controls', async ({ page }) => {
      // Check for zoom buttons
      const zoomInButton = page.locator('button').filter({ has: page.locator('[class*="zoom"]') }).first();
      await expect(zoomInButton).toBeVisible();
    });
  });

  test.describe('Experiment Recommendations', () => {
    test('should display Experiment Recommendations section', async ({ page }) => {
      await expect(page.getByText(/Experiment Recommendations/i)).toBeVisible();
    });
  });

  test.describe('Heterogeneous Treatment Effects', () => {
    test('should display HTE section', async ({ page }) => {
      await expect(page.getByText('Heterogeneous Treatment Effects', { exact: true })).toBeVisible();
    });

    test('should show CATE info', async ({ page }) => {
      await expect(page.getByText('CATE Analysis:', { exact: true })).toBeVisible();
    });
  });

  test.describe('System Health Score', () => {
    test('should display System Health Score section', async ({ page }) => {
      // CardTitle renders as div, so use text matcher with first()
      await expect(page.getByText('System Health Score').first()).toBeVisible();
    });

    test('should display health metrics', async ({ page }) => {
      // Check for at least one health metric
      await expect(page.getByText('Model Drift')).toBeVisible();
    });
  });

  test.describe('Responsive Design', () => {
    test('should work on mobile viewport', async ({ page }) => {
      await page.setViewportSize({ width: 375, height: 812 });
      await expect(page.getByRole('heading', { name: /AI Agent Insights/i })).toBeVisible();
    });

    test('should work on tablet viewport', async ({ page }) => {
      await page.setViewportSize({ width: 768, height: 1024 });
      await expect(page.getByRole('heading', { name: /AI Agent Insights/i })).toBeVisible();
    });

    test('should work on desktop viewport', async ({ page }) => {
      await page.setViewportSize({ width: 1920, height: 1080 });
      await expect(page.getByRole('heading', { name: /AI Agent Insights/i })).toBeVisible();
    });
  });
});
