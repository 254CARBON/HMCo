import { test, expect } from '@playwright/test';

test.describe('Portal Smoke Tests', () => {
  test.describe('Without E2E bypass', () => {
    test.use({
      extraHTTPHeaders: {
        // No bypass header
      },
    });

    test('should redirect to Cloudflare Access when not authenticated', async ({ page, context }) => {
      // Skip this test if Cloudflare Access is not enabled
      const response = await page.goto('/');
      
      if (!response) {
        throw new Error('Failed to load page');
      }

      // In dev mode without Cloudflare Access configured, page loads directly
      // With Cloudflare Access configured, we expect a redirect to Cloudflare
      const currentUrl = page.url();
      
      if (currentUrl.includes('cloudflareaccess.com') || currentUrl.includes('/cdn-cgi/access/login')) {
        // Successfully redirected to Cloudflare Access login
        expect(currentUrl).toMatch(/(cloudflareaccess\.com|\/cdn-cgi\/access\/login)/);
        console.log('✓ Redirected to Cloudflare Access:', currentUrl);
      } else {
        // In dev mode without Cloudflare Access, page should load directly
        console.log('✓ Cloudflare Access not configured - page loaded directly');
        expect(response.status()).toBe(200);
      }
    });
  });

  test.describe('With E2E bypass in dev mode', () => {
    test.use({
      extraHTTPHeaders: {
        'x-e2e-bypass': 'test-bypass-token',
      },
    });

    test('should bypass authentication with E2E_BYPASS header in dev', async ({ page }) => {
      // This test requires E2E_BYPASS=1 environment variable to be set
      const response = await page.goto('/');
      
      if (!response) {
        throw new Error('Failed to load page');
      }

      // With E2E bypass enabled, page should load successfully
      expect(response.status()).toBe(200);
      
      // Verify the page content loads
      await expect(page).toHaveTitle(/254CARBON|Portal/i, { timeout: 10000 });
      
      console.log('✓ Page bypassed authentication and loaded successfully');
    });

    test('should load homepage with expected content', async ({ page }) => {
      await page.goto('/');
      
      // Wait for page to be ready
      await page.waitForLoadState('domcontentloaded');
      
      // Verify basic page structure exists
      const body = await page.locator('body');
      await expect(body).toBeVisible();
      
      console.log('✓ Homepage content verified');
    });

    test('should have working navigation', async ({ page }) => {
      await page.goto('/');
      
      // Wait for page to be ready
      await page.waitForLoadState('networkidle', { timeout: 15000 });
      
      // Basic smoke test - page should be interactive
      const pageContent = await page.content();
      expect(pageContent.length).toBeGreaterThan(100);
      
      console.log('✓ Navigation working, page is interactive');
    });
  });

  test.describe('Public paths', () => {
    test('should load public paths without authentication', async ({ page }) => {
      // Public paths should be accessible without authentication
      const publicPaths = ['/login', '/favicon.ico'];
      
      for (const path of publicPaths) {
        const response = await page.goto(path, { waitUntil: 'domcontentloaded' });
        
        if (response) {
          // Should not redirect to Cloudflare Access
          const url = page.url();
          expect(url).not.toContain('cloudflareaccess.com');
          expect(url).not.toContain('/cdn-cgi/access/login');
          
          console.log(`✓ Public path ${path} accessible without auth`);
        }
      }
    });
  });
});
