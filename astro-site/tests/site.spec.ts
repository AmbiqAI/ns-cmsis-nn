// SPDX-License-Identifier: LicenseRef-Ambiq-Apollo-SDK
import { test, expect } from '@playwright/test';

for (const colorScheme of ['light', 'dark'] as const) {
  for (const width of [1437, 390, 320]) {
    test(`navigation and API remain usable at ${width}px in ${colorScheme}`, async ({ page }) => {
      await page.emulateMedia({ colorScheme });
      const errors: string[] = [];
      page.on('pageerror', error => errors.push(error.message));
      await page.setViewportSize({ width, height: 998 });
      for (const route of ['', 'getting-started/first-kernel/', 'guide/', 'guide/coverage/data-types-by-family/', 'reference/kernel-index/', 'reference/api/heliacore/nnconv/']) {
        const response = await page.goto(`/ns-cmsis-nn/${route}`);
        expect(response?.status()).toBe(200);
        await expect(page.locator('h1')).toBeVisible();
        await page.evaluate(() => document.fonts.ready);
        expect(await page.evaluate(() => document.documentElement.scrollWidth > innerWidth)).toBe(false);
      }
      await page.screenshot({ path: `test-results/api-${width}-${colorScheme}.png`, fullPage: false });
      expect(errors).toEqual([]);
    });
  }
}

test('unknown legacy page returns 404 and leads to Home', async ({ page }) => {
  const response = await page.goto('/ns-cmsis-nn/api/function_old_kernel.html');
  expect(response?.status()).toBe(404);
  await expect(page.getByRole('heading', { name: 'This page has moved' })).toBeVisible();
  await expect(page.getByRole('link', { name: 'API reference', exact: true }).last()).toHaveAttribute('href', '/ns-cmsis-nn/reference/');
  await page.screenshot({ path: 'test-results/not-found.png' });
  await page.waitForURL('**/ns-cmsis-nn/', { timeout: 10000 });
  await expect(page.locator('h1')).toBeVisible();
});

test('authored legacy pages keep their destination', async ({ page }) => {
  await page.goto('/ns-cmsis-nn/getting-started/cmake.html');
  await page.waitForURL('**/ns-cmsis-nn/getting-started/cmake/');
  await expect(page.locator('h1')).toBeVisible();
});
