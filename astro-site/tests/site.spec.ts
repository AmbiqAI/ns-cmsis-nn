// SPDX-License-Identifier: LicenseRef-Ambiq-Apollo-SDK
import { test, expect } from '@playwright/test';
import type { RefModule, RefSymbol } from '@ambiqai/helia-ui/reference-model';

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


test('API Markdown and agent bundle preserve C contracts', async ({ request }) => {
  const modelResponse = await request.get('/ns-cmsis-nn/reference/api/reference.json');
  expect(modelResponse.ok()).toBe(true);
  const model = await modelResponse.json();
  const bundleResponse = await request.get('/ns-cmsis-nn/llms-full.txt');
  expect(bundleResponse.ok()).toBe(true);
  const bundle = await bundleResponse.text();
  const modules = (entries: RefModule[]): RefModule[] => entries.flatMap(module => [module, ...modules(module.submodules ?? [])]);
  const symbols = (entries: RefSymbol[]): RefSymbol[] => entries.flatMap(symbol => [symbol, ...symbols(symbol.members ?? [])]);
  for (const module of modules(model.modules)) {
    const route = module.path.toLowerCase().replaceAll('.', '/');
    const response = await request.get(`/ns-cmsis-nn/reference/api/${route}/index.md`);
    expect(response.ok(), module.path).toBe(true);
    const markdown = await response.text();
    for (const symbol of symbols(module.symbols ?? [])) {
      expect(markdown, symbol.id).toContain(symbol.signature);
      expect(bundle, symbol.id).toContain(symbol.signature);
    }
  }
  const convolution = await request.get('/ns-cmsis-nn/reference/api/heliacore/nnconv/index.md');
  const contract = await convolution.text();
  expect(contract).toContain('arm_convolve_nhwc_f32');
  expect(contract).toContain('**Parameters**');
  expect(contract).toContain('| Direction |');
  expect(contract).toContain('**Returns**');
});


test('kernel catalog combines search and facets and resets results', async ({ page }) => {
  await page.goto('/ns-cmsis-nn/reference/kernel-index/');
  const catalog = page.getByRole('region', { name: 'Find a kernel', exact: true });
  await expect(catalog).toHaveAttribute('data-ready', 'true');
  await catalog.getByRole('searchbox').fill('arm_add');
  await catalog.getByLabel('Operator group', { exact: true }).selectOption('Elementwise');
  await catalog.getByLabel('Data type', { exact: true }).selectOption('s8');
  await expect(catalog.getByRole('link', { name: 'arm_add_s8', exact: true })).toBeVisible();
  await expect(catalog.getByRole('link', { name: 'arm_add_s16', exact: true })).toHaveCount(0);
  await catalog.getByRole('searchbox').fill('no-such-kernel');
  await expect(catalog.getByText('No matching functions.')).toBeVisible();
  await catalog.getByRole('button', { name: 'Clear filters' }).first().click();
  await expect(catalog.locator('tbody tr')).toHaveCount(25);
  await catalog.getByRole('link', { name: 'arm_abs_s16', exact: true }).click();
  await expect(page).toHaveURL(/#arm_abs_s16$/);
});


test('kernel index Markdown lists the rendered catalog', async ({ request, page }) => {
  await page.goto('/ns-cmsis-nn/reference/kernel-index/');
  const sidecar = await page.locator('script[data-helia-rendition="reference-browser"]').textContent();
  const response = await request.get('/ns-cmsis-nn/reference/kernel-index/index.md');
  expect(response.ok()).toBe(true);
  const markdown = await response.text();
  expect(sidecar).toBeTruthy();
  for (const name of sidecar!.matchAll(/\[(arm_[^\]]+)\]/g)) {
    expect(markdown).toContain(`[${name[1]}]`);
  }
  expect(markdown).toContain('## Functions');
});
