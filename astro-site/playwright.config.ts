// SPDX-License-Identifier: LicenseRef-Ambiq-Apollo-SDK
import { defineConfig } from '@playwright/test';

export default defineConfig({
  testDir: './tests',
  use: { baseURL: 'http://127.0.0.1:14473', headless: true },
  webServer: {
    command: 'helia-ui-serve-dist --dist dist --base /ns-cmsis-nn --port 14473',
    url: 'http://127.0.0.1:14473/ns-cmsis-nn/',
    reuseExistingServer: false,
  },
});
