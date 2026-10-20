// @ts-check
import { defineConfig } from 'astro/config';

// GitHub Pages project site: https://ameyamprojects.github.io/nba-predictor/
// Every internal link goes through src/lib/url.ts so the base path is never hard-coded.
export default defineConfig({
  site: 'https://ameyamprojects.github.io',
  base: '/nba-predictor',
  trailingSlash: 'always',
  prefetch: { prefetchAll: true, defaultStrategy: 'hover' },
  build: { inlineStylesheets: 'always', format: 'directory' },
});
