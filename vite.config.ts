import { fileURLToPath, URL } from 'node:url';
// `vitest/config` re-exports Vite's `defineConfig` widened with the `test`
// block, so a single config file serves both the dev server and the test run.
import { defineConfig } from 'vitest/config';

/**
 * Vite configuration for d2rim.
 *
 * Notes:
 * - `three/webgpu` is the single three.js entry point used by game code. The
 *   classic `three` build is never imported at runtime (only `import type`),
 *   which keeps exactly one copy of the core classes in the bundle so that
 *   `instanceof` checks stay sound.
 * - `@dimforge/rapier3d-compat` ships a base64-inlined WASM blob and must be
 *   excluded from dependency pre-bundling optimisation quirks; it is listed in
 *   `optimizeDeps.include` so its CJS interop is resolved once, up front.
 * - `target: 'esnext'` is required for top-level `await`, which the renderer
 *   bootstrap uses.
 */
export default defineConfig({
  resolve: {
    alias: {
      '@': fileURLToPath(new URL('./src', import.meta.url)),
    },
  },
  build: {
    target: 'esnext',
    sourcemap: true,
    chunkSizeWarningLimit: 2048,
    rollupOptions: {
      output: {
        manualChunks: {
          three: ['three/webgpu', 'three/tsl'],
        },
      },
    },
  },
  esbuild: {
    target: 'esnext',
  },
  optimizeDeps: {
    include: ['three/webgpu', 'three/tsl', '@dimforge/rapier3d-compat'],
  },
  server: {
    host: '127.0.0.1',
    port: 5173,
  },
  preview: {
    host: '127.0.0.1',
    port: 4173,
  },
  test: {
    environment: 'node',
    include: ['tests/**/*.test.ts'],
  },
});
