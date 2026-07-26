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
    // Array form so the `three` entry can be an exact-match regex. The object
    // form does prefix matching, which would rewrite `three/webgpu` itself into
    // `three/webgpu/webgpu`.
    alias: [
      { find: '@', replacement: fileURLToPath(new URL('./src', import.meta.url)) },
      // ONE copy of three, and therefore one set of class identities.
      //
      // `three/webgpu` is a complete build: it contains the whole classic core
      // *plus* the node/TSL renderer. `three/examples/jsm/*` — GLTFLoader,
      // RGBELoader, KTX2Loader — import bare `'three'`, which resolves to the
      // classic build. Without this alias the bundle ships both, and every
      // object the loaders produce is an instance of the *other* copy's
      // classes. `mesh instanceof THREE.Mesh` is then false for every mesh in
      // every loaded model, silently, and so is every `instanceof` on their
      // materials — which is exactly the kind of failure that shows up as
      // "the props are the wrong colour" three layers away from its cause.
      //
      // Aliasing the bare specifier onto the webgpu build collapses the two
      // into one module, and drops the duplicated core from the bundle.
      { find: /^three$/, replacement: 'three/webgpu' },
    ],
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
