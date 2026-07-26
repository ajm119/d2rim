/**
 * Tests for the AssetManager's cache, deduplication and licence-tier guard.
 *
 * These exercise the paths that do not need a GPU: registry lookup, URL
 * resolution, the LRU, pinning, in-flight coalescing, and the guard that stops
 * a licence-unresolved asset being loaded by accident. Decoding itself (EXR,
 * GLTF, KTX2) needs a real renderer and is covered by the headless capture.
 *
 * `loadBinary` is the vehicle throughout because it goes through plain `fetch`,
 * which can be stubbed, while exercising exactly the same `#acquire` path that
 * every typed loader uses.
 */

import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import { AssetManager } from '../src/assets/AssetManager';

const FOX = 'creature.fox';
const ROBOT = 'character.robot';

/** Stub `fetch` with a per-URL byte count, and count how often it is called. */
function stubFetch(bytesPerCall = 1024): { calls: () => number; release: () => void } {
  let calls = 0;
  let releaseAll: () => void = () => {};
  const gate = new Promise<void>((resolve) => {
    releaseAll = resolve;
  });
  let gated = false;

  vi.stubGlobal('fetch', async (url: string) => {
    calls++;
    if (gated) await gate;
    return {
      ok: true,
      status: 200,
      statusText: 'OK',
      url,
      arrayBuffer: async () => new ArrayBuffer(bytesPerCall),
    };
  });

  return {
    calls: () => calls,
    release: () => {
      gated = false;
      releaseAll();
    },
  };
}

describe('AssetManager registry', () => {
  const manager = new AssetManager({ baseUrl: '/' });

  it('recognises registered keys and rejects unknown ones', () => {
    expect(manager.has(FOX)).toBe(true);
    expect(manager.has('not.a.real.key')).toBe(false);
  });

  it('resolves URLs against the base URL', () => {
    expect(manager.url(FOX)).toBe('/assets/models/Fox.glb');
    expect(new AssetManager({ baseUrl: '/game/' }).url(FOX)).toBe('/game/assets/models/Fox.glb');
    // A base URL without a trailing slash must not produce a glued path.
    expect(new AssetManager({ baseUrl: '/game' }).url(FOX)).toBe('/game/assets/models/Fox.glb');
  });

  it('exposes licence metadata for every registered asset', () => {
    for (const key of manager.keys()) {
      const entry = manager.entry(key);
      expect(entry.license).toMatch(/\S/);
      expect(entry.path.startsWith('assets/')).toBe(true);
    }
  });

  it('reports the Fox as CC BY 4.0, which requires attribution', () => {
    expect(manager.entry(FOX).license).toBe('CC-BY-4.0');
  });
});

describe('AssetManager caching', () => {
  let manager: AssetManager;

  beforeEach(() => {
    manager = new AssetManager({ baseUrl: '/' });
  });

  afterEach(() => {
    manager.dispose();
    vi.unstubAllGlobals();
  });

  it('fetches once and serves subsequent requests from cache', async () => {
    const stub = stubFetch();
    await manager.loadBinary(FOX);
    await manager.loadBinary(FOX);
    await manager.loadBinary(FOX);

    expect(stub.calls()).toBe(1);
    const stats = manager.stats();
    expect(stats.misses).toBe(1);
    expect(stats.hits).toBe(2);
    expect(stats.entries).toBe(1);
  });

  it('coalesces concurrent requests for the same key into one fetch', async () => {
    const stub = stubFetch();
    // The motivating case: a scene full of identical props all asking at once.
    const results = await Promise.all(Array.from({ length: 25 }, () => manager.loadBinary(FOX)));
    expect(stub.calls()).toBe(1);
    // All callers must receive the very same object, not copies.
    for (const result of results) expect(result).toBe(results[0]);
    expect(manager.stats().inFlight).toBe(0);
  });

  it('rejects unknown keys with a clear message', async () => {
    stubFetch();
    await expect(manager.loadBinary('nope' as never)).rejects.toThrow(/unknown asset key/);
  });

  it('refuses licence-review-required assets that were not fetched', async () => {
    stubFetch();
    // The guard must fire before any network access, and must say why.
    await expect(manager.loadBinary('character.bigvegas')).rejects.toThrow(
      /licence-review-required/,
    );
    await expect(manager.loadBinary('animation.xbot')).rejects.toThrow(
      /--include-review-required/,
    );
  });

  it('surfaces HTTP failures as errors naming the key and URL', async () => {
    vi.stubGlobal('fetch', async () => ({ ok: false, status: 404, statusText: 'Not Found' }));
    await expect(manager.loadBinary(FOX)).rejects.toThrow(/creature\.fox.*Fox\.glb/s);
  });

  it('does not cache a failed load', async () => {
    vi.stubGlobal('fetch', async () => ({ ok: false, status: 500, statusText: 'Server Error' }));
    await expect(manager.loadBinary(FOX)).rejects.toThrow();
    expect(manager.stats().entries).toBe(0);
    expect(manager.stats().inFlight).toBe(0);
  });

  it('peeks without loading', async () => {
    const stub = stubFetch();
    expect(manager.peek(FOX)).toBeUndefined();
    await manager.loadBinary(FOX);
    expect(manager.peek(FOX)).toBeDefined();
    expect(stub.calls()).toBe(1);
  });

  it('unloads and reloads', async () => {
    const stub = stubFetch();
    await manager.loadBinary(FOX);
    expect(manager.unload(FOX)).toBe(true);
    expect(manager.unload(FOX)).toBe(false);
    expect(manager.stats().entries).toBe(0);
    await manager.loadBinary(FOX);
    expect(stub.calls()).toBe(2);
  });

  it('evicts least-recently-used entries when over budget', async () => {
    stubFetch();
    // Budget below the recorded size of either asset, so the second load must
    // evict the first.
    const tight = new AssetManager({ baseUrl: '/', cacheBudgetBytes: 200_000 });
    await tight.loadBinary(FOX); // 162_852 bytes recorded in the registry
    expect(tight.stats().entries).toBe(1);

    await tight.loadBinary(ROBOT); // 463_988 bytes
    const stats = tight.stats();
    expect(stats.evictions).toBeGreaterThan(0);
    expect(stats.entries).toBeLessThan(2);
    tight.dispose();
  });

  it('never evicts a pinned entry', async () => {
    stubFetch();
    const tight = new AssetManager({ baseUrl: '/', cacheBudgetBytes: 200_000 });
    await tight.loadBinary(FOX);
    tight.pin(FOX);

    await tight.loadBinary(ROBOT);
    // The pinned entry survives even though the budget is blown.
    expect(tight.peek(FOX)).toBeDefined();
    expect(tight.stats().pinned).toBe(1);

    tight.unpin(FOX);
    expect(tight.stats().pinned).toBe(0);
    tight.dispose();
  });

  it('clear() spares pinned entries unless told otherwise', async () => {
    stubFetch();
    await manager.loadBinary(FOX);
    await manager.loadBinary(ROBOT);
    manager.pin(FOX);

    manager.clear();
    expect(manager.peek(FOX)).toBeDefined();
    expect(manager.peek(ROBOT)).toBeUndefined();

    manager.clear({ includePinned: true });
    expect(manager.peek(FOX)).toBeUndefined();
  });

  it('refuses to be used after dispose', async () => {
    stubFetch();
    manager.dispose();
    await expect(manager.loadBinary(FOX)).rejects.toThrow(/after dispose/);
  });

  it('accounts for every key and never short-circuits on a failure', async () => {
    // Both keys are role `gltf`, so `preload` routes them to `loadGLTF`, which
    // needs a renderer this test does not have. That makes this a clean test of
    // the aggregation contract specifically: a loading screen must get a verdict
    // for every asset it asked for, not stall on the first one that fails.
    vi.stubGlobal('fetch', async () => ({ ok: false, status: 404, statusText: 'Not Found' }));

    const { loaded, failed } = await manager.preload([FOX, ROBOT]);

    expect(loaded.length + failed.length).toBe(2);
    expect([...loaded, ...failed].sort()).toEqual([FOX, ROBOT].sort());
    expect(manager.stats().inFlight).toBe(0);
  });

  it('resolves an empty preload immediately', async () => {
    await expect(manager.preload([])).resolves.toEqual({ loaded: [], failed: [] });
  });

  it('dispatches by role: a gltf key requested as a texture is refused', async () => {
    stubFetch();
    await expect(manager.loadTexture(FOX)).rejects.toThrow(/is not a texture/);
    await expect(manager.loadEnvironment(FOX)).rejects.toThrow(/expected "hdri"/);
    await expect(manager.loadGLTF('env.overcast')).rejects.toThrow(/expected "gltf"/);
  });
});
