/**
 * Unit tests for the stack's buffer management and tier ordering.
 *
 * `RenderTargetPool` is the piece of `PostStack` that can be exercised without
 * a GPU: `THREE.RenderTarget` is an ordinary object until something renders
 * into it, so allocation, reuse and format separation are all testable here.
 * The invariant that matters is that a chain of N passes allocates a bounded
 * number of buffers, not N of them.
 */

import { describe, expect, it } from 'vitest';
import * as THREE from 'three/webgpu';

import { RenderTargetPool, tierAtLeast } from '../src/render/post/PostStack';

describe('tierAtLeast', () => {
  it('orders the tiers', () => {
    expect(tierAtLeast('ultra', 'low')).toBe(true);
    expect(tierAtLeast('high', 'high')).toBe(true);
    expect(tierAtLeast('medium', 'high')).toBe(false);
    expect(tierAtLeast('low', 'medium')).toBe(false);
  });
});

describe('RenderTargetPool', () => {
  it('hands back the same buffer once it has been released', () => {
    const pool = new RenderTargetPool(64, 32);
    const first = pool.acquire('hdr');
    pool.release(first);
    expect(pool.acquire('hdr')).toBe(first);
    expect(pool.count).toBe(1);
    pool.dispose();
  });

  it('allocates a second buffer only while the first is still in use', () => {
    const pool = new RenderTargetPool(64, 32);
    const a = pool.acquire('hdr');
    const b = pool.acquire('hdr');
    expect(b).not.toBe(a);
    expect(pool.count).toBe(2);
    pool.dispose();
  });

  it('keeps a ping-pong chain of any length at two buffers', () => {
    // The invariant the whole design exists for: passes come and go, buffers do
    // not. Simulate ten chained passes, releasing the input after each.
    const pool = new RenderTargetPool(1920, 1080);
    let held: THREE.RenderTarget | null = pool.acquire('hdr');
    for (let i = 0; i < 10; i++) {
      const output = pool.acquire('hdr');
      pool.release(held);
      held = output;
    }
    expect(pool.count).toBe(2);
    pool.dispose();
  });

  it('does not mix HDR and LDR buffers', () => {
    const pool = new RenderTargetPool(64, 32);
    const hdr = pool.acquire('hdr');
    pool.release(hdr);
    const ldr = pool.acquire('ldr');
    expect(ldr).not.toBe(hdr);
    expect(hdr.texture.type).toBe(THREE.HalfFloatType);
    expect(ldr.texture.type).toBe(THREE.UnsignedByteType);
    pool.dispose();
  });

  it('falls back to 8-bit when half float is unavailable', () => {
    const pool = new RenderTargetPool(64, 32, false);
    expect(pool.acquire('hdr').texture.type).toBe(THREE.UnsignedByteType);
    pool.dispose();
  });

  it('tags intermediates as linear so sampling applies no transfer function', () => {
    // Every buffer in the chain holds linear data, or — for the composite's
    // hand-off to a spatial AA pass — bytes that must be read back exactly as
    // stored. Either way, a decode on sample would be wrong.
    const pool = new RenderTargetPool(64, 32);
    for (const domain of ['hdr', 'ldr'] as const) {
      expect(pool.acquire(domain).texture.colorSpace).toBe(THREE.LinearSRGBColorSpace);
    }
    pool.dispose();
  });

  it('carries no depth or MSAA on intermediates', () => {
    const pool = new RenderTargetPool(64, 32);
    const target = pool.acquire('hdr');
    expect(target.depthBuffer).toBe(false);
    expect(target.samples).toBe(0);
    expect(target.texture.generateMipmaps).toBe(false);
    pool.dispose();
  });

  it('resizes every buffer it owns, including the free ones', () => {
    const pool = new RenderTargetPool(64, 32);
    const a = pool.acquire('hdr');
    const b = pool.acquire('ldr');
    pool.release(a);
    pool.setSize(128, 96);
    expect(a.width).toBe(128);
    expect(a.height).toBe(96);
    expect(b.width).toBe(128);
    pool.dispose();
  });

  it('reports its footprint from the real formats', () => {
    const pool = new RenderTargetPool(100, 100);
    pool.acquire('hdr'); // 100 * 100 * 8
    pool.acquire('ldr'); // 100 * 100 * 4
    expect(pool.bytes).toBe(100 * 100 * 8 + 100 * 100 * 4);
    pool.dispose();
  });

  it('releaseAll recovers buffers a throwing pass forgot to hand back', () => {
    const pool = new RenderTargetPool(64, 32);
    const a = pool.acquire('hdr');
    const b = pool.acquire('hdr');
    pool.releaseAll();
    // Both come back to the free list, so the next two acquisitions reuse them
    // rather than allocating. The order they return in is not part of the
    // contract; the count is.
    const first = pool.acquire('hdr');
    const second = pool.acquire('hdr');
    expect(new Set([first, second])).toEqual(new Set([a, b]));
    expect(pool.count).toBe(2);
    pool.dispose();
  });

  it('ignores a double release rather than duplicating a buffer', () => {
    const pool = new RenderTargetPool(64, 32);
    const a = pool.acquire('hdr');
    pool.release(a);
    pool.release(a);
    const first = pool.acquire('hdr');
    const second = pool.acquire('hdr');
    // If the double release had duplicated the entry, these would be the same
    // buffer being written by two passes at once.
    expect(first).toBe(a);
    expect(second).not.toBe(a);
    pool.dispose();
  });

  it('tolerates a null release', () => {
    const pool = new RenderTargetPool(64, 32);
    expect(() => pool.release(null)).not.toThrow();
    pool.dispose();
  });

  it('clamps to a non-degenerate size', () => {
    const pool = new RenderTargetPool(0, -4);
    expect(pool.width).toBeGreaterThan(0);
    expect(pool.height).toBeGreaterThan(0);
    pool.dispose();
  });
});
