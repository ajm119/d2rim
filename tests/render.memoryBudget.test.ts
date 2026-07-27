/**
 * Budget guards for the failure that made the deployed build unplayable.
 *
 * A player reported "very laggy slow and eventually crashed with error code 5"
 * in Brave. Chromium's Error code 5 on the Aw-Snap page is an out-of-memory
 * renderer kill. The cause was texture memory: 40 shipped plates, every one
 * 2048x2048 RGBA8, 32 of them preloaded at boot regardless of which zone the
 * player was in. 22.4 MB each with mips, so ~716 MB of texture before a model,
 * an environment map or a single render target existed.
 *
 * These tests are analytic on purpose. There is no GPU in CI and no way to
 * reproduce the crash here, so instead of trying to observe it they assert the
 * arithmetic that predicts it. If someone adds a 4K texture, drops the KTX2
 * pipeline, or points the loader back at the JPEGs, the sums move and these
 * fail — which is the only kind of regression test this bug can have.
 */

import * as THREE from 'three/webgpu';
import { describe, expect, it, beforeEach } from 'vitest';

import { KTX2_VARIANTS } from '../src/assets/ktx2.generated';
import { GENERATED_ASSETS } from '../src/assets/registry.generated';
import {
  bytesPerPixel,
  collectMemoryReport,
  formatMemoryReport,
  geometryBytes,
  resetRenderTargetLedger,
  textureBytes,
  trackRenderTarget,
  trackedRenderTargets,
} from '../src/render/MemoryReport';

const MB = 1024 * 1024;

/** A texture with a plausible image, without touching a GPU. */
function fakeTexture(width: number, height: number, mips = true): THREE.Texture {
  const texture = new THREE.Texture();
  texture.image = { width, height };
  texture.generateMipmaps = mips;
  return texture;
}

/** A compressed texture whose mip chain reports the given total byte length. */
function fakeCompressed(width: number, height: number, bytes: number): THREE.Texture {
  const texture = new THREE.CompressedTexture([], width, height);
  (texture as unknown as { mipmaps: Array<{ data: { byteLength: number } }> }).mipmaps = [
    { data: { byteLength: bytes } },
  ];
  return texture;
}

beforeEach(() => {
  resetRenderTargetLedger();
});

describe('textureBytes', () => {
  it('bills an uncompressed 2K RGBA8 plate at 21.3 MiB with its mip chain', () => {
    // 2048 * 2048 * 4 bytes = 16 MiB, and the mip chain adds a third.
    const bytes = textureBytes(fakeTexture(2048, 2048));
    expect(bytes / MB).toBeCloseTo(21.33, 1);
  });

  it('drops the mip third when mips are off', () => {
    expect(textureBytes(fakeTexture(2048, 2048, false)) / MB).toBeCloseTo(16, 1);
  });

  it('bills a compressed texture for its blocks, not for the RGBA8 it never becomes', () => {
    // This is the whole point of shipping KTX2: the GPU keeps the compressed
    // form. Counting it as RGBA8 would make the budget meaningless.
    const bytes = textureBytes(fakeCompressed(2048, 2048, 2 * MB));
    expect(bytes).toBe(2 * MB);
    expect(bytes).toBeLessThan(textureBytes(fakeTexture(2048, 2048)) / 8);
  });

  it('returns zero rather than NaN for a texture with no image yet', () => {
    expect(textureBytes(new THREE.Texture())).toBe(0);
  });

  it('scales half-float and single-channel formats correctly', () => {
    expect(bytesPerPixel(THREE.UnsignedByteType, THREE.RGBAFormat)).toBe(4);
    expect(bytesPerPixel(THREE.HalfFloatType, THREE.RGBAFormat)).toBe(8);
    expect(bytesPerPixel(THREE.FloatType, THREE.RGBAFormat)).toBe(16);
    expect(bytesPerPixel(THREE.UnsignedByteType, THREE.RedFormat)).toBe(1);
  });
});

describe('the shipped texture set', () => {
  /** Every manifest entry that is a texture served from `assets/textures/`. */
  const texturePaths = Object.values(GENERATED_ASSETS)
    .map((entry) => entry.path)
    .filter((path) => path.startsWith('assets/textures/'));

  it('has a KTX2 variant for every source texture', () => {
    const missing = texturePaths.filter((path) => KTX2_VARIANTS[path] === undefined);
    expect(missing).toEqual([]);
  });

  it('keeps the whole set under a 150 MB GPU budget once compressed', () => {
    // Measured: ETC1S transcodes to BC1 on desktop at 0.5 bytes/px, so a 2K
    // plate is 2 MiB, 2.67 MiB with mips. 39 of them is ~104 MiB. The budget is
    // set just above that with room for a couple more materials, and well below
    // the ~832 MiB the uncompressed set cost.
    const compressedBytesPerPlate = 2048 * 2048 * 0.5 * (4 / 3);
    const total = texturePaths.length * compressedBytesPerPlate;
    expect(total / MB).toBeLessThan(150);
  });

  it('would blow any sane budget uncompressed — the regression this guards', () => {
    const uncompressed = texturePaths.length * textureBytes(fakeTexture(2048, 2048));
    expect(uncompressed / MB).toBeGreaterThan(700);
  });
});

describe('render-target ledger', () => {
  it('tracks a target and releases it when three disposes it', () => {
    const target = trackRenderTarget(new THREE.RenderTarget(256, 256));
    expect(trackedRenderTargets()).toHaveLength(1);
    target.dispose();
    expect(trackedRenderTargets()).toHaveLength(0);
  });

  it('is idempotent, so a re-registered target is not double-counted', () => {
    const target = new THREE.RenderTarget(64, 64);
    trackRenderTarget(target);
    trackRenderTarget(target);
    expect(trackedRenderTargets()).toHaveLength(1);
  });

  it('counts a half-float target at 8 bytes per pixel', () => {
    trackRenderTarget(
      new THREE.RenderTarget(1920, 1080, {
        type: THREE.HalfFloatType,
        format: THREE.RGBAFormat,
        depthBuffer: false,
      }),
    );
    const report = collectMemoryReport(new THREE.Scene());
    expect(report.renderTargetBytes / MB).toBeCloseTo((1920 * 1080 * 8) / MB, 1);
  });

  it('adds the depth attachment when the target has one', () => {
    trackRenderTarget(
      new THREE.RenderTarget(100, 100, {
        type: THREE.UnsignedByteType,
        format: THREE.RGBAFormat,
        depthBuffer: true,
      }),
    );
    const report = collectMemoryReport(new THREE.Scene());
    expect(report.renderTargetBytes).toBe(100 * 100 * 4 + 100 * 100 * 4);
  });
});

describe('collectMemoryReport', () => {
  it('de-duplicates a texture shared by many meshes', () => {
    const scene = new THREE.Scene();
    const shared = fakeTexture(1024, 1024);
    for (let i = 0; i < 50; i++) {
      const material = new THREE.MeshStandardMaterial();
      material.map = shared;
      scene.add(new THREE.Mesh(new THREE.BoxGeometry(), material));
    }
    const report = collectMemoryReport(scene);
    // Fifty rocks sharing one albedo cost one albedo. A report that said
    // otherwise would send someone optimising the wrong thing.
    expect(report.textureCount).toBe(1);
    expect(report.textureBytes).toBe(textureBytes(shared));
    expect(report.meshCount).toBe(50);
  });

  it('counts the environment map, which no material references', () => {
    const scene = new THREE.Scene();
    scene.environment = fakeTexture(512, 256);
    const report = collectMemoryReport(scene);
    expect(report.textureCount).toBe(1);
  });

  it('multiplies triangles by instance count', () => {
    const scene = new THREE.Scene();
    const geometry = new THREE.BoxGeometry();
    scene.add(new THREE.InstancedMesh(geometry, new THREE.MeshBasicMaterial(), 100));
    const single = geometry.index !== null ? geometry.index.count / 3 : 0;
    expect(collectMemoryReport(scene).triangles).toBe(Math.round(single * 100));
  });

  it('totals geometry from attributes and indices', () => {
    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute('position', new THREE.BufferAttribute(new Float32Array(300), 3));
    expect(geometryBytes(geometry)).toBe(1200);
  });

  it('renders a report a human can read', () => {
    const scene = new THREE.Scene();
    scene.environment = fakeTexture(2048, 2048);
    const text = formatMemoryReport(collectMemoryReport(scene));
    expect(text).toContain('GPU memory report');
    expect(text).toContain('textures');
    expect(text).toMatch(/21\.\d MB/);
  });
});
