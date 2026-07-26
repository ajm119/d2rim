/**
 * Guards the orientation of fetched HDR environment plates.
 *
 * An upside-down IBL is one of the nastiest bugs in this area because it never
 * looks like an orientation bug. The scene simply goes flat and slightly
 * uplit — contact shadows fill in and the key-to-fill ratio collapses — which
 * reads as a tone-mapping or exposure problem and sends you looking in entirely
 * the wrong place.
 *
 * These tests parse the actual shipped `.exr` files, so they fail if a future
 * re-fetch, a different plate, or an EXRLoader change alters the convention.
 */

import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';

import { EXRLoader } from 'three/examples/jsm/loaders/EXRLoader.js';
import * as THREE from 'three/webgpu';
import { describe, expect, it } from 'vitest';

import { flipEquirectVertically } from '../src/assets/AssetManager';

const ROOT = resolve(__dirname, '..');

/**
 * Parse a shipped plate into the same `DataTexture` shape `EXRLoader.load`
 * produces.
 *
 * `parse()` alone returns a bare `{ data, width, height, format, type }`
 * record; it is `load()` that wraps it in a texture. Reproducing that wrapping
 * here is what makes this a test of the real runtime path rather than of a
 * convenient approximation of it.
 */
function loadPlate(
  name: string,
  /**
   * `FloatType` yields real radiance, which comparisons of absolute brightness
   * need. The default `HalfFloatType` — what the runtime actually uses — yields
   * a `Uint16Array` of raw half-float bit patterns; those happen to sort in the
   * same order as their values for positive floats, so they are still valid for
   * *ordering* tests, but arithmetic on them is meaningless.
   */
  dataType: typeof THREE.FloatType | typeof THREE.HalfFloatType = THREE.FloatType,
): THREE.DataTexture {
  const buffer = readFileSync(resolve(ROOT, `public/assets/hdri/${name}.exr`));
  const loader = new EXRLoader();
  loader.setDataType(dataType);
  const parsed = loader.parse(
    buffer.buffer.slice(buffer.byteOffset, buffer.byteOffset + buffer.byteLength),
  ) as unknown as {
    /** `Float32Array` for `FloatType`, `Uint16Array` for `HalfFloatType`. */
    data: Float32Array | Uint16Array;
    width: number;
    height: number;
    format: THREE.PixelFormat;
    type: THREE.TextureDataType;
  };
  return new THREE.DataTexture(
    parsed.data,
    parsed.width,
    parsed.height,
    parsed.format,
    parsed.type,
  );
}

/**
 * Solid-angle-weighted mean luminance of the top and bottom halves.
 *
 * Weighted by `sin(theta)` because an equirect wildly over-samples the poles;
 * an unweighted mean would let a small bright zenith patch dominate.
 */
function hemisphereLuminance(texture: THREE.DataTexture): { top: number; bottom: number } {
  const image = texture.image as unknown as {
    data: ArrayLike<number>;
    width: number;
    height: number;
  };
  const { data, width, height } = image;
  const channels = data.length / (width * height);

  let top = 0;
  let topWeight = 0;
  let bottom = 0;
  let bottomWeight = 0;

  for (let y = 0; y < height; y++) {
    const weight = Math.sin(((y + 0.5) / height) * Math.PI);
    for (let x = 0; x < width; x++) {
      const i = (y * width + x) * channels;
      const luminance = 0.2126 * data[i]! + 0.7152 * data[i + 1]! + 0.0722 * data[i + 2]!;
      if (y < height / 2) {
        top += luminance * weight;
        topWeight += weight;
      } else {
        bottom += luminance * weight;
        bottomWeight += weight;
      }
    }
  }
  return { top: top / topWeight, bottom: bottom / bottomWeight };
}

describe('HDRI orientation', () => {
  it('decodes EXR plates with the ground at row 0, before correction', () => {
    // Documents the upstream convention this code compensates for. If a future
    // three.js changes it, this test fails first and explains why.
    const raw = loadPlate('overcast');
    const { top, bottom } = hemisphereLuminance(raw);
    expect(bottom).toBeGreaterThan(top);
  });

  it('puts the bright sky hemisphere at row 0 after correction', () => {
    // Row 0 is the zenith for `scene.environment`, so the sky must end up there
    // or the scene is lit from below.
    for (const name of ['overcast', 'dusk']) {
      const texture = loadPlate(name);
      flipEquirectVertically(texture);
      const { top, bottom } = hemisphereLuminance(texture);
      expect(top, `${name}: zenith half should be brighter than the nadir half`).toBeGreaterThan(
        bottom,
      );
    }
  });

  it('is an involution — flipping twice restores the original', () => {
    // Deliberately the runtime dtype (half-float), so the row-swap is proven
    // against the typed array the game actually loads.
    const once = loadPlate('overcast', THREE.HalfFloatType);
    const twice = loadPlate('overcast', THREE.HalfFloatType);
    flipEquirectVertically(twice);
    flipEquirectVertically(twice);

    const a = (once.image as unknown as { data: ArrayLike<number> }).data;
    const b = (twice.image as unknown as { data: ArrayLike<number> }).data;
    expect(a.length).toBe(b.length);
    // Spot-check rather than compare 512*256*4 values one by one.
    for (let i = 0; i < a.length; i += 997) expect(b[i]).toBe(a[i]);
  });

  it('keeps the cave plate far darker than the exterior plates', () => {
    // Guards the selection rationale, not just the plumbing: the Den of Evil
    // needs near-zero ambient so torches dominate.
    const cave = loadPlate('cave');
    const overcast = loadPlate('overcast');
    flipEquirectVertically(cave);
    flipEquirectVertically(overcast);
    const caveLuminance = hemisphereLuminance(cave);
    const overcastLuminance = hemisphereLuminance(overcast);
    expect(caveLuminance.top + caveLuminance.bottom).toBeLessThan(
      (overcastLuminance.top + overcastLuminance.bottom) * 0.25,
    );
  });
});
