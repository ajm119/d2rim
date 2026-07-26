/**
 * @module render/ProceduralTextures
 *
 * CPU-side procedural texture synthesis. The project's asset policy is hybrid —
 * fetch CC0 material where it exists, generate the rest — and this module is
 * the "generate the rest" half for surfaces.
 *
 * Everything here is deterministic (seeded integer hashing, never
 * `Math.random`) so that golden-image capture stays reproducible, and every
 * generator produces seamlessly tiling output by wrapping its noise lattice.
 */

import * as THREE from 'three/webgpu';

/** Deterministic 32-bit hash -> `[0, 1)`. */
function hash2(x: number, y: number, seed: number): number {
  let h = (x * 1597334677 + y * 3812015801 + seed * 2654435761) | 0;
  h = Math.imul(h ^ (h >>> 15), 2246822519);
  h = Math.imul(h ^ (h >>> 13), 3266489917);
  h = h ^ (h >>> 16);
  return (h >>> 0) / 4294967296;
}

function smoothstep(t: number): number {
  return t * t * (3 - 2 * t);
}

/** Value noise on a lattice wrapping every `period` cells. */
function valueNoise(x: number, y: number, period: number, seed: number): number {
  const xi = Math.floor(x);
  const yi = Math.floor(y);
  const u = smoothstep(x - xi);
  const v = smoothstep(y - yi);

  const wrap = (n: number): number => ((n % period) + period) % period;
  const x0 = wrap(xi);
  const x1 = wrap(xi + 1);
  const y0 = wrap(yi);
  const y1 = wrap(yi + 1);

  const n00 = hash2(x0, y0, seed);
  const n10 = hash2(x1, y0, seed);
  const n01 = hash2(x0, y1, seed);
  const n11 = hash2(x1, y1, seed);

  return (n00 * (1 - u) + n10 * u) * (1 - v) + (n01 * (1 - u) + n11 * u) * v;
}

/** Tiling fBm in `[0, 1]`. */
function fbm(x: number, y: number, period: number, seed: number, octaves: number): number {
  let sum = 0;
  let total = 0;
  let amplitude = 0.5;
  let freq = 1;
  for (let i = 0; i < octaves; i++) {
    sum += valueNoise(x * freq, y * freq, period * freq, seed + i * 131) * amplitude;
    total += amplitude;
    amplitude *= 0.5;
    freq *= 2;
  }
  return sum / total;
}

/**
 * Tiling Worley (cellular) noise returning distance to the nearest feature
 * point, normalised to roughly `[0, 1]`. Used for the pebble/aggregate look of
 * packed dirt; pure fBm alone reads as fog rather than ground.
 */
function worley(x: number, y: number, period: number, seed: number): number {
  const xi = Math.floor(x);
  const yi = Math.floor(y);
  let best = 8;
  for (let oy = -1; oy <= 1; oy++) {
    for (let ox = -1; ox <= 1; ox++) {
      const cx = xi + ox;
      const cy = yi + oy;
      const wx = ((cx % period) + period) % period;
      const wy = ((cy % period) + period) % period;
      const px = cx + hash2(wx, wy, seed);
      const py = cy + hash2(wx, wy, seed + 977);
      const dx = px - x;
      const dy = py - y;
      const d = dx * dx + dy * dy;
      if (d < best) best = d;
    }
  }
  return Math.min(1, Math.sqrt(best));
}

export interface GroundMaterialMaps {
  /** sRGB base colour. */
  readonly map: THREE.DataTexture;
  /** Linear roughness; the green channel is what three.js reads. */
  readonly roughnessMap: THREE.DataTexture;
  /** Tangent-space normal map. */
  readonly normalMap: THREE.DataTexture;
  dispose(): void;
}

export interface GroundOptions {
  /** Texture edge length in texels. Default 512. */
  size?: number;
  seed?: number;
  /** How many times the maps repeat across the plane. Default 24. */
  repeat?: number;
  /** Height-field to normal-map strength. Default 2.2. */
  normalStrength?: number;
}

/**
 * Generate a coherent albedo/roughness/normal set for weathered packed earth —
 * the ground of the Blood Moor.
 *
 * The three maps share one height field, which is what makes the result read as
 * a single surface: crevices are simultaneously darker, rougher and lower.
 * Independently generated maps look like three unrelated materials stacked.
 */
export function createGroundMaterialMaps(options: GroundOptions = {}): GroundMaterialMaps {
  const size = options.size ?? 512;
  const seed = options.seed ?? 20250725;
  const repeat = options.repeat ?? 24;
  const normalStrength = options.normalStrength ?? 2.2;

  const texels = size * size;
  const height = new Float32Array(texels);
  const albedo = new Uint8Array(texels * 4);
  const roughness = new Uint8Array(texels * 4);
  const normal = new Uint8Array(texels * 4);

  // Lattice periods, in cells across the whole texture. All divide evenly so
  // the tiling stays seamless.
  const coarsePeriod = 4;
  const detailPeriod = 16;
  const grainPeriod = 64;

  for (let y = 0; y < size; y++) {
    for (let x = 0; x < size; x++) {
      const u = x / size;
      const v = y / size;

      const coarse = fbm(u * coarsePeriod, v * coarsePeriod, coarsePeriod, seed, 4);
      const detail = fbm(u * detailPeriod, v * detailPeriod, detailPeriod, seed + 17, 4);
      const pebbles = 1 - worley(u * grainPeriod, v * grainPeriod, grainPeriod, seed + 91);

      // Broad undulation, medium clumping, small aggregate.
      const h = coarse * 0.55 + detail * 0.30 + pebbles * 0.15;
      height[y * size + x] = h;
    }
  }

  // Two earth tones interpolated by height: damp dark soil in the hollows,
  // sun-bleached dust on the high points.
  const lowColor = { r: 0.055, g: 0.043, b: 0.032 };
  const highColor = { r: 0.30, g: 0.245, b: 0.175 };

  const toSrgbByte = (linear: number): number => {
    const c = Math.min(1, Math.max(0, linear));
    const s = c <= 0.0031308 ? c * 12.92 : 1.055 * Math.pow(c, 1 / 2.4) - 0.055;
    return Math.round(s * 255);
  };

  for (let y = 0; y < size; y++) {
    for (let x = 0; x < size; x++) {
      const idx = y * size + x;
      const h = height[idx] ?? 0;
      const t = Math.min(1, Math.max(0, (h - 0.28) / 0.5));
      const shade = 0.75 + 0.25 * t;

      const i4 = idx * 4;
      albedo[i4] = toSrgbByte((lowColor.r + (highColor.r - lowColor.r) * t) * shade);
      albedo[i4 + 1] = toSrgbByte((lowColor.g + (highColor.g - lowColor.g) * t) * shade);
      albedo[i4 + 2] = toSrgbByte((lowColor.b + (highColor.b - lowColor.b) * t) * shade);
      albedo[i4 + 3] = 255;

      // Packed hollows are polished smooth by traffic and water; raised dust is
      // fully rough. Never let it reach 1.0 — a perfectly rough dielectric loses
      // all grazing-angle sheen and looks like flat paper.
      const rough = 0.96 - 0.30 * t;
      const byte = Math.round(Math.min(1, Math.max(0, rough)) * 255);
      roughness[i4] = byte;
      roughness[i4 + 1] = byte;
      roughness[i4 + 2] = byte;
      roughness[i4 + 3] = 255;
    }
  }

  // Sobel the height field into a tangent-space normal map. Wrapping the sample
  // indices keeps the normals continuous across the tile seam.
  const at = (x: number, y: number): number =>
    height[(((y % size) + size) % size) * size + (((x % size) + size) % size)] ?? 0;

  for (let y = 0; y < size; y++) {
    for (let x = 0; x < size; x++) {
      const dx =
        at(x + 1, y - 1) + 2 * at(x + 1, y) + at(x + 1, y + 1) -
        (at(x - 1, y - 1) + 2 * at(x - 1, y) + at(x - 1, y + 1));
      const dy =
        at(x - 1, y + 1) + 2 * at(x, y + 1) + at(x + 1, y + 1) -
        (at(x - 1, y - 1) + 2 * at(x, y - 1) + at(x + 1, y - 1));

      let nx = -dx * normalStrength;
      let ny = -dy * normalStrength;
      let nz = 1;
      const inv = 1 / Math.hypot(nx, ny, nz);
      nx *= inv;
      ny *= inv;
      nz *= inv;

      const i4 = (y * size + x) * 4;
      normal[i4] = Math.round((nx * 0.5 + 0.5) * 255);
      normal[i4 + 1] = Math.round((ny * 0.5 + 0.5) * 255);
      normal[i4 + 2] = Math.round((nz * 0.5 + 0.5) * 255);
      normal[i4 + 3] = 255;
    }
  }

  const build = (data: Uint8Array, colorSpace: string): THREE.DataTexture => {
    const texture = new THREE.DataTexture(data, size, size, THREE.RGBAFormat, THREE.UnsignedByteType);
    texture.colorSpace = colorSpace;
    texture.wrapS = THREE.RepeatWrapping;
    texture.wrapT = THREE.RepeatWrapping;
    texture.repeat.set(repeat, repeat);
    texture.magFilter = THREE.LinearFilter;
    texture.minFilter = THREE.LinearMipmapLinearFilter;
    texture.generateMipmaps = true;
    // Grazing-angle ground is the worst case for mip blur; 8x anisotropy is the
    // difference between crisp terrain and a smear at the horizon.
    texture.anisotropy = 8;
    texture.needsUpdate = true;
    return texture;
  };

  const map = build(albedo, THREE.SRGBColorSpace);
  const roughnessMap = build(roughness, THREE.NoColorSpace);
  const normalMap = build(normal, THREE.NoColorSpace);

  return {
    map,
    roughnessMap,
    normalMap,
    dispose(): void {
      map.dispose();
      roughnessMap.dispose();
      normalMap.dispose();
    },
  };
}
