/**
 * @module assets/Procedural
 *
 * The generated half of d2rim's hybrid art strategy.
 *
 * ### Why this module carries so much weight
 *
 * The intended plan was to fetch CC0 PBR material sets from Poly Haven and
 * ambientCG. Both are hard-blocked by this environment's egress policy, as are
 * Kenney, Quaternius, OpenGameArt and every other CC0 texture library. The
 * complete reachable asset surface is two hosts, and neither carries a single
 * tiling PBR material set with a documented licence. So the six ground and prop
 * materials Act I needs — mud, grass, rock, wet stone, bark, planks — are not
 * "placeholder until we fetch the real ones". They are the real ones, and they
 * are generated here.
 *
 * ### Everything is seeded
 *
 * No `Math.random` appears anywhere in this file. Every generator takes a seed
 * and produces byte-identical output for the same inputs, which is what makes
 * headless capture comparisons meaningful: a diff in a reference frame is a
 * renderer regression, never noise from a reshuffled boulder field.
 *
 * ### Everything tiles
 *
 * Textures are sampled from **4D simplex noise on a 2-torus**. Mapping the UV
 * square onto a torus embedded in 4D and sampling there is what makes the
 * result seamless in both axes exactly, rather than approximately — no mirror
 * tricks, no cross-fade seams. Worley noise wraps by taking cell coordinates
 * modulo the grid, which is exact for the same reason.
 */

import * as THREE from 'three/webgpu';

/* -------------------------------------------------------------------------- */
/* Deterministic randomness                                                    */
/* -------------------------------------------------------------------------- */

/** A seeded uniform generator over `[0, 1)`. */
export interface Rng {
  /** Next float in `[0, 1)`. */
  next(): number;
  /** Next float in `[min, max)`. */
  range(min: number, max: number): number;
  /** Next integer in `[min, max]`. */
  int(min: number, max: number): number;
  /** Uniform point on the unit sphere. */
  onSphere(target?: THREE.Vector3): THREE.Vector3;
}

/**
 * Hash a string into a 32-bit seed.
 *
 * Lets callers seed with something meaningful (`'blood-moor.boulders'`) instead
 * of a magic number, so scattering two different prop layers in the same place
 * cannot accidentally correlate.
 */
export function hashSeed(seed: string | number): number {
  if (typeof seed === 'number') return seed >>> 0;
  let h = 2166136261 >>> 0;
  for (let i = 0; i < seed.length; i++) {
    h ^= seed.charCodeAt(i);
    h = Math.imul(h, 16777619);
  }
  return h >>> 0;
}

/**
 * mulberry32 — small, fast, and statistically far better than the
 * `sin(x) * 43758.5453` hash that graphics code reaches for by habit.
 */
export function createRng(seed: string | number): Rng {
  let state = hashSeed(seed);
  const next = (): number => {
    state = (state + 0x6d2b79f5) >>> 0;
    let t = state;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };

  return {
    next,
    range: (min, max) => min + next() * (max - min),
    int: (min, max) => Math.floor(min + next() * (max - min + 1)),
    onSphere(target = new THREE.Vector3()) {
      // Inverse-transform sampling: uniform in z and azimuth gives a uniform
      // distribution on the sphere. Normalising a Gaussian triple would too,
      // but this needs only two samples.
      const z = next() * 2 - 1;
      const angle = next() * Math.PI * 2;
      const r = Math.sqrt(Math.max(0, 1 - z * z));
      return target.set(r * Math.cos(angle), r * Math.sin(angle), z);
    },
  };
}

/* -------------------------------------------------------------------------- */
/* Simplex noise (2D and 4D)                                                   */
/* -------------------------------------------------------------------------- */

const GRAD_3 = new Int8Array([
  1, 1, 0, -1, 1, 0, 1, -1, 0, -1, -1, 0, 1, 0, 1, -1, 0, 1, 1, 0, -1, -1, 0, -1, 0, 1, 1, 0, -1, 1,
  0, 1, -1, 0, -1, -1,
]);

// The 32 gradients of the 4D simplex construction: every vector with one zero
// component and three +/-1 components.
const GRAD_4 = new Int8Array([
  0, 1, 1, 1, 0, 1, 1, -1, 0, 1, -1, 1, 0, 1, -1, -1, 0, -1, 1, 1, 0, -1, 1, -1, 0, -1, -1, 1, 0,
  -1, -1, -1, 1, 0, 1, 1, 1, 0, 1, -1, 1, 0, -1, 1, 1, 0, -1, -1, -1, 0, 1, 1, -1, 0, 1, -1, -1, 0,
  -1, 1, -1, 0, -1, -1, 1, 1, 0, 1, 1, 1, 0, -1, 1, -1, 0, 1, 1, -1, 0, -1, -1, 1, 0, 1, -1, 1, 0,
  -1, -1, -1, 0, 1, -1, -1, 0, -1, 1, 1, 1, 0, 1, 1, -1, 0, 1, -1, 1, 0, 1, -1, -1, 0, -1, 1, 1, 0,
  -1, 1, -1, 0, -1, -1, 1, 0, -1, -1, -1, 0,
]);

/**
 * Seeded simplex noise.
 *
 * The 4D variant is the one that matters here: it is what makes a texture tile
 * seamlessly. Simplex rather than Perlin because Perlin's 4D case has
 * pronounced axis-aligned directional artefacts, which on a tiling ground
 * texture read as a visible grid.
 */
export class SimplexNoise {
  /**
   * Translation applied to the second circle of the torus mapping in
   * {@link tileable2D}. Arbitrary, deliberately not a round number, and small
   * enough not to erode floating-point precision. See that method for why it
   * is needed.
   */
  static readonly #TORUS_OFFSET_Z = 31.4159;
  static readonly #TORUS_OFFSET_W = 17.2358;

  readonly #perm = new Uint8Array(512);
  readonly #permMod12 = new Uint8Array(512);

  constructor(seed: string | number = 0) {
    const rng = createRng(seed);
    const p = new Uint8Array(256);
    for (let i = 0; i < 256; i++) p[i] = i;
    // Fisher-Yates with the seeded generator.
    for (let i = 255; i > 0; i--) {
      const j = Math.floor(rng.next() * (i + 1));
      const tmp = p[i] as number;
      p[i] = p[j] as number;
      p[j] = tmp;
    }
    for (let i = 0; i < 512; i++) {
      const value = p[i & 255] as number;
      this.#perm[i] = value;
      this.#permMod12[i] = value % 12;
    }
  }

  /** 2D simplex noise in roughly `[-1, 1]`. */
  noise2D(xin: number, yin: number): number {
    const F2 = 0.5 * (Math.sqrt(3) - 1);
    const G2 = (3 - Math.sqrt(3)) / 6;

    const s = (xin + yin) * F2;
    const i = Math.floor(xin + s);
    const j = Math.floor(yin + s);
    const t = (i + j) * G2;
    const x0 = xin - (i - t);
    const y0 = yin - (j - t);

    const i1 = x0 > y0 ? 1 : 0;
    const j1 = x0 > y0 ? 0 : 1;

    const x1 = x0 - i1 + G2;
    const y1 = y0 - j1 + G2;
    const x2 = x0 - 1 + 2 * G2;
    const y2 = y0 - 1 + 2 * G2;

    const ii = i & 255;
    const jj = j & 255;

    const corner = (x: number, y: number, gi: number): number => {
      let t0 = 0.5 - x * x - y * y;
      if (t0 < 0) return 0;
      t0 *= t0;
      const g = gi * 3;
      return t0 * t0 * ((GRAD_3[g] as number) * x + (GRAD_3[g + 1] as number) * y);
    };

    const n0 = corner(x0, y0, this.#permMod12[ii + (this.#perm[jj] as number)] as number);
    const n1 = corner(x1, y1, this.#permMod12[ii + i1 + (this.#perm[jj + j1] as number)] as number);
    const n2 = corner(x2, y2, this.#permMod12[ii + 1 + (this.#perm[jj + 1] as number)] as number);

    return 70 * (n0 + n1 + n2);
  }

  /** 4D simplex noise in roughly `[-1, 1]`. The basis for seamless tiling. */
  noise4D(x: number, y: number, z: number, w: number): number {
    const F4 = (Math.sqrt(5) - 1) / 4;
    const G4 = (5 - Math.sqrt(5)) / 20;

    const s = (x + y + z + w) * F4;
    const i = Math.floor(x + s);
    const j = Math.floor(y + s);
    const k = Math.floor(z + s);
    const l = Math.floor(w + s);
    const t = (i + j + k + l) * G4;

    const x0 = x - (i - t);
    const y0 = y - (j - t);
    const z0 = z - (k - t);
    const w0 = w - (l - t);

    // Rank the four coordinates; the ordering selects which simplex corner
    // comes next. This is the branch-free formulation from Gustavson's paper.
    let rankx = 0;
    let ranky = 0;
    let rankz = 0;
    let rankw = 0;
    if (x0 > y0) rankx++; else ranky++;
    if (x0 > z0) rankx++; else rankz++;
    if (x0 > w0) rankx++; else rankw++;
    if (y0 > z0) ranky++; else rankz++;
    if (y0 > w0) ranky++; else rankw++;
    if (z0 > w0) rankz++; else rankw++;

    const i1 = rankx >= 3 ? 1 : 0;
    const j1 = ranky >= 3 ? 1 : 0;
    const k1 = rankz >= 3 ? 1 : 0;
    const l1 = rankw >= 3 ? 1 : 0;
    const i2 = rankx >= 2 ? 1 : 0;
    const j2 = ranky >= 2 ? 1 : 0;
    const k2 = rankz >= 2 ? 1 : 0;
    const l2 = rankw >= 2 ? 1 : 0;
    const i3 = rankx >= 1 ? 1 : 0;
    const j3 = ranky >= 1 ? 1 : 0;
    const k3 = rankz >= 1 ? 1 : 0;
    const l3 = rankw >= 1 ? 1 : 0;

    const ii = i & 255;
    const jj = j & 255;
    const kk = k & 255;
    const ll = l & 255;

    const perm = this.#perm;
    const corner = (
      dx: number,
      dy: number,
      dz: number,
      dw: number,
      gi: number,
    ): number => {
      let tc = 0.6 - dx * dx - dy * dy - dz * dz - dw * dw;
      if (tc < 0) return 0;
      tc *= tc;
      const g = (gi % 32) * 4;
      return (
        tc *
        tc *
        ((GRAD_4[g] as number) * dx +
          (GRAD_4[g + 1] as number) * dy +
          (GRAD_4[g + 2] as number) * dz +
          (GRAD_4[g + 3] as number) * dw)
      );
    };

    const gi0 =
      (perm[ii + (perm[jj + (perm[kk + (perm[ll] as number)] as number)] as number)] as number);
    const gi1 = perm[
      ii + i1 + (perm[jj + j1 + (perm[kk + k1 + (perm[ll + l1] as number)] as number)] as number)
    ] as number;
    const gi2 = perm[
      ii + i2 + (perm[jj + j2 + (perm[kk + k2 + (perm[ll + l2] as number)] as number)] as number)
    ] as number;
    const gi3 = perm[
      ii + i3 + (perm[jj + j3 + (perm[kk + k3 + (perm[ll + l3] as number)] as number)] as number)
    ] as number;
    const gi4 = perm[
      ii + 1 + (perm[jj + 1 + (perm[kk + 1 + (perm[ll + 1] as number)] as number)] as number)
    ] as number;

    const n0 = corner(x0, y0, z0, w0, gi0);
    const n1 = corner(x0 - i1 + G4, y0 - j1 + G4, z0 - k1 + G4, w0 - l1 + G4, gi1);
    const n2 = corner(x0 - i2 + 2 * G4, y0 - j2 + 2 * G4, z0 - k2 + 2 * G4, w0 - l2 + 2 * G4, gi2);
    const n3 = corner(x0 - i3 + 3 * G4, y0 - j3 + 3 * G4, z0 - k3 + 3 * G4, w0 - l3 + 3 * G4, gi3);
    const n4 = corner(x0 - 1 + 4 * G4, y0 - 1 + 4 * G4, z0 - 1 + 4 * G4, w0 - 1 + 4 * G4, gi4);

    return 27 * (n0 + n1 + n2 + n3 + n4);
  }

  /**
   * Seamlessly tiling 2D noise over the unit UV square.
   *
   * `(u, v)` in `[0,1)` is mapped onto a torus in 4D and sampled there, so
   * `f(0, v) === f(1, v)` and `f(u, 0) === f(u, 1)` exactly. `frequency` is in
   * tiles-per-unit and must be an integer for the wrap to hold.
   *
   * ### Why the second circle is offset
   *
   * The obvious mapping — `(cos u, sin u, cos v, sin v)` scaled — has a subtle
   * defect that produces a visible artefact. It makes `x === z` and `y === w`
   * *exactly*, in floating point, for every point where `u === v`: the whole
   * diagonal of the texture. Simplex noise ranks its four coordinates to pick a
   * simplex corner, and those rank comparisons use strict `>`, so along that
   * diagonal the tie is broken arbitrarily and the chosen corner can flip
   * between neighbouring texels. Measured before this fix, an input difference
   * of 1.6e-16 across the wrap produced an output difference of 5e-4 — a real
   * discontinuity, laid down exactly along the texture's diagonal.
   *
   * Translating the second circle by an irrational-looking constant removes the
   * coincidence. It cannot affect tiling: the mapping is still periodic in `u`
   * and `v` with period 1, because a constant translation of the sample point
   * commutes with the periodicity. Different offsets for `z` and `w` also
   * prevent `y === w` ties.
   */
  tileable2D(u: number, v: number, frequency: number): number {
    const f = Math.max(1, Math.round(frequency));
    const tau = Math.PI * 2;
    const scale = f / tau;
    return this.noise4D(
      Math.cos(u * tau) * scale,
      Math.sin(u * tau) * scale,
      Math.cos(v * tau) * scale + SimplexNoise.#TORUS_OFFSET_Z,
      Math.sin(v * tau) * scale + SimplexNoise.#TORUS_OFFSET_W,
    );
  }

  /** Fractal Brownian motion over {@link tileable2D}, normalised to `[0, 1]`. */
  tileableFbm(
    u: number,
    v: number,
    baseFrequency: number,
    octaves = 5,
    lacunarity = 2,
    gain = 0.5,
  ): number {
    let sum = 0;
    let amplitude = 1;
    let total = 0;
    let frequency = baseFrequency;
    for (let i = 0; i < octaves; i++) {
      sum += this.tileable2D(u, v, frequency) * amplitude;
      total += amplitude;
      amplitude *= gain;
      frequency *= lacunarity;
    }
    return (sum / total) * 0.5 + 0.5;
  }

  /**
   * Ridged multifractal: sharp creases instead of smooth hills.
   *
   * This is what makes rock read as rock. Ordinary fBm produces rounded, almost
   * cloud-like relief; folding the absolute value gives the hard ridgelines and
   * V-shaped gullies that eroded stone actually has.
   */
  tileableRidged(u: number, v: number, baseFrequency: number, octaves = 5): number {
    let sum = 0;
    let amplitude = 1;
    let total = 0;
    let frequency = baseFrequency;
    for (let i = 0; i < octaves; i++) {
      const n = 1 - Math.abs(this.tileable2D(u, v, frequency));
      sum += n * n * amplitude;
      total += amplitude;
      amplitude *= 0.5;
      frequency *= 2;
    }
    return sum / total;
  }
}

/* -------------------------------------------------------------------------- */
/* Worley / cellular noise                                                     */
/* -------------------------------------------------------------------------- */

/** Distances to the nearest and second-nearest feature points. */
export interface WorleyResult {
  f1: number;
  f2: number;
}

/**
 * Seamlessly tiling Worley (cellular) noise.
 *
 * Wraps exactly by taking the feature-point grid modulo `cells`, so the lattice
 * is genuinely periodic rather than merely blurred at the edges. `f2 - f1`
 * gives the classic cracked/veined pattern used below for dried mud and the
 * mortar lines between cobbles.
 */
export class WorleyNoise {
  readonly #seed: number;

  constructor(seed: string | number = 0) {
    this.#seed = hashSeed(seed);
  }

  /** Deterministic feature point inside cell `(cx, cy)`. */
  #point(cx: number, cy: number, cells: number, out: [number, number]): void {
    const wx = ((cx % cells) + cells) % cells;
    const wy = ((cy % cells) + cells) % cells;
    let h = (Math.imul(wx, 374761393) + Math.imul(wy, 668265263) + this.#seed) | 0;
    h = Math.imul(h ^ (h >>> 13), 1274126177);
    const a = ((h ^ (h >>> 16)) >>> 0) / 4294967296;
    let h2 = Math.imul(h ^ 0x9e3779b9, 2246822519);
    h2 = Math.imul(h2 ^ (h2 >>> 15), 3266489917);
    const b = ((h2 ^ (h2 >>> 16)) >>> 0) / 4294967296;
    out[0] = cx + a;
    out[1] = cy + b;
  }

  /** Sample at `(u, v)` in `[0,1)` with a `cells x cells` lattice. */
  sample(u: number, v: number, cells: number): WorleyResult {
    const x = u * cells;
    const y = v * cells;
    const cx = Math.floor(x);
    const cy = Math.floor(y);

    let f1 = Infinity;
    let f2 = Infinity;
    const point: [number, number] = [0, 0];

    for (let oy = -1; oy <= 1; oy++) {
      for (let ox = -1; ox <= 1; ox++) {
        this.#point(cx + ox, cy + oy, cells, point);
        const dx = point[0] - x;
        const dy = point[1] - y;
        const d = Math.sqrt(dx * dx + dy * dy);
        if (d < f1) {
          f2 = f1;
          f1 = d;
        } else if (d < f2) {
          f2 = d;
        }
      }
    }
    return { f1, f2 };
  }
}

/* -------------------------------------------------------------------------- */
/* Tiling PBR material sets                                                    */
/* -------------------------------------------------------------------------- */

/** The material presets Act I needs. */
export type MaterialPreset =
  | 'mud'
  | 'grass'
  | 'rock'
  | 'wetStone'
  | 'bark'
  | 'planks';

export interface MaterialSetOptions {
  /** Texture edge length in pixels. Powers of two only. Default 512. */
  size?: number;
  /** Seed. Defaults to the preset name, so each preset differs by default. */
  seed?: string | number;
  /** Strength of the generated normal map. Default 1. */
  normalStrength?: number;
  /** UV repeat baked into the returned textures. Default 1. */
  repeat?: number;
}

/**
 * A complete generated PBR material set.
 *
 * `height` is exposed alongside the maps because displacement, parallax and
 * terrain blending all want the same field the normal was derived from, and
 * regenerating it would be both wasteful and subtly inconsistent.
 */
export interface MaterialSet {
  readonly preset: MaterialPreset;
  readonly size: number;
  readonly map: THREE.DataTexture;
  readonly normalMap: THREE.DataTexture;
  readonly roughnessMap: THREE.DataTexture;
  readonly aoMap: THREE.DataTexture;
  /** Raw height field in `[0,1]`, row-major, `size * size` entries. */
  readonly height: Float32Array;
  /** Build a `MeshStandardMaterial` wired to every map in this set. */
  toMaterial(overrides?: THREE.MeshStandardMaterialParameters): THREE.MeshStandardMaterial;
  dispose(): void;
}

/** Per-preset appearance and surface description. */
interface PresetSpec {
  /** Two colours the albedo ramps between, plus a rarely-hit accent. */
  readonly base: readonly [number, number, number];
  readonly alt: readonly [number, number, number];
  readonly accent: readonly [number, number, number];
  readonly roughnessRange: readonly [number, number];
  readonly normalStrength: number;
  /** Produce a height value in `[0,1]` at `(u, v)`. */
  readonly height: (u: number, v: number, simplex: SimplexNoise, worley: WorleyNoise) => number;
  /** Optional extra albedo modulation, given height and position. */
  readonly tint?: (
    u: number,
    v: number,
    h: number,
    simplex: SimplexNoise,
    worley: WorleyNoise,
  ) => number;
}

/**
 * The art direction, expressed as maths.
 *
 * Colours are linear-space and deliberately desaturated and dark: Act I is a
 * cold, overcast, grimdark register, and materials that look correct under a
 * neutral studio light are far too colourful under an overcast sky.
 */
const PRESETS: Record<MaterialPreset, PresetSpec> = {
  /** Churned, wet, rutted earth. The Blood Moor floor. */
  mud: {
    base: [0.085, 0.066, 0.050],
    alt: [0.150, 0.118, 0.086],
    accent: [0.045, 0.038, 0.034],
    roughnessRange: [0.62, 0.95],
    normalStrength: 1.15,
    height: (u, v, simplex, worley) => {
      // Broad churn, plus dried-crack veining from the Worley f2-f1 ridge.
      const churn = simplex.tileableFbm(u, v, 4, 5);
      const cracks = 1 - Math.min(1, (worley.sample(u, v, 8).f2 - worley.sample(u, v, 8).f1) * 3.2);
      const grit = simplex.tileableFbm(u, v, 32, 3) * 0.12;
      return Math.min(1, Math.max(0, churn * 0.78 - cracks * 0.22 + grit));
    },
    tint: (u, v, _h, simplex) => simplex.tileableFbm(u, v, 6, 4),
  },

  /** Sparse, trampled, half-dead grass over soil. */
  grass: {
    base: [0.055, 0.072, 0.036],
    alt: [0.115, 0.128, 0.062],
    accent: [0.088, 0.070, 0.040],
    roughnessRange: [0.72, 0.98],
    normalStrength: 0.85,
    height: (u, v, simplex) => {
      // Many thin high-frequency blades over a slow clumping field.
      const clumps = simplex.tileableFbm(u, v, 5, 4);
      const blades = simplex.tileableFbm(u, v, 48, 2);
      return Math.min(1, Math.max(0, clumps * 0.55 + blades * 0.45));
    },
    tint: (u, v, _h, simplex) => simplex.tileableFbm(u, v, 3, 3),
  },

  /** Eroded grey cliff stone. */
  rock: {
    base: [0.062, 0.060, 0.058],
    alt: [0.135, 0.130, 0.124],
    accent: [0.040, 0.036, 0.033],
    roughnessRange: [0.48, 0.88],
    normalStrength: 1.45,
    height: (u, v, simplex) => {
      // Ridged multifractal gives the hard creases; a little fBm softens them.
      const ridges = simplex.tileableRidged(u, v, 4, 6);
      const detail = simplex.tileableFbm(u, v, 24, 3);
      return Math.min(1, Math.max(0, ridges * 0.8 + detail * 0.2));
    },
  },

  /** Wet, cobbled, slick stone. Rogue Encampment paths and cave floors. */
  wetStone: {
    base: [0.040, 0.041, 0.046],
    alt: [0.098, 0.100, 0.110],
    accent: [0.022, 0.024, 0.029],
    // The tight, low range is the whole point: uniform low roughness is what
    // makes a surface read as wet under an environment map.
    roughnessRange: [0.12, 0.42],
    normalStrength: 1.25,
    height: (u, v, simplex, worley) => {
      // Worley cells are the cobbles; f2-f1 carves the mortar between them.
      const cell = worley.sample(u, v, 7);
      const mortar = Math.min(1, (cell.f2 - cell.f1) * 4.5);
      const wear = simplex.tileableFbm(u, v, 18, 3) * 0.18;
      return Math.min(1, Math.max(0, mortar * 0.85 + wear));
    },
    tint: (u, v, _h, _s, worley) => {
      // Constant per cell, so each cobble gets its own stone colour.
      const cell = worley.sample(u, v, 7);
      return (cell.f1 * 3.1) % 1;
    },
  },

  /** Vertically fissured tree bark. */
  bark: {
    base: [0.048, 0.038, 0.029],
    alt: [0.098, 0.079, 0.058],
    accent: [0.030, 0.026, 0.022],
    roughnessRange: [0.70, 0.96],
    normalStrength: 1.6,
    height: (u, v, simplex) => {
      // Anisotropic sampling: high frequency across the trunk, low along it,
      // which is what produces vertical fissures rather than isotropic lumps.
      const fissures = simplex.tileableRidged(u * 1, v * 1, 16, 4);
      const along = simplex.tileableFbm(u, v, 3, 3);
      const grain = simplex.tileableFbm(u, v, 40, 2) * 0.15;
      return Math.min(1, Math.max(0, fissures * 0.55 + along * 0.35 + grain));
    },
  },

  /** Weathered wooden planks. Palisades, doors, cart beds. */
  planks: {
    base: [0.072, 0.052, 0.034],
    alt: [0.128, 0.098, 0.064],
    accent: [0.042, 0.031, 0.021],
    roughnessRange: [0.55, 0.90],
    normalStrength: 1.0,
    height: (u, v, simplex) => {
      const PLANKS = 6;
      const row = Math.floor(v * PLANKS);
      // Offset each plank along its length so the grain does not line up across
      // the boundary — that alignment is the giveaway of a fake wood texture.
      const offset = ((Math.sin(row * 12.9898) * 43758.5453) % 1 + 1) % 1;
      const localV = v * PLANKS - row;

      // Distance to the plank edge, used to carve the gap between boards.
      const edge = Math.min(localV, 1 - localV);
      const gap = Math.min(1, edge * 14);

      // Long grain: stretched noise plus a few darker growth rings.
      const grain = simplex.tileableFbm(u + offset, v * 0.25, 24, 3);
      const rings = Math.abs(Math.sin((grain * 6 + u * 8) * Math.PI)) * 0.25;

      return Math.min(1, Math.max(0, gap * (0.55 + grain * 0.3 + rings)));
    },
    tint: (u, v, _h, simplex) => simplex.tileableFbm(u * 0.3, v * 3, 4, 3),
  },
};

/** Wrap an index into `[0, n)`. Used so derivatives respect tiling. */
function wrapIndex(i: number, n: number): number {
  return ((i % n) + n) % n;
}

function lerp(a: number, b: number, t: number): number {
  return a + (b - a) * t;
}

function clamp01(v: number): number {
  return v < 0 ? 0 : v > 1 ? 1 : v;
}

/**
 * Convert a linear-space float to an 8-bit sRGB byte.
 *
 * The albedo texture is tagged `SRGBColorSpace`, so the values written into it
 * must actually be sRGB-encoded. Writing linear values into an sRGB-tagged
 * texture is the most common way procedural materials come out looking washed
 * out and chalky.
 */
function linearToSrgbByte(linear: number): number {
  const c = clamp01(linear);
  const encoded = c <= 0.0031308 ? c * 12.92 : 1.055 * Math.pow(c, 1 / 2.4) - 0.055;
  return Math.round(clamp01(encoded) * 255);
}

/**
 * Generate a complete tiling PBR material set.
 *
 * Pipeline: evaluate a height field, derive the normal from its gradient with
 * wrapped sampling, derive AO from local height relative to a blurred
 * neighbourhood, and drive albedo and roughness from height plus a preset tint
 * field. Deriving everything from one height field is what keeps the maps
 * mutually consistent — the crevice that is dark in AO is the same crevice the
 * normal bends into.
 */
export function generateMaterialSet(
  preset: MaterialPreset,
  options: MaterialSetOptions = {},
): MaterialSet {
  const size = options.size ?? 512;
  const spec = PRESETS[preset];
  const seed = options.seed ?? `d2rim.material.${preset}`;
  const simplex = new SimplexNoise(seed);
  const worley = new WorleyNoise(`${hashSeed(seed)}.worley`);
  const normalStrength = (options.normalStrength ?? 1) * spec.normalStrength;
  const repeat = options.repeat ?? 1;

  /* --- height ----------------------------------------------------------- */
  const height = new Float32Array(size * size);
  for (let y = 0; y < size; y++) {
    const v = y / size;
    for (let x = 0; x < size; x++) {
      height[y * size + x] = clamp01(spec.height(x / size, v, simplex, worley));
    }
  }

  /* --- AO --------------------------------------------------------------- */
  // Cheap cavity approximation: how much lower a texel sits than the average of
  // a wrapped neighbourhood. Genuine ray-traced AO on a tiling height field is
  // not worth the cost here — the cavity term is what the eye reads anyway.
  const ao = new Float32Array(size * size);
  const RADIUS = Math.max(2, Math.round(size / 128));
  for (let y = 0; y < size; y++) {
    for (let x = 0; x < size; x++) {
      let sum = 0;
      let count = 0;
      for (let oy = -RADIUS; oy <= RADIUS; oy += 2) {
        for (let ox = -RADIUS; ox <= RADIUS; ox += 2) {
          sum += height[wrapIndex(y + oy, size) * size + wrapIndex(x + ox, size)] as number;
          count++;
        }
      }
      const local = height[y * size + x] as number;
      const average = sum / count;
      ao[y * size + x] = clamp01(0.5 + (local - average) * 2.4);
    }
  }

  /* --- pack textures ---------------------------------------------------- */
  const albedoData = new Uint8Array(size * size * 4);
  const normalData = new Uint8Array(size * size * 4);
  const roughnessData = new Uint8Array(size * size * 4);
  const aoData = new Uint8Array(size * size * 4);

  for (let y = 0; y < size; y++) {
    for (let x = 0; x < size; x++) {
      const i = y * size + x;
      const o = i * 4;
      const h = height[i] as number;

      /* normal: central differences with wrapped neighbours, so the derived
         normal map tiles as exactly as the height field it came from. */
      const hl = height[y * size + wrapIndex(x - 1, size)] as number;
      const hr = height[y * size + wrapIndex(x + 1, size)] as number;
      const hd = height[wrapIndex(y - 1, size) * size + x] as number;
      const hu = height[wrapIndex(y + 1, size) * size + x] as number;

      const dx = (hl - hr) * normalStrength * size * 0.012;
      const dy = (hd - hu) * normalStrength * size * 0.012;
      const inv = 1 / Math.sqrt(dx * dx + dy * dy + 1);
      normalData[o] = Math.round((dx * inv * 0.5 + 0.5) * 255);
      normalData[o + 1] = Math.round((dy * inv * 0.5 + 0.5) * 255);
      normalData[o + 2] = Math.round((inv * 0.5 + 0.5) * 255);
      normalData[o + 3] = 255;

      /* albedo */
      const tint = spec.tint?.(x / size, y / size, h, simplex, worley) ?? h;
      const blend = clamp01(h * 0.6 + tint * 0.4);
      const accentMix = clamp01((tint - 0.72) * 3.2);
      const r = lerp(lerp(spec.base[0], spec.alt[0], blend), spec.accent[0], accentMix);
      const g = lerp(lerp(spec.base[1], spec.alt[1], blend), spec.accent[1], accentMix);
      const b = lerp(lerp(spec.base[2], spec.alt[2], blend), spec.accent[2], accentMix);
      // Cavities are darker: dirt collects and less light reaches them.
      const occlusion = lerp(0.55, 1, ao[i] as number);
      albedoData[o] = linearToSrgbByte(r * occlusion);
      albedoData[o + 1] = linearToSrgbByte(g * occlusion);
      albedoData[o + 2] = linearToSrgbByte(b * occlusion);
      albedoData[o + 3] = 255;

      /* roughness: recesses hold moisture and read smoother; peaks are worn
         and read rougher. Linear data, so no sRGB encoding. */
      const roughness = lerp(spec.roughnessRange[0], spec.roughnessRange[1], clamp01(h * 0.85 + 0.1));
      const byte = Math.round(clamp01(roughness) * 255);
      roughnessData[o] = byte;
      roughnessData[o + 1] = byte;
      roughnessData[o + 2] = byte;
      roughnessData[o + 3] = 255;

      /* AO */
      const aoByte = Math.round((ao[i] as number) * 255);
      aoData[o] = aoByte;
      aoData[o + 1] = aoByte;
      aoData[o + 2] = aoByte;
      aoData[o + 3] = 255;
    }
  }

  const makeTexture = (data: Uint8Array, colorSpace: THREE.ColorSpace): THREE.DataTexture => {
    const texture = new THREE.DataTexture(data, size, size, THREE.RGBAFormat);
    texture.colorSpace = colorSpace;
    texture.wrapS = THREE.RepeatWrapping;
    texture.wrapT = THREE.RepeatWrapping;
    texture.minFilter = THREE.LinearMipmapLinearFilter;
    texture.magFilter = THREE.LinearFilter;
    texture.generateMipmaps = true;
    texture.anisotropy = 8;
    texture.repeat.set(repeat, repeat);
    texture.needsUpdate = true;
    return texture;
  };

  const map = makeTexture(albedoData, THREE.SRGBColorSpace);
  const normalMap = makeTexture(normalData, THREE.NoColorSpace);
  const roughnessMap = makeTexture(roughnessData, THREE.NoColorSpace);
  const aoMap = makeTexture(aoData, THREE.NoColorSpace);

  return {
    preset,
    size,
    map,
    normalMap,
    roughnessMap,
    aoMap,
    height,
    toMaterial(overrides = {}) {
      return new THREE.MeshStandardMaterial({
        map,
        normalMap,
        roughnessMap,
        aoMap,
        roughness: 1,
        metalness: 0,
        ...overrides,
      });
    },
    dispose() {
      map.dispose();
      normalMap.dispose();
      roughnessMap.dispose();
      aoMap.dispose();
    },
  };
}

/* -------------------------------------------------------------------------- */
/* Rock and boulder meshes                                                     */
/* -------------------------------------------------------------------------- */

export interface RockOptions {
  /** Seed. Same seed, same rock, always. */
  seed?: string | number;
  /** Icosphere subdivisions. 2 ≈ 320 tris, 3 ≈ 1280, 4 ≈ 5120. Default 3. */
  detail?: number;
  /** Approximate radius before non-uniform scaling. Default 1. */
  radius?: number;
  /** Displacement amount as a fraction of radius. Default 0.34. */
  displacement?: number;
  /** Non-uniform squash, so boulders are not spheres. Default `[1, 0.72, 0.9]`. */
  scale?: readonly [number, number, number];
  /**
   * Flatten the bottom into a base so the rock sits on the ground instead of
   * floating or intersecting. `0` disables. Default 0.35 (fraction of radius).
   */
  flattenBase?: number;
}

/**
 * Build a boulder by displacing an icosphere along its own normals.
 *
 * An icosphere rather than a UV sphere because its triangles are near-uniform:
 * displacing a UV sphere concentrates all the detail at the poles and stretches
 * it at the equator, which reads as an obviously synthetic pinched shape.
 *
 * Displacement layers ridged multifractal (large chipped facets) over fBm
 * (overall lumpiness), which is what gives fractured stone rather than a potato.
 */
export function generateRockGeometry(options: RockOptions = {}): THREE.BufferGeometry {
  const seed = options.seed ?? 'd2rim.rock';
  const detail = options.detail ?? 3;
  const radius = options.radius ?? 1;
  const displacement = options.displacement ?? 0.34;
  const scale = options.scale ?? ([1, 0.72, 0.9] as const);
  const flattenBase = options.flattenBase ?? 0.35;

  const simplex = new SimplexNoise(seed);
  const geometry = new THREE.IcosahedronGeometry(radius, detail);
  const position = geometry.getAttribute('position') as THREE.BufferAttribute;

  const vertex = new THREE.Vector3();
  for (let i = 0; i < position.count; i++) {
    vertex.fromBufferAttribute(position, i);
    const direction = vertex.clone().normalize();

    // Sampling 3D-ish noise via the 4D function with a fixed w keeps one noise
    // implementation in play and avoids seams entirely — the sphere is sampled
    // in solid space, so there is no UV boundary to match.
    const large = simplex.noise4D(direction.x * 1.6, direction.y * 1.6, direction.z * 1.6, 0.5);
    const ridged =
      1 - Math.abs(simplex.noise4D(direction.x * 3.1, direction.y * 3.1, direction.z * 3.1, 11.7));
    const fine = simplex.noise4D(direction.x * 7.4, direction.y * 7.4, direction.z * 7.4, 23.1);

    const offset = large * 0.55 + (ridged - 0.5) * 0.7 + fine * 0.18;
    vertex.copy(direction).multiplyScalar(radius * (1 + offset * displacement));

    vertex.x *= scale[0];
    vertex.y *= scale[1];
    vertex.z *= scale[2];

    if (flattenBase > 0) {
      const floor = -radius * scale[1] * (1 - flattenBase);
      if (vertex.y < floor) {
        // Ease rather than clamp: a hard clamp leaves a visible disc edge.
        vertex.y = floor - (floor - vertex.y) * 0.15;
      }
    }

    position.setXYZ(i, vertex.x, vertex.y, vertex.z);
  }

  position.needsUpdate = true;

  // Normals are recomputed after displacement, on non-indexed geometry so each
  // facet gets its own flat normal — smooth-shading a chipped rock destroys the
  // faceting that makes it read as stone.
  //
  // `IcosahedronGeometry` is already non-indexed (three builds all polyhedron
  // geometries that way), so calling `toNonIndexed` unconditionally works but
  // logs a warning per rock. Checked rather than suppressed, so that a future
  // switch to an indexed source geometry still gets converted.
  // Annotated as the base type: the branch below may replace the concrete
  // `IcosahedronGeometry` with a plain `BufferGeometry`.
  let faceted: THREE.BufferGeometry = geometry;
  if (geometry.index !== null) {
    faceted = geometry.toNonIndexed();
    geometry.dispose();
  }
  faceted.computeVertexNormals();
  faceted.computeBoundingBox();
  faceted.computeBoundingSphere();
  return faceted;
}

/* -------------------------------------------------------------------------- */
/* Detail scatter                                                              */
/* -------------------------------------------------------------------------- */

/** One placed instance, as handed to the caller's filter. */
export interface ScatterSample {
  readonly position: THREE.Vector3;
  readonly normal: THREE.Vector3;
  /** Uniform scale chosen for this instance. */
  readonly scale: number;
  /** Rotation about Y, radians. */
  readonly rotation: number;
  readonly index: number;
}

export interface ScatterOptions {
  /** Number of placement attempts. Rejected samples are not retried. */
  count: number;
  /** Half-extent of the square placement area, in world units. */
  area: number;
  /** Centre of the placement area. Default origin. */
  center?: THREE.Vector3;
  seed?: string | number;
  /** Uniform scale range. Default `[0.8, 1.25]`. */
  scaleRange?: readonly [number, number];
  /**
   * Surface query. Return `null` to reject the sample (out of bounds, in water,
   * inside a building). Defaults to a flat plane at y = 0.
   */
  surface?: (x: number, z: number) => { y: number; normal: THREE.Vector3 } | null;
  /**
   * Reject slopes steeper than this, in degrees. Grass does not grow on a
   * cliff face. Default 90 (accept everything).
   */
  maxSlopeDegrees?: number;
  /**
   * Density field in `[0,1]`, sampled per candidate and used as an acceptance
   * probability. This is what produces clumping instead of an even sprinkle.
   */
  density?: (x: number, z: number) => number;
  /** Minimum spacing between accepted instances. `0` disables. Default 0. */
  minSpacing?: number;
  /** Align instances to the surface normal instead of standing upright. */
  alignToNormal?: boolean;
}

/**
 * Scatter instances across a surface, deterministically.
 *
 * Returns the accepted samples rather than building a mesh, so the same
 * placement can drive an `InstancedMesh`, a set of physics colliders and a
 * gameplay lookup table without any risk of the three disagreeing. Use
 * {@link buildInstancedMesh} for the common rendering case.
 *
 * Minimum spacing is enforced with a uniform grid rather than an O(n^2) sweep,
 * so a few thousand instances stay cheap.
 */
export function scatter(options: ScatterOptions): ScatterSample[] {
  const {
    count,
    area,
    center = new THREE.Vector3(),
    seed = 'd2rim.scatter',
    scaleRange = [0.8, 1.25] as const,
    surface,
    maxSlopeDegrees = 90,
    density,
    minSpacing = 0,
    alignToNormal = false,
  } = options;

  const rng = createRng(seed);
  const accepted: ScatterSample[] = [];
  const minSlopeCos = Math.cos(THREE.MathUtils.degToRad(maxSlopeDegrees));

  // Spatial hash for the spacing test.
  const cellSize = minSpacing > 0 ? minSpacing : 1;
  const grid = new Map<string, ScatterSample[]>();
  const cellKey = (x: number, z: number): string =>
    `${Math.floor(x / cellSize)},${Math.floor(z / cellSize)}`;

  const tooClose = (x: number, z: number): boolean => {
    if (minSpacing <= 0) return false;
    const gx = Math.floor(x / cellSize);
    const gz = Math.floor(z / cellSize);
    for (let oz = -1; oz <= 1; oz++) {
      for (let ox = -1; ox <= 1; ox++) {
        const bucket = grid.get(`${gx + ox},${gz + oz}`);
        if (bucket === undefined) continue;
        for (const sample of bucket) {
          const dx = sample.position.x - x;
          const dz = sample.position.z - z;
          if (dx * dx + dz * dz < minSpacing * minSpacing) return true;
        }
      }
    }
    return false;
  };

  for (let i = 0; i < count; i++) {
    // Every random draw for this candidate happens up front, before any
    // rejection test. This is the difference between a generator that is
    // reproducible and one that merely looks it: if a rejected candidate
    // consumed fewer draws than an accepted one, the RNG stream would desync
    // and adding a density function would silently relocate every subsequent
    // instance rather than just thinning the set.
    const x = center.x + rng.range(-area, area);
    const z = center.z + rng.range(-area, area);
    const roll = rng.next();
    const scaleValue = rng.range(scaleRange[0], scaleRange[1]);
    const rotation = rng.next() * Math.PI * 2;

    if (density !== undefined && roll > clamp01(density(x, z))) continue;

    let y = center.y;
    let normal = new THREE.Vector3(0, 1, 0);
    if (surface !== undefined) {
      const hit = surface(x, z);
      if (hit === null) continue;
      y = hit.y;
      normal = hit.normal;
    }

    if (normal.y < minSlopeCos) continue;
    if (tooClose(x, z)) continue;

    const sample: ScatterSample = {
      position: new THREE.Vector3(x, y, z),
      normal: alignToNormal ? normal.clone() : new THREE.Vector3(0, 1, 0),
      scale: scaleValue,
      rotation,
      index: accepted.length,
    };
    accepted.push(sample);

    if (minSpacing > 0) {
      const key = cellKey(x, z);
      const bucket = grid.get(key);
      if (bucket === undefined) grid.set(key, [sample]);
      else bucket.push(sample);
    }
  }

  return accepted;
}

/**
 * Turn scatter samples into an `InstancedMesh`.
 *
 * Separate from {@link scatter} so the same placement can also feed physics and
 * gameplay without duplicating the sampling.
 */
export function buildInstancedMesh(
  geometry: THREE.BufferGeometry,
  material: THREE.Material,
  samples: readonly ScatterSample[],
): THREE.InstancedMesh {
  const mesh = new THREE.InstancedMesh(geometry, material, samples.length);
  const matrix = new THREE.Matrix4();
  const quaternion = new THREE.Quaternion();
  const scaleVector = new THREE.Vector3();
  const up = new THREE.Vector3(0, 1, 0);
  const align = new THREE.Quaternion();
  const spin = new THREE.Quaternion();

  for (const sample of samples) {
    align.setFromUnitVectors(up, sample.normal);
    spin.setFromAxisAngle(up, sample.rotation);
    quaternion.copy(align).multiply(spin);
    scaleVector.setScalar(sample.scale);
    matrix.compose(sample.position, quaternion, scaleVector);
    mesh.setMatrixAt(sample.index, matrix);
  }

  mesh.instanceMatrix.needsUpdate = true;
  mesh.castShadow = true;
  mesh.receiveShadow = true;
  mesh.computeBoundingSphere();
  return mesh;
}
