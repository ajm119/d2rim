/**
 * Tests for the procedural generators.
 *
 * Two properties matter more than anything else here and are both easy to break
 * without noticing:
 *
 * 1. **Determinism.** Reference-frame comparison is the project's regression
 *    signal. If a boulder field reshuffles between runs, every capture diff
 *    becomes noise and the signal is gone.
 * 2. **Seamless tiling.** A ground texture with a visible seam every few metres
 *    is the single most obvious "this is procedural" tell. The 4D-noise-on-a-
 *    torus construction is supposed to make wrapping exact, so that claim is
 *    asserted numerically rather than trusted.
 */

import { describe, expect, it } from 'vitest';

import {
  SimplexNoise,
  WorleyNoise,
  buildInstancedMesh,
  createRng,
  generateMaterialSet,
  generateRockGeometry,
  hashSeed,
  scatter,
} from '../src/assets/Procedural';

describe('seeded randomness', () => {
  it('produces identical sequences for identical seeds', () => {
    const a = createRng('blood-moor');
    const b = createRng('blood-moor');
    const seqA = Array.from({ length: 64 }, () => a.next());
    const seqB = Array.from({ length: 64 }, () => b.next());
    expect(seqA).toEqual(seqB);
  });

  it('decorrelates different seeds', () => {
    const a = createRng('rocks');
    const b = createRng('grass');
    const seqA = Array.from({ length: 32 }, () => a.next());
    const seqB = Array.from({ length: 32 }, () => b.next());
    expect(seqA).not.toEqual(seqB);
  });

  it('stays within [0, 1)', () => {
    const rng = createRng(7);
    for (let i = 0; i < 4096; i++) {
      const v = rng.next();
      expect(v).toBeGreaterThanOrEqual(0);
      expect(v).toBeLessThan(1);
    }
  });

  it('is roughly uniform', () => {
    const rng = createRng('uniformity');
    const buckets = new Array<number>(10).fill(0);
    const samples = 100_000;
    for (let i = 0; i < samples; i++) buckets[Math.floor(rng.next() * 10)]!++;
    for (const count of buckets) {
      // +/-5% of the expected 10% share is a generous band that still catches a
      // genuinely broken generator.
      expect(count).toBeGreaterThan(samples * 0.095);
      expect(count).toBeLessThan(samples * 0.105);
    }
  });

  it('hashes seeds stably across calls', () => {
    expect(hashSeed('den-of-evil')).toBe(hashSeed('den-of-evil'));
    expect(hashSeed('den-of-evil')).not.toBe(hashSeed('den-of-evi1'));
  });

  it('generates points on the unit sphere', () => {
    const rng = createRng('sphere');
    for (let i = 0; i < 256; i++) {
      expect(rng.onSphere().length()).toBeCloseTo(1, 6);
    }
  });
});

describe('simplex noise', () => {
  const noise = new SimplexNoise('test');

  it('is deterministic for a given seed', () => {
    const other = new SimplexNoise('test');
    for (let i = 0; i < 32; i++) {
      const u = i / 32;
      expect(noise.tileable2D(u, 0.3, 4)).toBe(other.tileable2D(u, 0.3, 4));
    }
  });

  it('stays in range', () => {
    for (let i = 0; i < 1000; i++) {
      const v = noise.noise4D(i * 0.13, i * 0.29, i * 0.07, i * 0.51);
      expect(v).toBeGreaterThanOrEqual(-1.2);
      expect(v).toBeLessThanOrEqual(1.2);
    }
  });

  it('wraps exactly at the tile boundary', () => {
    // u = 0 and u = 1 are literally the same point on the torus, so this is an
    // exact equality, not an approximation.
    for (let i = 0; i < 16; i++) {
      const v = i / 16;
      expect(noise.tileable2D(0, v, 4)).toBeCloseTo(noise.tileable2D(1, v, 4), 10);
      expect(noise.tileable2D(v, 0, 4)).toBeCloseTo(noise.tileable2D(v, 1, 4), 10);
    }
  });

  it('is continuous across the wrap, not merely equal at it', () => {
    // Equality at the boundary would also hold for a function that jumps just
    // before it. Continuity is the property that actually removes the seam.
    const step = 1 / 512;
    for (let i = 0; i < 8; i++) {
      const v = i / 8;
      const justBefore = noise.tileable2D(1 - step, v, 4);
      const justAfter = noise.tileable2D(step, v, 4);
      const interior = noise.tileable2D(0.5, v, 4);
      const interiorNeighbour = noise.tileable2D(0.5 + step, v, 4);
      const seamJump = Math.abs(justAfter - justBefore);
      const typicalJump = Math.abs(interiorNeighbour - interior);
      // The step across the seam must be the same order as any other step.
      expect(seamJump).toBeLessThan(typicalJump + 0.05);
    }
  });

  it('produces ridged output in [0, 1]', () => {
    for (let i = 0; i < 200; i++) {
      const value = noise.tileableRidged(i / 200, 0.4, 4);
      expect(value).toBeGreaterThanOrEqual(0);
      expect(value).toBeLessThanOrEqual(1);
    }
  });

  it('produces fbm output in [0, 1]', () => {
    for (let i = 0; i < 200; i++) {
      const value = noise.tileableFbm(i / 200, 0.7, 4);
      expect(value).toBeGreaterThanOrEqual(0);
      expect(value).toBeLessThanOrEqual(1);
    }
  });
});

describe('worley noise', () => {
  const worley = new WorleyNoise('cells');

  it('is deterministic', () => {
    const other = new WorleyNoise('cells');
    expect(worley.sample(0.31, 0.62, 8)).toEqual(other.sample(0.31, 0.62, 8));
  });

  it('orders f1 below f2', () => {
    for (let i = 0; i < 200; i++) {
      const { f1, f2 } = worley.sample(i / 200, (i * 7) % 200 / 200, 8);
      expect(f1).toBeLessThanOrEqual(f2);
    }
  });

  it('wraps at the tile boundary', () => {
    for (let i = 0; i < 16; i++) {
      const v = i / 16;
      expect(worley.sample(0, v, 8).f1).toBeCloseTo(worley.sample(1, v, 8).f1, 10);
      expect(worley.sample(v, 0, 8).f1).toBeCloseTo(worley.sample(v, 1, 8).f1, 10);
    }
  });
});

describe('material sets', () => {
  const PRESETS = ['mud', 'grass', 'rock', 'wetStone', 'bark', 'planks'] as const;

  it('generates every Act I preset', () => {
    for (const preset of PRESETS) {
      const set = generateMaterialSet(preset, { size: 64 });
      expect(set.preset).toBe(preset);
      expect(set.size).toBe(64);
      expect(set.height.length).toBe(64 * 64);
      expect((set.map.image.data as Uint8Array).length).toBe(64 * 64 * 4);
      set.dispose();
    }
  });

  it('is byte-identical for the same seed', () => {
    const a = generateMaterialSet('rock', { size: 64, seed: 'boulder' });
    const b = generateMaterialSet('rock', { size: 64, seed: 'boulder' });
    expect(Array.from(a.map.image.data as Uint8Array)).toEqual(
      Array.from(b.map.image.data as Uint8Array),
    );
    expect(Array.from(a.normalMap.image.data as Uint8Array)).toEqual(
      Array.from(b.normalMap.image.data as Uint8Array),
    );
    a.dispose();
    b.dispose();
  });

  it('differs between seeds', () => {
    const a = generateMaterialSet('rock', { size: 64, seed: 'one' });
    const b = generateMaterialSet('rock', { size: 64, seed: 'two' });
    expect(Array.from(a.height)).not.toEqual(Array.from(b.height));
    a.dispose();
    b.dispose();
  });

  it('produces a height field with no seam at the wrap', () => {
    // Compare the discontinuity across the tile edge against the typical
    // interior discontinuity. A seam shows up as an edge step far larger than
    // the interior noise floor.
    for (const preset of PRESETS) {
      const size = 64;
      const set = generateMaterialSet(preset, { size });
      const h = set.height;

      let edgeDelta = 0;
      let interiorDelta = 0;
      for (let y = 0; y < size; y++) {
        edgeDelta += Math.abs(h[y * size + (size - 1)]! - h[y * size + 0]!);
        interiorDelta += Math.abs(h[y * size + 31]! - h[y * size + 32]!);
      }
      edgeDelta /= size;
      interiorDelta /= size;

      expect(
        edgeDelta,
        `${preset}: horizontal seam (edge ${edgeDelta.toFixed(4)} vs interior ${interiorDelta.toFixed(4)})`,
      ).toBeLessThan(Math.max(interiorDelta * 3, 0.05));
      set.dispose();
    }
  });

  it('tags colour space by role: albedo sRGB, data maps linear', () => {
    const set = generateMaterialSet('mud', { size: 32 });
    // 'srgb' / '' are three's ColorSpace string constants.
    expect(set.map.colorSpace).toBe('srgb');
    expect(set.normalMap.colorSpace).toBe('');
    expect(set.roughnessMap.colorSpace).toBe('');
    expect(set.aoMap.colorSpace).toBe('');
    set.dispose();
  });

  it('keeps normal maps dominated by +Z, as tangent space requires', () => {
    const set = generateMaterialSet('rock', { size: 64 });
    const data = set.normalMap.image.data as Uint8Array;
    for (let i = 0; i < data.length; i += 4) {
      // B is the Z component; a tangent-space normal must point out of the
      // surface, so encoded B is always above the 128 midpoint.
      expect(data[i + 2]!).toBeGreaterThan(128);
    }
    set.dispose();
  });

  it('respects the wetStone preset being smoother than the mud preset', () => {
    // Guards the art direction, not just the plumbing: wet stone must read as
    // slick or the "wet" in its name is a lie.
    const wet = generateMaterialSet('wetStone', { size: 64 });
    const mud = generateMaterialSet('mud', { size: 64 });
    const mean = (t: Uint8Array): number => {
      let sum = 0;
      for (let i = 0; i < t.length; i += 4) sum += t[i]!;
      return sum / (t.length / 4);
    };
    expect(mean(wet.roughnessMap.image.data as Uint8Array)).toBeLessThan(
      mean(mud.roughnessMap.image.data as Uint8Array),
    );
    wet.dispose();
    mud.dispose();
  });

  it('builds a material wired to all four maps', () => {
    const set = generateMaterialSet('planks', { size: 32 });
    const material = set.toMaterial();
    expect(material.map).toBe(set.map);
    expect(material.normalMap).toBe(set.normalMap);
    expect(material.roughnessMap).toBe(set.roughnessMap);
    expect(material.aoMap).toBe(set.aoMap);
    material.dispose();
    set.dispose();
  });
});

describe('rock geometry', () => {
  it('is deterministic for a seed', () => {
    const a = generateRockGeometry({ seed: 'rock-1', detail: 2 });
    const b = generateRockGeometry({ seed: 'rock-1', detail: 2 });
    expect(Array.from(a.getAttribute('position').array)).toEqual(
      Array.from(b.getAttribute('position').array),
    );
    a.dispose();
    b.dispose();
  });

  it('varies with the seed', () => {
    const a = generateRockGeometry({ seed: 'rock-1', detail: 2 });
    const b = generateRockGeometry({ seed: 'rock-2', detail: 2 });
    expect(Array.from(a.getAttribute('position').array)).not.toEqual(
      Array.from(b.getAttribute('position').array),
    );
    a.dispose();
    b.dispose();
  });

  it('carries normals and bounds, and is faceted (non-indexed)', () => {
    const rock = generateRockGeometry({ detail: 2 });
    expect(rock.getAttribute('normal')).toBeDefined();
    expect(rock.boundingBox).not.toBeNull();
    expect(rock.boundingSphere).not.toBeNull();
    expect(rock.index).toBeNull();
    rock.dispose();
  });

  it('scales subdivision as expected', () => {
    const coarse = generateRockGeometry({ detail: 1 });
    const fine = generateRockGeometry({ detail: 3 });
    expect(fine.getAttribute('position').count).toBeGreaterThan(
      coarse.getAttribute('position').count,
    );
    coarse.dispose();
    fine.dispose();
  });

  it('flattens the base so boulders sit on the ground', () => {
    const rock = generateRockGeometry({ seed: 'seated', detail: 3, radius: 1, flattenBase: 0.4 });
    const position = rock.getAttribute('position');
    let min = Infinity;
    for (let i = 0; i < position.count; i++) min = Math.min(min, position.getY(i));
    // With flattenBase the lowest point is pulled well above the un-flattened
    // extent of the squashed sphere.
    expect(min).toBeGreaterThan(-1);
    rock.dispose();
  });
});

describe('scatter', () => {
  it('is deterministic', () => {
    const options = { count: 200, area: 50, seed: 'trees' };
    const a = scatter(options).map((s) => [s.position.x, s.position.z, s.scale]);
    const b = scatter(options).map((s) => [s.position.x, s.position.z, s.scale]);
    expect(a).toEqual(b);
  });

  it('stays inside the requested area', () => {
    for (const sample of scatter({ count: 500, area: 20, seed: 'bounds' })) {
      expect(Math.abs(sample.position.x)).toBeLessThanOrEqual(20);
      expect(Math.abs(sample.position.z)).toBeLessThanOrEqual(20);
    }
  });

  it('honours minimum spacing', () => {
    const samples = scatter({ count: 400, area: 30, seed: 'spaced', minSpacing: 4 });
    for (let i = 0; i < samples.length; i++) {
      for (let j = i + 1; j < samples.length; j++) {
        expect(samples[i]!.position.distanceTo(samples[j]!.position)).toBeGreaterThanOrEqual(4);
      }
    }
    expect(samples.length).toBeGreaterThan(0);
  });

  it('rejects samples the surface query refuses', () => {
    // Only accept the +x half-plane; nothing should land on the -x side.
    const samples = scatter({
      count: 400,
      area: 25,
      seed: 'halfplane',
      surface: (x) => (x > 0 ? { y: 0, normal: { x: 0, y: 1, z: 0 } as never } : null),
    });
    expect(samples.length).toBeGreaterThan(0);
    for (const sample of samples) expect(sample.position.x).toBeGreaterThan(0);
  });

  it('rejects slopes steeper than the limit', () => {
    // A 60-degree surface against a 30-degree limit must reject everything.
    const steep = Math.cos(Math.PI / 3);
    const samples = scatter({
      count: 100,
      area: 10,
      seed: 'slope',
      maxSlopeDegrees: 30,
      surface: () => ({ y: 0, normal: { x: 0, y: steep, z: 0 } as never }),
    });
    expect(samples).toHaveLength(0);
  });

  it('keeps determinism independent of the density function', () => {
    // The density roll is drawn unconditionally, so the positions of accepted
    // samples must be a subset of the undensified run rather than a reshuffle.
    const base = scatter({ count: 300, area: 40, seed: 'density' });
    const dense = scatter({ count: 300, area: 40, seed: 'density', density: () => 0.5 });
    const baseKeys = new Set(base.map((s) => `${s.position.x},${s.position.z}`));
    expect(dense.length).toBeLessThan(base.length);
    for (const sample of dense) {
      expect(baseKeys.has(`${sample.position.x},${sample.position.z}`)).toBe(true);
    }
  });

  it('builds an instanced mesh with one transform per sample', () => {
    const samples = scatter({ count: 50, area: 10, seed: 'instances' });
    const geometry = generateRockGeometry({ detail: 1 });
    const set = generateMaterialSet('rock', { size: 32 });
    const material = set.toMaterial();
    const mesh = buildInstancedMesh(geometry, material, samples);

    expect(mesh.count).toBe(samples.length);
    expect(mesh.instanceMatrix.array.length).toBe(samples.length * 16);
    // A zero matrix would mean a transform was never written.
    expect(Array.from(mesh.instanceMatrix.array).some((v) => v !== 0)).toBe(true);

    mesh.dispose();
    material.dispose();
    set.dispose();
    geometry.dispose();
  });
});
