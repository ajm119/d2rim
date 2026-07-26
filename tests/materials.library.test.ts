/**
 * Integration tests for the material library.
 *
 * These do two things a GPU is not needed for.
 *
 * First, they assemble the actual TSL node graph for every archetype and both
 * projections. Node construction is where TSL API misuse shows up — a swizzle
 * that does not exist, an operator applied to the wrong node type, a `Loop`
 * built outside a stack — and catching it here is much cheaper than catching it
 * as a blank frame in a capture. Everything outside an `Fn` body is built
 * eagerly, so this exercises triplanar projection, hex-tiling, the detail layer,
 * macro variation, the wetness model and the two-archetype blend.
 *
 * Second, they audit the archetype table itself. Every number in it is art
 * direction, and a typo in a reflectance or a porosity is not a crash — it is a
 * substance that behaves wrongly forever. The bounds asserted here are the
 * physical ones.
 */

import { describe, expect, it } from 'vitest';

import { ARCHETYPE_SPECS, resolveSpec } from '../src/render/materials/Archetypes';
import { MATERIAL_ARCHETYPES, QUALITY_TIERS } from '../src/render/materials/types';
import type { MaterialArchetype, MaterialQuality } from '../src/render/materials/types';
import { GENERATED_ASSETS } from '../src/assets/registry.generated';
import { MaterialLibrary } from '../src/render/MaterialLibrary';

const QUALITIES: readonly MaterialQuality[] = ['low', 'medium', 'high', 'ultra'];

/**
 * A library with no engine and no AssetManager.
 *
 * This is the degraded path on purpose: every texture slot falls back to a
 * placeholder or to a procedural set, which is exactly the configuration the
 * game must still boot in when an asset checkout is incomplete.
 */
function makeLibrary(quality: MaterialQuality = 'high'): MaterialLibrary {
  return new MaterialLibrary({ quality, proceduralSize: 32, derivedHeightSize: 32 });
}

describe('archetype table', () => {
  it('covers every declared archetype exactly once', () => {
    expect(Object.keys(ARCHETYPE_SPECS).sort()).toEqual([...MATERIAL_ARCHETYPES].sort());
  });

  it('has a self-consistent archetype field on every row', () => {
    for (const name of MATERIAL_ARCHETYPES) {
      expect(ARCHETYPE_SPECS[name].archetype).toBe(name);
    }
  });

  it('references only texture keys that exist in the generated registry', () => {
    // The whole point of going through semantic keys is that a missing asset is
    // a compile-or-test-time failure, not a 404 at runtime.
    for (const name of MATERIAL_ARCHETYPES) {
      const spec = ARCHETYPE_SPECS[name];
      for (const key of Object.values(spec.textures)) {
        expect(Object.hasOwn(GENERATED_ASSETS, key)).toBe(true);
      }
      if (spec.detail !== null) {
        expect(Object.hasOwn(GENERATED_ASSETS, spec.detail.normalKey)).toBe(true);
      }
    }
  });

  it('keeps every physical parameter inside its physical range', () => {
    for (const name of MATERIAL_ARCHETYPES) {
      const spec = ARCHETYPE_SPECS[name];

      expect(spec.porosity).toBeGreaterThanOrEqual(0);
      expect(spec.porosity).toBeLessThanOrEqual(1);
      expect(spec.wetnessExposure).toBeGreaterThanOrEqual(0);
      expect(spec.wetnessExposure).toBeLessThanOrEqual(1);
      expect(spec.metalness).toBeGreaterThanOrEqual(0);
      expect(spec.metalness).toBeLessThanOrEqual(1);
      expect(spec.clearcoat).toBeGreaterThanOrEqual(0);
      expect(spec.clearcoat).toBeLessThanOrEqual(1);

      // Dielectric F0 for real-world materials spans roughly 0.02 (water) to
      // 0.08 (diamond-like). Anything outside that is a typo.
      expect(spec.reflectance).toBeGreaterThanOrEqual(0.02);
      expect(spec.reflectance).toBeLessThanOrEqual(0.08);

      const [rLo, rHi] = spec.roughnessRange;
      expect(rLo).toBeGreaterThanOrEqual(0);
      expect(rHi).toBeLessThanOrEqual(1);
      expect(rHi).toBeGreaterThan(rLo);

      expect(spec.tiling).toBeGreaterThan(0);
      expect(spec.triplanarSharpness).toBeGreaterThanOrEqual(1);
      expect(spec.normalStrength).toBeGreaterThan(0);
    }
  });

  it('keeps every albedo tint in a plausible linear range', () => {
    for (const name of MATERIAL_ARCHETYPES) {
      for (const c of ARCHETYPE_SPECS[name].albedoTint) {
        expect(c).toBeGreaterThan(0);
        // No real diffuse surface reflects more than ~0.9; the tint is a grade
        // on top of a texture, so it should never brighten either.
        expect(c).toBeLessThanOrEqual(1);
      }
    }
  });

  it('keeps macro tints dark, so variation separates hue without saturating', () => {
    for (const name of MATERIAL_ARCHETYPES) {
      const macro = ARCHETYPE_SPECS[name].macro;
      if (macro === null) continue;
      expect(macro.metres).toBeGreaterThan(0);
      expect(macro.tintAmount).toBeGreaterThanOrEqual(0);
      expect(macro.tintAmount).toBeLessThanOrEqual(1);
      for (const c of macro.tint) {
        expect(c).toBeGreaterThanOrEqual(0);
        // The art direction is desaturated grimdark: a bright macro tint would
        // read as a colour filter over the whole world.
        expect(c).toBeLessThan(0.15);
      }
    }
  });

  it('gives every terrain-scale archetype some anti-tiling', () => {
    // Visible repetition on a large surface is an automatic fail. Anything
    // projected triplanarly is by definition covering a lot of ground.
    for (const name of MATERIAL_ARCHETYPES) {
      const spec = ARCHETYPE_SPECS[name];
      if (spec.projection !== 'triplanar') continue;
      expect(spec.antiTile).toBe('hex');
      expect(spec.macro).not.toBeNull();
    }
  });

  it('only enables parallax where a UV projection can support it', () => {
    for (const name of MATERIAL_ARCHETYPES) {
      const spec = ARCHETYPE_SPECS[name];
      if (spec.parallax === null) continue;
      expect(spec.projection).toBe('uv');
      expect(spec.parallax.scale).toBeGreaterThan(0);
      expect(spec.parallax.scale).toBeLessThan(0.2);
      expect(spec.parallax.maxSteps).toBeGreaterThanOrEqual(spec.parallax.minSteps);
      expect(spec.parallax.fadeEnd).toBeGreaterThan(0);
    }
  });

  it('orders every detail fade correctly', () => {
    for (const name of MATERIAL_ARCHETYPES) {
      const detail = ARCHETYPE_SPECS[name].detail;
      if (detail === null) continue;
      expect(detail.fadeEnd).toBeGreaterThan(detail.fadeStart);
      expect(detail.tiling).toBeGreaterThan(ARCHETYPE_SPECS[name].tiling);
      expect(detail.normalStrength).toBeGreaterThan(0);
    }
  });

  it('encodes the porosity contrast the wetness model depends on', () => {
    // These orderings are the art direction, not incidental values: if mud were
    // not far more porous than iron, rain would look identical on both.
    expect(ARCHETYPE_SPECS.wetMud.porosity).toBeGreaterThan(0.8);
    expect(ARCHETYPE_SPECS.ironClean.porosity).toBeLessThan(0.1);
    expect(ARCHETYPE_SPECS.wetMud.porosity).toBeGreaterThan(ARCHETYPE_SPECS.rock.porosity);
    expect(ARCHETYPE_SPECS.mossyRock.porosity).toBeGreaterThan(ARCHETYPE_SPECS.rock.porosity);
    // And only the non-porous substances are allowed a water film clearcoat.
    expect(ARCHETYPE_SPECS.ironClean.clearcoat).toBeGreaterThan(0);
    expect(ARCHETYPE_SPECS.wetMud.clearcoat).toBe(0);
  });
});

describe('spec resolution', () => {
  it('returns the table entry unchanged with no overrides', () => {
    expect(resolveSpec('rock')).toBe(ARCHETYPE_SPECS.rock);
  });

  it('merges overrides without mutating the table', () => {
    const before = ARCHETYPE_SPECS.rock.tiling;
    const merged = resolveSpec('rock', { tiling: 4, parallax: null });
    expect(merged.tiling).toBe(4);
    expect(merged.parallax).toBeNull();
    expect(ARCHETYPE_SPECS.rock.tiling).toBe(before);
  });

  it('never lets an override change which archetype a spec is', () => {
    const merged = resolveSpec('rock', { archetype: 'skin' } as Partial<
      typeof ARCHETYPE_SPECS.rock
    >);
    expect(merged.archetype).toBe('rock');
  });
});

describe('quality tiers', () => {
  it('are ordered from cheapest to most expensive', () => {
    const order: MaterialQuality[] = ['low', 'medium', 'high', 'ultra'];
    for (let i = 1; i < order.length; i++) {
      const previous = QUALITY_TIERS[order[i - 1] as MaterialQuality];
      const current = QUALITY_TIERS[order[i] as MaterialQuality];
      expect(current.parallaxStepScale).toBeGreaterThanOrEqual(previous.parallaxStepScale);
      expect(current.detailRangeScale).toBeGreaterThanOrEqual(previous.detailRangeScale);
    }
  });

  it('turns the expensive features off at the lowest tier', () => {
    expect(QUALITY_TIERS.low.parallax).toBe(false);
    expect(QUALITY_TIERS.low.hexTiling).toBe(false);
  });
});

describe('material construction', () => {
  it('builds a complete node graph for every archetype', () => {
    const library = makeLibrary();
    try {
      for (const archetype of MATERIAL_ARCHETYPES) {
        const material = library.create(archetype);
        expect(material.colorNode, archetype).not.toBeNull();
        expect(material.normalNode, archetype).not.toBeNull();
        expect(material.roughnessNode, archetype).not.toBeNull();
        expect(material.metalnessNode, archetype).not.toBeNull();
        expect(material.aoNode, archetype).not.toBeNull();
        expect(material.specularIntensityNode, archetype).not.toBeNull();
        expect(material.name).toBe(`d2rim.${archetype}`);
      }
    } finally {
      library.dispose();
    }
  });

  it('builds every archetype at every quality tier', () => {
    // The tiers change the *structure* of the shader — parallax and hex-tiling
    // appear and disappear — so each is a distinct node graph to get wrong.
    for (const quality of QUALITIES) {
      const library = makeLibrary(quality);
      try {
        for (const archetype of MATERIAL_ARCHETYPES) {
          expect(library.create(archetype).colorNode, `${archetype}@${quality}`).not.toBeNull();
        }
      } finally {
        library.dispose();
      }
    }
  });

  it('caches shared materials and hands out fresh ones on request', () => {
    const library = makeLibrary();
    try {
      expect(library.get('rock')).toBe(library.get('rock'));
      expect(library.create('rock')).not.toBe(library.create('rock'));
    } finally {
      library.dispose();
    }
  });

  it('honours a projection override, building the other code path', () => {
    const library = makeLibrary();
    try {
      const asUv = library.create('rock', { projection: 'uv', parallax: null });
      expect(asUv.colorNode).not.toBeNull();
      const asTriplanar = library.create('wetStone', {
        projection: 'triplanar',
        parallax: null,
      });
      expect(asTriplanar.colorNode).not.toBeNull();
    } finally {
      library.dispose();
    }
  });

  it('enables clearcoat only where the archetype asks for it', () => {
    const library = makeLibrary();
    try {
      expect(library.create('ironClean').clearcoatNode).not.toBeNull();
      // A material that never gets a water film must not compile the clearcoat
      // branch at all; mud with a lacquer coat is the classic wetness failure.
      expect(library.create('wetMud').clearcoatNode).toBeNull();
    } finally {
      library.dispose();
    }
  });

  it('enables sheen only on fabrics', () => {
    const library = makeLibrary();
    try {
      expect(library.create('cloth').sheenNode).not.toBeNull();
      expect(library.create('rock').sheenNode).toBeNull();
    } finally {
      library.dispose();
    }
  });

  it('keeps the IOR at the value the reflectance maths assumes', () => {
    // `specularIntensityNode` is `reflectance / 0.04`, and 0.04 is only the
    // right divisor while `ior` is 1.5. If one moves the other must.
    const library = makeLibrary();
    try {
      expect(library.create('rock').ior).toBeCloseTo(1.5, 10);
    } finally {
      library.dispose();
    }
  });

  it('builds a height-blended two-archetype material', () => {
    const library = makeLibrary();
    try {
      const blended = library.createBlended({ base: 'rock', overlay: 'deadGrass' });
      expect(blended.colorNode).not.toBeNull();
      expect(blended.normalNode).not.toBeNull();
      expect(blended.roughnessNode).not.toBeNull();
      expect(blended.name).toBe('d2rim.rock+deadGrass');
    } finally {
      library.dispose();
    }
  });

  it('builds a blend across the projection boundary', () => {
    // Mud is triplanar and masonry is UV; a courtyard needs exactly this.
    const library = makeLibrary();
    try {
      const blended = library.createBlended({
        base: 'wetStone',
        overlay: 'wetMud',
        depth: 0.05,
      });
      expect(blended.colorNode).not.toBeNull();
    } finally {
      library.dispose();
    }
  });
});

describe('wetness control', () => {
  it('clamps and propagates the global wetness', () => {
    const library = makeLibrary();
    try {
      library.setWetness(0.42);
      expect(library.wetness).toBeCloseTo(0.42, 10);
      library.setWetness(5);
      expect(library.wetness).toBe(1);
      library.setWetness(-2);
      expect(library.wetness).toBe(0);
    } finally {
      library.dispose();
    }
  });

  it('couples the puddle level to wetness by default', () => {
    const library = makeLibrary();
    try {
      library.setWetness(1);
      expect(library.stats().puddleLevel).toBeGreaterThan(0);
      library.setWetness(0);
      expect(library.stats().puddleLevel).toBe(0);
    } finally {
      library.dispose();
    }
  });

  it('starts rain-soaked, because the Blood Moor is', () => {
    const library = new MaterialLibrary();
    try {
      expect(library.wetness).toBeGreaterThan(0.5);
    } finally {
      library.dispose();
    }
  });
});

describe('quality switching', () => {
  it('rebuilds cached materials in place so live meshes stay valid', () => {
    const library = makeLibrary('low');
    try {
      const material = library.get('wetStone');
      const before = material.colorNode;
      library.setQuality('ultra');
      expect(library.quality).toBe('ultra');
      // Same object — meshes already referencing it must not go stale…
      expect(library.get('wetStone')).toBe(material);
      // …but a different graph, since low disables parallax and ultra does not.
      expect(material.colorNode).not.toBe(before);
    } finally {
      library.dispose();
    }
  });

  it('is a no-op when the tier does not change', () => {
    const library = makeLibrary('high');
    try {
      const before = library.get('rock').colorNode;
      library.setQuality('high');
      expect(library.get('rock').colorNode).toBe(before);
    } finally {
      library.dispose();
    }
  });
});

describe('texture resolution without an AssetManager', () => {
  it('settles every requested archetype and reports it', async () => {
    const library = makeLibrary();
    try {
      const wanted: MaterialArchetype[] = ['rock', 'leather'];
      for (const a of wanted) library.create(a);
      await library.ready(wanted);
      const stats = library.stats();
      expect(stats.textureSets).toBeGreaterThanOrEqual(wanted.length);
      expect(stats.settledSets).toBeGreaterThanOrEqual(wanted.length);
      expect(stats.weatherProvider).toBe(false);
    } finally {
      library.dispose();
    }
  });
});
