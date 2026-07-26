/**
 * @module render/materials/Archetypes
 *
 * The art direction, as data.
 *
 * Every number in this file is a decision about how Act I looks, and this is
 * the only place any of them are made. The register is cold, overcast, medieval
 * grimdark: rain-soaked mud, wet stone, dead grass, bare wood, rusted iron.
 * Desaturated — but the separation between cold blue-grey shadow and warm
 * firelight, between moss green and rust and dried blood, has to survive, so
 * "desaturated" is achieved by keeping *luminance* low and letting the macro
 * tint push hue apart, never by desaturating the albedo towards grey.
 *
 * ### How to read a row
 *
 * - `albedoTint` multiplies the sampled albedo. For archetypes with an albedo
 *   map it is a grade, and lives near 1. For archetypes without one it *is* the
 *   albedo, because the placeholder is white.
 * - `roughnessRange` remaps the sampled roughness. Free CC0 sets are almost
 *   universally authored too glossy for an overcast key light; the low end of
 *   these ranges is where "wet stone" stops looking like polished granite.
 * - `reflectance` is normal-incidence dielectric F₀. 0.04 is the everyday
 *   default (IOR 1.5). Organics sit slightly below, dense wet minerals above.
 * - `porosity` decides how wetness reads. See
 *   {@link module:render/materials/Wetness}; it is the single most consequential
 *   number in this file.
 * - `tiling` is repeats per UV unit for `uv` projection, and repeats per *metre*
 *   for `triplanar`.
 *
 * ### Texture keys
 *
 * Only keys that exist in the generated asset registry appear here, and only as
 * semantic keys. Archetypes with no reachable CC0 set either synthesise their
 * base maps procedurally (`useProceduralBase`) or run on an authored flat colour
 * plus a detail normal, which for skin is the honest answer anyway since real
 * skin maps arrive with the character mesh.
 */

import type { MaterialArchetype, SurfaceSpec } from './types';

/** Shared detail-layer presets, so the fade distances stay consistent. */
const GROUND_DETAIL = {
  normalKey: 'detail.normal.organic',
  tiling: 14,
  normalStrength: 0.55,
  albedoStrength: 0.35,
  fadeStart: 5,
  fadeEnd: 14,
} as const;

const STONE_DETAIL = {
  normalKey: 'detail.normal.coarse',
  tiling: 10,
  normalStrength: 0.6,
  albedoStrength: 0.3,
  fadeStart: 4,
  fadeEnd: 11,
} as const;

const FINE_DETAIL = {
  normalKey: 'detail.normal.fine',
  tiling: 18,
  normalStrength: 0.45,
  albedoStrength: 0.25,
  fadeStart: 2.5,
  fadeEnd: 7,
} as const;

/**
 * The archetype table.
 *
 * Frozen at module scope: these are defaults, and a caller that wants something
 * different gets a merged copy from {@link resolveSpec} rather than the ability
 * to mutate the art direction out from under every other material in the world.
 */
export const ARCHETYPE_SPECS: Readonly<Record<MaterialArchetype, SurfaceSpec>> = {
  /* --- Ground -------------------------------------------------------- */

  /**
   * The Blood Moor floor. Churned, rutted, permanently saturated.
   *
   * Porosity 0.95 is the highest in the table and it is what makes rain read
   * correctly here: mud drinks the water, so it goes much darker and much
   * richer in colour without ever going glossy. The gloss shows up only where
   * the puddle mask says water is actually standing.
   */
  wetMud: {
    archetype: 'wetMud',
    textures: {
      albedo: 'terrain.mud.albedo',
      normal: 'terrain.mud.normal',
      roughness: 'terrain.mud.roughness',
      ao: 'terrain.mud.ao',
    },
    proceduralFallback: 'mud',
    useProceduralBase: false,
    albedoTint: [0.52, 0.55, 0.58],
    roughnessRange: [0.55, 0.96],
    metalness: 0,
    reflectance: 0.042,
    aoStrength: 1,
    normalStrength: 1.05,
    projection: 'triplanar',
    tiling: 0.45,
    triplanarSharpness: 6,
    antiTile: 'hex',
    detail: GROUND_DETAIL,
    macro: {
      metres: 34,
      albedoAmount: 0.3,
      roughnessAmount: 0.14,
      // A rust-brown that reads as iron-rich soil and dried blood, not as a
      // warm filter. Very dark, so it darkens *and* separates hue.
      tint: [0.05, 0.036, 0.028],
      tintAmount: 0.28,
    },
    parallax: null,
    porosity: 0.95,
    wetnessExposure: 1,
    sheen: 0,
    sheenRoughness: 1,
    clearcoat: 0,
  },

  /** Trampled, half-dead grass over soil. Never a bright fantasy meadow. */
  deadGrass: {
    archetype: 'deadGrass',
    textures: {
      albedo: 'terrain.grass.albedo',
      normal: 'terrain.grass.normal',
      roughness: 'terrain.grass.roughness',
      ao: 'terrain.grass.ao',
    },
    proceduralFallback: 'grass',
    useProceduralBase: false,
    albedoTint: [0.56, 0.60, 0.50],
    roughnessRange: [0.68, 0.99],
    metalness: 0,
    reflectance: 0.035,
    aoStrength: 1,
    normalStrength: 0.9,
    projection: 'triplanar',
    // Grass belongs to the ground plane; a high sharpness keeps the Y
    // projection dominant and stops it from creeping up steep banks, where the
    // rock archetype should be taking over anyway.
    tiling: 0.5,
    triplanarSharpness: 9,
    antiTile: 'hex',
    detail: GROUND_DETAIL,
    macro: {
      metres: 42,
      albedoAmount: 0.34,
      roughnessAmount: 0.1,
      tint: [0.032, 0.048, 0.02],
      tintAmount: 0.42,
    },
    parallax: null,
    porosity: 0.88,
    wetnessExposure: 1,
    sheen: 0,
    sheenRoughness: 1,
    clearcoat: 0,
  },

  /* --- Rock ---------------------------------------------------------- */

  /** Bare cliff rock. Cold, slightly blue in shadow. */
  rock: {
    archetype: 'rock',
    textures: {
      albedo: 'rock.cliff.albedo',
      normal: 'rock.cliff.normal',
      roughness: 'rock.cliff.roughness',
      ao: 'rock.cliff.ao',
    },
    proceduralFallback: 'rock',
    useProceduralBase: false,
    albedoTint: [0.7, 0.71, 0.75],
    roughnessRange: [0.45, 0.92],
    metalness: 0,
    reflectance: 0.045,
    aoStrength: 1,
    normalStrength: 1.1,
    projection: 'triplanar',
    tiling: 0.32,
    triplanarSharpness: 5,
    antiTile: 'hex',
    detail: STONE_DETAIL,
    macro: {
      metres: 28,
      albedoAmount: 0.28,
      roughnessAmount: 0.16,
      tint: [0.042, 0.05, 0.038],
      tintAmount: 0.3,
    },
    parallax: null,
    porosity: 0.35,
    wetnessExposure: 1,
    sheen: 0,
    sheenRoughness: 1,
    clearcoat: 0,
  },

  /** Rock the moor has been growing on for a century. */
  mossyRock: {
    archetype: 'mossyRock',
    textures: {
      albedo: 'rock.mossy.albedo',
      normal: 'rock.mossy.normal',
      roughness: 'rock.mossy.roughness',
      ao: 'rock.mossy.ao',
    },
    proceduralFallback: 'rock',
    useProceduralBase: false,
    albedoTint: [0.66, 0.72, 0.6],
    roughnessRange: [0.5, 0.95],
    metalness: 0,
    reflectance: 0.042,
    aoStrength: 1,
    normalStrength: 1.05,
    projection: 'triplanar',
    tiling: 0.34,
    triplanarSharpness: 5,
    antiTile: 'hex',
    detail: STONE_DETAIL,
    macro: {
      metres: 24,
      albedoAmount: 0.3,
      roughnessAmount: 0.14,
      tint: [0.026, 0.04, 0.016],
      tintAmount: 0.5,
    },
    parallax: null,
    // Moss holds water like a sponge; bare rock does not. The blend is why this
    // archetype exists separately rather than being a tint on `rock`.
    porosity: 0.6,
    wetnessExposure: 1,
    sheen: 0,
    sheenRoughness: 1,
    clearcoat: 0,
  },

  /* --- Built ---------------------------------------------------------- */

  /**
   * Rain-blackened masonry. The showcase for parallax: mortar courses are deep,
   * regular and always seen at an angle, which is exactly the case POM was
   * invented for.
   */
  wetStone: {
    archetype: 'wetStone',
    textures: {
      albedo: 'stone.masonry.albedo',
      normal: 'stone.masonry.normal',
      roughness: 'stone.masonry.roughness',
      ao: 'stone.masonry.ao',
    },
    proceduralFallback: 'wetStone',
    useProceduralBase: false,
    albedoTint: [0.44, 0.47, 0.52],
    roughnessRange: [0.32, 0.88],
    metalness: 0,
    reflectance: 0.05,
    aoStrength: 1,
    normalStrength: 1,
    projection: 'uv',
    tiling: 1,
    triplanarSharpness: 6,
    // `macro`, not `hex`: this archetype already spends its texture budget on
    // the parallax march, and masonry's regular courses make lattice offsets
    // conspicuous rather than concealing.
    antiTile: 'macro',
    detail: STONE_DETAIL,
    macro: {
      metres: 22,
      albedoAmount: 0.24,
      roughnessAmount: 0.14,
      tint: [0.03, 0.036, 0.032],
      tintAmount: 0.32,
    },
    parallax: {
      // 16 rather than the 24 the relief could use: three emits the march once
      // in the normal sub-build and once in the main flow, so the shader
      // actually runs 32 samples at grazing incidence. See the note in
      // `Parallax.ts` — the step counts here are the *emitted* ones, and the
      // real cost is double.
      scale: 0.045,
      maxSteps: 16,
      minSteps: 6,
      fadeEnd: 14,
      clipSilhouette: false,
    },
    porosity: 0.4,
    wetnessExposure: 1,
    sheen: 0,
    sheenRoughness: 1,
    clearcoat: 0,
  },

  /* --- Wood ----------------------------------------------------------- */

  bark: {
    archetype: 'bark',
    textures: {
      albedo: 'wood.bark.albedo',
      normal: 'wood.bark.normal',
      roughness: 'wood.bark.roughness',
      ao: 'wood.bark.ao',
    },
    proceduralFallback: 'bark',
    useProceduralBase: false,
    albedoTint: [0.62, 0.60, 0.56],
    roughnessRange: [0.6, 0.97],
    metalness: 0,
    reflectance: 0.038,
    aoStrength: 1,
    normalStrength: 1.15,
    projection: 'uv',
    tiling: 1,
    triplanarSharpness: 6,
    antiTile: 'macro',
    detail: STONE_DETAIL,
    macro: {
      metres: 9,
      albedoAmount: 0.26,
      roughnessAmount: 0.1,
      tint: [0.024, 0.033, 0.018],
      tintAmount: 0.38,
    },
    parallax: null,
    porosity: 0.78,
    wetnessExposure: 1,
    sheen: 0,
    sheenRoughness: 1,
    clearcoat: 0,
  },

  /** Weathered plank: fences, palisades, shutters. */
  plank: {
    archetype: 'plank',
    textures: {
      albedo: 'wood.plank.albedo',
      normal: 'wood.plank.normal',
      roughness: 'wood.plank.roughness',
      ao: 'wood.plank.ao',
    },
    proceduralFallback: 'planks',
    useProceduralBase: false,
    albedoTint: [0.64, 0.62, 0.58],
    roughnessRange: [0.5, 0.92],
    metalness: 0,
    reflectance: 0.04,
    aoStrength: 1,
    normalStrength: 1.05,
    projection: 'uv',
    tiling: 1,
    triplanarSharpness: 6,
    antiTile: 'macro',
    detail: FINE_DETAIL,
    macro: {
      metres: 11,
      albedoAmount: 0.22,
      roughnessAmount: 0.1,
      tint: [0.028, 0.03, 0.02],
      tintAmount: 0.3,
    },
    parallax: {
      scale: 0.03,
      maxSteps: 12,
      minSteps: 5,
      fadeEnd: 10,
      clipSilhouette: false,
    },
    porosity: 0.72,
    wetnessExposure: 1,
    sheen: 0,
    sheenRoughness: 1,
    clearcoat: 0,
  },

  /** Interior boarding. Drier, so a lower wetness exposure. */
  woodFloor: {
    archetype: 'woodFloor',
    textures: {
      albedo: 'wood.floor.albedo',
      normal: 'wood.floor.normal',
      roughness: 'wood.floor.roughness',
      ao: 'wood.floor.ao',
    },
    proceduralFallback: 'planks',
    useProceduralBase: false,
    albedoTint: [0.76, 0.71, 0.63],
    roughnessRange: [0.42, 0.85],
    metalness: 0,
    reflectance: 0.042,
    aoStrength: 1,
    normalStrength: 0.95,
    projection: 'uv',
    tiling: 1,
    triplanarSharpness: 6,
    antiTile: 'macro',
    detail: FINE_DETAIL,
    macro: {
      metres: 14,
      albedoAmount: 0.2,
      roughnessAmount: 0.1,
      tint: [0.03, 0.026, 0.018],
      tintAmount: 0.25,
    },
    parallax: {
      scale: 0.022,
      maxSteps: 10,
      minSteps: 4,
      fadeEnd: 8,
      clipSilhouette: false,
    },
    porosity: 0.6,
    wetnessExposure: 0.35,
    sheen: 0,
    sheenRoughness: 1,
    clearcoat: 0,
  },

  /* --- Metal ---------------------------------------------------------- */

  /**
   * Rusted iron: hinges, banding, weapon racks, the state most metal in Act I
   * is actually in.
   *
   * Metalness is deliberately below 1. Rust is iron *oxide*, a dielectric, so a
   * rusted surface is a spatial mix of conductor and insulator, and forcing
   * metalness to 1 makes the rust read as painted metal instead of as
   * corrosion. Its porosity is high for the same reason: rust drinks water.
   */
  ironRusted: {
    archetype: 'ironRusted',
    textures: {
      albedo: 'metal.iron.albedo',
      normal: 'metal.iron.normal',
      roughness: 'metal.iron.roughness',
      metalness: 'metal.iron.metalness',
    },
    proceduralFallback: 'rock',
    useProceduralBase: false,
    albedoTint: [0.72, 0.5, 0.36],
    roughnessRange: [0.42, 0.95],
    metalness: 0.7,
    reflectance: 0.05,
    aoStrength: 0.6,
    normalStrength: 1.1,
    projection: 'uv',
    tiling: 1,
    triplanarSharpness: 6,
    antiTile: 'macro',
    detail: FINE_DETAIL,
    macro: {
      metres: 6,
      albedoAmount: 0.32,
      roughnessAmount: 0.16,
      tint: [0.09, 0.035, 0.012],
      tintAmount: 0.45,
    },
    parallax: null,
    porosity: 0.55,
    wetnessExposure: 1,
    sheen: 0,
    sheenRoughness: 1,
    clearcoat: 0,
  },

  /**
   * Clean iron: a drawn blade, a fresh buckle.
   *
   * Porosity 0.02 and a clearcoat of 0.6. This is the archetype that makes rain
   * legible — it is the only common substance that turns genuinely mirror-like
   * when wet, and it is what gives SSR something to work with in a scene
   * otherwise made of mud.
   */
  ironClean: {
    archetype: 'ironClean',
    textures: {
      albedo: 'metal.iron.albedo',
      normal: 'metal.iron.normal',
      roughness: 'metal.iron.roughness',
      metalness: 'metal.iron.metalness',
    },
    proceduralFallback: 'rock',
    useProceduralBase: false,
    albedoTint: [0.62, 0.63, 0.66],
    roughnessRange: [0.16, 0.5],
    metalness: 1,
    reflectance: 0.05,
    aoStrength: 0.5,
    normalStrength: 0.8,
    projection: 'uv',
    tiling: 1,
    triplanarSharpness: 6,
    antiTile: 'off',
    detail: FINE_DETAIL,
    macro: {
      metres: 4,
      albedoAmount: 0.12,
      roughnessAmount: 0.12,
      tint: [0.06, 0.05, 0.045],
      tintAmount: 0.15,
    },
    parallax: null,
    porosity: 0.02,
    wetnessExposure: 1,
    sheen: 0,
    sheenRoughness: 1,
    clearcoat: 0.6,
  },

  /* --- Soft ----------------------------------------------------------- */

  /**
   * Leather: belts, jerkins, boots, scabbards.
   *
   * No reachable CC0 leather set, so the base maps are synthesised from the
   * `bark` preset — grain structure of roughly the right frequency and
   * anisotropy — and graded hard into a dark oiled brown.
   */
  leather: {
    archetype: 'leather',
    textures: {},
    proceduralFallback: 'bark',
    useProceduralBase: true,
    albedoTint: [0.34, 0.22, 0.15],
    roughnessRange: [0.34, 0.72],
    metalness: 0,
    reflectance: 0.045,
    aoStrength: 0.9,
    normalStrength: 0.75,
    projection: 'uv',
    tiling: 3,
    triplanarSharpness: 6,
    antiTile: 'macro',
    detail: FINE_DETAIL,
    macro: {
      metres: 3,
      albedoAmount: 0.2,
      roughnessAmount: 0.14,
      tint: [0.05, 0.03, 0.02],
      tintAmount: 0.25,
    },
    parallax: null,
    // Oiled leather sheds most water but the grain still drinks some.
    porosity: 0.35,
    wetnessExposure: 0.85,
    sheen: 0.12,
    sheenRoughness: 0.6,
    clearcoat: 0.25,
  },

  /** Cloth: cloaks, tunics, banners, tent canvas. */
  cloth: {
    archetype: 'cloth',
    textures: {
      albedo: 'cloth.albedo',
      normal: 'cloth.normal',
      roughness: 'cloth.roughness',
    },
    proceduralFallback: 'planks',
    useProceduralBase: false,
    albedoTint: [0.52, 0.5, 0.48],
    roughnessRange: [0.6, 0.98],
    metalness: 0,
    reflectance: 0.035,
    aoStrength: 0.8,
    normalStrength: 0.9,
    projection: 'uv',
    tiling: 2,
    triplanarSharpness: 6,
    antiTile: 'macro',
    detail: FINE_DETAIL,
    macro: {
      metres: 2.5,
      albedoAmount: 0.18,
      roughnessAmount: 0.08,
      tint: [0.045, 0.04, 0.038],
      tintAmount: 0.2,
    },
    parallax: null,
    porosity: 0.88,
    // Cloth is mostly worn under something. Full exposure would leave every
    // character looking as though they had been thrown in a river.
    wetnessExposure: 0.6,
    sheen: 0.5,
    sheenRoughness: 0.7,
    clearcoat: 0,
  },

  /**
   * Skin.
   *
   * No base maps by design: real skin maps arrive with the character mesh, and
   * a procedural stand-in for skin looks worse than a well-chosen flat albedo.
   * What this archetype contributes is the *response* — a low reflectance
   * (0.028, skin's measured F₀), a narrow roughness range, a very fine detail
   * normal that only appears inside a couple of metres, and a small clearcoat
   * so sweat and rain read on the face.
   */
  skin: {
    archetype: 'skin',
    textures: {},
    proceduralFallback: 'mud',
    useProceduralBase: false,
    albedoTint: [0.56, 0.4, 0.34],
    roughnessRange: [0.42, 0.62],
    metalness: 0,
    reflectance: 0.028,
    aoStrength: 0.7,
    normalStrength: 1,
    projection: 'uv',
    tiling: 1,
    triplanarSharpness: 6,
    antiTile: 'off',
    detail: {
      normalKey: 'detail.normal.fine',
      tiling: 30,
      normalStrength: 0.35,
      albedoStrength: 0.15,
      fadeStart: 1.5,
      fadeEnd: 4,
    },
    macro: {
      metres: 1.5,
      albedoAmount: 0.1,
      roughnessAmount: 0.06,
      tint: [0.08, 0.03, 0.025],
      tintAmount: 0.18,
    },
    parallax: null,
    porosity: 0.15,
    wetnessExposure: 0.8,
    sheen: 0,
    sheenRoughness: 1,
    clearcoat: 0.25,
  },
};

/**
 * Merge per-instance overrides over an archetype's defaults.
 *
 * Shallow by design. The nested blocks (`detail`, `macro`, `parallax`,
 * `textures`) are replaced wholesale rather than deep-merged, because a partial
 * `parallax` override that silently inherited a step count from the archetype
 * is exactly the kind of action-at-a-distance this table exists to avoid.
 * `null` is a meaningful override: it switches the feature off.
 */
export function resolveSpec(
  archetype: MaterialArchetype,
  overrides?: Partial<SurfaceSpec>,
): SurfaceSpec {
  const base = ARCHETYPE_SPECS[archetype];
  if (overrides === undefined) return base;
  return { ...base, ...overrides, archetype };
}
