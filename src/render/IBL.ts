/**
 * @module render/IBL
 *
 * Image-based lighting: turns whatever the sky module produced into the ambient
 * diffuse and glossy specular that everything in the world is lit by, and
 * supplies the occlusion terms that stop cavities and enclosed interiors from
 * glowing with sky light they cannot possibly see.
 *
 * ### Split-sum, and which half three.js already owns
 *
 * The split-sum approximation (Karis, "Real Shading in Unreal Engine 4",
 * SIGGRAPH 2013 course notes) factors the specular IBL integral into
 *
 * ```
 * ∫ L(l) f(l,v) cosθ dl  ≈  [ ∫ L(l) D(l) dl ] · [ ∫ f(l,v) cosθ dl ]
 *                            ^ prefiltered radiance   ^ environment BRDF (DFG)
 * ```
 *
 * - The **first** term is a GGX-importance-sampled prefilter of the environment
 *   into a roughness-indexed mip chain. three.js's `PMREMGenerator` /
 *   `PMREMNode` implement exactly this, and `EnvironmentNode` samples it with a
 *   roughness-driven LOD and a reflection vector blended toward the normal by
 *   `roughness⁴` (so rough surfaces stop gathering light from behind their own
 *   tangent plane). Reusing it is the right call: it is well tested on both
 *   backends and re-implementing the prefilter would gain nothing.
 * - The **second** term is three's `DFGLUT`, Karis's analytic mobile
 *   approximation of the environment BRDF, applied inside
 *   `PhysicalLightingModel.computeMultiscattering` together with Fdez-Agüera's
 *   multi-scattering energy compensation.
 *
 * So this module's job is *not* to reimplement split-sum. It is to (a) route
 * the sky's radiance into that machinery correctly and at the right intensity,
 * and (b) supply the occlusion terms three cannot infer on its own.
 *
 * ### Occlusion
 *
 * A prefiltered environment map has no idea that a fragment sits at the bottom
 * of a crack. Two terms fix that, both from Lagarde & de Rousiers, "Moving
 * Frostbite to Physically Based Rendering" (SIGGRAPH 2014 course), §4.10:
 *
 * 1. **Specular occlusion** `computeSpecularOcclusion(N·V, AO, roughness)` —
 *    derives a specular visibility from the diffuse AO term, biased by
 *    roughness and view angle, because a mirror lobe sees far less of the
 *    hemisphere than a cosine lobe does. three's `PhysicalLightingModel`
 *    *already* applies this once `context.ambientOcclusion` is populated, which
 *    happens when a material has an `aoNode`. {@link IBLService.applyOcclusion}
 *    installs one.
 * 2. **Horizon occlusion** — when a normal map bends the shading normal far
 *    enough that the reflection vector points below the *geometric* surface,
 *    the reflection is physically impossible. Fading it out removes the bright
 *    rim that otherwise appears along every strongly normal-mapped silhouette.
 *
 * Neither term is a substitute for real SSAO/GTAO; both compose with it. If an
 * ambient-occlusion service is registered (see {@link AmbientOcclusionKey}) its
 * node is multiplied in.
 *
 * ### What this module expects from other systems
 *
 * Everything below is resolved at runtime through the `ServiceLocator` and
 * degrades gracefully when absent — see {@link SkyEnvironmentKey} and
 * {@link AmbientOcclusionKey}.
 */

import * as THREE from 'three/webgpu';
import {
  Fn,
  clamp,
  dot,
  exp2,
  float,
  materialAO,
  normalWorldGeometry,
  normalView,
  pmremTexture,
  positionViewDirection,
  reflect,
  vec3,
} from 'three/tsl';

import { AssetManagerKey } from '../assets/AssetManager';
import { serviceKey, type ServiceLocator } from '../core/ServiceLocator';
import type { GameContext, GameModule } from '../core/types';

/* ------------------------------------------------------------------------- *
 * Contracts with modules owned by other agents
 * ------------------------------------------------------------------------- */

/**
 * What this module needs from the sky/atmosphere system.
 *
 * The sky module is expected to register an object matching this shape under
 * {@link SkyEnvironmentKey}. `environmentVersion` must increment whenever
 * `environmentTexture` is regenerated in place (time-of-day changes, weather
 * transitions) so that the PMREM chain can be rebuilt; if it is omitted, the
 * texture identity alone is used and in-place edits will not be noticed.
 *
 * If no such service is registered, this module falls back to the
 * `env.overcast` HDRI from the AssetManager, and failing that leaves whatever
 * `scene.environment` already holds alone.
 */
export interface SkyEnvironmentProvider {
  /** Equirectangular or cube HDR radiance. Linear colour space, values > 1. */
  readonly environmentTexture: THREE.Texture | null;
  /** Bumped when the texture contents change without the object identity changing. */
  readonly environmentVersion?: number;
  /** Optional Y-rotation of the environment, in radians. */
  readonly environmentRotationY?: number;
}

/** Service key the sky module should register itself under. */
export const SkyEnvironmentKey = serviceKey<SkyEnvironmentProvider>('render.sky');

/**
 * What this module can consume from a screen-space occlusion system.
 *
 * An SSAO/GTAO module may register an object of this shape under
 * {@link AmbientOcclusionKey}. `occlusionNode` must be a TSL float in `[0, 1]`
 * where 1 means unoccluded. When present it multiplies the material AO before
 * the specular-occlusion term is derived, so screen-space occlusion darkens
 * glossy reflections as well as ambient diffuse.
 */
export interface AmbientOcclusionProvider {
  readonly occlusionNode: THREE.Node<'float'> | null;
}

/** Service key an SSAO/GTAO module should register itself under. */
export const AmbientOcclusionKey = serviceKey<AmbientOcclusionProvider>('render.ao');

/* ------------------------------------------------------------------------- *
 * Public service
 * ------------------------------------------------------------------------- */

export interface IBLOptions {
  /**
   * Multiplier on all image-based lighting. Written to
   * `scene.environmentIntensity`, which three folds into both the diffuse and
   * specular IBL terms. Default 1.
   */
  intensity?: number;
  /** Y-rotation of the environment, in radians. Default 0. */
  rotationY?: number;
  /**
   * Strength of the horizon-occlusion fade in `[0, 1]`. 0 disables it.
   * Default 1.
   */
  horizonOcclusion?: number;
  /**
   * AssetManager key used when no sky service is registered.
   * Default `'env.overcast'` — the cold grey key this game is authored for.
   */
  fallbackEnvironmentKey?: string;
}

export interface IBLStats {
  /** Which source the current environment came from. */
  readonly source: 'sky' | 'asset' | 'scene' | 'none';
  /** `scene.environmentIntensity`. */
  readonly intensity: number;
  /** Whether an AO service was found and is being composed in. */
  readonly ambientOcclusionService: boolean;
  /** Number of prefiltered roughness levels the PMREM chain provides. */
  readonly prefilteredMipCount: number;
}

export interface IBLService {
  /**
   * Install `texture` as the scene's image-based light source.
   *
   * Passing an equirectangular or cube HDR texture is enough; three prefilters
   * it into a PMREM chain lazily on first use and caches it per renderer, so
   * calling this with the same texture twice is free.
   *
   * Passing `null` removes image-based lighting entirely (useful for interiors
   * lit purely by local lights).
   */
  setEnvironment(texture: THREE.Texture | null): void;

  /** The texture currently driving IBL, if any. */
  readonly environment: THREE.Texture | null;

  /** Multiplier on both IBL lobes. Written to `scene.environmentIntensity`. */
  setIntensity(intensity: number): void;

  /** Y-rotation of the environment, in radians. */
  setRotationY(radians: number): void;

  /**
   * Compose this module's occlusion terms onto a material.
   *
   * Sets `material.aoNode` to `existingAO · serviceAO · horizonOcclusion`.
   * three's `PhysicalLightingModel` then multiplies ambient diffuse by that
   * value and ambient specular by `computeSpecularOcclusion` of it, which is
   * the term that stops cavities glowing.
   *
   * Safe to call more than once on the same material: the previous `aoNode` is
   * folded in rather than replaced.
   */
  applyOcclusion(material: THREE.NodeMaterial): void;

  /**
   * The occlusion node on its own, for callers that want to compose it
   * themselves (for example into a material that already builds a complex AO
   * expression).
   */
  occlusionNode(baseAO?: THREE.Node<'float'>): THREE.Node<'float'>;

  /**
   * A standalone prefiltered-radiance lookup, for effects that need to sample
   * the environment outside the standard PBR path — water, glass, a rim term on
   * a character shader.
   *
   * @param direction world-space direction to sample.
   * @param roughness perceptual roughness in `[0, 1]`; selects the PMREM mip.
   */
  prefilteredRadiance(
    direction: THREE.Node<'vec3'>,
    roughness: THREE.Node<'float'>,
  ): THREE.Node<'vec3'>;

  readonly stats: IBLStats;
}

/** Service key other modules resolve this module by. */
export const IBLKey = serviceKey<IBLService>('render.ibl');

/* ------------------------------------------------------------------------- *
 * Pure occlusion math (unit-tested in tests/ibl.occlusion.test.ts)
 * ------------------------------------------------------------------------- */

/**
 * Lagarde & de Rousiers specular occlusion, Frostbite PBR course notes (2014),
 * listing 26.
 *
 * ```
 * saturate( pow(N·V + AO, exp2(-16·roughness - 1)) - 1 + AO )
 * ```
 *
 * The intuition: a smooth surface's specular lobe is narrow, so a diffuse
 * occlusion value computed over the whole hemisphere massively over-darkens it;
 * the roughness-driven exponent walks the term from "barely occluded" at
 * roughness 0 to "as occluded as the diffuse" at roughness 1. Grazing angles
 * (small `N·V`) see more of the occluding geometry, so they darken faster.
 *
 * This is the same expression three.js's `PhysicalLightingModel` applies
 * internally; it is duplicated here in plain JS so the behaviour is pinned by a
 * unit test and so tools can reason about it outside a shader.
 */
export function computeSpecularOcclusion(
  nDotV: number,
  ambientOcclusion: number,
  roughness: number,
): number {
  const exponent = Math.pow(2, -16 * roughness - 1);
  const value = Math.pow(Math.max(0, nDotV + ambientOcclusion), exponent) - 1 + ambientOcclusion;
  return Math.min(1, Math.max(0, value));
}

/**
 * Horizon occlusion (Lagarde, "Image-based Lighting approaches and
 * parallax-corrected cubemap", SIGGRAPH 2012 talk; restated in the Frostbite
 * notes, §4.10.3).
 *
 * When a normal map tilts the shading normal enough that the mirror reflection
 * vector `R` dips below the *geometric* surface, that reflection would have to
 * pass through the object. `saturate(1 + fade · R·Ng)²` fades it out smoothly.
 * Squaring makes the falloff perceptually gentle rather than a hard edge.
 *
 * @param rDotNg dot product of the reflection vector with the geometric normal.
 * @param fade   strength in `[0, 1]`; 0 disables the term.
 */
export function horizonOcclusion(rDotNg: number, fade = 1): number {
  const horizon = Math.min(1, Math.max(0, 1 + fade * rDotNg));
  return horizon * horizon;
}

/**
 * Mip level of the prefiltered radiance chain for a given perceptual roughness.
 *
 * three's PMREM chain is authored so that level `i` corresponds to roughness
 * `i / (levels - 1)`; the mapping is linear in *perceptual* roughness, which is
 * why materials store `roughness` and not `alpha = roughness²`.
 *
 * Exposed mainly so tools and tests can reason about the chain; the shader path
 * lets `PMREMNode` do the lookup itself.
 */
export function roughnessToMip(roughness: number, mipCount: number): number {
  const levels = Math.max(1, mipCount);
  return Math.min(levels - 1, Math.max(0, roughness)) * (levels - 1);
}

/**
 * Number of usable prefiltered roughness levels for a PMREM chain built from a
 * source of the given size. three's `PMREMGenerator` builds down to a 16x16
 * base face plus a fixed extra tail of blurred levels.
 */
export function prefilteredMipCount(sourceSize: number): number {
  const size = Math.max(16, sourceSize);
  return Math.max(1, Math.floor(Math.log2(size)) - 3);
}

/* ------------------------------------------------------------------------- *
 * TSL nodes
 * ------------------------------------------------------------------------- */

/**
 * TSL form of {@link horizonOcclusion}.
 *
 * Uses the *geometric* normal deliberately: the whole point of the term is to
 * catch cases where the shading normal has been bent away from the geometry.
 */
const horizonOcclusionNode = Fn(([fade]: [THREE.Node<'float'>]) => {
  // Reflection about the shading normal, in view space, then compared against
  // the geometric normal. Both are unit vectors, so the dot is a cosine.
  const reflectVec = reflect(positionViewDirection.negate(), normalView);
  const horizon = clamp(
    float(1).add(fade.mul(dot(reflectVec, normalWorldGeometry))),
    0,
    1,
  );
  return horizon.mul(horizon);
});

/* ------------------------------------------------------------------------- *
 * Module
 * ------------------------------------------------------------------------- */

/** `Scene.environmentNode`, absent from the `Scene` class in @types/three r185. */
interface SceneWithEnvironmentNode {
  environmentNode: THREE.Node | null;
  environmentIntensity: number;
  environmentRotation: THREE.Euler;
}

/** `NodeMaterial.aoNode`, likewise absent from the public class declaration. */
interface MaterialWithAO {
  aoNode: THREE.Node<'float'> | null;
  aoMap: THREE.Texture | null;
}

/** Minimal AssetManager surface this module needs. */
interface EnvironmentSource {
  has(key: string): boolean;
  peek(key: string): unknown;
  loadEnvironment(key: string): Promise<THREE.DataTexture>;
}

const DEFAULTS: Required<IBLOptions> = {
  intensity: 1,
  rotationY: 0,
  horizonOcclusion: 1,
  fallbackEnvironmentKey: 'env.overcast',
};

/**
 * Registers the {@link IBLService} and keeps `scene.environment` in sync with
 * whatever the sky module is currently producing.
 *
 * Register it *after* the AssetManager and, ideally, after the sky module; it
 * re-resolves both every frame, so ordering only affects the first frame.
 */
export class IBLModule implements GameModule {
  readonly name = 'IBL';

  readonly #options: Required<IBLOptions>;

  #scene: THREE.Scene | null = null;
  #services: ServiceLocator | null = null;
  #environment: THREE.Texture | null = null;
  #source: IBLStats['source'] = 'none';
  #lastSkyVersion = -1;
  #lastSkyTexture: THREE.Texture | null = null;
  #fallbackRequested = false;
  #mipCount = 1;

  readonly #rotation = new THREE.Euler(0, 0, 0);

  constructor(options: IBLOptions = {}) {
    this.#options = { ...DEFAULTS, ...options };
  }

  init(ctx: GameContext): void {
    this.#scene = ctx.scene;
    this.#services = ctx.services;

    this.#applyIntensity(this.#options.intensity);
    this.#applyRotationY(this.#options.rotationY);

    // Adopt whatever is already on the scene so that a sky module which ran
    // first is not clobbered, and so the reference scene keeps working if this
    // module is added to an already-lit scene.
    const existing = ctx.scene.environment;
    if (existing !== null) {
      this.#environment = existing;
      this.#source = 'scene';
      this.#mipCount = prefilteredMipCount(textureSize(existing));
    }

    this.#service = this.#createService();
    ctx.services.register(IBLKey, this.#service);
  }

  /**
   * Runs in `lateUpdate` so that a sky module which regenerates its texture in
   * `update` is observed in the same frame rather than one frame late — a
   * one-frame lag is invisible for a slow time-of-day cycle but very visible
   * for a lightning flash.
   */
  lateUpdate(): void {
    this.#syncEnvironment();
  }

  dispose(): void {
    this.#services?.unregister(IBLKey);
    this.#scene = null;
    this.#services = null;
  }

  // -- environment plumbing ----------------------------------------------

  #syncEnvironment(): void {
    const services = this.#services;
    const scene = this.#scene;
    if (services === null || scene === null) return;

    const sky = services.tryGet(SkyEnvironmentKey);
    if (sky !== undefined && sky.environmentTexture !== null) {
      const version = sky.environmentVersion ?? 0;
      if (sky.environmentTexture !== this.#lastSkyTexture || version !== this.#lastSkyVersion) {
        this.#lastSkyTexture = sky.environmentTexture;
        this.#lastSkyVersion = version;
        this.#install(sky.environmentTexture, 'sky');
      }
      if (sky.environmentRotationY !== undefined) this.#applyRotationY(sky.environmentRotationY);
      return;
    }

    if (this.#environment !== null) return;

    // No sky service and nothing already installed: fall back to a shipped HDRI
    // so the world is never lit by nothing at all.
    const assets = services.tryGet(AssetManagerKey) as unknown as EnvironmentSource | undefined;
    if (assets === undefined) return;

    const key = this.#options.fallbackEnvironmentKey;
    if (!assets.has(key)) return;

    const cached = assets.peek(key);
    if (cached instanceof THREE.Texture) {
      this.#install(cached, 'asset');
      return;
    }

    if (this.#fallbackRequested) return;
    this.#fallbackRequested = true;
    assets
      .loadEnvironment(key)
      .then((texture) => {
        // A sky service may have appeared while the HDRI was in flight.
        if (this.#source === 'sky') return;
        this.#install(texture, 'asset');
      })
      .catch((error: unknown) => {
        console.warn(`[IBL] fallback environment "${key}" failed to load:`, error);
      });
  }

  #install(texture: THREE.Texture | null, source: IBLStats['source']): void {
    const scene = this.#scene;
    if (scene === null) return;

    this.#environment = texture;
    this.#source = texture === null ? 'none' : source;
    this.#mipCount = texture === null ? 1 : prefilteredMipCount(textureSize(texture));

    // `scene.environment` is the supported hook: three lazily prefilters it
    // through `PMREMNode`, caches the result per renderer, and `EnvironmentNode`
    // samples it with the roughness-driven LOD that the split-sum prefiltered
    // term requires. Assigning the raw texture rather than a pre-built PMREM
    // render target also means the chain is rebuilt automatically if the sky
    // texture is replaced.
    scene.environment = texture;

    // Clear any node override a previous install may have left, so the two
    // paths cannot disagree about which environment is live.
    (scene as unknown as SceneWithEnvironmentNode).environmentNode = null;
  }

  #applyRotationY(radians: number): void {
    const scene = this.#scene;
    if (scene === null) return;
    this.#options.rotationY = radians;
    this.#rotation.set(0, radians, 0);
    (scene as unknown as SceneWithEnvironmentNode).environmentRotation = this.#rotation;
  }

  #applyIntensity(intensity: number): void {
    const scene = this.#scene;
    if (scene === null) return;
    const clamped = Math.max(0, intensity);
    this.#options.intensity = clamped;
    (scene as unknown as SceneWithEnvironmentNode).environmentIntensity = clamped;
  }

  // -- service -----------------------------------------------------------

  #service: IBLService | null = null;

  /**
   * Build the service object.
   *
   * Written as a factory over a captured `self` rather than as a class-field
   * object literal because the getters need access to the module's private
   * fields, and `this` inside an object literal is the literal, not the class.
   */
  #createService(): IBLService {
    // eslint-disable-next-line @typescript-eslint/no-this-alias
    const self = this;

    return {
      setEnvironment(texture: THREE.Texture | null): void {
        self.#install(texture, texture === null ? 'none' : 'scene');
      },

      get environment(): THREE.Texture | null {
        return self.#environment;
      },

      setIntensity(intensity: number): void {
        self.#applyIntensity(intensity);
      },

      setRotationY(radians: number): void {
        self.#applyRotationY(radians);
      },

      applyOcclusion(material: THREE.NodeMaterial): void {
        const target = material as unknown as MaterialWithAO;
        const existing = target.aoNode;
        target.aoNode = this.occlusionNode(existing ?? undefined);
      },

      occlusionNode(baseAO?: THREE.Node<'float'>): THREE.Node<'float'> {
        // Start from whatever the material already had, else its AO map, else 1.
        let ao: THREE.Node<'float'> = baseAO ?? (materialAO as THREE.Node<'float'>);

        const provider = self.#services?.tryGet(AmbientOcclusionKey);
        if (provider !== undefined && provider.occlusionNode !== null) {
          ao = ao.mul(provider.occlusionNode);
        }

        const fade = self.#options.horizonOcclusion;
        if (fade > 0) ao = ao.mul(horizonOcclusionNode(float(fade)));

        return clamp(ao, 0, 1);
      },

      prefilteredRadiance(
        direction: THREE.Node<'vec3'>,
        roughness: THREE.Node<'float'>,
      ): THREE.Node<'vec3'> {
        const environment = self.#environment;
        if (environment === null) return vec3(0, 0, 0);
        // `PMREMNode` reads the sample direction and LOD from the surrounding
        // node context, which is how `EnvironmentNode` drives it. Supplying them
        // explicitly gives callers a direct lookup with the same prefiltering.
        return pmremTexture(environment, direction, roughness) as unknown as THREE.Node<'vec3'>;
      },

      get stats(): IBLStats {
        return {
          source: self.#source,
          intensity: self.#options.intensity,
          ambientOcclusionService: self.#services?.has(AmbientOcclusionKey) ?? false,
          prefilteredMipCount: self.#mipCount,
        };
      },
    };
  }
}

/** Longest edge of a texture's image, in texels; 0 when it cannot be determined. */
function textureSize(texture: THREE.Texture): number {
  const image = texture.image as { width?: number; height?: number } | null | undefined;
  if (image === null || image === undefined) return 0;
  return Math.max(image.width ?? 0, image.height ?? 0);
}

/**
 * Convenience entry point mirroring the rest of `src/render`.
 *
 * Prefer registering {@link IBLModule} with the engine so it participates in the
 * normal lifecycle; this exists for tools and tests that build a context by
 * hand.
 */
export function registerIBL(ctx: GameContext, options: IBLOptions = {}): IBLModule {
  const module = new IBLModule(options);
  ctx.engine.add(module);
  return module;
}

// `exp2` is imported for the documented TSL form of specular occlusion below;
// three applies its own copy internally, so the node is exported rather than
// installed, for materials that need the term outside the standard PBR path.
/**
 * TSL form of {@link computeSpecularOcclusion}.
 *
 * three's `PhysicalLightingModel` applies this automatically to indirect
 * specular once a material has an `aoNode`, so most call sites want
 * {@link IBLService.applyOcclusion} instead. This export exists for custom
 * lighting models (water, eyes, hair) that bypass `PhysicalLightingModel`.
 */
export const specularOcclusionNode = Fn(
  ([nDotV, ambientOcclusion, roughness]: [
    THREE.Node<'float'>,
    THREE.Node<'float'>,
    THREE.Node<'float'>,
  ]) => {
    const exponent = exp2(roughness.mul(-16).sub(1));
    return clamp(
      nDotV.add(ambientOcclusion).max(0).pow(exponent).sub(1).add(ambientOcclusion),
      0,
      1,
    );
  },
);
