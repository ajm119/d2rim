/**
 * @module render/FrameGraph
 *
 * One renderer, assembled from six independently-authored subsystems.
 *
 * Each subsystem was written in isolation against interfaces it *declared* and
 * expected somebody else to satisfy. Several of those declarations collide on
 * the same `ServiceLocator` id with different shapes, several were never
 * registered by anyone, and two subsystems model the same physical phenomenon.
 * This module is where those seams are closed, and it is deliberately the only
 * place in the project that knows about all of them at once.
 *
 * ---
 *
 * ## Frame order
 *
 * `Engine` runs `fixedUpdate*` → `update` → `lateUpdate` → `renderer.render`,
 * each phase in module registration order, and `PostStack` wraps
 * `renderer.render`. So the ordered graph below is produced by (a) the order
 * modules are added here and (b) where each one does its work.
 *
 * ```
 *  update      ── camera / animation settle
 *              ── TimeOfDay      sun + moon ephemeris
 *              ── Sky            sky-view LUT, environment probe, scene.fogNode
 *
 *  lateUpdate  ── Lighting       key light from the sky, CSM cascade fit
 *              ── GuideBuffer    DEPTH + NORMAL PREPASS  (shared, one draw)
 *              ── GTAO           horizon-search AO  → 'render.ao'
 *              ── IBL            environment → scene.environment, AO folded in
 *              ── SSR            hi-Z trace against LAST frame's colour
 *              ── Volumetrics    froxel scatter + transmittance integration
 *              ── LightShafts    low-res radial shafts buffer
 *
 *  render      ── PostStack.render
 *                 ├─ MOTION      jitter the projection, bind velocity MRT
 *                 ├─ SHADOWS     CSM cascade passes (three, during scene draw)
 *                 ├─ OPAQUE      scene draw → HDR colour + velocity + depth
 *                 ├─ TRANSPARENT (same draw; sorted after opaque by three)
 *                 ├─ post.ssr        composite reflection delta      [HDR]
 *                 ├─ lightshafts     composite volumetrics + shafts  [HDR]
 *                 ├─ post.taa        temporal resolve                [HDR]
 *                 ├─ post.bloom      producer, writes its own pyramid
 *                 ├─ post.composite  EXPOSURE → AgX → GRADE          [HDR→LDR]
 *                 └─ post.fxaa       (FXAA tiers only)               [LDR]
 *
 *  DOM         ── DebugOverlay   UI, composited by the browser above the canvas
 * ```
 *
 * ## Seams, and how each is resolved
 *
 * **Two fog models.** `Sky`/`Atmosphere` install `scene.fogNode` — real
 * multiple-scattering aerial perspective, correct at kilometres. `Volumetrics`
 * integrates a froxel volume — correct at metres, and the only one of the two
 * that can do god rays, shadowed in-scatter or a local fog volume. Naively
 * enabling both double-counts every metre of air in front of the camera.
 * Resolution: they are given disjoint domains. The froxel volume owns
 * `[near, volumeDistance]` and the aerial perspective owns everything beyond,
 * with the *mist* term (the near-ground component) zeroed out of the
 * atmosphere's aerial settings — see {@link handoffDistance}. There is still
 * exactly one atmosphere: `AtmosphereService` is the single source of the
 * ambient in-scatter colour the froxel integrator uses, so the fog in the
 * hollows and the haze on the horizon are the same air.
 *
 * **Two exposure authorities.** `RendererFactory` sets
 * `renderer.toneMapping = ACESFilmic` with `toneMappingExposure`, and
 * `CompositePass` applies its own exposure + AgX curve. `PostStack` forces
 * `NoToneMapping` while installed, which makes the renderer's setting dead
 * code, but "dead by side effect" is not a contract. Resolution: the composite
 * pass is the sole authority and this module asserts it at boot, logging if
 * the renderer is left in a tone-mapping state.
 *
 * **Three noise sources.** `Motion` owns the Halton sequence for TAA jitter,
 * `Volumetrics` re-declared `radicalInverse`/`haltonJitter3D` for its temporal
 * froxel offset, and `Procedural` has the seeded value/simplex noise the
 * materials and scatter use. These are not interchangeable — a 2D sub-pixel
 * sequence, a 3D froxel offset and a spatial hash solve different problems —
 * but they must not *correlate*, or TAA will resolve the fog dither into a
 * standing pattern. Resolution: `Motion` is the only consumer of base (2,3),
 * `Volumetrics` is seeded onto base (2,3,5) with an offset, and
 * {@link SCATTER_SEED} namespaces every world-space `Procedural` draw. One
 * generator per domain, no shared state, no accidental phase lock.
 *
 * **Colliding service ids.** Four ids are declared by more than one module:
 *
 * | id | declared by | registered by | consumed by |
 * |---|---|---|---|
 * | `render.sky` | `Sky`, `IBL` | `Sky` | `IBL` (structurally compatible) |
 * | `render.sky.sun` | `Sky`, `Lighting`, `Volumetrics` | `Sky` | both |
 * | `lighting.celestial` | `Sky`, `Lighting` | `Lighting` | `Sky` (push) |
 * | `render.ao` | `IBL`, `GTAO` | `GTAO` (`publishToIBL`) | `IBL` |
 *
 * All four are structurally compatible, which is not luck — each pair was
 * authored as a deliberate mirror. What was missing is that nothing enforced
 * it, so {@link auditServices} checks every one at boot and reports what is
 * connected, what degraded, and to what.
 *
 * **Interfaces nobody implemented.** `render.sceneColor`,
 * `render.gbuffer.surface`, `render.shadow.volumetric`,
 * `render.volumetrics.lights`, `render.volumetrics.ambient`, `render.quality`
 * and `world.weather` were all declared and never registered, so seven code
 * paths were silently taking a fallback. {@link RenderBridges} implements six
 * of them; `RenderSettings` implements the last two. The one that remains
 * genuinely unsatisfied is `render.gbuffer.surface` — see its note below.
 */

import * as THREE from 'three/webgpu';
import { texture as textureNodeFactory } from 'three/tsl';

import type { GameContext, GameModule } from '../core/types';

import { AtmosphereKey, type AtmosphereService } from './Atmosphere';
import type { CascadedShadowMapNode } from './CascadedShadowMaps';
import { IBLModule, IBLKey, type IBLService } from './IBL';
import { LightingModule, LightingKey, lightImportance, type LightingService } from './Lighting';
import { MaterialLibrary, MaterialLibraryKey } from './MaterialLibrary';
import { RenderSettings, type RenderQuality, type RenderTier } from './RenderSettings';
import { Sky, SkyKey } from './Sky';
import { TimeOfDay, TimeOfDayKey } from './TimeOfDay';
import {
  VolumetricsModule,
  VolumetricsKey,
  VolumetricAmbientKey,
  VolumetricLightsKey,
  VolumetricShadowKey,
  VolumetricSunKey,
  type VolumetricAmbientProvider,
  type VolumetricLightProvider,
  type VolumetricPointLight,
  type VolumetricShadowProvider,
} from './Volumetrics';
import { RenderQualityKey, WeatherStateKey } from './materials/types';
import { GTAOModule, GTAOKey, AMBIENT_OCCLUSION_SERVICE_ID } from './post/GTAO';
import { LightShaftsModule, LightShaftsKey } from './post/LightShafts';
import { MotionVectorsKey } from './post/Motion';
import {
  PostStack,
  PostStackKey,
  type PostCapabilities,
  type PostFrame,
  type PostPass,
  type QualityTier,
} from './post/PostStack';
import {
  SSRModule,
  SSRKey,
  SCENE_COLOR_SERVICE_ID,
  SURFACE_PARAMETERS_SERVICE_ID,
  type SceneColorProvider,
  type SSRService,
  type SurfaceParameterProvider,
} from './post/SSR';

/**
 * Where the froxel volume stops and aerial perspective takes over, in metres.
 *
 * Chosen, not tuned: at 96 m the froxel grid's furthest slice is ~4 m deep at
 * the default 64-slice exponential distribution, which is still fine enough
 * that a shaft edge does not stair-step, and it is comfortably past the far
 * treeline in the Blood Moor composition so the handoff never lands on a
 * silhouette. Pushing it further buys nothing — beyond ~100 m single-scatter
 * froxel fog and the atmosphere's aerial perspective agree to within a
 * rounding error anyway, and the froxel version costs a 3D texture.
 */
export const handoffDistance = 96;

/** Namespace for every world-space procedural draw, so scatters never phase-lock. */
export const SCATTER_SEED = 'd2rim.bloodMoor';

/* -------------------------------------------------------------------------- */
/* SSR composite                                                              */
/* -------------------------------------------------------------------------- */

function makeSourceNode(name: string): THREE.TextureNode {
  const placeholder = new THREE.DataTexture(new Uint8Array([0, 0, 0, 255]), 1, 1);
  placeholder.needsUpdate = true;
  const node = textureNodeFactory(placeholder);
  node.name = name;
  return node;
}

/**
 * Adds the screen-space reflection delta back into the HDR chain.
 *
 * `SSRModule` traces and denoises but deliberately does not composite: it
 * exposes `composite(sceneColour)` and leaves placement to whoever owns the
 * frame, because the same delta is wanted in a different slot depending on
 * whether the renderer is forward or deferred. This is that slot — before TAA,
 * so reflections are temporally filtered with everything else rather than
 * shimmering on top of a resolved image.
 *
 * The pass owns a stable texture node and rebinds its `.value` per frame. That
 * matters: `frame.input` is a pool buffer whose *identity* alternates between
 * frames, and a TSL graph binds texture objects, not slots. Rebuilding the
 * graph every frame would recompile a shader every frame.
 */
class SSRCompositePass implements PostPass {
  readonly id = 'post.ssr';
  readonly kind = 'chain' as const;
  readonly outputDomain = 'hdr' as const;
  enabled = true;

  readonly #ssr: SSRService;
  readonly #source = makeSourceNode('ssrSceneColor');
  readonly #copySource = makeSourceNode('ssrCopySource');
  #material: THREE.NodeMaterial | null = null;
  #copyMaterial: THREE.NodeMaterial | null = null;
  #boundReflection: THREE.Texture | null = null;

  constructor(ssr: SSRService) {
    this.#ssr = ssr;
  }

  isAvailable(_quality: QualityTier, _capabilities: PostCapabilities): boolean {
    return this.#ssr.quality !== 'off';
  }

  configure(_quality: QualityTier, _capabilities: PostCapabilities): void {
    // The composite graph is supplied by SSRModule and does not vary by tier;
    // the tier changes trace cost, which SSRModule applies itself.
  }

  setSize(_width: number, _height: number): void {
    // Full-screen and resolution-independent: the reflection buffer is sized
    // by SSRModule against its own resolutionScale.
  }

  render(frame: PostFrame): THREE.Texture | null | void {
    // Close the scene-colour loop for the *next* frame's trace. SSR runs in
    // lateUpdate, before the scene draw, so this is the freshest fully-lit
    // radiance it can possibly see.
    this.#ssr.setSceneColor(frame.input, true);

    const reflection = this.#ssr.reflectionTexture;
    if (reflection === null) {
      this.#copySource.value = frame.input;
      frame.blit(this.#ensureCopyMaterial(), frame.output, 'post.ssr.passthrough');
      return null;
    }

    if (this.#material === null || this.#boundReflection !== reflection) {
      this.#material?.dispose();
      const material = new THREE.NodeMaterial();
      material.name = 'post.ssr';
      material.fragmentNode = this.#ssr.composite(
        this.#source as unknown as THREE.Node<'vec4'>,
      );
      material.depthTest = false;
      material.depthWrite = false;
      material.transparent = false;
      this.#material = material;
      this.#boundReflection = reflection;
    }

    this.#source.value = frame.input;
    frame.blit(this.#material, frame.output, 'post.ssr');
    return null;
  }

  #ensureCopyMaterial(): THREE.NodeMaterial {
    if (this.#copyMaterial === null) {
      const material = new THREE.NodeMaterial();
      material.name = 'post.ssr.passthrough';
      material.fragmentNode = this.#copySource as unknown as THREE.Node<'vec4'>;
      material.depthTest = false;
      material.depthWrite = false;
      material.transparent = false;
      this.#copyMaterial = material;
    }
    return this.#copyMaterial;
  }

  dispose(): void {
    this.#material?.dispose();
    this.#material = null;
    this.#copyMaterial?.dispose();
    this.#copyMaterial = null;
  }
}

/* -------------------------------------------------------------------------- */
/* Bridges                                                                    */
/* -------------------------------------------------------------------------- */

/**
 * Implements the six service contracts that were declared but never provided,
 * and wires the two passes that own no module.
 *
 * Everything here is a *live view* rather than a snapshot: `cascadeMatrices`
 * is rewritten in place by the CSM node every frame, and the point-light set
 * changes as the scene streams. Registering getters instead of values is what
 * lets a consumer hold the service object for the lifetime of the run.
 *
 * Ordering: this module is registered after every producer it reads from and
 * before every consumer that resolves at `init`, which is the entire reason
 * the frame graph is built as an explicit ordered list rather than by letting
 * each subsystem add itself.
 */
class RenderBridges implements GameModule {
  readonly name = 'render.bridges';

  readonly #post: PostStack;
  readonly #ssr: SSRModule;
  readonly #lightShafts: LightShaftsModule;
  readonly #tier: RenderTier;

  #lighting: LightingService | null = null;
  #atmosphere: AtmosphereService | null = null;
  #ssrPass: SSRCompositePass | null = null;
  #installedShafts = false;

  /** Reused scratch so `collectVolumetricLights` allocates nothing per frame. */
  readonly #lightScratch: THREE.PointLight[] = [];
  readonly #ambient = new THREE.Color();
  readonly #groundAmbient = new THREE.Color();

  constructor(parts: {
    post: PostStack;
    ssr: SSRModule;
    lightShafts: LightShaftsModule;
    tier: RenderTier;
  }) {
    this.#post = parts.post;
    this.#ssr = parts.ssr;
    this.#lightShafts = parts.lightShafts;
    this.#tier = parts.tier;
  }

  init(ctx: GameContext): void {
    this.#lighting = ctx.services.tryGet(LightingKey) ?? null;
    this.#atmosphere = ctx.services.tryGet(AtmosphereKey) ?? null;

    this.#registerSceneColor(ctx);
    this.#registerSurfaceParameters(ctx);
    this.#registerVolumetricShadows(ctx);
    this.#registerVolumetricLights(ctx);
    this.#registerVolumetricAmbient(ctx);
    this.#installPasses();
    this.#enforceSingleExposureAuthority(ctx);
  }

  /**
   * `render.sceneColor` — last frame's HDR radiance, straight off the scene
   * attachment. `isPreviousFrame` is true and that is not a compromise: a
   * same-frame reflection source does not exist in a forward renderer.
   */
  #registerSceneColor(ctx: GameContext): void {
    const post = this.#post;
    const provider: SceneColorProvider = {
      get sceneColorTexture(): THREE.Texture | null {
        return post.sceneColorTexture;
      },
      get isPreviousFrame(): boolean {
        return true;
      },
    };
    ctx.services.register<SceneColorProvider>(SCENE_COLOR_SERVICE_ID, provider);
  }

  /**
   * `render.gbuffer.surface` — **deliberately not satisfied.**
   *
   * SSR wants per-pixel roughness and metalness. This renderer is forward: the
   * only prepass it runs is the shared depth+normal guide buffer, and widening
   * that to a third attachment costs a full-resolution RGBA16F target plus the
   * bandwidth to write it, on every frame, for one effect. Fabricating a fake
   * one would be worse than not having it — SSR would trace confidently
   * against numbers nobody computed.
   *
   * Instead SSR runs on its scalar defaults, which are set from the scene's
   * dominant surface (wet mud and wet masonry: dielectric, roughness ~0.34)
   * rather than from three.js's neutral 0.5/0. The registration below records
   * the absence explicitly so `auditServices` reports it as degraded instead
   * of as missing, and phase 3 can drop in a real provider with no other
   * change.
   */
  #registerSurfaceParameters(ctx: GameContext): void {
    const provider: SurfaceParameterProvider = {
      get surfaceTexture(): THREE.Texture | null {
        return null;
      },
      get surfaceVersion(): number {
        return 0;
      },
    };
    ctx.services.register<SurfaceParameterProvider>(SURFACE_PARAMETERS_SERVICE_ID, provider);
    console.warn(
      '[FrameGraph] no G-buffer surface parameters: this is a forward renderer. ' +
        'SSR falls back to scalar roughness/metalness defaults tuned for wet ' +
        'dielectric ground. Reflections on metal props will be approximate.',
    );
  }

  /**
   * `render.shadow.volumetric` — the cascades, for shadowed in-scatter.
   *
   * `CascadedShadowMapNode` is a `ShadowBaseNode`: its shadow test only ever
   * runs inside a surface material's setup. The froxel integrator needs sun
   * visibility at arbitrary points *in the air*, which never reaches a
   * surface, so it samples the cascade atlas directly. This is the adapter
   * that hands it the pieces.
   */
  #registerVolumetricShadows(ctx: GameContext): void {
    const lighting = this.#lighting;
    if (lighting === null) {
      console.warn(
        '[FrameGraph] no LightingService: volumetric fog will not be shadowed, ' +
          'so god rays will read as uniform haze.',
      );
      return;
    }
    const provider: VolumetricShadowProvider = {
      get cascadeCount(): number {
        return lighting.sunShadows?.cascadeCount ?? 0;
      },
      get shadowDepthTexture(): THREE.Texture | null {
        return lighting.sunShadows?.shadowDepthTexture ?? null;
      },
      get cascadeMatrices(): readonly THREE.Matrix4[] {
        return lighting.sunShadows?.cascadeMatrices ?? EMPTY_MATRICES;
      },
      get shadowMapSize(): number {
        return lighting.sunShadows?.shadowMapSize ?? 0;
      },
      get shadowDistance(): number {
        return lighting.sunShadows?.shadowDistance ?? 0;
      },
      get reversedDepth(): boolean {
        return false;
      },
    };
    ctx.services.register<VolumetricShadowProvider>(VolumetricShadowKey, provider);
  }

  /**
   * `render.volumetrics.lights` — which local lights get volumetric scatter.
   *
   * `LightingService` has no enumeration API (it hands out opaque
   * `LightHandle`s), so the set is gathered from the scene graph. That is the
   * more robust source anyway: a light added directly to the scene by content
   * — the campfire — is picked up without having to route it through the
   * clustered rig first.
   *
   * The budget is small on purpose. Every scattering light is a full extra
   * evaluation per froxel, which at the `high` tier is 128×72×64 ≈ 590k
   * invocations. Ranking by {@link lightImportance} keeps the campfire in and
   * distant torches out.
   */
  #registerVolumetricLights(ctx: GameContext): void {
    const scene = ctx.scene;
    const camera = ctx.camera;
    const scratch = this.#lightScratch;

    const provider: VolumetricLightProvider = {
      collectVolumetricLights(out: VolumetricPointLight[], max: number): number {
        scratch.length = 0;
        scene.traverseVisible((object) => {
          if (object instanceof THREE.PointLight && object.intensity > 0) {
            scratch.push(object);
          }
        });
        if (scratch.length > max) {
          const eye = camera.position;
          scratch.sort(
            (a, b) =>
              lightImportance(b.intensity, b.distance, b.position.distanceTo(eye)) -
              lightImportance(a.intensity, a.distance, a.position.distanceTo(eye)),
          );
        }
        const count = Math.min(scratch.length, max);
        for (let i = 0; i < count; i++) {
          const light = scratch[i];
          if (light === undefined) continue;
          out[i] = {
            position: light.position,
            color: light.color,
            intensity: light.intensity,
            // `distance: 0` means "infinite" in three; the froxel integrator
            // needs a finite radius to bound its influence, so fall back to a
            // torch-sized 12 m rather than lighting the whole volume.
            radius: light.distance > 0 ? light.distance : 12,
            volumetricScale: 1,
          };
        }
        return count;
      },
    };
    ctx.services.register<VolumetricLightProvider>(VolumetricLightsKey, provider);
  }

  /**
   * `render.volumetrics.ambient` — and the enforcement point for
   * "exactly one atmosphere model".
   *
   * The froxel integrator needs the ambient radiance scattering into every
   * froxel from the sky. Reading it from {@link AtmosphereService.aerial}
   * rather than from a second sky model is what makes the fog in the hollows
   * and the haze on the horizon provably the same air: both terms come from
   * the same multiple-scattering LUT, updated once per frame by `Sky`.
   */
  #registerVolumetricAmbient(ctx: GameContext): void {
    const atmosphere = this.#atmosphere;
    if (atmosphere === null) {
      console.warn(
        '[FrameGraph] no AtmosphereService: volumetric ambient falls back to a ' +
          'constant, so fog colour will not track the sky.',
      );
      return;
    }
    const ambient = this.#ambient;
    const ground = this.#groundAmbient;
    const provider: VolumetricAmbientProvider = {
      get volumetricAmbient(): THREE.Color {
        const radiance = atmosphere.aerial.ambientRadiance;
        return ambient.setRGB(radiance.x, radiance.y, radiance.z);
      },
      get volumetricGroundAmbient(): THREE.Color {
        // The moor is wet mud and dead grass: a dark, slightly warm bounce,
        // scaled off the same sky term so it tracks the time of day.
        const radiance = atmosphere.aerial.ambientRadiance;
        return ground
          .setRGB(radiance.x, radiance.y, radiance.z)
          .multiply(GROUND_BOUNCE_ALBEDO);
      },
    };
    ctx.services.register<VolumetricAmbientProvider>(VolumetricAmbientKey, provider);
  }

  /**
   * Insert the two chain passes whose modules own no slot in `PostStack`.
   *
   * Order inside the HDR half of the chain is `ssr → lightshafts → taa`.
   * Reflections go in before the fog composite so that a reflected sky is
   * itself fogged; both go in before TAA so the whole image is temporally
   * filtered once rather than each effect fighting its own shimmer.
   */
  #installPasses(): void {
    if (this.#tier.ssr !== 'off') {
      const pass = new SSRCompositePass(this.#ssr);
      this.#post.addPass(pass, { before: 'post.taa' });
      this.#ssrPass = pass;
    }
    this.#post.addPass(this.#lightShafts.pass, { before: 'post.taa' });
    this.#installedShafts = true;
  }

  /**
   * There is exactly one exposure authority: `CompositePass`.
   *
   * `PostStack` sets `NoToneMapping` when it installs, so the renderer-level
   * ACES curve `RendererFactory` configures is inert — but only as a side
   * effect of installation order. Checking it here turns an implicit
   * invariant into a loud one.
   */
  #enforceSingleExposureAuthority(ctx: GameContext): void {
    const renderer = ctx.renderer.three as { toneMapping?: THREE.ToneMapping };
    if (renderer.toneMapping !== undefined && renderer.toneMapping !== THREE.NoToneMapping) {
      console.warn(
        '[FrameGraph] the renderer still has tone mapping enabled while PostStack ' +
          'is installed; the image would be tone-mapped twice. Forcing NoToneMapping. ' +
          'Exposure belongs to PostStack.composite and nowhere else.',
      );
      renderer.toneMapping = THREE.NoToneMapping;
    }
  }

  dispose(): void {
    if (this.#ssrPass !== null) {
      this.#post.removePass(this.#ssrPass.id);
      this.#ssrPass.dispose();
      this.#ssrPass = null;
    }
    if (this.#installedShafts) {
      this.#post.removePass(this.#lightShafts.pass.id);
      this.#installedShafts = false;
    }
  }
}

const EMPTY_MATRICES: readonly THREE.Matrix4[] = [];

/** Wet mud and dead grass: dark, faintly warm. Multiplies the sky term. */
const GROUND_BOUNCE_ALBEDO = new THREE.Color(0.16, 0.14, 0.11);

/* -------------------------------------------------------------------------- */
/* Audit                                                                      */
/* -------------------------------------------------------------------------- */

interface ServiceExpectation {
  readonly id: string;
  readonly what: string;
  /** What is lost if it is absent. Printed at `warn` when it is. */
  readonly degradesTo: string;
}

const EXPECTED_SERVICES: readonly ServiceExpectation[] = [
  { id: 'assets', what: 'AssetManager', degradesTo: 'procedural placeholder textures only' },
  { id: 'render.quality', what: 'RenderQualityProvider', degradesTo: 'materials pin to "high"' },
  { id: 'world.weather', what: 'WeatherStateProvider', degradesTo: 'authored dry wetness' },
  { id: 'render.post', what: 'PostStack', degradesTo: 'no tone mapping, no AA, no grade' },
  { id: 'render.motion', what: 'MotionVectorProvider', degradesTo: 'no TAA, no temporal AO/SSR' },
  { id: 'render.timeOfDay', what: 'TimeOfDay', degradesTo: 'sun frozen at the default hour' },
  { id: 'render.sky', what: 'Sky / SkyEnvironmentProvider', degradesTo: 'no IBL from the sky' },
  { id: 'render.atmosphere', what: 'AtmosphereService', degradesTo: 'no aerial perspective' },
  { id: 'render.sky.sun', what: 'SkySunProvider', degradesTo: 'lighting and fog lose the sun' },
  { id: 'lighting.celestial', what: 'CelestialLightSink', degradesTo: 'key light stops tracking' },
  { id: 'render.lighting', what: 'LightingService', degradesTo: 'no clustered lights, no CSM' },
  { id: 'render.materials', what: 'MaterialLibraryService', degradesTo: 'no PBR archetypes' },
  { id: 'render.ibl', what: 'IBLService', degradesTo: 'flat ambient, no specular probe' },
  { id: AMBIENT_OCCLUSION_SERVICE_ID, what: 'GTAO → IBL AO', degradesTo: 'no contact occlusion' },
  { id: 'render.gtao', what: 'GTAOService', degradesTo: 'no ambient occlusion at all' },
  { id: 'render.guideBuffer', what: 'GuideBufferProvider', degradesTo: 'AO and SSR disabled' },
  { id: 'render.ssr', what: 'SSRService', degradesTo: 'no wet-surface reflections' },
  { id: 'render.sceneColor', what: 'SceneColorProvider', degradesTo: 'SSR traces against black' },
  {
    id: 'render.gbuffer.surface',
    what: 'SurfaceParameterProvider',
    degradesTo: 'SSR uses scalar roughness defaults',
  },
  { id: 'render.volumetrics', what: 'VolumetricsService', degradesTo: 'no fog, no god rays' },
  { id: 'render.shadow.volumetric', what: 'CSM → fog', degradesTo: 'unshadowed, flat in-scatter' },
  { id: 'render.volumetrics.lights', what: 'local lights → fog', degradesTo: 'no campfire glow' },
  { id: 'render.volumetrics.ambient', what: 'atmosphere → fog', degradesTo: 'constant fog colour' },
  { id: 'render.lightShafts', what: 'LightShaftsService', degradesTo: 'fog never composited' },
];

/**
 * Report every declared dependency and whether it was actually satisfied.
 *
 * The point is the *negative* space. A subsystem that resolves its inputs with
 * `tryGet` and shrugs when they are missing is impossible to debug from the
 * image alone — a scene with no god rays looks like a scene with the fog
 * density set low. Printing the table at boot turns "why is it flat" into a
 * single line of console output.
 */
export function auditServices(ctx: GameContext): { connected: number; degraded: string[] } {
  const degraded: string[] = [];
  let connected = 0;
  for (const expectation of EXPECTED_SERVICES) {
    if (ctx.services.has(expectation.id)) {
      connected++;
    } else {
      degraded.push(expectation.id);
      console.warn(
        `[FrameGraph] MISSING "${expectation.id}" (${expectation.what}) — ` +
          `degrading to: ${expectation.degradesTo}`,
      );
    }
  }
  // A surface provider that answers `null` is registered but not useful; it is
  // reported by its own warning at registration, so it is not double-counted.
  console.info(
    `[FrameGraph] services ${connected}/${EXPECTED_SERVICES.length} connected` +
      (degraded.length === 0 ? '' : `; degraded: ${degraded.join(', ')}`),
  );
  return { connected, degraded };
}

/* -------------------------------------------------------------------------- */
/* Assembly                                                                   */
/* -------------------------------------------------------------------------- */

/** The subsystem instances, for a scene or a capture script that wants one. */
export interface FrameGraph {
  readonly settings: RenderSettings;
  readonly post: PostStack;
  readonly timeOfDay: TimeOfDay;
  readonly sky: Sky;
  readonly lighting: LightingModule;
  readonly gtao: GTAOModule;
  readonly ibl: IBLModule;
  readonly ssr: SSRModule;
  readonly materials: MaterialLibrary;
  readonly volumetrics: VolumetricsModule;
  readonly lightShafts: LightShaftsModule;
  /** In `Engine.add` order. Registration order *is* frame order. */
  readonly modules: readonly GameModule[];
}

export interface FrameGraphOptions {
  readonly quality?: RenderQuality;
  /** Time-of-day preset. Blood Moor's cold overcast morning by default. */
  readonly preset?: 'bloodMoor' | 'dawn' | 'clearNoon' | 'dusk' | 'night' | 'storm';
}

/**
 * Build the whole renderer, in the one order that satisfies every `init`-time
 * dependency.
 *
 * The ordering constraints, all of which are load-bearing:
 *
 * 1. `RenderSettings` first — `MaterialLibrary` reads quality and weather.
 * 2. `PostStack` before `GTAO`/`SSR` — it registers `render.motion`, and both
 *    of them bind the jitter and velocity at `init`.
 * 3. `TimeOfDay` before `Sky` — otherwise `Sky` constructs a second clock.
 * 4. `Sky` before `Lighting` — `Sky` registers `render.sky.sun`.
 * 5. `Lighting` before `RenderBridges` — the bridge reads `sunShadows`.
 * 6. `GTAO` before `IBL` — `GTAO` registers `render.ao`, which `IBL` folds in.
 * 7. `IBL` before `SSR` — `SSR` resolves `render.ibl` for its miss fallback.
 * 8. `RenderBridges` before `Volumetrics` — it provides all three volumetric
 *    providers, every one of which `Volumetrics` resolves at `init`.
 * 9. `Volumetrics` before `LightShafts` — the shaft pass composites its buffer.
 *
 * The scene is added by the caller, after this list, so that content resolves
 * a fully-built renderer.
 */
export function buildFrameGraph(options: FrameGraphOptions = {}): FrameGraph {
  const settings =
    options.quality === undefined
      ? new RenderSettings()
      : new RenderSettings({ quality: options.quality });
  const tier = settings.tier;
  const preset = options.preset ?? 'bloodMoor';

  const post = new PostStack({
    quality: tier.post,
    tonemap: {
      // AgX with the project's own restrained CDL. Not ACES: ACES's notorious
      // hue skew turns exactly the two colours this frame lives on — orange
      // firelight and blue-grey shade — into yellow and cyan.
      look: 'grimdark',
      autoExposure: true,
      // Middle grey below the photographic 0.18. The moor is genuinely dark:
      // wet mud sits near 0.09 albedo and the brief asks for grimdark, so
      // metering the frame to 18% grey would be metering it to the wrong
      // picture. This is the one exposure authority in the renderer.
      middleGrey: 0.115,
      exposureCompensation: -0.2,
      // Exposure *multiplier* bounds, not stops. Deliberately tighter than the
      // default [0.25, 6]: the moor has one very bright thing in it (the fire)
      // and a wide open sky, and a metering range that generous lets a pan
      // toward either of them re-expose the whole moor. Clamped like this the
      // scene stays the brightness it was authored at, and the fire is allowed
      // to be the brightest thing in the frame rather than being normalised
      // back into the midtones.
      autoExposureRange: [0.4, 4],
      // Slow, and slower to open up than to stop down. A camera that
      // re-exposes as it turns reads as a phone, not as a game.
      adaptationUp: 0.9,
      adaptationDown: 2.0,
    },
    bloom: {
      // Threshold above the diffuse range so only the fire and the brightest
      // sky can bloom. Anything lower and an overcast sky glows, which is the
      // "rainbow bloom" failure the brief rules out explicitly.
      threshold: 1.35,
      knee: 0.55,
      intensity: 0.34,
      radius: 0.62,
    },
    grade: {
      // −520 K: cool the whole image toward an overcast white point, then
      // hold a trace of green out of it with a positive tint so the mud and
      // moss stay separable instead of collapsing into the blue.
      temperature: -520,
      tint: 0.015,
      // Lift the shadows into blue and pull a little warmth out of the
      // highlights. This is the split-tone that keeps a desaturated image from
      // being grey mush: the shade is blue, the firelight is orange, and the
      // gap between them survives the saturation cut.
      lift: [-0.004, 0.0, 0.012],
      gamma: [1.0, 1.0, 0.985],
      gain: [1.02, 1.0, 0.97],
      saturation: 0.86,
      contrast: 1.09,
      contrastPivot: 0.4,
      // Just enough to keep the eye off the corners. Any more and it reads as
      // a filter rather than as a lens.
      vignette: 0.24,
      vignetteStart: 0.55,
      grain: 0.012,
    },
  });

  const timeOfDay = new TimeOfDay({
    preset,
    // The `bloodMoor` preset's 10.30 is a beautiful hour and the wrong one for
    // this brief: at latitude 51 in early November it puts the sun at ~15°,
    // and 15° of atmosphere reddens direct sunlight into gold no matter how
    // much cloud is in front of it. Late morning holds the long shadows the
    // composition wants while keeping the key neutral enough to read as
    // overcast rather than as sunset.
    hours: 12.15,
  });

  const sky = new Sky({
    timeOfDay,
    preset,
    skyViewWidth: tier.skyViewWidth,
    environmentWidth: tier.environmentWidth,
    installBackground: true,
    installEnvironment: true,
    // Aerial perspective on every material. This is the long-range half of the
    // fog handoff; the froxel volume owns everything nearer than
    // `handoffDistance`.
    installFog: true,
    stars: false,
  });

  const lighting = new LightingModule({
    keyLight: 'sun',
    ambientFromSky: true,
    shadows: {
      cascades: tier.shadowCascades,
      mapSize: tier.shadowMapSize,
      shadowDistance: tier.shadowDistance,
      // An overcast sky is a big soft source. 2.4° is far wider than the sun's
      // true 0.53° disc and it is the single strongest cue that the light is
      // coming through cloud rather than from a clear sky.
      sunAngularRadius: 2.4,
      lambda: 0.6,
    },
  });

  const gtao = new GTAOModule({
    quality: tier.gtao,
    resolutionScale: tier.gtaoScale,
    // 1.1 m: large enough to darken the hollow a rock sits in, small enough
    // that it does not turn the whole treeline into a smudge.
    radius: 1.1,
    intensity: 1.15,
    multiBounce: 0.7,
    publishToIBL: true,
  });

  const ibl = new IBLModule({
    // Slightly under unity. A fully overcast sky is a very large, very bright
    // source, and at 1.0 it lifts every upward-facing surface far enough that
    // the tone curve has nothing left to do — the frame goes milky, which is
    // the opposite of the brief.
    intensity: 0.82,
    // Kills the light leaking up from below the horizon that a prefiltered
    // equirect otherwise gives every downward-facing surface.
    horizonOcclusion: 1,
    fallbackEnvironmentKey: 'env.overcast.grey',
  });

  const ssr = new SSRModule({
    quality: tier.ssr,
    resolutionScale: tier.ssrScale,
    // See RenderBridges.#registerSurfaceParameters: with no G-buffer these
    // scalars *are* the surface model, so they are set from the scene's
    // dominant material rather than from a neutral default.
    defaultRoughness: 0.34,
    defaultMetalness: 0,
    defaultReflectance: 0.5,
    maxRoughness: 0.62,
    maxDistance: 40,
    intensity: 0.9,
    fallbackRadiance: 0x2b3138,
  });

  const materials = new MaterialLibrary({
    quality: tier.materials,
    anisotropy: 8,
    preload: ['wetMud', 'deadGrass', 'rock', 'mossyRock', 'wetStone', 'bark', 'plank', 'ironRusted'],
  });

  const volumetrics = new VolumetricsModule({
    quality: tier.volumetrics,
    ...(tier.froxelDimensions === null ? {} : { froxelDimensions: tier.froxelDimensions }),
    // Base (2,3,5) with a per-frame offset; TAA owns base (2,3) in 2D. See the
    // "three noise sources" note at the top of this file.
    noiseSeed: `${SCATTER_SEED}.fog`,
    maxLights: 8,
    params: {
      // Thin air, thick ground layer. Blood Moor fog is a pool in the hollows,
      // not a uniform grey wash — `heightFalloff` is what makes it pool, and
      // the density is deliberately low: at 0.010/m the optical depth across
      // the 96 m volume is under 1, so the far ridge is *veiled* rather than
      // erased. Fog that hides the composition is not atmosphere, it is a
      // missing composition.
      density: 0.0105,
      height: 2.2,
      heightFalloff: 0.5,
      // Cold and very slightly blue. Water droplets scatter almost neutrally;
      // the blue comes from what is lighting them, which is the sky.
      albedo: new THREE.Color(0.80, 0.83, 0.88),
      // Forward-scattering, so looking toward the sun blooms and looking away
      // stays dark. That asymmetry is the entire god-ray effect.
      anisotropy: 0.7,
      backAnisotropy: -0.22,
      lobeBlend: 0.78,
      // Restraint, all three of them. The sun is behind a full cloud slab, so
      // a scattering scale above 1 would be modelling light that is not there;
      // the ambient term is the one that would turn the frame into grey soup
      // if it were allowed to dominate.
      sunScatteringScale: 0.7,
      lightScatteringScale: 0.85,
      ambientScatteringScale: 0.5,
      noiseStrength: 0.42,
      noiseScale: 0.055,
      wind: new THREE.Vector3(0.35, 0.02, 0.18),
      volumeDistance: handoffDistance,
    },
  });

  const lightShafts = new LightShaftsModule({
    resolutionScale: tier.lightShaftScale,
    intensity: 1,
    radialShafts: true,
    radialIntensity: 0.55,
    radialSamples: tier.post === 'low' ? 24 : 48,
    radialLength: 0.72,
    radialDecay: 0.94,
  });

  const bridges = new RenderBridges({ post, ssr, lightShafts, tier });

  // The froxel volume now owns the near-ground mist, so the atmosphere must
  // stop modelling it or the first 96 m are fogged twice. Haze — the
  // long-range term — stays exactly as the preset authored it.
  timeOfDay.mood.mistDensity = 0;

  const modules: GameModule[] = [
    settings,
    post,
    timeOfDay,
    sky,
    lighting,
    gtao,
    ibl,
    ssr,
    materials,
    bridges,
    volumetrics,
    lightShafts,
  ];

  return {
    settings,
    post,
    timeOfDay,
    sky,
    lighting,
    gtao,
    ibl,
    ssr,
    materials,
    volumetrics,
    lightShafts,
    modules,
  };
}

/** Every key the audit knows about, so a test can assert the list is honest. */
export const FRAME_GRAPH_SERVICE_IDS: readonly string[] = EXPECTED_SERVICES.map(
  (expectation) => expectation.id,
);

export {
  AtmosphereKey,
  GTAOKey,
  IBLKey,
  LightShaftsKey,
  LightingKey,
  MaterialLibraryKey,
  MotionVectorsKey,
  PostStackKey,
  RenderQualityKey,
  SSRKey,
  SkyKey,
  TimeOfDayKey,
  VolumetricSunKey,
  VolumetricsKey,
  WeatherStateKey,
};
export type { CascadedShadowMapNode, IBLService, LightingService };
