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
 * 45 m, not 96. The froxel volume owns *near-ground* fog: the pool in the
 * basin, the shafts through the treeline, the warm air over the fire. It is a
 * single-scatter model against a locally sampled source function, and it is
 * very good at all three. What it is not is aerial perspective — it has no
 * knowledge of the sky's radiance distribution, so at range it veils distance
 * toward its own albedo rather than toward the horizon colour, and the result
 * is a far ridge that goes *darker* with distance instead of lighter. That
 * inversion destroys depth ordering, which is the single thing aerial
 * perspective exists to establish.
 *
 * At 96 m the froxel volume owned the entire authored composition — ridge,
 * treeline, escarpment, all of it — so there was no aerial perspective in the
 * frame at all. 45 m puts the handoff just past the campfire and the fence
 * line, and hands everything beyond to `Sky`'s aerial-perspective fog node,
 * which integrates against the actual sky-view LUT and therefore lifts the
 * ridge toward the horizon colour the way it must.
 *
 * The froxel grid's furthest slice is ~1.8 m deep at 45 m on the default
 * 64-slice exponential distribution, so shaft edges are if anything crisper.
 */
export const handoffDistance = 45;

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
      // **Off, and this is an art-direction decision, not a limitation.**
      //
      // Auto exposure was the root cause of the frame's worst defect: the same
      // scene reading as two different times of day. The meter converges over
      // frames, so `wide-establishing` (20 warmup frames) settled dark and
      // murky while `webgpu-backend-check` (0 warmup frames, because presenting
      // to the WebGPU canvas loses the device in this container) photographed
      // the unconverged value and came out a stop and a half brighter. The tier
      // table made it worse again by switching metering off at `low` and on
      // everywhere else, so `quality-low` was a third exposure. Three capture
      // paths, three different pictures, one scene.
      //
      // A metered exposure is a *gameplay* feature — it stops a player walking
      // out of a cellar into a white-out. A composed frame must not have one,
      // for the same reason a cinematographer does not shoot a set on aperture
      // priority: the key is a decision, and a decision that changes when the
      // camera turns is not one. `PostStack` now treats an explicit setting
      // here as outranking the tier, so this locks every backend, every tier
      // and every warmup length to the same response.
      autoExposure: false,
      // The locked key, as a linear multiplier on scene radiance. Placed so the
      // wet mud in open sky lands near 0.16–0.20 in display space — the low
      // third of the histogram, with the fire and the sky above it and real
      // shadow below — which is where a grimdark frame's midtone population
      // belongs.
      // 1.35. Locked exposure exposed (as it should) how much brighter the
      // scene had quietly become: the overcast deck was thinned from optical
      // depth 62 to 20 and the cloud's forward-scattering memory more than
      // doubled, between them raising the key by something like two and a half
      // stops. Auto exposure had been hiding all of it by stopping down, which
      // is exactly why a metered frame is not an art-directed one — the
      // brightness was never a decision, it was a feedback loop. At 1.35 the
      // masonry and every prop clipped: not "looked bright", *clipped*, which
      // is why the wall read as a flat cream slab with no visible stone in it
      // at all. Texture detail cannot survive a clipped highlight. 0.45 was the
      // overcorrection — a frame that is *black* is not grimdark, it is
      // unexposed, and it throws away the shadow colour the whole palette
      // depends on. Measured off the capture rather than guessed at: at 1.0 the
      // sky sat at 190/255 and the fire at 186, while the mud, the grass, the
      // Barbarian and the whole foreground were all under 30 — a bimodal
      // histogram with nothing in the middle, which is the textbook description
      // of a frame with no midtones. 1.35, with the ambient raised to match,
      // puts the ground population back into the 40-80 band where a grade can
      // actually work on it, and the material tints have come down far enough
      // (masonry 0.44 -> 0.245, and every prop desaturated in-shader) that
      // nothing clips there any more.
      exposure: 1.35,
      // Retained for the metering path, which is still live for gameplay
      // scenes; it has no effect while `autoExposure` is off.
      middleGrey: 0.17,
      exposureCompensation: 0.0,
      autoExposureRange: [0.8, 2.5],
      adaptationUp: 0.9,
      adaptationDown: 2.0,
    },
    bloom: {
      // Threshold above the diffuse range so only the fire and the brightest
      // sky can bloom. Anything lower and an overcast sky glows, which is the
      // "rainbow bloom" failure the brief rules out explicitly.
      // Threshold just above the diffuse range so only the fire and the
      // brightest sky can bloom. Dropped from 1.35 and the intensity raised
      // from 0.34 because the ember's radiance came down by a factor of ~2.6
      // at the same time (BloodMoor's `emberMaterial`): the fire has to get its
      // halo from the bloom pyramid now instead of from being blown out, which
      // is what gives it a soft edge rather than a hard elliptical one.
      threshold: 1.0,
      knee: 0.55,
      intensity: 0.55,
      radius: 0.66,
    },
    grade: {
      // −520 K: cool the whole image toward an overcast white point, then
      // hold a trace of green out of it with a positive tint so the mud and
      // moss stay separable instead of collapsing into the blue.
      temperature: -520,
      // Negative now, i.e. a touch toward magenta. The positive (green) tint
      // was fighting the cold white balance and the result on screen was a
      // ground that read teal — a hue that belongs to no part of this palette.
      // Pulling it the other way lets the mud stay a cold *brown-grey*.
      tint: -0.012,
      // Lift the shadows into blue and pull a little warmth out of the
      // highlights. This is the split-tone that keeps a desaturated image from
      // being grey mush: the shade is blue, the firelight is orange, and the
      // gap between them survives the saturation cut.
      // Pushed hard, in both directions at once, because the target moved:
      // this is no longer chasing photoreal grimdark, it is chasing *stylized*
      // grimdark — Torchlight, Fable, the Diablo III cinematics — and stylized
      // work is built on confident colour separation, not on the absence of
      // colour. Grey mush is the failure mode of a timid grade, and the fix for
      // it is a split-tone with real amplitude: shadows a clear blue, highlights
      // a clear amber, so the cold sky-lit half of every object and the warm
      // fire-lit half of it are two different colours rather than two
      // brightnesses of the same one.
      lift: [-0.007, 0.0, 0.022],
      gamma: [1.0, 1.0, 0.982],
      gain: [1.055, 1.005, 0.945],
      // 0.94, up from 0.86. Desaturating a stylized frame is exactly backwards.
      // The saturation *discipline* this scene needs — nothing chromatic but
      // the fire, the rust and the moss — now lives where it belongs, in the
      // material albedos (`SurfaceSpec.albedoSaturation`) and in the prop
      // weathering pass, both of which can be selective. A global cut cannot:
      // it takes the chroma out of the firelight just as hard as it takes it
      // out of a salmon-pink barrel.
      saturation: 0.94,
      // 1.16. With a locked exposure there is finally a stable histogram to
      // apply contrast to, and a stylized image wants more of it than a
      // photographic one — the read has to survive being looked at for a third
      // of a second.
      contrast: 1.12,
      // Pivot at the frame's actual midtone, not above it. At 0.4 the contrast
      // operator only ever *darkened*, because with a correct exposure the mud,
      // the wall and the fog all sit between 0.12 and 0.30 in display space and
      // every one of them was below the pivot. 0.2 puts roughly half the ground
      // population on each side, which is what a contrast adjustment is for.
      // 0.17, following the exposure lock down. The pivot has to sit at the
      // *frame's* midtone or contrast becomes a brightness control.
      contrastPivot: 0.15,
      // Just enough to keep the eye off the corners. Any more and it reads as
      // a filter rather than as a lens — and with a correctly exposed frame far
      // less of it is needed to feel moody.
      vignette: 0.17,
      vignetteStart: 0.58,
      // 0.12 px at the corner, down from the look's 0.35. Lateral aberration is
      // a lens artefact and it is a *fine* one; at 0.35 it was painting visible
      // magenta and cyan fringes onto every twig on the ridge, because a bare
      // tree against a bright overcast sky is the highest-contrast,
      // highest-frequency edge in the entire image and radial RGB offsets land
      // hardest exactly there. Stylized art direction cannot afford a coloured
      // fringe on a silhouette — the silhouette *is* the art.
      chromaticAberration: 0.12,
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
    // The froxel volume owns the near-ground mist. Declared, not poked: see
    // `SkyOptions.nearMistExternal`.
    nearMistExternal: true,
    stars: false,
  });

  const lighting = new LightingModule({
    keyLight: 'sun',
    // 1.55. The key's *colour* is physically derived and must stay that way —
    // it is what keeps the overcast cast honest — but its *strength* is an art
    // direction decision, and the two should not be welded together. Thickening
    // the cloud deck back to optical depth 40 was needed to get the skylight
    // cold again (see `TimeOfDay`), and it took the key down with it. This
    // scales the key back up without touching the hue, which is exactly the
    // separation of concerns a cinematographer has between a gel and a dimmer.
    keyLightScale: 1.55,
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
      // Both biases roughly doubled from the module defaults (1.2 / 1.6).
      //
      // The escarpment is displaced rock geometry: large, near-planar facets
      // meeting at sharp creases, lit at a raking angle by a low November sun.
      // That is the worst case for shadow acne, and at the default bias it
      // showed as hard-edged black wedges with dithered borders across the
      // whole cliff — the single most obvious rendering artefact left in the
      // frame, and one that reads as damage rather than as shading.
      //
      // The cost of buying it off is peter-panning: contact shadows detach by
      // the normal-offset distance. That is affordable here and only here,
      // because 3.2 texels at cascade 0's resolution is a couple of
      // centimetres, and because the contact darkening that actually grounds
      // objects in this scene is GTAO's job, not the sun's — see the GTAO
      // radius note below.
      depthBiasTexels: 3.0,
      // 6.5 texels. 3.2 was measured and found insufficient: the acne wedges
      // survived on the escarpment's largest facets, which is exactly where the
      // displaced geometry departs furthest from its own interpolated normal.
      // At cascade 0's ~1 cm texel this is 6-7 cm of receiver offset — visible
      // as peter-panning only on contacts smaller than that, and there are
      // none in this scene that the GTAO pass is not already grounding.
      normalBiasTexels: 6.5,
    },
  });

  const gtao = new GTAOModule({
    quality: tier.gtao,
    resolutionScale: tier.gtaoScale,
    // 0.55 m, down from 1.1. The contact seam where a barrel meets mud lives at
    // the 0.1–0.3 m scale, and a metre-wide horizon search at half resolution
    // averages that seam away into a broad, weak darkening that reads as dirt
    // rather than as contact. Halving the world radius doubles the screen-space
    // sampling density over the range that actually grounds objects. The broad
    // occlusion the larger radius was buying — the hollow a rock sits in — is
    // now the sun's job again, which is the correct owner of it and is only
    // available because the key light has been restored (see TimeOfDay's
    // `cloudOpticalDepth`).
    radius: 0.55,
    intensity: 1.25,
    multiBounce: 0.7,
    publishToIBL: true,
  });

  const ibl = new IBLModule({
    // Slightly under unity. A fully overcast sky is a very large, very bright
    // source, and at 1.0 it lifts every upward-facing surface far enough that
    // the tone curve has nothing left to do — the frame goes milky, which is
    // the opposite of the brief.
    // 1.05, up from 0.82. The earlier cut to 0.58 was correct in *direction*
    // and wrong in magnitude: ambient was flattening the frame, but ambient is
    // also the only thing lighting the shaded two-thirds of an overcast scene,
    // and cutting it crushed the mud and the foreground to black. The flatness
    // is now fixed where it should have been fixed in the first place — with a
    // directional rim and a directional fill in `BloodMoor.#buildLightRig` —
    // which leaves the ambient free to do its actual job of holding shadow
    // detail. Ambient is what was flattening the frame: under a
    // full overcast dome the sky term is close to omnidirectional, so every
    // stop of it goes on uniformly and buys no form. Cutting it and leaving the
    // key and the rim alone is the whole difference between "lit from
    // everywhere" and "lit". The shadows do not go black — the cold shadow fill
    // in `BloodMoor.#buildLightRig` is there to put a hue in them.
    intensity: 1.05,
    // Kills the light leaking up from below the horizon that a prefiltered
    // equirect otherwise gives every downward-facing surface. Below unity: at
    // 1.0 the term is a hard cut, and a heightfield has a great many surfaces
    // tilted a few degrees below horizontal that should be dimmed, not
    // switched off. 0.7 keeps the leak suppressed and the far bank lit.
    horizonOcclusion: 0.7,
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
      // 0.035/m over the 45 m volume is an optical depth of ~1.6 at the
      // handoff, which is a fog you can see. At 0.016 over 96 m the extinction
      // was comparable but the *source function* was near zero (see
      // `ambientScatteringScale`), so it was an invisible veil: it removed
      // contrast from the distance without adding any light back, which is
      // exactly backwards for an overcast day.
      density: 0.035,
      height: 2.0,
      heightFalloff: 0.62,
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
      // Unity, not 0.5. Under a full overcast the *ambient* term is not the
      // one that would turn the frame to grey soup — it is the only in-scatter
      // source there is, because the beam is behind the deck. Halving it was
      // halving the fog's entire source function, which is why the volume
      // extinguished the distance without ever lighting it.
      ambientScatteringScale: 1.0,
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
