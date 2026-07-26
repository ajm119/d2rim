/**
 * @module render/Lighting
 *
 * The single owner of every light in the game.
 *
 * No other module ever adds a `THREE.Light` to the scene. Gameplay, VFX and
 * level code describe what they want — "a torch here, warm, three metres of
 * reach" — and get back a {@link LightHandle}. This module decides what is
 * actually bound to the GPU each frame, which lights get shadows, and how the
 * whole set is culled.
 *
 * ### Why a service and not just `scene.add(light)`
 *
 * In three.js's node renderer, the set of lights bound to a draw call is part of
 * the shader's cache key. A combat spell that adds four point lights and removes
 * them 300 ms later would trigger a shader rebuild on every material in view,
 * twice — a multi-frame hitch, at exactly the moment the player is watching
 * closely. Everything below exists to make that impossible:
 *
 * - **Fixed slot pools.** The number of light *objects* in the scene never
 *   changes after boot. Registering and releasing a light binds and unbinds a
 *   slot; it never creates or destroys one. Unused slots are parked with a
 *   sub-millimetre radius so they are culled before they cost anything.
 * - **Clustered culling on WebGPU.** `ClusteredLightsNode` (Forward+: a 3D grid
 *   of screen tiles × exponential depth slices, filled by a compute pass, one
 *   fragment reading only its own cluster's list) moves shadowless point lights
 *   out of the shader's light list entirely and into a data texture. The shader
 *   is then invariant to how many torches are lit. This follows the standard
 *   Forward+ formulation — Harada, McKee & Yang, "Forward+: Bringing Deferred
 *   Lighting to the Next Level" (Eurographics 2012), extended to 3D clusters by
 *   Olsson, Billeter & Assarsson, "Clustered Deferred and Forward Shading"
 *   (HPG 2012).
 * - **CPU pre-cull on both backends.** Before any of that, registered lights are
 *   culled against the camera frustum and ranked by an analytic screen
 *   importance, so only the strongest `maxPointLights` reach the GPU. This is
 *   what makes the WebGL2 path viable at all — there is no compute shader
 *   there, so the shader loops over every bound light and the bound count has to
 *   stay small and, crucially, *constant*.
 *
 * ### Shadow budget
 *
 * Local shadows are strictly rationed. A shadow-casting point light costs six
 * shadow-map faces per frame; a spot light costs one. Slots are allocated at
 * boot and arbitrated per frame by priority, then by screen importance, so the
 * campfire the player is standing next to always wins and the one across the
 * moor does not. See {@link LightingOptions.maxShadowedPointLights}.
 *
 * ### Sun
 *
 * The sun is a `DirectionalLight` whose shadows come from
 * {@link CascadedShadowMapNode}. Its direction can be driven directly or from
 * elevation/azimuth, and it is kept in sync with the sky module's sun direction
 * when one is registered (see {@link SkySunProvider}).
 */

import * as THREE from 'three/webgpu';
import { clusteredLights } from 'three/addons/tsl/lighting/ClusteredLightsNode.js';

import { serviceKey, type ServiceLocator } from '../core/ServiceLocator';
import type { GameContext, GameModule } from '../core/types';
import {
  attachCascadedShadowMaps,
  type CascadedShadowMapNode,
  type CascadedShadowMapOptions,
} from './CascadedShadowMaps';

declare module '../core/EventBus' {
  interface GameEvents {
    /** Emitted once the lighting rig exists and the service is registered. */
    'lighting:ready': { clustered: boolean; maxPointLights: number };
    /** Emitted when a registered light is granted or loses its shadow slot. */
    'lighting:shadowSlot': { id: number; granted: boolean };
  }
}

/* ------------------------------------------------------------------------- *
 * Contracts with modules owned by other agents
 * ------------------------------------------------------------------------- */

/**
 * What this module reads from the sky/atmosphere system, if it exists.
 *
 * Register an object of this shape under {@link SkySunKey} and the sun light
 * will track it every frame — direction, colour and relative intensity — so a
 * time-of-day cycle drives the key light and the shadows without either module
 * importing the other. Absent that service, the sun is whatever
 * {@link LightingService.setSun} last set.
 *
 * `sunDirection` points **from the origin toward the sun** (the same convention
 * `src/render/ProceduralSky.ts` already uses), i.e. the light travels along
 * `-sunDirection`.
 */
export interface SkySunProvider {
  readonly sunDirection: THREE.Vector3;
  /** Linear-space colour of direct sunlight. */
  readonly sunColor?: THREE.Color;
  /** Scalar in `[0, 1]`; 0 at or below the horizon. */
  readonly sunIntensity?: number;
}

/** Service key the sky module should register itself under for sun tracking. */
export const SkySunKey = serviceKey<SkySunProvider>('render.sky.sun');

/* ------------------------------------------------------------------------- *
 * Public types
 * ------------------------------------------------------------------------- */

export type LightKind = 'point' | 'spot';

/** A position or direction accepted by the light API. */
export interface Vec3Like {
  readonly x: number;
  readonly y: number;
  readonly z: number;
}

export interface LightDesc {
  readonly kind: LightKind;
  /** World-space position. */
  readonly position: Vec3Like;
  /** Linear-space colour. A number is read as `0xRRGGBB`. */
  readonly color?: THREE.ColorRepresentation;
  /** Radiant intensity. Default 1. */
  readonly intensity?: number;
  /**
   * Influence radius in world units. Beyond it the light contributes nothing
   * and it is culled. This is the sphere clustered culling uses, so it must be
   * honest — an over-large radius costs real performance. Default 8.
   */
  readonly radius?: number;
  /** Inverse-square falloff exponent. Default 2 (physical). */
  readonly decay?: number;
  /** Spot only: cone half-angle in radians. Default 0.5. */
  readonly angle?: number;
  /** Spot only: cone edge softness in `[0, 1]`. Default 0.4. */
  readonly penumbra?: number;
  /** Spot only: world-space point the cone aims at. */
  readonly target?: Vec3Like;
  /**
   * Request a shadow slot. Grants are budgeted and arbitrated every frame; the
   * request is a preference, never a guarantee. Read
   * {@link LightHandle.shadowed} to find out what actually happened.
   */
  readonly castShadow?: boolean;
  /**
   * Arbitration weight, applied before screen importance. Use it to pin story
   * lights (the Rogue Encampment fire) above incidental ones. Default 0.
   */
  readonly priority?: number;
  /** Diagnostic name; shows up in the stats readout. */
  readonly name?: string;
}

export interface LightHandle {
  readonly id: number;
  readonly kind: LightKind;
  /** False once {@link release} has been called. */
  readonly alive: boolean;
  /** Whether the light was bound to a GPU slot on the most recent frame. */
  readonly active: boolean;
  /** Whether the light currently holds a shadow slot. */
  readonly shadowed: boolean;

  setPosition(x: number, y: number, z: number): void;
  setTarget(x: number, y: number, z: number): void;
  setColor(color: THREE.ColorRepresentation): void;
  setIntensity(intensity: number): void;
  setRadius(radius: number): void;
  setPriority(priority: number): void;
  /** Change the shadow request. Subject to the same budget as the initial one. */
  setCastShadow(castShadow: boolean): void;

  /** Return the light's slot to the pool. Idempotent. */
  release(): void;
}

export interface SunConfig {
  /**
   * Direction **toward** the sun. Normalised internally. Mutually exclusive
   * with `elevation`/`azimuth`; whichever is supplied last wins.
   */
  readonly direction?: Vec3Like;
  /** Degrees above the horizon. */
  readonly elevation?: number;
  /** Compass degrees. */
  readonly azimuth?: number;
  readonly color?: THREE.ColorRepresentation;
  readonly intensity?: number;
  /** Turn sun shadows on or off wholesale. */
  readonly castShadow?: boolean;
}

export interface AmbientConfig {
  /** Colour arriving from above. */
  readonly skyColor?: THREE.ColorRepresentation;
  /** Colour bouncing off the ground. */
  readonly groundColor?: THREE.ColorRepresentation;
  /**
   * Strength of the hemisphere fill. Defaults to 0 because image-based lighting
   * from the sky is the primary ambient term and adding both double-counts the
   * sky. Raise it for interiors or for scenes with no environment map.
   */
  readonly intensity?: number;
}

export interface LightingStats {
  /** Lights currently registered, whether bound this frame or not. */
  readonly registered: number;
  /** Registered lights that survived frustum and importance culling. */
  readonly visible: number;
  /** Lights bound to a GPU slot this frame. */
  readonly bound: number;
  /** Lights dropped because every slot was already taken by a stronger one. */
  readonly starved: number;
  /** Shadow-casting local lights this frame. */
  readonly shadowed: number;
  /** Whether the GPU clustered-culling compute pass is in use. */
  readonly clustered: boolean;
  /** Slot pool sizes, for budget readouts. */
  readonly capacity: {
    readonly point: number;
    readonly spot: number;
    readonly shadowedPoint: number;
    readonly shadowedSpot: number;
  };
}

export interface LightingService {
  /** Register a light and receive a handle. Never returns null. */
  addLight(desc: LightDesc): LightHandle;

  /** Configure the sun. Fields left undefined keep their current value. */
  setSun(config: SunConfig): void;

  /** Unit vector pointing from the world toward the sun. Copy; safe to keep. */
  getSunDirection(target?: THREE.Vector3): THREE.Vector3;

  /** Configure the hemisphere fill light. */
  setAmbient(config: AmbientConfig): void;

  /**
   * The sun's cascaded shadow map node, for debug overlays and quality
   * settings. `null` when sun shadows are disabled.
   */
  readonly sunShadows: CascadedShadowMapNode | null;

  readonly stats: LightingStats;
}

/** Service key other modules resolve this module by. */
export const LightingKey = serviceKey<LightingService>('render.lighting');

/* ------------------------------------------------------------------------- *
 * Options
 * ------------------------------------------------------------------------- */

export interface LightingOptions {
  /**
   * Shadowless point-light slots. The default is backend-dependent: 128 when a
   * compute-driven clustered pass is available (the shader never iterates them
   * individually), 8 otherwise (every bound light is a loop iteration in every
   * fragment).
   */
  maxPointLights?: number;
  /** Shadowless spot-light slots. Default 4. */
  maxSpotLights?: number;
  /**
   * Shadow-casting point-light slots. Each one costs a **six-face** cube shadow
   * render every frame it is active, so this is deliberately tiny. Default 1.
   */
  maxShadowedPointLights?: number;
  /** Shadow-casting spot-light slots. One face each. Default 1. */
  maxShadowedSpotLights?: number;
  /** Cube face resolution for local point shadows. Default 512. */
  pointShadowMapSize?: number;
  /** Map resolution for local spot shadows. Default 1024. */
  spotShadowMapSize?: number;
  /** Screen tile size, in pixels, for the clustered grid. Default 32. */
  clusterTileSize?: number;
  /** Exponential depth slices in the clustered grid. Default 24. */
  clusterZSlices?: number;
  /** Per-cluster light capacity. Default 32. */
  maxLightsPerCluster?: number;
  /** Force the clustered path off even on a compute-capable backend. */
  disableClustered?: boolean;
  /** Cascaded shadow map configuration for the sun. */
  shadows?: CascadedShadowMapOptions;
  /** Sun defaults, applied at init. */
  sun?: SunConfig;
  /** Ambient defaults, applied at init. */
  ambient?: AmbientConfig;
}

/* ------------------------------------------------------------------------- *
 * Importance ranking (unit-tested in tests/lighting.culling.test.ts)
 * ------------------------------------------------------------------------- */

/**
 * Analytic screen importance of a point light.
 *
 * The idea is to rank lights by how much of the frame they can plausibly
 * change, not by raw distance. A light's peak irradiance falls off as
 * `intensity / d²`, and the solid angle it subtends falls off as `r² / d²`, so
 * the product `intensity · r² / d⁴` orders lights by "how much visible
 * brightness is at stake". `d` is clamped below by the radius: once the camera
 * is *inside* the light's sphere, moving closer does not make it more
 * important, it is already filling the screen.
 *
 * `priority` is added as a multiplicative bump (2^priority) rather than an
 * additive one so a single priority step reliably outranks distance noise while
 * still letting a much closer light of the same priority win.
 */
export function lightImportance(
  intensity: number,
  radius: number,
  distance: number,
  priority = 0,
): number {
  if (!(intensity > 0) || !(radius > 0)) return 0;
  const d = Math.max(distance, radius, 1e-3);
  const d2 = d * d;
  return (intensity * radius * radius) / (d2 * d2) * Math.pow(2, priority);
}

/* ------------------------------------------------------------------------- *
 * Internals
 * ------------------------------------------------------------------------- */

interface LightRecord {
  readonly id: number;
  readonly kind: LightKind;
  readonly name: string;
  readonly position: THREE.Vector3;
  readonly target: THREE.Vector3;
  readonly color: THREE.Color;
  intensity: number;
  radius: number;
  decay: number;
  angle: number;
  penumbra: number;
  castShadow: boolean;
  priority: number;
  alive: boolean;
  /** Recomputed every frame by the culler. */
  importance: number;
  active: boolean;
  shadowed: boolean;
  /** Shadow state on the previous frame, used to emit slot-change edges. */
  wasShadowed: boolean;
}

/**
 * `Light.shadow`, which `@types/three` r185 declares only on the concrete light
 * subclasses and not on the `Light` base class.
 */
interface ShadowCastingLight extends THREE.Light {
  readonly shadow: THREE.LightShadow;
}

/** One pooled GPU light object plus the record currently bound to it. */
interface Slot<T extends THREE.Light> {
  readonly light: T;
  readonly castsShadow: boolean;
  bound: LightRecord | null;
}

const PARK_POSITION = new THREE.Vector3(0, -100000, 0);
/** Non-zero, because `distance === 0` means "unbounded" in three.js. */
const PARK_RADIUS = 0.001;

const _sphere = new THREE.Sphere();
const _frustum = new THREE.Frustum();
const _projScreen = new THREE.Matrix4();
const _cameraPosition = new THREE.Vector3();
const _direction = new THREE.Vector3();

const DEFAULT_SUN_DIRECTION = new THREE.Vector3(0.35, 0.42, 0.84).normalize();

/**
 * `Renderer.lighting`, whose `createNode` hook is how a custom `LightsNode` —
 * the clustered one — is installed for a scene.
 */
interface LightingManagerLike {
  createNode(lights?: THREE.Light[]): THREE.LightsNode;
  getNode(scene: THREE.Scene): THREE.LightsNode;
}

/* ------------------------------------------------------------------------- *
 * Module
 * ------------------------------------------------------------------------- */

export class LightingModule implements GameModule {
  readonly name = 'Lighting';

  readonly #options: LightingOptions;

  #scene: THREE.Scene | null = null;
  #camera: THREE.PerspectiveCamera | null = null;
  #services: ServiceLocator | null = null;
  #events: GameContext['events'] | null = null;

  readonly #sun = new THREE.DirectionalLight(0xffffff, 3);
  readonly #sunTarget = new THREE.Object3D();
  readonly #sunDirection = DEFAULT_SUN_DIRECTION.clone();
  #sunShadows: CascadedShadowMapNode | null = null;
  #sunBaseIntensity = 3;

  readonly #hemisphere = new THREE.HemisphereLight(0x9fb4c8, 0x3a3226, 0);

  readonly #pointSlots: Slot<THREE.PointLight>[] = [];
  readonly #spotSlots: Slot<THREE.SpotLight>[] = [];
  readonly #shadowedPointSlots: Slot<THREE.PointLight>[] = [];
  readonly #shadowedSpotSlots: Slot<THREE.SpotLight>[] = [];

  readonly #records: LightRecord[] = [];
  readonly #visible: LightRecord[] = [];
  #nextId = 1;

  #clusteredActive = false;
  #detachResize: (() => void) | null = null;

  #stats: LightingStats = {
    registered: 0,
    visible: 0,
    bound: 0,
    starved: 0,
    shadowed: 0,
    clustered: false,
    capacity: { point: 0, spot: 0, shadowedPoint: 0, shadowedSpot: 0 },
  };

  constructor(options: LightingOptions = {}) {
    this.#options = options;
  }

  init(ctx: GameContext): void {
    this.#scene = ctx.scene;
    this.#camera = ctx.camera;
    this.#services = ctx.services;
    this.#events = ctx.events;

    const compute = ctx.renderer.capabilities.compute && this.#options.disableClustered !== true;

    // Sun. The light's *position* is irrelevant to a directional light and to
    // the cascade fitting (which derives its own eye per cascade); only the
    // position -> target vector matters.
    this.#sun.name = 'Sun';
    this.#sun.position.set(0, 0, 0);
    this.#sunTarget.name = 'SunTarget';
    this.#sun.target = this.#sunTarget;
    ctx.scene.add(this.#sun, this.#sunTarget);

    this.#hemisphere.name = 'AmbientFill';
    ctx.scene.add(this.#hemisphere);

    this.#applySunDirection(this.#sunDirection);
    if (this.#options.sun !== undefined) this.#setSun(this.#options.sun);
    if (this.#options.ambient !== undefined) this.#setAmbient(this.#options.ambient);

    if (this.#options.sun?.castShadow !== false) {
      this.#sunShadows = attachCascadedShadowMaps(this.#sun, this.#options.shadows ?? {});
    } else {
      this.#sun.castShadow = false;
    }

    this.#buildPools(compute);
    this.#clusteredActive = compute ? this.#installClustered(ctx) : false;

    this.#stats = {
      ...this.#stats,
      clustered: this.#clusteredActive,
      capacity: {
        point: this.#pointSlots.length,
        spot: this.#spotSlots.length,
        shadowedPoint: this.#shadowedPointSlots.length,
        shadowedSpot: this.#shadowedSpotSlots.length,
      },
    };

    ctx.services.register(LightingKey, this.#createService());
    ctx.events.emit('lighting:ready', {
      clustered: this.#clusteredActive,
      maxPointLights: this.#pointSlots.length,
    });
  }

  /**
   * Culling and slot assignment run in `lateUpdate` so that every gameplay
   * module has already moved its lights for this frame. Doing it in `update`
   * would bind slots against last frame's positions and produce a one-frame lag
   * on anything attached to the player.
   */
  lateUpdate(): void {
    this.#trackSky();
    this.#cull();
    this.#assignSlots();
  }

  dispose(): void {
    const scene = this.#scene;
    if (scene !== null) {
      scene.remove(this.#sun, this.#sunTarget, this.#hemisphere);
      for (const slot of this.#allSlots()) scene.remove(slot.light);
    }
    for (const slot of this.#allSlots()) slot.light.dispose();
    this.#sunShadows?.dispose();
    this.#sunShadows = null;
    this.#detachResize?.();
    this.#detachResize = null;
    this.#events = null;
    this.#services?.unregister(LightingKey);
    this.#records.length = 0;
    this.#scene = null;
    this.#camera = null;
    this.#services = null;
  }

  // -- pools --------------------------------------------------------------

  *#allSlots(): Generator<Slot<THREE.Light>> {
    yield* this.#pointSlots;
    yield* this.#spotSlots;
    yield* this.#shadowedPointSlots;
    yield* this.#shadowedSpotSlots;
  }

  /**
   * Allocate every light object the game will ever use, up front.
   *
   * The pool sizes are the actual performance contract of this module: on the
   * WebGL2 path every bound light is an unconditional loop iteration in every
   * lit fragment, so the default of 8 is chosen to keep the forward shading
   * cost bounded. On WebGPU the clustered pass reduces the *per-fragment* count
   * to whatever actually overlaps that pixel's cluster, so the pool can be far
   * larger without the shader getting slower.
   */
  #buildPools(compute: boolean): void {
    const scene = this.#scene;
    if (scene === null) return;

    const maxPoint = this.#options.maxPointLights ?? (compute ? 128 : 8);
    const maxSpot = this.#options.maxSpotLights ?? 4;
    const maxShadowedPoint = this.#options.maxShadowedPointLights ?? 1;
    const maxShadowedSpot = this.#options.maxShadowedSpotLights ?? 1;
    const pointShadowSize = this.#options.pointShadowMapSize ?? 512;
    const spotShadowSize = this.#options.spotShadowMapSize ?? 1024;

    for (let i = 0; i < maxPoint; i++) {
      this.#pointSlots.push(this.#makePointSlot(`PointSlot${i}`, false, pointShadowSize));
    }
    for (let i = 0; i < maxShadowedPoint; i++) {
      this.#shadowedPointSlots.push(
        this.#makePointSlot(`ShadowedPointSlot${i}`, true, pointShadowSize),
      );
    }
    for (let i = 0; i < maxSpot; i++) {
      this.#spotSlots.push(this.#makeSpotSlot(`SpotSlot${i}`, false, spotShadowSize));
    }
    for (let i = 0; i < maxShadowedSpot; i++) {
      this.#shadowedSpotSlots.push(
        this.#makeSpotSlot(`ShadowedSpotSlot${i}`, true, spotShadowSize),
      );
    }

    for (const slot of this.#allSlots()) {
      scene.add(slot.light);
      this.#park(slot);
    }
  }

  #makePointSlot(name: string, castShadow: boolean, mapSize: number): Slot<THREE.PointLight> {
    const light = new THREE.PointLight(0xffffff, 0, PARK_RADIUS, 2);
    light.name = name;
    light.castShadow = castShadow;
    if (castShadow) {
      light.shadow.mapSize.set(mapSize, mapSize);
      light.shadow.camera.near = 0.1;
      light.shadow.camera.far = 20;
      // Cube shadows have no cascade to scale bias against, so both terms are
      // absolute. Normal bias does the heavy lifting; the depth bias is kept
      // small so a torch does not detach from the ground it sits on.
      light.shadow.bias = -0.0008;
      light.shadow.normalBias = 0.04;
      light.shadow.radius = 3;
    }
    return { light, castsShadow: castShadow, bound: null };
  }

  #makeSpotSlot(name: string, castShadow: boolean, mapSize: number): Slot<THREE.SpotLight> {
    const light = new THREE.SpotLight(0xffffff, 0, PARK_RADIUS, 0.5, 0.4, 2);
    light.name = name;
    light.castShadow = castShadow;
    // Every spot needs its own target object, and it must be in the scene graph
    // for its world matrix to update.
    light.target.name = `${name}Target`;
    light.add(light.target);
    light.target.position.set(0, 0, -1);
    if (castShadow) {
      light.shadow.mapSize.set(mapSize, mapSize);
      light.shadow.camera.near = 0.1;
      light.shadow.camera.far = 40;
      light.shadow.bias = -0.0005;
      light.shadow.normalBias = 0.03;
      light.shadow.radius = 3;
    }
    return { light, castsShadow: castShadow, bound: null };
  }

  /**
   * Return a slot to its inert state.
   *
   * Parked lights stay in the scene — removing them would change the light list
   * and rebuild every shader — but are pushed far below the world with a
   * millimetre radius and zero intensity, so both the CPU frustum test and the
   * GPU cluster test reject them immediately.
   */
  #park(slot: Slot<THREE.Light>): void {
    slot.bound = null;
    slot.light.intensity = 0;
    slot.light.position.copy(PARK_POSITION);
    const ranged = slot.light as THREE.PointLight;
    if (typeof ranged.distance === 'number') ranged.distance = PARK_RADIUS;
    if (slot.castsShadow) {
      // A parked shadow caster must not render its six cube faces.
      const shadow = (slot.light as ShadowCastingLight).shadow;
      shadow.autoUpdate = false;
      shadow.needsUpdate = false;
    }
  }

  // -- clustered culling --------------------------------------------------

  /**
   * Install `ClusteredLightsNode` as the scene's lights node.
   *
   * three caches one `LightsNode` per scene inside its private `Lighting`
   * manager, created lazily through `createNode()`. Temporarily replacing that
   * factory and forcing the cache to populate is the supported way to install a
   * custom implementation. If the cache was already populated — which happens
   * if a frame rendered before this module initialised — the swap is refused
   * rather than silently ignored, and the CPU-only path is used instead.
   */
  #installClustered(ctx: GameContext): boolean {
    const renderer = ctx.renderer.three as unknown as { lighting?: LightingManagerLike };
    const lighting = renderer.lighting;
    if (lighting === undefined) return false;

    const node = clusteredLights(
      Math.max(1, this.#options.maxPointLights ?? 128),
      this.#options.clusterTileSize ?? 32,
      this.#options.clusterZSlices ?? 24,
      this.#options.maxLightsPerCluster ?? 32,
    );

    const original = lighting.createNode.bind(lighting);
    lighting.createNode = () => node;
    let installed: THREE.LightsNode;
    try {
      installed = lighting.getNode(ctx.scene);
    } finally {
      lighting.createNode = original;
    }

    if (installed !== (node as unknown as THREE.LightsNode)) {
      console.warn(
        '[Lighting] a lights node already existed for this scene; clustered culling is off. ' +
          'Register LightingModule before the first frame is rendered.',
      );
      return false;
    }

    // No explicit `setSize` here: `ClusteredLightsNode.updateProgram()` reads the
    // renderer's drawing-buffer size itself before every compute dispatch and
    // rebuilds the grid when it changes, so pushing a size in from the resize
    // event would only add a second, racier source of truth.

    return true;
  }

  // -- per-frame ----------------------------------------------------------

  /** Track the sky module's sun, when one is registered. */
  #trackSky(): void {
    const sky = this.#services?.tryGet(SkySunKey);
    if (sky === undefined) return;

    if (sky.sunDirection.lengthSq() > 1e-8) this.#applySunDirection(sky.sunDirection);
    if (sky.sunColor !== undefined) this.#sun.color.copy(sky.sunColor);
    if (sky.sunIntensity !== undefined) {
      this.#sun.intensity = this.#sunBaseIntensity * Math.max(0, sky.sunIntensity);
    }
  }

  /**
   * Frustum-cull registered lights and rank the survivors.
   *
   * Culling is done against the light's influence sphere, which is the same
   * volume the clustered pass uses, so a light that survives here is guaranteed
   * to be able to affect at least one visible pixel.
   */
  #cull(): void {
    const camera = this.#camera;
    this.#visible.length = 0;
    if (camera === null) return;

    camera.updateMatrixWorld();
    _projScreen.multiplyMatrices(camera.projectionMatrix, camera.matrixWorldInverse);
    _frustum.setFromProjectionMatrix(_projScreen, camera.coordinateSystem);
    camera.getWorldPosition(_cameraPosition);

    // Compact dead records in the same pass, so `release()` stays O(1).
    let write = 0;
    for (let read = 0; read < this.#records.length; read++) {
      const record = this.#records[read];
      if (record === undefined) continue;
      if (!record.alive) continue;
      this.#records[write++] = record;

      record.active = false;
      record.shadowed = false;
      record.importance = 0;
      if (record.intensity <= 0 || record.radius <= 0) continue;

      _sphere.center.copy(record.position);
      _sphere.radius = record.radius;
      if (!_frustum.intersectsSphere(_sphere)) continue;

      record.importance = lightImportance(
        record.intensity,
        record.radius,
        record.position.distanceTo(_cameraPosition),
        record.priority,
      );
      if (record.importance > 0) this.#visible.push(record);
    }
    this.#records.length = write;

    this.#visible.sort((a, b) => b.importance - a.importance);
  }

  /**
   * Bind the ranked survivors to slots.
   *
   * Shadow slots are handed out first, and only to lights that asked for them,
   * so a bright unshadowed light can never displace the campfire the shadow
   * budget was reserved for. Everything else fills the plain pools in
   * importance order; whatever is left over is dropped and counted as starved.
   */
  #assignSlots(): void {
    for (const slot of this.#allSlots()) {
      slot.bound = null;
    }
    for (const record of this.#visible) record.wasShadowed = record.shadowed;

    let shadowedPoint = 0;
    let shadowedSpot = 0;
    let point = 0;
    let spot = 0;
    let starved = 0;

    for (const record of this.#visible) {
      let slot: Slot<THREE.Light> | undefined;

      if (record.castShadow) {
        slot =
          record.kind === 'point'
            ? this.#shadowedPointSlots[shadowedPoint]
            : this.#shadowedSpotSlots[shadowedSpot];
        if (slot !== undefined) {
          if (record.kind === 'point') shadowedPoint++;
          else shadowedSpot++;
        }
      }

      if (slot === undefined) {
        slot =
          record.kind === 'point' ? this.#pointSlots[point] : this.#spotSlots[spot];
        if (slot !== undefined) {
          if (record.kind === 'point') point++;
          else spot++;
        }
      }

      if (slot === undefined) {
        starved++;
        continue;
      }

      this.#bind(slot, record);
    }

    // Park everything that did not get a tenant this frame.
    for (const slot of this.#allSlots()) {
      if (slot.bound === null) this.#park(slot);
    }

    const shadowed = shadowedPoint + shadowedSpot;

    // Report slot transitions so a VFX system can, for example, swap a torch's
    // fake blob shadow in when its real shadow is taken away by a stronger
    // light. Only edges are emitted, never per-frame state.
    const events = this.#events;
    if (events !== null) {
      for (const record of this.#visible) {
        if (record.shadowed !== record.wasShadowed) {
          events.emit('lighting:shadowSlot', { id: record.id, granted: record.shadowed });
        }
      }
    }

    this.#stats = {
      ...this.#stats,
      registered: this.#records.length,
      visible: this.#visible.length,
      bound: point + spot + shadowed,
      starved,
      shadowed,
    };
  }

  #bind(slot: Slot<THREE.Light>, record: LightRecord): void {
    slot.bound = record;
    record.active = true;
    record.shadowed = slot.castsShadow;

    const light = slot.light;
    light.color.copy(record.color);
    light.intensity = record.intensity;
    light.position.copy(record.position);

    if (record.kind === 'point') {
      const pointLight = light as THREE.PointLight;
      pointLight.distance = record.radius;
      pointLight.decay = record.decay;
      if (slot.castsShadow) {
        pointLight.shadow.camera.far = Math.max(1, record.radius);
        pointLight.shadow.camera.updateProjectionMatrix();
      }
    } else {
      const spotLight = light as THREE.SpotLight;
      spotLight.distance = record.radius;
      spotLight.decay = record.decay;
      spotLight.angle = record.angle;
      spotLight.penumbra = record.penumbra;
      // `target` is parented to the light, so the offset must be local.
      spotLight.target.position.copy(record.target).sub(record.position);
      spotLight.target.updateMatrixWorld(true);
      if (slot.castsShadow) {
        spotLight.shadow.camera.far = Math.max(1, record.radius);
        spotLight.shadow.camera.updateProjectionMatrix();
      }
    }

    if (slot.castsShadow) {
      const shadow = (slot.light as ShadowCastingLight).shadow;
      shadow.autoUpdate = true;
      shadow.needsUpdate = true;
    }
  }

  // -- sun / ambient ------------------------------------------------------

  #applySunDirection(direction: Vec3Like): void {
    _direction.set(direction.x, direction.y, direction.z);
    if (_direction.lengthSq() < 1e-8) return;
    _direction.normalize();
    this.#sunDirection.copy(_direction);

    // Light travels *from* the sun, so the target sits opposite the sun
    // direction. Distance is arbitrary for a directional light; 1 keeps the
    // numbers well conditioned.
    this.#sun.position.set(0, 0, 0);
    this.#sunTarget.position.copy(_direction).multiplyScalar(-1);
    this.#sunTarget.updateMatrixWorld(true);
    this.#sun.updateMatrixWorld(true);
  }

  #setSun(config: SunConfig): void {
    if (config.elevation !== undefined || config.azimuth !== undefined) {
      const elevation = THREE.MathUtils.degToRad(config.elevation ?? 30);
      const azimuth = THREE.MathUtils.degToRad(config.azimuth ?? 0);
      this.#applySunDirection({
        x: Math.cos(elevation) * Math.sin(azimuth),
        y: Math.sin(elevation),
        z: Math.cos(elevation) * Math.cos(azimuth),
      });
    }
    if (config.direction !== undefined) this.#applySunDirection(config.direction);
    if (config.color !== undefined) this.#sun.color.set(config.color);
    if (config.intensity !== undefined) {
      this.#sunBaseIntensity = Math.max(0, config.intensity);
      this.#sun.intensity = this.#sunBaseIntensity;
    }
    if (config.castShadow !== undefined) this.#sun.castShadow = config.castShadow;
  }

  #setAmbient(config: AmbientConfig): void {
    if (config.skyColor !== undefined) this.#hemisphere.color.set(config.skyColor);
    if (config.groundColor !== undefined) this.#hemisphere.groundColor.set(config.groundColor);
    if (config.intensity !== undefined) this.#hemisphere.intensity = Math.max(0, config.intensity);
  }

  // -- service ------------------------------------------------------------

  #createService(): LightingService {
    // eslint-disable-next-line @typescript-eslint/no-this-alias
    const self = this;

    return {
      addLight(desc: LightDesc): LightHandle {
        const record: LightRecord = {
          id: self.#nextId++,
          kind: desc.kind,
          name: desc.name ?? `${desc.kind}${self.#nextId}`,
          position: new THREE.Vector3(desc.position.x, desc.position.y, desc.position.z),
          target: new THREE.Vector3(
            desc.target?.x ?? desc.position.x,
            desc.target?.y ?? desc.position.y - 1,
            desc.target?.z ?? desc.position.z,
          ),
          color: new THREE.Color(desc.color ?? 0xffffff),
          intensity: desc.intensity ?? 1,
          radius: desc.radius ?? 8,
          decay: desc.decay ?? 2,
          angle: desc.angle ?? 0.5,
          penumbra: desc.penumbra ?? 0.4,
          castShadow: desc.castShadow ?? false,
          priority: desc.priority ?? 0,
          alive: true,
          importance: 0,
          active: false,
          shadowed: false,
          wasShadowed: false,
        };
        self.#records.push(record);
        return createHandle(record);
      },

      setSun(config: SunConfig): void {
        self.#setSun(config);
      },

      getSunDirection(target = new THREE.Vector3()): THREE.Vector3 {
        return target.copy(self.#sunDirection);
      },

      setAmbient(config: AmbientConfig): void {
        self.#setAmbient(config);
      },

      get sunShadows(): CascadedShadowMapNode | null {
        return self.#sunShadows;
      },

      get stats(): LightingStats {
        return self.#stats;
      },
    };
  }
}

/**
 * Wrap a record in the handle callers see.
 *
 * The handle deliberately exposes no `THREE.Light`: a caller that could reach
 * the underlying object could move it, change its shadow settings or reparent
 * it, and the slot arbitration would silently stop being true.
 */
function createHandle(record: LightRecord): LightHandle {
  return {
    id: record.id,
    kind: record.kind,
    get alive(): boolean {
      return record.alive;
    },
    get active(): boolean {
      return record.active;
    },
    get shadowed(): boolean {
      return record.shadowed;
    },
    setPosition(x: number, y: number, z: number): void {
      record.position.set(x, y, z);
    },
    setTarget(x: number, y: number, z: number): void {
      record.target.set(x, y, z);
    },
    setColor(color: THREE.ColorRepresentation): void {
      record.color.set(color);
    },
    setIntensity(intensity: number): void {
      record.intensity = Math.max(0, intensity);
    },
    setRadius(radius: number): void {
      record.radius = Math.max(0, radius);
    },
    setPriority(priority: number): void {
      record.priority = priority;
    },
    setCastShadow(castShadow: boolean): void {
      record.castShadow = castShadow;
    },
    release(): void {
      record.alive = false;
      record.active = false;
      record.shadowed = false;
      record.intensity = 0;
    },
  };
}

/**
 * Convenience entry point mirroring the rest of `src/render`.
 *
 * Prefer registering {@link LightingModule} with the engine directly so it
 * participates in the normal module lifecycle.
 */
export function registerLighting(ctx: GameContext, options: LightingOptions = {}): LightingModule {
  const module = new LightingModule(options);
  ctx.engine.add(module);
  return module;
}
