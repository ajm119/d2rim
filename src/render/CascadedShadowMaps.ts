/**
 * @module render/CascadedShadowMaps
 *
 * Stable cascaded shadow maps (CSM) with contact-hardening (PCSS) filtering for
 * the sun, implemented as a three.js node-renderer `ShadowBaseNode` so that it
 * works on both the WebGPU and WebGL2 backends from one code path.
 *
 * ### Technique and references
 *
 * - **Cascade splits** use the *practical split scheme* of Zhang, Sun, Xu and
 *   Lu, "Parallel-Split Shadow Maps for Large-scale Virtual Environments"
 *   (VRCIA 2006): a `lambda`-weighted blend of the logarithmic and uniform
 *   schemes. Pure logarithmic wastes almost the whole first cascade on the near
 *   plane; pure uniform starves the foreground. `lambda ≈ 0.55` is the usual
 *   compromise and is the default here.
 *
 * - **Stable fitting** follows the standard "stable cascaded shadow maps"
 *   construction popularised by Valient, "Stable Rendering of Cascaded Shadow
 *   Maps" (ShaderX6, 2008) and by Michal Valient / Matt Pettineo's `Shadows`
 *   sample. Each cascade is fitted to the **bounding sphere** of its view
 *   frustum slice rather than to the slice's AABB. A sphere is invariant under
 *   camera rotation, so the cascade extent never changes as the player looks
 *   around; the ortho volume is then **snapped to whole shadow texels** in light
 *   space, so it never changes as the player *walks* either. Without both of
 *   these, shadow edges crawl and shimmer with every camera movement — the
 *   single most recognisable amateur tell in a real-time renderer.
 *
 * - **Bias** combines a small constant depth bias with *normal-offset* bias
 *   (Holbert, "Saying Goodbye to Shadow Acne", GDC 2011): the receiver position
 *   is pushed along the geometric normal by a fraction of the cascade's world
 *   texel size before projection. Normal offset scales naturally with cascade
 *   resolution, kills acne on grazing surfaces without the depth-bias
 *   peter-panning that would detach every object from its own contact shadow.
 *
 * - **Filtering** is percentage-closer soft shadows (Fernando, "Percentage-Closer
 *   Soft Shadows", SIGGRAPH 2005 sketch) adapted for a directional light: a
 *   blocker search estimates the average occluder depth, the penumbra width is
 *   `2 · tan(θ_sun) · blockerDistance` (the geometric penumbra cast by a disc
 *   light of angular radius `θ_sun`), and the PCF kernel radius is set from it.
 *   Shadows are therefore *sharp at the contact point and soften with occluder
 *   distance*. Taps use a Vogel (golden-angle) disc rotated per pixel by
 *   interleaved gradient noise (Jimenez, "Next Generation Post Processing in
 *   Call of Duty: Advanced Warfare", SIGGRAPH 2014) so undersampling shows as
 *   fine dither rather than banding.
 *
 * - **Cascade blending** cross-fades the last `cascadeBlend` fraction of each
 *   cascade into the next one, so no seam is visible at a split boundary. Only
 *   fragments inside the blend band pay for the second lookup.
 *
 * ### Storage
 *
 * All cascades live in a single depth **array texture** (one layer per cascade)
 * rendered in one `renderer.render()` call through an `ArrayCamera`, the same
 * mechanism three.js's own `TileShadowNode` addon uses. That keeps the shader to
 * exactly one texture binding and one sampler regardless of cascade count, and
 * lets the cascade index be an ordinary arithmetic value instead of a shader
 * branch over N textures.
 *
 * The depth texture deliberately has **no** `compareFunction`: PCSS needs to read
 * raw occluder depth for the blocker search, which a comparison sampler cannot
 * do. Depth comparison is done in the shader instead. We lose the free 2x2
 * hardware PCF that `textureSampleCompare` gives, and pay for it with more taps.
 *
 * ### Ownership
 *
 * This module does not touch the scene graph on its own. `Lighting.ts` owns the
 * sun light and installs this node on it. Use {@link attachCascadedShadowMaps}.
 */

import * as THREE from 'three/webgpu';
import {
  Fn,
  If,
  clamp,
  cos,
  float,
  int,
  interleavedGradientNoise,
  min,
  mix,
  normalWorldGeometry,
  positionView,
  screenCoordinate,
  shadowPositionWorld,
  sin,
  smoothstep,
  step,
  texture,
  uniform,
  uniformArray,
  vec2,
  vec4,
  getShadowMaterial,
  getShadowRenderObjectFunction,
} from 'three/tsl';

/* ------------------------------------------------------------------------- *
 * Options
 * ------------------------------------------------------------------------- */

export interface CascadedShadowMapOptions {
  /** Number of cascades, clamped to `[1, 4]`. Default 4. */
  cascades?: number;
  /** Square resolution of **each** cascade layer. Default 2048. */
  mapSize?: number;
  /**
   * Furthest view distance, in world units, that receives sun shadows.
   * Beyond this the shadow term fades to fully lit. Default 180.
   */
  shadowDistance?: number;
  /**
   * Practical-split blend in `[0, 1]`. 0 = uniform splits, 1 = logarithmic.
   * Default 0.55.
   */
  lambda?: number;
  /**
   * Fraction of each cascade's depth range spent cross-fading into the next
   * cascade. 0 disables blending (and shows a hard seam). Default 0.12.
   */
  cascadeBlend?: number;
  /**
   * Extra distance, in world units, placed between the cascade's ortho near
   * plane and the front of the fitted sphere. Casters standing between the sun
   * and the visible slice must be inside this margin or their shadows pop in.
   * Default 80.
   */
  lightMargin?: number;
  /**
   * Apparent angular *radius* of the sun disc, in degrees, used to drive
   * penumbra width. The physical sun is 0.27 deg, which produces shadows far
   * crisper than the overcast, heavily-scattered key this game is authored for.
   * Default 1.6 deg — still physically motivated (a bright patch of overcast sky
   * subtends roughly this much), just a larger source. Raise to soften.
   */
  sunAngularRadius?: number;
  /** PCF taps per cascade lookup. Default 16. */
  filterSamples?: number;
  /** Blocker-search taps. Default 8. */
  blockerSamples?: number;
  /** Minimum PCF radius, in shadow texels. Keeps contacts from aliasing. Default 1. */
  minFilterTexels?: number;
  /** Maximum PCF radius, in shadow texels. Bounds worst-case bandwidth. Default 12. */
  maxFilterTexels?: number;
  /**
   * Radius of the blocker search, in shadow texels. Bounds how far away an
   * occluder may be and still widen the penumbra. Default 12.
   */
  blockerSearchTexels?: number;
  /** Constant depth bias, expressed in shadow texels of slope. Default 1.2. */
  depthBiasTexels?: number;
  /** Normal-offset distance, in shadow texels. Default 1.6. */
  normalBiasTexels?: number;
  /** Shadow darkness in `[0, 1]`. 1 = fully occluded is black. Default 1. */
  intensity?: number;
}

interface ResolvedOptions extends Required<CascadedShadowMapOptions> {}

const DEFAULTS: ResolvedOptions = {
  cascades: 4,
  mapSize: 2048,
  shadowDistance: 180,
  lambda: 0.55,
  cascadeBlend: 0.12,
  lightMargin: 80,
  sunAngularRadius: 1.6,
  filterSamples: 16,
  blockerSamples: 8,
  minFilterTexels: 1,
  maxFilterTexels: 12,
  blockerSearchTexels: 12,
  depthBiasTexels: 1.2,
  normalBiasTexels: 1.6,
  intensity: 1,
};

function resolveOptions(options: CascadedShadowMapOptions): ResolvedOptions {
  const merged: ResolvedOptions = { ...DEFAULTS, ...options };
  merged.cascades = Math.max(1, Math.min(4, Math.round(merged.cascades)));
  merged.mapSize = Math.max(256, 1 << Math.round(Math.log2(merged.mapSize)));
  merged.lambda = Math.max(0, Math.min(1, merged.lambda));
  merged.cascadeBlend = Math.max(0, Math.min(0.5, merged.cascadeBlend));
  merged.filterSamples = Math.max(4, Math.round(merged.filterSamples));
  merged.blockerSamples = Math.max(4, Math.round(merged.blockerSamples));
  return merged;
}

/* ------------------------------------------------------------------------- *
 * Pure geometry helpers (unit-tested in tests/csm.math.test.ts)
 * ------------------------------------------------------------------------- */

/**
 * Practical split scheme (Zhang et al. 2006).
 *
 * Returns the `count` **far** distances of the cascades, in view-space units
 * along the camera forward axis. The last entry always equals `far`.
 *
 * `lambda = 0` gives the uniform scheme `near + (far - near) · i/N`;
 * `lambda = 1` gives the logarithmic scheme `near · (far/near)^(i/N)`;
 * intermediate values linearly interpolate the two, which is what makes the
 * near cascades dense enough to resolve contact shadows without collapsing the
 * far cascade into a sliver.
 */
export function practicalSplits(
  near: number,
  far: number,
  count: number,
  lambda: number,
): number[] {
  const n = Math.max(1, Math.round(count));
  const safeNear = Math.max(1e-4, near);
  const safeFar = Math.max(safeNear + 1e-4, far);
  const ratio = safeFar / safeNear;
  const splits: number[] = [];
  for (let i = 1; i <= n; i++) {
    const p = i / n;
    const log = safeNear * Math.pow(ratio, p);
    const uniformSplit = safeNear + (safeFar - safeNear) * p;
    splits.push(lambda * log + (1 - lambda) * uniformSplit);
  }
  // Guard the last split against floating-point drift so the outermost cascade
  // provably reaches the shadow distance.
  splits[n - 1] = safeFar;
  return splits;
}

/** A bounding sphere of a view-frustum slice, expressed in view space. */
export interface FrustumSliceSphere {
  /** Distance from the eye to the sphere centre along the forward axis (positive). */
  readonly distance: number;
  readonly radius: number;
}

/**
 * Exact bounding sphere of a symmetric perspective frustum slice.
 *
 * Derivation: with `k² = tan²(fovY/2)·(1 + aspect²)`, the near-plane corners sit
 * at radius `n·k` and the far-plane corners at `f·k` from the axis. The
 * minimum-radius sphere centred on the axis at distance `d` satisfies
 * `(d-n)² + n²k² = (d-f)² + f²k²`, giving
 * `d = ½(n+f)(1+k²)`. If that centre falls beyond the far plane — which happens
 * for very shallow slices — the far face alone bounds the slice and the sphere
 * is the circumsphere of the far quad.
 *
 * Using a sphere (rather than the slice AABB) is what makes the cascade extent
 * independent of camera orientation, and therefore what makes the shadow map
 * stable under rotation.
 */
export function frustumSliceSphere(
  near: number,
  far: number,
  fovYRadians: number,
  aspect: number,
): FrustumSliceSphere {
  const n = Math.max(1e-4, near);
  const f = Math.max(n + 1e-4, far);
  const t = Math.tan(fovYRadians * 0.5);
  const k2 = t * t * (1 + aspect * aspect);

  if (k2 * (f + n) >= f - n) {
    // Centre would land at or past the far plane: bound by the far quad.
    return { distance: f, radius: f * Math.sqrt(k2) };
  }

  const distance = 0.5 * (f + n) * (1 + k2);
  const dx = distance - f;
  const radius = Math.sqrt(dx * dx + f * f * k2);
  return { distance, radius };
}

/**
 * Snap a light-space position to a whole-texel grid.
 *
 * The cascade's ortho volume spans `2·radius` world units across `mapSize`
 * texels, so one texel is `2·radius/mapSize` world units. Quantising the volume
 * centre to that grid means a one-texel camera movement shifts the shadow map by
 * exactly one texel instead of a fraction of one, which is the difference
 * between a shadow edge that translates cleanly and one that boils.
 *
 * Only the two axes perpendicular to the light matter; depth is not snapped.
 */
export function snapToTexelGrid(x: number, y: number, texelWorldSize: number): [number, number] {
  if (!(texelWorldSize > 0)) return [x, y];
  return [Math.floor(x / texelWorldSize) * texelWorldSize, Math.floor(y / texelWorldSize) * texelWorldSize];
}

/**
 * `index`-th point of a Vogel (golden-angle / sunflower) disc of `count` points,
 * in the unit disc.
 *
 * Vogel discs have near-optimal blue-noise-like spacing for any count, unlike a
 * fixed Poisson table which only looks good at the size it was generated for.
 * The rotation is applied per pixel in the shader; the JS side only produces the
 * canonical unrotated pattern.
 */
export function vogelDiskSample(index: number, count: number): [number, number] {
  const goldenAngle = 2.399963229728653; // π(3 − √5)
  const r = Math.sqrt((index + 0.5) / count);
  const theta = index * goldenAngle;
  return [r * Math.cos(theta), r * Math.sin(theta)];
}

/* ------------------------------------------------------------------------- *
 * Boundary types
 *
 * @types/three r185 does not describe every internal three.js surface this node
 * needs. Rather than reach for `any`, each gap gets a narrow structural
 * interface and exactly one cast at the boundary.
 * ------------------------------------------------------------------------- */

/** `RenderTarget` options that create a layered target with array depth. */
interface ArrayDepthTargetOptions {
  format: THREE.PixelFormat;
  type: THREE.TextureDataType;
  depth: number;
  useArrayDepthTexture: boolean;
  depthBuffer: boolean;
}

/** `Camera._reversedDepth`, which `Renderer._updateCamera` writes before drawing. */
interface ReversibleCamera {
  _reversedDepth: boolean;
}

/** `Light.shadow`, declared only on the concrete light subclasses in @types/three. */
interface LightWithShadow {
  shadow: THREE.LightShadow | undefined;
}

/**
 * The real runtime signatures of three's renderer/scene state helpers.
 *
 * `@types/three` r185 declares `resetRendererAndSceneState` and
 * `restoreRendererAndSceneState` without their `scene` parameter, which does not
 * match `RendererUtils.js`. The structural interface below matches the source.
 */
interface RendererSceneStateApi {
  resetRendererAndSceneState(renderer: THREE.Renderer, scene: THREE.Scene, state: unknown): unknown;
  restoreRendererAndSceneState(renderer: THREE.Renderer, scene: THREE.Scene, state: unknown): void;
}

const rendererUtils = THREE.RendererUtils as unknown as RendererSceneStateApi;

/** The exact callback shape `Renderer.setRenderObjectFunction` accepts. */
type RenderObjectFunction = Parameters<THREE.Renderer['setRenderObjectFunction']>[0];

/** Minimal view of the frame object handed to `Node.updateBefore`. */
interface ShadowFrame {
  renderer: THREE.Renderer;
  scene: THREE.Scene;
  camera: THREE.Camera;
  frameId: number;
}

/**
 * A featherweight stand-in for a `DirectionalLight`.
 *
 * `LightShadow.updateMatrices()` only reads `matrixWorld` and `target.matrixWorld`,
 * so a bare `Object3D` pair is enough to drive one cascade's shadow camera —
 * and it keeps N-1 extra real lights out of the scene's light list, where they
 * would each add a full lighting term to every shader.
 */
type CascadeAnchor = THREE.DirectionalLight;

/**
 * Create a cascade anchor.
 *
 * A `DirectionalLight` is used purely as a convenient carrier for an
 * `Object3D` + `target` pair and a ready-made `DirectionalLightShadow` (whose
 * `OrthographicCamera` and `updateMatrices()` are exactly what a cascade needs).
 * It is deliberately **never added to the scene**, so three's light list never
 * sees it and no extra lighting term reaches any shader. `@types/three` exports
 * `DirectionalLightShadow` as a type only, so this is also the only way to
 * construct one without a cast.
 */
function createCascadeAnchor(index: number): CascadeAnchor {
  const anchor = new THREE.DirectionalLight(0xffffff, 0);
  anchor.name = `CSMCascade${index}`;
  anchor.castShadow = true;
  return anchor;
}

/* ------------------------------------------------------------------------- *
 * The node
 * ------------------------------------------------------------------------- */

interface Cascade {
  readonly anchor: CascadeAnchor;
  readonly shadow: THREE.DirectionalLightShadow;
  /** World-space size of one shadow texel for this cascade. */
  texelWorldSize: number;
  /** `far - near` of this cascade's ortho camera, in world units. */
  depthRange: number;
  /** View-space distance at which this cascade ends. */
  splitFar: number;
  /** View-space distance at which the cross-fade into the next cascade begins. */
  blendStart: number;
}

const _lightDirection = new THREE.Vector3();
const _lightWorld = new THREE.Vector3();
const _targetWorld = new THREE.Vector3();
const _sliceCentre = new THREE.Vector3();
const _lightBasis = new THREE.Matrix4();
const _lightBasisInverse = new THREE.Matrix4();
const _up = new THREE.Vector3(0, 1, 0);
const _altUp = new THREE.Vector3(0, 0, 1);
const _eye = new THREE.Vector3();
const _zero = new THREE.Vector3(0, 0, 0);

let _rendererState: unknown;

/**
 * Cascaded shadow map node for a directional light.
 *
 * Install it with `light.shadow.shadowNode = node` (see
 * {@link attachCascadedShadowMaps}); three's `AnalyticLightNode.setupShadow`
 * picks a custom `shadowNode` up automatically and multiplies the light colour
 * by whatever float this node returns.
 */
export class CascadedShadowMapNode extends THREE.ShadowBaseNode {
  static get type(): string {
    return 'CascadedShadowMapNode';
  }

  readonly options: ResolvedOptions;

  readonly #cascades: Cascade[] = [];
  #arrayCamera: THREE.ArrayCamera | null = null;
  #shadowMap: THREE.RenderTarget | null = null;
  #depthTexture: THREE.DepthTexture | null = null;

  /** `world -> [0,1]^3` per cascade. Uploaded every frame. */
  readonly #matrixValues: THREE.Matrix4[] = [];
  /** `(texelWorldSize, depthRange, depthBias, normalOffset)` per cascade. */
  readonly #paramValues: THREE.Vector4[] = [];
  /** `(splitFar, blendStart, invBlendRange, unused)` per cascade. */
  readonly #bandValues: THREE.Vector4[] = [];

  readonly #uFade = uniform(1).setName('csmFadeStart');
  readonly #uDistance = uniform(1).setName('csmShadowDistance');
  readonly #uTanSun = uniform(0.03).setName('csmTanSunRadius');
  readonly #uIntensity = uniform(1).setName('csmIntensity');

  /** Set once `setup()` has built the render target. */
  #built = false;
  #reversedDepth = false;
  #node: THREE.Node | null = null;
  readonly #cameraFrameId = new WeakMap<THREE.Camera, number>();

  constructor(light: THREE.Light, options: CascadedShadowMapOptions = {}) {
    super(light);
    this.options = resolveOptions(options);
    this.updateBeforeType = 'render';

    for (let i = 0; i < this.options.cascades; i++) {
      this.#matrixValues.push(new THREE.Matrix4());
      this.#paramValues.push(new THREE.Vector4(1, 1, 0, 0));
      this.#bandValues.push(new THREE.Vector4(1, 1, 0, 0));
    }

    this.#uIntensity.value = this.options.intensity;
    this.#uDistance.value = this.options.shadowDistance;
    this.#uTanSun.value = Math.tan(THREE.MathUtils.degToRad(this.options.sunAngularRadius));
  }

  /** Number of cascades actually in use. */
  get cascadeCount(): number {
    return this.options.cascades;
  }

  /** The depth array texture holding every cascade. `null` before first build. */
  get shadowMap(): THREE.RenderTarget | null {
    return this.#shadowMap;
  }

  /** View-space far distance of each cascade, for debug overlays. */
  get splitDistances(): number[] {
    return this.#cascades.map((cascade) => cascade.splitFar);
  }

  /**
   * Change how soft the sun's shadows are without rebuilding anything.
   *
   * @param degrees apparent angular radius of the light source, in degrees.
   */
  setSunAngularRadius(degrees: number): void {
    this.options.sunAngularRadius = Math.max(0.01, degrees);
    this.#uTanSun.value = Math.tan(THREE.MathUtils.degToRad(this.options.sunAngularRadius));
  }

  /** Change the furthest shadowed distance. Takes effect next frame. */
  setShadowDistance(distance: number): void {
    this.options.shadowDistance = Math.max(1, distance);
    this.#uDistance.value = this.options.shadowDistance;
  }

  /** Shadow darkness in `[0, 1]`. */
  setIntensity(intensity: number): void {
    this.options.intensity = Math.max(0, Math.min(1, intensity));
    this.#uIntensity.value = this.options.intensity;
  }

  // -- resources ----------------------------------------------------------

  #build(renderer: THREE.Renderer): void {
    if (this.#built) return;

    const { cascades, mapSize } = this.options;
    this.#reversedDepth = renderer.reversedDepthBuffer === true;

    // One depth array texture, one layer per cascade. No `compareFunction`:
    // the PCSS blocker search must read raw occluder depth, which a comparison
    // sampler cannot provide. Nearest filtering because depth textures are not
    // linearly filterable with a non-comparison sampler on WebGPU anyway, and
    // the PCF kernel does its own reconstruction.
    const depthTexture = new THREE.DepthTexture(
      mapSize,
      mapSize,
      undefined,
      undefined,
      undefined,
      undefined,
      THREE.NearestFilter,
      THREE.NearestFilter,
      undefined,
      undefined,
      cascades,
    );
    depthTexture.name = 'CSMDepthArray';
    depthTexture.compareFunction = null;

    const targetOptions: ArrayDepthTargetOptions = {
      // A colour attachment is still required by both backends even though the
      // shadow pass writes nothing to it; R8 is the cheapest legal choice.
      format: THREE.RedFormat,
      type: THREE.UnsignedByteType,
      depth: cascades,
      useArrayDepthTexture: true,
      depthBuffer: true,
    };
    const shadowMap = new THREE.RenderTarget(
      mapSize,
      mapSize,
      targetOptions as unknown as THREE.RenderTargetOptions,
    );
    shadowMap.texture.name = 'CSMShadowColour';
    shadowMap.depthTexture = depthTexture;

    this.#shadowMap = shadowMap;
    this.#depthTexture = depthTexture;

    const cameras: THREE.OrthographicCamera[] = [];
    for (let i = 0; i < cascades; i++) {
      const anchor = createCascadeAnchor(i);
      const shadow = anchor.shadow;
      shadow.mapSize.set(mapSize, mapSize);
      shadow.camera.coordinateSystem = renderer.coordinateSystem;
      (shadow.camera as unknown as ReversibleCamera)._reversedDepth = this.#reversedDepth;
      cameras.push(shadow.camera);

      this.#cascades.push({
        anchor,
        shadow,
        texelWorldSize: 1,
        depthRange: 1,
        splitFar: 1,
        blendStart: 1,
      });
    }

    // `ArrayCamera` is typed for perspective sub-cameras because its usual job
    // is stereo XR views; three's renderer only ever reads `projectionMatrix`,
    // `matrixWorldInverse`, `layers` and `viewport` off them, all of which
    // `OrthographicCamera` provides. `TileShadowNode` in three's own addons does
    // exactly this.
    this.#arrayCamera = new THREE.ArrayCamera(
      cameras as unknown as THREE.PerspectiveCamera[],
    );
    this.#arrayCamera.name = 'CSMCascadeCameras';

    this.#built = true;
  }

  // -- per-frame cascade fitting -----------------------------------------

  /**
   * Refit every cascade to the current camera. Called once per frame, before
   * the shadow maps are drawn.
   *
   * The fit is deliberately independent of anything but the camera transform,
   * the light direction and the option set, so it is deterministic: the same
   * camera pose always produces byte-identical shadow matrices, which is what
   * lets golden-image capture work.
   */
  #fitCascades(camera: THREE.Camera): void {
    const { cascades, mapSize, lambda, cascadeBlend, lightMargin } = this.options;

    const perspective = camera as THREE.PerspectiveCamera;
    const fovY = THREE.MathUtils.degToRad(
      typeof perspective.fov === 'number' && perspective.fov > 0 ? perspective.fov : 60,
    );
    const aspect =
      typeof perspective.aspect === 'number' && perspective.aspect > 0 ? perspective.aspect : 16 / 9;
    const near = Math.max(1e-3, perspective.near ?? 0.1);
    const far = Math.min(perspective.far ?? this.options.shadowDistance, this.options.shadowDistance);

    // Light direction: from the light toward its target, in world space.
    const light = this.light;
    light.getWorldPosition(_lightWorld);
    const lightTarget = (light as THREE.DirectionalLight).target;
    if (lightTarget !== undefined) {
      lightTarget.getWorldPosition(_targetWorld);
    } else {
      _targetWorld.copy(_zero);
    }
    _lightDirection.subVectors(_targetWorld, _lightWorld);
    if (_lightDirection.lengthSq() < 1e-8) _lightDirection.set(0, -1, 0);
    _lightDirection.normalize();

    // A light pointing straight up or down makes `lookAt` with a +Y up vector
    // degenerate; swap to +Z so the light basis stays well conditioned.
    const up = Math.abs(_lightDirection.y) > 0.999 ? _altUp : _up;

    const splits = practicalSplits(near, far, cascades, lambda);
    camera.updateMatrixWorld();

    let sliceNear = near;
    for (let i = 0; i < cascades; i++) {
      const cascade = this.#cascades[i];
      const splitFar = splits[i];
      if (cascade === undefined || splitFar === undefined) continue;

      const sphere = frustumSliceSphere(sliceNear, splitFar, fovY, aspect);

      // Centre of the slice sphere in world space. View space looks down -Z.
      _sliceCentre.set(0, 0, -sphere.distance).applyMatrix4(camera.matrixWorld);

      const radius = Math.max(1e-3, sphere.radius);
      const texelWorldSize = (2 * radius) / mapSize;

      // Build the light-space basis (rotation only) so the centre can be
      // quantised on the shadow-map texel grid. `Matrix4.lookAt` produces the
      // same basis `Object3D.lookAt` will later use inside
      // `LightShadow.updateMatrices`, so the snap and the render agree.
      _eye.copy(_sliceCentre).addScaledVector(_lightDirection, -(radius + lightMargin));
      _lightBasis.lookAt(_eye, _sliceCentre, up);
      _lightBasisInverse.copy(_lightBasis).invert();

      _sliceCentre.applyMatrix4(_lightBasisInverse);
      const [snappedX, snappedY] = snapToTexelGrid(_sliceCentre.x, _sliceCentre.y, texelWorldSize);
      _sliceCentre.set(snappedX, snappedY, _sliceCentre.z).applyMatrix4(_lightBasis);

      _eye.copy(_sliceCentre).addScaledVector(_lightDirection, -(radius + lightMargin));

      cascade.anchor.position.copy(_eye);
      cascade.anchor.updateMatrixWorld(true);
      cascade.anchor.target.position.copy(_sliceCentre);
      cascade.anchor.target.updateMatrixWorld(true);

      const shadowCamera = cascade.shadow.camera;
      shadowCamera.left = -radius;
      shadowCamera.right = radius;
      shadowCamera.top = radius;
      shadowCamera.bottom = -radius;
      shadowCamera.near = 0;
      // The eye sits `radius + lightMargin` in front of the sphere centre, so
      // the whole sphere ends before `2·radius + lightMargin`. Everything nearer
      // than the sphere (between the sun and the visible slice) still casts.
      shadowCamera.far = 2 * radius + lightMargin;
      shadowCamera.up.copy(up);
      shadowCamera.coordinateSystem = this.#coordinateSystem;
      (shadowCamera as unknown as ReversibleCamera)._reversedDepth = this.#reversedDepth;
      shadowCamera.updateProjectionMatrix();

      cascade.shadow.updateMatrices(cascade.anchor);

      cascade.texelWorldSize = texelWorldSize;
      cascade.depthRange = shadowCamera.far - shadowCamera.near;
      cascade.splitFar = splitFar;
      cascade.blendStart =
        i === cascades - 1 ? splitFar : splitFar - (splitFar - sliceNear) * cascadeBlend;

      // Upload.
      const matrix = this.#matrixValues[i];
      const params = this.#paramValues[i];
      const bands = this.#bandValues[i];
      if (matrix !== undefined) matrix.copy(cascade.shadow.matrix);
      if (params !== undefined) {
        params.set(
          texelWorldSize,
          cascade.depthRange,
          // Constant bias is authored in texels of world size and converted into
          // the cascade's normalised depth units, so it stays visually constant
          // as cascades change scale.
          (this.options.depthBiasTexels * texelWorldSize) / cascade.depthRange,
          this.options.normalBiasTexels * texelWorldSize,
        );
      }
      if (bands !== undefined) {
        const blendRange = Math.max(1e-4, cascade.splitFar - cascade.blendStart);
        bands.set(
          cascade.splitFar,
          cascade.blendStart,
          i === cascades - 1 ? 0 : 1 / blendRange,
          0,
        );
      }

      sliceNear = splitFar;
    }

    this.#uDistance.value = far;
    // Fade the last 15% of the shadowed range so the terminator is a soft
    // gradient rather than a visible arc drawn across the ground.
    this.#uFade.value = far * 0.85;
  }

  #coordinateSystem: THREE.CoordinateSystem = THREE.WebGLCoordinateSystem;

  // -- shader -------------------------------------------------------------

  override setup(builder: THREE.NodeBuilder): THREE.Node | null {
    const renderer = builder.renderer;
    if (renderer.shadowMap.enabled === false) return null;

    this.#coordinateSystem = renderer.coordinateSystem;
    this.#build(renderer);

    // Assigns `shadowPositionWorld`, honouring any per-material override.
    this.setupShadowPosition(builder);

    if (this.#node === null) this.#node = this.#buildShadowNode();
    return this.#node;
  }

  #buildShadowNode(): THREE.Node {
    const depthTexture = this.#depthTexture;
    if (depthTexture === null) return float(1);

    const {
      cascades,
      mapSize,
      filterSamples,
      blockerSamples,
      minFilterTexels,
      maxFilterTexels,
      blockerSearchTexels,
    } = this.options;

    const matrices = uniformArray<'mat4'>(this.#matrixValues, 'mat4');
    const params = uniformArray<'vec4'>(this.#paramValues, 'vec4');
    const bands = uniformArray<'vec4'>(this.#bandValues, 'vec4');

    const invMapSize = 1 / mapSize;
    const reversed = this.#reversedDepth;

    // Pre-computed, un-rotated Vogel discs. Rotating a fixed table by one
    // sin/cos pair per pixel costs two transcendentals for the whole kernel
    // instead of two per tap.
    const filterDisc: Array<[number, number]> = [];
    for (let i = 0; i < filterSamples; i++) filterDisc.push(vogelDiskSample(i, filterSamples));
    const blockerDisc: Array<[number, number]> = [];
    for (let i = 0; i < blockerSamples; i++) blockerDisc.push(vogelDiskSample(i, blockerSamples));

    /**
     * Sample one cascade with PCSS.
     *
     * Returns 1 where the fragment is lit and 0 where it is fully occluded.
     * Every texture read uses an explicit LOD so the function stays legal
     * inside non-uniform control flow (WGSL forbids implicit derivatives there,
     * and the cascade-blend branch is exactly that).
     */
    const sampleCascade = Fn(
      ([layer, worldPosition, worldNormal, rotSin, rotCos]: [
        THREE.Node<'float'>,
        THREE.Node<'vec3'>,
        THREE.Node<'vec3'>,
        THREE.Node<'float'>,
        THREE.Node<'float'>,
      ]) => {
        const layerIndex = int(layer);
        const cascadeParams = params.element(layerIndex);
        const depthRange = cascadeParams.y;
        const depthBias = cascadeParams.z;
        const normalOffset = cascadeParams.w;

        // Normal-offset bias: displace the *receiver* along its geometric
        // normal before projecting, rather than pushing depth. Surfaces at
        // grazing angles to the light — where acne lives — move furthest,
        // surfaces facing the light barely move, and nothing detaches from its
        // own contact shadow.
        const offsetPosition = worldPosition.add(worldNormal.mul(normalOffset));
        const projected = matrices.element(layerIndex).mul(vec4(offsetPosition, 1.0));

        // Ortho projection: w is 1, but divide anyway so the node is correct if
        // a perspective cascade is ever introduced.
        const ndc = projected.xyz.div(projected.w);
        // three's node renderer stores shadow maps with the WebGPU texture
        // convention, so v is flipped relative to the [0,1] clip mapping.
        const uv = vec2(ndc.x, ndc.y.oneMinus()).toVar('csmUV');
        const receiver = ndc.z.toVar('csmReceiver');

        const lit = float(1).toVar('csmLit');

        // Outside this cascade's volume there is nothing to test.
        const inside = uv.x
          .greaterThanEqual(0)
          .and(uv.x.lessThanEqual(1))
          .and(uv.y.greaterThanEqual(0))
          .and(uv.y.lessThanEqual(1))
          .and(receiver.greaterThanEqual(0))
          .and(receiver.lessThanEqual(1));

        If(inside, () => {
          const readDepth = (offset: THREE.Node<'vec2'>) =>
            texture(depthTexture, uv.add(offset), 0).depth(layerIndex).r;

          // -- blocker search ------------------------------------------
          //
          // Average the depth of everything in front of the receiver inside a
          // fixed search radius. Done branchlessly with `step` so the loop is
          // straight-line code.
          const searchRadius = float(blockerSearchTexels * invMapSize);
          const blockerSum = float(0).toVar('csmBlockerSum');
          const blockerCount = float(0).toVar('csmBlockerCount');

          for (const [dx, dy] of blockerDisc) {
            const rx = rotCos.mul(dx).sub(rotSin.mul(dy));
            const ry = rotSin.mul(dx).add(rotCos.mul(dy));
            const d = readDepth(vec2(rx, ry).mul(searchRadius));
            // "In front of the receiver" flips sign under a reversed-Z buffer.
            const isBlocker = reversed
              ? step(receiver.add(depthBias), d)
              : step(d, receiver.sub(depthBias));
            blockerSum.addAssign(d.mul(isBlocker));
            blockerCount.addAssign(isBlocker);
          }

          If(blockerCount.greaterThan(0.5), () => {
            const avgBlocker = blockerSum.div(blockerCount);

            // Penumbra of a disc light of angular radius θ at distance z behind
            // the blocker: w = 2·z·tan(θ). Both terms are in world units, so
            // dividing by the cascade's world texel size gives a radius in
            // texels that is automatically consistent across cascades.
            const blockerDistance = reversed
              ? avgBlocker.sub(receiver).max(0).mul(depthRange)
              : receiver.sub(avgBlocker).max(0).mul(depthRange);
            const penumbraWorld = blockerDistance.mul(this.#uTanSun).mul(2);
            const penumbraTexels = penumbraWorld.div(cascadeParams.x);
            const radiusUV = clamp(
              penumbraTexels,
              float(minFilterTexels),
              float(maxFilterTexels),
            ).mul(invMapSize);

            // -- PCF ---------------------------------------------------
            const visible = float(0).toVar('csmVisible');
            for (const [dx, dy] of filterDisc) {
              const rx = rotCos.mul(dx).sub(rotSin.mul(dy));
              const ry = rotSin.mul(dx).add(rotCos.mul(dy));
              const d = readDepth(vec2(rx, ry).mul(radiusUV));
              visible.addAssign(
                reversed
                  ? step(d, receiver.add(depthBias))
                  : step(receiver.sub(depthBias), d),
              );
            }
            lit.assign(visible.div(filterSamples));
          });
        });

        return lit;
      },
    );

    return Fn(() => {
      // `shadowPositionWorld` is declared as a bare `Node` upstream; it is a
      // vec3 property node by construction (see `ShadowBaseNode`).
      const worldPosition = shadowPositionWorld as THREE.Node<'vec3'>;
      const worldNormal = normalWorldGeometry;

      // View-space distance from the eye, the quantity the splits are in.
      const viewDepth = positionView.z.negate().toVar('csmViewDepth');

      // Cascade selection without a branch: count how many split planes the
      // fragment is behind. Straight-line code, one `step` per boundary.
      const index = float(0).toVar('csmCascade');
      for (let i = 0; i < cascades - 1; i++) {
        index.addAssign(step(bands.element(i).x, viewDepth));
      }

      // Per-pixel kernel rotation. IGN is cheap, temporally stable for a static
      // camera and decorrelates neighbouring pixels, which turns undersampling
      // into fine grain instead of concentric banding.
      const phi = interleavedGradientNoise(screenCoordinate.xy).mul(Math.PI * 2);
      const rotSin = sin(phi).toVar('csmRotSin');
      const rotCos = cos(phi).toVar('csmRotCos');

      const result = sampleCascade(
        index,
        worldPosition,
        worldNormal,
        rotSin,
        rotCos,
      ).toVar('csmShadow');

      if (cascades > 1) {
        const band = bands.element(int(index));
        const blend = clamp(viewDepth.sub(band.y).mul(band.z), 0, 1).toVar('csmBlend');

        // Only the sliver of the frame inside a blend band pays for the second
        // cascade lookup. `sampleCascade` is a real shader function, so calling
        // it inside a branch emits a call, not a duplicated body.
        If(blend.greaterThan(0), () => {
          const next = min(index.add(1), float(cascades - 1));
          result.assign(
            mix(
              result,
              sampleCascade(next, worldPosition, worldNormal, rotSin, rotCos),
              blend,
            ),
          );
        });
      }

      // Fade out at the shadow distance so the outermost cascade does not end
      // in a hard arc drawn across the ground.
      const fade = smoothstep(this.#uFade, this.#uDistance, viewDepth);
      const faded = mix(result, float(1), fade);

      return mix(float(1), faded, this.#uIntensity);
    })();
  }

  // -- rendering ----------------------------------------------------------

  override updateBefore(frame: THREE.NodeFrame): boolean | undefined {
    const shadowFrame = frame as unknown as ShadowFrame;
    const { renderer, scene, camera } = shadowFrame;
    if (renderer === null || scene === null || camera === null) return undefined;
    if (!this.#built || this.#arrayCamera === null || this.#shadowMap === null) return undefined;

    const shadow = (this.light as unknown as LightWithShadow).shadow;
    let needsUpdate = shadow === undefined || shadow.needsUpdate || shadow.autoUpdate;
    if (needsUpdate) {
      // Guard against the same camera being rendered twice in a frame (the
      // capture path renders once to a target and once to the canvas).
      if (this.#cameraFrameId.get(camera) === shadowFrame.frameId) needsUpdate = false;
      this.#cameraFrameId.set(camera, shadowFrame.frameId);
    }
    if (!needsUpdate) return undefined;

    this.#fitCascades(camera);
    this.#renderShadowMaps(renderer, scene, camera);

    if (shadow !== undefined) shadow.needsUpdate = false;
    return undefined;
  }

  /**
   * Draw every cascade in one pass.
   *
   * All cascades share a layered render target, so a single `ArrayCamera`
   * render walks the sub-cameras and writes each into its own depth layer. That
   * is one scene traversal and one render-state setup for the whole CSM instead
   * of N of each — the dominant CPU cost of cascaded shadows in a naive
   * implementation.
   */
  #renderShadowMaps(renderer: THREE.Renderer, scene: THREE.Scene, camera: THREE.Camera): void {
    const shadowMap = this.#shadowMap;
    const arrayCamera = this.#arrayCamera;
    if (shadowMap === null || arrayCamera === null) return;

    const previousRenderObjectFunction = renderer.getRenderObjectFunction();
    const mrt = renderer.getMRT();
    const useVelocity = mrt !== null ? mrt.has('velocity') : false;
    const shadowType = renderer.shadowMap.type;

    _rendererState = rendererUtils.resetRendererAndSceneState(renderer, scene, _rendererState);

    const restoreLayers: number[] = [];
    try {
      scene.overrideMaterial = getShadowMaterial(this.light);

      for (const cascade of this.#cascades) {
        const shadowCamera = cascade.shadow.camera;
        restoreLayers.push(shadowCamera.layers.mask);
        // An untouched shadow camera sees only layer 0; inherit the view
        // camera's layer mask so that anything the player can see can also
        // cast.
        if ((shadowCamera.layers.mask & 0xfffffffe) === 0) {
          shadowCamera.layers.mask = camera.layers.mask;
        }
      }

      const lightShadow = (this.light as unknown as LightWithShadow).shadow;
      if (lightShadow === undefined) return;
      renderer.setRenderObjectFunction(
        getShadowRenderObjectFunction(
          renderer,
          lightShadow,
          shadowType,
          useVelocity,
        ) as unknown as RenderObjectFunction,
      );
      renderer.setClearColor(0x000000, 0);
      renderer.setRenderTarget(shadowMap);
      renderer.render(scene, arrayCamera);
    } finally {
      renderer.setRenderObjectFunction(previousRenderObjectFunction);
      for (let i = 0; i < this.#cascades.length; i++) {
        const cascade = this.#cascades[i];
        const mask = restoreLayers[i];
        if (cascade !== undefined && mask !== undefined) cascade.shadow.camera.layers.mask = mask;
      }
      rendererUtils.restoreRendererAndSceneState(renderer, scene, _rendererState);
    }
  }

  override dispose(): void {
    this.#shadowMap?.dispose();
    this.#shadowMap = null;
    this.#depthTexture = null;
    this.#arrayCamera = null;
    this.#cascades.length = 0;
    this.#built = false;
    this.#node = null;
    super.dispose();
  }
}

/**
 * Build a {@link CascadedShadowMapNode} and install it on `light`.
 *
 * three's `AnalyticLightNode.setupShadow` looks for `light.shadow.shadowNode`
 * and uses it in place of the built-in single-map `ShadowNode`, so this is the
 * whole integration: no renderer flags, no scene traversal.
 *
 * The light must have `castShadow = true` or three never evaluates the shadow
 * node at all; this function sets it.
 */
export function attachCascadedShadowMaps(
  light: THREE.DirectionalLight,
  options: CascadedShadowMapOptions = {},
): CascadedShadowMapNode {
  const node = new CascadedShadowMapNode(light, options);
  light.castShadow = true;
  // three's own shadow bookkeeping is bypassed entirely by the custom node, but
  // `LightShadow.autoUpdate`/`needsUpdate` still gate `updateBefore`.
  light.shadow.autoUpdate = true;
  (light.shadow as unknown as { shadowNode: THREE.Node }).shadowNode = node;
  return node;
}
