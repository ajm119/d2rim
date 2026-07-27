/**
 * @module scene/BloodMoor
 *
 * The vertical slice: a composed view of Diablo II Act I's Blood Moor, built
 * from the project's real photoscanned material sets and real prop models, and
 * used as the calibration target for the whole renderer.
 *
 * This replaces the phase-1 material chart. That scene existed to prove the
 * BRDF; this one exists to prove the *art direction*, which is a different and
 * harder claim: cold overcast grimdark, desaturated but with real colour
 * separation between blue-grey shadow and orange firelight, wet ground, low
 * sky, fog pooling in the hollows.
 *
 * ## Composition
 *
 * Nothing here is randomly placed at the top level. The frame is built as three
 * depth planes with one focal point and one leading line:
 *
 * ```
 *   background   the ridge crest at z ≈ −40, carrying a treeline of procedural
 *                dead hardwoods (see scene/DeadTree) whose broken silhouettes
 *                let the sky through; aerial perspective and haze separate it
 *   midground    the ruined masonry wall (left), the campfire (right of
 *                centre) — the focal point, and the only warm light in the
 *                image — with the Barbarian standing between them
 *   foreground   a rock shelf and rubble in the lower left, dark and in
 *                shadow, framing the eye into the frame
 *   framing      three oversized dead trees at the left and right edges and
 *                one behind the fire: the proscenium arch. They are cut by the
 *                frame edge on purpose — a framing tree that fits inside the
 *                frame is a prop, one that runs off the corner is architecture
 *   leading lines two of them, converging on the fire from opposite sides of
 *                the frame: the broken fence run from the lower-left
 *                foreground, and a worn mud track written into the ground
 *                material itself, entering at the lower right
 * ```
 *
 * ## The palette rule
 *
 * One rule, enforced everywhere, and every colour decision in this file is
 * downstream of it: **the campfire is the only saturated warm thing in the
 * frame.** Everything else is cold blue-grey, with restrained accents of rust,
 * moss and dried blood and nothing else. It is enforced in four places, because
 * one place is never enough:
 *
 * - in the *geometry*, by there being no foliage in the scene at all;
 * - in the *archetypes*, via `SurfaceSpec.albedoSaturation`, which pulls a warm
 *   photoscan toward grey before the cold tint multiplies it — a tint alone
 *   preserves channel ratios and cannot change a hue;
 * - in the *prop weathering*, which desaturates the kit atlases in the shader;
 * - in the *lighting*, where the only warm source in the rig is the fire.
 *
 * The grade is deliberately *not* one of those places. A global saturation cut
 * takes the chroma out of the firelight exactly as hard as it takes it out of a
 * salmon-pink barrel, which is how a grimdark frame becomes grey mush.
 *
 * ## Lighting
 *
 * Three sources, and only one of them is physical. The overcast key and its sky
 * dome come from `Sky`; on top of them sits a shadowless cold rim from beyond
 * the ridge and a much weaker cold fill from camera-right, both authored — see
 * `#buildLightRig` for why an art-directed backlight is not a cheat.
 *
 * The hollow the fire sits in is deliberate: `TerrainField` cuts a basin there,
 * the froxel fog pools in it because the fog has a height falloff, and the low
 * overcast sun rakes across it. That is what produces the god rays through the
 * trees on the ridge — the effect is a consequence of the terrain, not a
 * separately-authored flourish.
 *
 * ## Materials
 *
 * The ground is one mesh with a height-blended two-archetype material: wet mud
 * underneath, dead grass on top, with coverage driven by *world height and
 * slope* rather than by a painted mask, so mud collects in the hollows and
 * grass holds the rises exactly the way water would sort them.
 *
 * Exposed cliff rock is deliberately **not** a third splat layer. Terrain
 * splatting can only ever produce rock that lies flat against the heightfield,
 * which reads as a texture rather than as geology. The escarpment on the west
 * bank is real displaced rock geometry using the `rock` archetype (the
 * photoscanned cliff set, triplanar so it needs no UVs), which is how this is
 * done in production and which is also what lets it cast a real silhouette.
 *
 * The ruined wall is boxed courses carrying the `wetStone` archetype — the
 * masonry set with parallax occlusion mapping. It is UV-projected rather than
 * triplanar because parallax needs a tangent frame, and boxes give it a clean
 * one. Tiling is set per-block from the block's world size so the stone course
 * height stays constant across blocks of different sizes.
 *
 * ## Determinism
 *
 * Every placement derives from {@link SCATTER_SEED} through `Procedural`'s
 * seeded RNG, and every animation reads `ctx.time.elapsed`, never the wall
 * clock. `Engine.stepFrames(n)` therefore reproduces any frame exactly, which
 * is what makes the capture harness's baselines meaningful.
 */

import * as THREE from 'three/webgpu';
import {
  float,
  mix,
  mx_noise_float,
  normalView,
  normalWorldGeometry,
  oneMinus,
  positionViewDirection,
  positionLocal,
  positionWorld,
  vec2,
  saturate,
  sin,
  smoothstep,
  uniform,
  vec3,
  vec4,
} from 'three/tsl';

import { AssetManagerKey, type AssetKey, type AssetManager } from '../assets/AssetManager';
import {
  SimplexNoise,
  buildInstancedMesh,
  createRng,
  generateRockGeometry,
  scatter,
  type ScatterSample,
} from '../assets/Procedural';
import { BLOOD_MOOR_SPAWNS } from '../ai/EnemyDirector';
import type { GameContext } from '../core/types';
import type { PhysicsWorld } from '../physics/PhysicsWorld';
import { disposeZoneTree } from '../world/Zone';
import type { PortalSpec, Zone, ZoneEntryPoint } from '../world/Zone';
import { SCATTER_SEED, auditServices } from '../render/FrameGraph';
import { IBLKey, type IBLService } from '../render/IBL';
import { LightingKey, type LightHandle, type LightingService } from '../render/Lighting';
import { MaterialLibraryKey, type MaterialLibraryService } from '../render/MaterialLibrary';
import type { SurfaceSpec } from '../render/materials/types';
import { VolumetricsKey, type VolumetricsService } from '../render/Volumetrics';
import type { RenderSettings } from '../render/RenderSettings';
import { DEAD_TREE_VARIANTS, generateDeadTreeGeometry } from './DeadTree';
import { weatherMaterial, type WeatheringGrade } from './Weathering';

/* -------------------------------------------------------------------------- */
/* Terrain                                                                    */
/* -------------------------------------------------------------------------- */

/**
 * The single authority for "how high is the ground here".
 *
 * Both the terrain mesh and every scattered prop sample this, which is the
 * only reason props sit *on* the ground rather than hovering over it or
 * sinking into it. Duplicating the height function — once in the mesh builder,
 * once in the placement pass — is the classic way to get a floating tree.
 *
 * Three octaves of simplex for the general roll, plus two authored landforms
 * that the composition depends on and that noise would never reliably produce:
 * a basin for the fire to sit in and fog to pool in, and a bank on the west
 * side for the escarpment to cut out of.
 */
export class TerrainField {
  readonly #noise: SimplexNoise;

  constructor(seed: string | number = `${SCATTER_SEED}.terrain`) {
    this.#noise = new SimplexNoise(seed);
  }

  heightAt(x: number, z: number): number {
    const noise = this.#noise;
    let h = 2.05 * noise.noise2D(x * 0.0165, z * 0.0165);
    h += 0.92 * noise.noise2D(x * 0.0415 + 11.3, z * 0.0415 - 7.1);
    h += 0.31 * noise.noise2D(x * 0.113 - 3.7, z * 0.113 + 5.2);

    // The basin. Centred behind and right of the camera's look-at so the fire
    // sits in it and the fog has somewhere to collect.
    const bx = x - 3.0;
    const bz = z + 5.5;
    h -= 2.35 * Math.exp(-(bx * bx + bz * bz) / 210);

    // The west bank. A smooth ramp rather than a wall: the escarpment geometry
    // supplies the hard edge, and a heightfield cliff would only fight it.
    h += 4.6 * smootherstep(-8, -22, x);

    // The background ridge. Rises toward −z so the skyline sits above the
    // horizon and the bare trees on it read as silhouettes against the sky.
    h += 3.9 * smootherstep(-16, -40, z);

    return h;
  }

  /** Central-difference normal. `epsilon` is a metre — below the prop scale. */
  normalAt(x: number, z: number, epsilon = 1): THREE.Vector3 {
    const dx = this.heightAt(x + epsilon, z) - this.heightAt(x - epsilon, z);
    const dz = this.heightAt(x, z + epsilon) - this.heightAt(x, z - epsilon);
    return new THREE.Vector3(-dx, 2 * epsilon, -dz).normalize();
  }

  /** The surface sampler `Procedural.scatter` expects. */
  readonly surface = (x: number, z: number): { y: number; normal: THREE.Vector3 } => ({
    y: this.heightAt(x, z),
    normal: this.normalAt(x, z),
  });

  /**
   * Displace a plane into the heightfield.
   *
   * The plane is built in XY and rotated, so `position.y` is the plane's
   * second axis before rotation. Displacing before the rotation keeps the
   * vertex order — and therefore the index buffer — untouched.
   */
  buildGeometry(size: number, segments: number): THREE.BufferGeometry {
    const geometry = new THREE.PlaneGeometry(size, size, segments, segments);
    geometry.rotateX(-Math.PI / 2);
    const position = geometry.getAttribute('position');
    for (let i = 0; i < position.count; i++) {
      const x = position.getX(i);
      const z = position.getZ(i);
      position.setY(i, this.heightAt(x, z));
    }
    position.needsUpdate = true;
    geometry.computeVertexNormals();
    geometry.computeBoundingSphere();
    return geometry;
  }
}

/**
 * Coverage mask for a worn track running between two world-space XZ points.
 *
 * The analytic point-to-segment distance, softened. Evaluated per fragment on
 * the ground material: two `smoothstep`s and a dot product, which is cheaper
 * than the texture fetch a painted splat mask would cost and, unlike a splat
 * mask, is resolution-independent — the track's edge stays crisp with the
 * camera's nose on it.
 */
function trackMask(
  a: THREE.Vector2,
  b: THREE.Vector2,
  outerWidth: number,
  innerWidth: number,
): THREE.Node<'float'> {
  const point = vec2(positionWorld.x, positionWorld.z);
  const along = vec2(b.x - a.x, b.y - a.y);
  const offset = point.sub(vec2(a.x, a.y));
  const t = offset.dot(along).div(along.dot(along)).clamp(0, 1);
  const closest = vec2(a.x, a.y).add(along.mul(t));
  return smoothstep(outerWidth, innerWidth, point.distance(closest));
}

/** Ken Perlin's C2 smootherstep, remapping `[edge0, edge1]` to `[0, 1]`. */
function smootherstep(edge0: number, edge1: number, x: number): number {
  const t = Math.min(1, Math.max(0, (x - edge0) / (edge1 - edge0)));
  return t * t * t * (t * (t * 6 - 15) + 10);
}

/* -------------------------------------------------------------------------- */
/* Composition constants                                                      */
/* -------------------------------------------------------------------------- */

/**
 * The default camera.
 *
 * A 34° horizontal-ish FOV at 12 m is a long-ish lens for a third-person game:
 * it compresses the ridge toward the midground so the depth planes stack, which
 * is what makes the aerial perspective legible. A 60° FOV would push the ridge
 * to nothing and the frame would read as an empty field.
 */
const CAMERA = {
  position: new THREE.Vector3(9.4, 3.05, 11.2),
  target: new THREE.Vector3(0.6, 1.35, -3.4),
  fov: 34,
  near: 0.15,
  far: 900,
} as const;

/** Where the fire is. Everything in the midground is composed around it. */
const CAMPFIRE = new THREE.Vector2(3.1, -5.2);

/**
 * The track's control points, in world XZ.
 *
 * Chosen against the default camera: it enters bottom-right of frame, bends
 * left across the basin floor, and terminates at the fire. The last point is
 * the fire itself, so the line does not merely point at the focal object — it
 * arrives at it.
 */
const TRACK: readonly [THREE.Vector2, THREE.Vector2, THREE.Vector2] = [
  new THREE.Vector2(14.6, 17.5),
  new THREE.Vector2(7.4, 2.6),
  new THREE.Vector2(CAMPFIRE.x + 0.4, CAMPFIRE.y + 1.1),
];

/** Where the Barbarian stands: between the camera and the fire, off-centre. */
const HERO = new THREE.Vector2(1.1, 0.4);

/**
 * The moor's two doors, and the ground around them.
 *
 * Both are placed by the same rule: **off the composition's sightline, and out
 * of the encounter**. The wide establishing shot looks north-north-west from
 * (roughly) the south-east, past the fire to the ridge; a cave mouth or a gate
 * marker sitting in that corridor would compete with the fire for the eye and
 * would be the first thing every capture in `tools/capture/shots.json` framed.
 *
 * `BLOOD_MOOR_SPAWNS` puts skeletons between z = -15 and z = +5, mostly west
 * and north. The camp door is therefore south-east at (7, 17), well behind the
 * player's spawn, and the arrival point is 6 m short of it — a player stepping
 * out of the gate gets to see the moor before the moor sees him.
 *
 * The den is the opposite: it is the objective, so it is *past* the encounter,
 * out on the west bank at (-21, -4) where the escarpment already gives the
 * terrain a face for an opening to be cut into.
 */
const DEN_MOUTH = {
  /** Centre of the cave-mouth geometry, and of its portal volume. */
  portal: new THREE.Vector2(-21, -4),
  /** Where a player arriving from the Den lands: outside the mouth, facing east. */
  approach: new THREE.Vector2(-17.6, -3.2),
  /** The way back to the camp. */
  campPortal: new THREE.Vector2(7, 17),
  /** Where a player arriving from the camp lands. */
  campApproach: new THREE.Vector2(6, 11.5),
} as const;

/** The ruined wall's heading, radians. Shared by the wall and its supply stack. */
const WALL_YAW = THREE.MathUtils.degToRad(24);

/**
 * Where the four seats round the fire go, as bearings *relative to the camera*.
 *
 * None of them is within 60 deg of the lens bearing, so the arc the camera
 * looks through stays open. Uneven spacing on purpose: four props at 90 deg
 * intervals is a compass rose and reads as placed.
 */
const FIRE_SEAT_BEARINGS: readonly number[] = [1.22, 2.44, 3.49, 4.66];

/**
 * Is this ground position on the camera's sightline to the Barbarian?
 *
 * A capsule from the hero back toward the eye, 1.5 m in radius. `backlit-
 * silhouette` exists to prove rim separation on the subject and its pose is
 * deterministic, so a prop landing in this volume is a reproducible way to
 * ruin the one shot that has to work. Cheap enough to run per candidate at
 * build time — a point-segment distance and one compare.
 */
function isOnHeroSightline(x: number, z: number): boolean {
  const ax = HERO.x;
  const az = HERO.y;
  const bx = CAMERA.position.x;
  const bz = CAMERA.position.z;
  const dx = bx - ax;
  const dz = bz - az;
  const t = THREE.MathUtils.clamp(((x - ax) * dx + (z - az) * dz) / (dx * dx + dz * dz), 0, 1);
  const cx = ax + dx * t;
  const cz = az + dz * t;
  return Math.hypot(x - cx, z - cz) < 1.5;
}

/** Standing height of the figure, in metres. The scene's scale reference. */
const HERO_HEIGHT = 1.85;

/** Half-width of the playable ground mesh, in metres. */
const TERRAIN_SIZE = 260;

/**
 * Every prop model the scene places, grouped by role.
 *
 * Listed as data so the loader can preload the whole set in one batch and
 * report exactly which keys failed — a missing prop must degrade the scene by
 * one object, never by throwing out of `init`.
 */
const PROP_KEYS = {
  rocks: ['nature.rock.a', 'nature.rock.c'],
  fence: ['prop.fence.wood', 'prop.fence.wood.gate'],
  clutter: [
    'prop.barrel.large',
    'prop.barrel.small',
    'prop.crate.large',
    'prop.crate.open',
    'prop.sack',
    'prop.bucket',
  ],
  rubble: ['prop.rubble.large', 'prop.rubble.half'],
  logs: ['prop.lumber'],
} as const satisfies Record<string, readonly AssetKey[]>;

const CHARACTER_KEY: AssetKey = 'character.barbarian';

/* -------------------------------------------------------------------------- */
/* Module                                                                     */
/* -------------------------------------------------------------------------- */

export interface BloodMoorOptions {
  /** Drive the camera every frame. Capture shots override the pose anyway. */
  readonly driveCamera?: boolean;
  readonly settings?: RenderSettings;
  /**
   * Animate the Barbarian from this module. Default true.
   *
   * `false` still loads, scales, weathers and places him — the composition
   * needs its scale reference either way — but leaves the clips unplayed and
   * publishes them on {@link BloodMoor.heroClips} instead, so that
   * `character/PlayerController` can take ownership of the figure. Two mixers
   * on one skeleton is the one arrangement that cannot work.
   */
  readonly controlHero?: boolean;
  /**
   * Load, scale and weather the Barbarian. Default true.
   *
   * `false` when the scene is hosted by `world/ZoneManager`: the player module
   * loads its own figure and keeps it across zone transitions, so building a
   * second one here produces two Barbarians standing in the same mud.
   */
  readonly loadHero?: boolean;
}

export class BloodMoor implements Zone {
  readonly name = 'scene.bloodMoor';
  readonly zoneId = 'bloodMoor';
  readonly displayName = 'The Blood Moor';

  /**
   * The moor's trim on the frame. See `ZoneGrade`.
   *
   * The smallest of the three, because the base look in `buildFrameGraph` was
   * authored *on* this zone and is very nearly right for it: the honest capture
   * came back cold, overcast and bleak, which is the brief. What it also came
   * back was bimodal — luma p25 **0.090**, p50 0.188, p75 **0.618**, with the
   * band between 0.25 and 0.60 essentially unpopulated. The sky owns the top of
   * the histogram, the ground owns the bottom, and there is nothing in between,
   * so the near ground on the shaded side of the frame goes to unreadable
   * black while the overcast deck sits five stops above it.
   *
   * A third of a stop plus a blue shadow lift moves the ground population up
   * into the 0.13-0.20 band without touching the sky, and dropping the contrast
   * pivot to the moor's real midtone stops the contrast operator from pushing
   * that same population further down. The result should still be bleak — it
   * has to be — but a player has to be able to see a skeleton standing in it.
   */
  readonly grade = {
    stops: 0.35,
    grade: {
      lift: [-0.002, 0.008, 0.056] as const,
      gamma: [1.0, 1.0, 1.035] as const,
      contrast: 1.06,
      contrastPivot: 0.2,
    },
  };

  readonly field = new TerrainField();

  /** The moor's encounter table, owned by `ai/EnemyDirector`. */
  readonly enemySpawns = BLOOD_MOOR_SPAWNS;

  readonly #options: BloodMoorOptions;
  readonly #root = new THREE.Group();
  readonly #disposables: { dispose(): void }[] = [];

  #assets: AssetManager | null = null;
  #materials: MaterialLibraryService | null = null;
  #ibl: IBLService | null = null;
  #lighting: LightingService | null = null;
  #volumetrics: VolumetricsService | null = null;

  #fireLight: LightHandle | null = null;
  #fireGlow: THREE.PointLight | null = null;
  #fireFlame: THREE.Mesh | null = null;
  /**
   * Flame radiance multiplier and flame clock.
   *
   * Uniforms driven from `ctx.time.elapsed`, and deliberately *not* TSL's
   * `time` node — that node reads the renderer's own wall clock, and under
   * `Engine.stepFrames(n)` the renderer clock is not the simulation clock. A
   * fire that flickers off wall time makes every capture a different picture.
   * This is the determinism contract, in two variables.
   */
  readonly #fireFlicker = uniform(1);
  readonly #fireTime = uniform(0);
  #mixer: THREE.AnimationMixer | null = null;
  #hero: THREE.Object3D | null = null;
  #heroClips: THREE.AnimationClip[] = [];
  #terrainSegments = 192;
  #scatterDensity = 1;
  /** Measured normalising scale per asset key; see `#normalisingScale`. */
  readonly #scaleCache = new Map<AssetKey, number>();
  /** Original prop material -> its weathered clone; see `#weatherMaterials`. */
  readonly #materialCache = new Map<THREE.Material, THREE.Material>();

  constructor(options: BloodMoorOptions = {}) {
    this.#options = options;
  }

  async init(ctx: GameContext): Promise<void> {
    this.#root.name = 'BloodMoor';
    ctx.scene.add(this.#root);

    // Boot-time truth table for the whole renderer. Printed from the scene
    // rather than from the frame graph because this is the first moment every
    // module has finished initialising.
    auditServices(ctx);

    this.#assets = ctx.services.tryGet(AssetManagerKey) ?? null;
    this.#materials = ctx.services.tryGet(MaterialLibraryKey) ?? null;
    this.#ibl = ctx.services.tryGet(IBLKey) ?? null;
    this.#lighting = ctx.services.tryGet(LightingKey) ?? null;
    this.#volumetrics = ctx.services.tryGet(VolumetricsKey) ?? null;
    this.#scatterDensity = this.#options.settings?.tier.scatterDensity ?? 1;

    if (this.#materials === null) {
      console.error(
        '[BloodMoor] no MaterialLibrary: the ground and the wall cannot be built. ' +
          'The frame will be props on nothing.',
      );
    }

    // Wait for the photoscanned sets to settle before building anything that
    // uses them. `MaterialLibrary` resolves textures asynchronously and swaps
    // them in when they arrive, so a material built before its set has landed
    // renders the procedural fallback — a flat, untextured surface — for as
    // long as it takes. That is invisible in a live session, where the swap
    // happens in the first second, and permanently visible in a capture, which
    // steps a fixed number of frames as fast as it can and photographs
    // whatever state the loader happens to be in.
    await this.#materials?.ready();

    this.#configureCamera(ctx.camera);
    this.#buildLightRig();
    this.#buildTerrain();
    this.#buildEscarpment();
    this.#buildDenMouth();
    this.#buildRuinedWall();
    this.#buildCampfire(ctx);

    // Before the props: the treeline is the composition's backdrop and it is
    // procedural, so it must not be gated on the asset fetch succeeding.
    this.#placeDeadTrees();

    await this.#loadProps();
    // The hero is the *player's*, not the zone's, and he survives travel. When
    // the zone system is driving, `PlayerController` owns him outright and a
    // second copy built here would stand motionless next to the one being
    // played. See {@link BloodMoorOptions.loadHero}.
    if (this.#options.loadHero !== false) await this.#loadHero();

    // Fold the AO service into every material this scene owns. Materials the
    // library made are not automatically hooked up: `IBLService.applyOcclusion`
    // is an explicit call so that a material which wants its own `aoNode`
    // (skin, foliage) can opt out.
    this.#applyOcclusion();

    console.info(
      `[BloodMoor] ready: ${countRenderables(this.#root)} renderables, ` +
        `scatter density ${this.#scatterDensity.toFixed(2)}`,
    );
  }

  update(ctx: GameContext, _dt: number): void {
    const t = ctx.time.elapsed;

    if (this.#options.driveCamera !== false) {
      // Static framing by default. The composition is the point of this scene,
      // and an orbiting camera would make every capture a different picture.
      ctx.camera.position.copy(CAMERA.position);
      ctx.camera.lookAt(CAMERA.target);
    }

    // Firelight flicker. Two incommensurable frequencies so it never falls
    // into an audible-looking beat, and a shallow depth: a campfire's output
    // varies, but a light that swings 50% reads as a strobe, not as fire.
    const flicker =
      1 + 0.085 * Math.sin(t * 7.31) + 0.055 * Math.sin(t * 11.97 + 1.7) + 0.03 * Math.sin(t * 19.3);
    this.#fireLight?.setIntensity(FIRE_INTENSITY * flicker);
    if (this.#fireGlow !== null) this.#fireGlow.intensity = FIRE_SCATTER_INTENSITY * flicker;
    // The emissive geometry flickers on the same curve as the light it stands
    // for. If the two drift apart the fire reads as a lamp with a fire painted
    // on it — the single most common failure of this effect.
    this.#fireFlicker.value = flicker;
    this.#fireTime.value = t;
    // The flame also *breathes*: a campfire's plume changes height, not just
    // brightness. Scaled on Y only, so the base stays welded to the ember bed.
    this.#fireFlame?.scale.set(1, 0.9 + 0.22 * flicker, 1);

    // `setTime`, not `update(dt)`: absolute time makes the pose a pure function
    // of the frame index, so `stepFrames(n)` lands on the same pose every run.
    this.#mixer?.setTime(t);
  }

  dispose(): void {
    this.#fireLight?.release();
    // The scatter glow is the one object this zone parents to `ctx.scene`
    // rather than to its own root, because it lives on a layer the camera does
    // not render and `RenderBridges` gathers it by walking the scene. Being
    // outside the root means `disposeZoneTree` cannot see it, so it has to be
    // reclaimed by hand — measured as two orphaned `campfire.scatter` lights
    // on the scene after two visits to the moor.
    if (this.#fireGlow !== null) {
      this.#fireGlow.removeFromParent();
      this.#fireGlow.dispose();
      this.#fireGlow = null;
    }
    this.#root.removeFromParent();
    disposeTree(this.#root);
    for (const disposable of this.#disposables) disposable.dispose();
    this.#disposables.length = 0;
    this.#mixer = null;
    this.#hero = null;
  }

  /* -- camera ------------------------------------------------------------- */

  #configureCamera(camera: THREE.PerspectiveCamera): void {
    camera.fov = CAMERA.fov;
    camera.near = CAMERA.near;
    // 900 m: the background ridge is 40 m out, but the sky dome and the
    // atmosphere's aerial-perspective LUT are parameterised against the far
    // plane, so clipping it tight would quantise the horizon.
    camera.far = CAMERA.far;
    camera.position.copy(CAMERA.position);
    camera.lookAt(CAMERA.target);
    camera.updateProjectionMatrix();
  }

  /* -- light rig ---------------------------------------------------------- */

  /**
   * The rim light: one cold directional source from behind the subject.
   *
   * This is the difference between a frame that is *illuminated* and a frame
   * that is *lit*, and it is the one thing the physically-derived sky rig
   * cannot produce on its own. Under a full overcast slab the key is a huge
   * soft dome, and a huge soft dome wraps every object evenly — which is
   * correct, and which is why the previous frame had a Barbarian the same value
   * as the mud behind him. Stylized art direction does not accept that. Every
   * Fortnite, Torchlight or Diablo III cinematic frame separates its subject
   * from its background with a hard cold backlight, and none of them justify it
   * physically.
   *
   * The justification available here is that a break in the cloud behind the
   * ridge is a real and much brighter patch of sky, so the light is aimed down
   * the camera's own axis from beyond the ridge, at 18° elevation. It is:
   *
   * - **cold** (≈ 8500 K blue-white), because it is skylight and because the
   *   whole palette rests on the fire being the only warm source;
   * - **shadowless**, deliberately — a rim light that casts is a second key,
   *   and two keys is the flattest lighting there is. It costs one N·L term;
   * - **weak in absolute terms** (0.55) but *directional*, so it lands on the
   *   top-and-back facing surfaces the sky dome and the fire both miss: the
   *   Barbarian's shoulders, the top edge of the wall, the upper limbs of the
   *   framing trees, and the whole ridge treeline at once.
   *
   * Cost is one extra directional light in the lighting loop — a few ALU per
   * shaded pixel, no shadow map, no extra draw call. It is the highest
   * art-direction return per instruction in the entire scene.
   */
  #buildLightRig(): void {
    const rim = new THREE.DirectionalLight(RIM_COLOR, RIM_INTENSITY);
    rim.name = 'rim.skybreak';
    // Behind and above the composition, on roughly the camera's own bearing so
    // the rim lands on the silhouette edges the lens can actually see. A rim
    // light 90° off the lens axis rims a contour nobody is looking at.
    // Elevation dropped from 21° to 16°. A rim that comes in steeply lands on
    // the tops of things, and the tops of things are not the silhouette the
    // lens sees. At 16° it rakes the vertical edges — the Barbarian's shoulder
    // and arm, the wall's broken end, the trunks on the ridge — which is where
    // the separation actually has to happen. Bearing is unchanged: it is
    // already within 8° of directly opposed to the camera axis, which is what
    // makes it a backlight rather than a second key.
    rim.position.set(-12.5, 11.0, -34.0);
    rim.target.position.set(2.0, 0.9, -2.0);
    rim.castShadow = false;
    this.#root.add(rim);
    this.#root.add(rim.target);

    // A second, much weaker cold fill from camera-left at ground level, aimed
    // up the bank. Its only job is to keep the foreground rubble and the
    // escarpment from crushing to solid black — grimdark shadows must retain
    // *colour*, and a shadow with no light in it has no colour to retain.
    const bounce = new THREE.DirectionalLight(SHADOW_FILL_COLOR, SHADOW_FILL_INTENSITY);
    bounce.name = 'rim.shadowFill';
    bounce.position.set(16.0, 2.2, 14.0);
    bounce.target.position.set(-6.0, 1.0, -8.0);
    bounce.castShadow = false;
    this.#root.add(bounce);
    this.#root.add(bounce.target);
  }

  /* -- ground ------------------------------------------------------------- */

  /**
   * The ground: one mesh, one height-blended material.
   *
   * The blend weight is the interesting part. `heightBlend2` in the material
   * library interlocks the two sets by their *mesostructure* height maps, so a
   * weight of 0.5 does not cross-fade — the grass fills the low spots in the
   * mud's surface first and the transition looks like grass growing through
   * mud rather than like two textures dissolving.
   *
   * The weight itself is `f(world height) · f(slope)`:
   *
   * - height: grass takes over above the water line. The basin floor is at
   *   about −2.4 m and the rises reach +2 m, so the transition band is placed
   *   at −1.4 → 0.6 m, which puts the mud/grass line partway up the sides of
   *   the hollow where standing water would actually have left it.
   * - slope: anything steeper than ~40° loses its topsoil. This is what keeps
   *   grass off the bank and what makes the escarpment read as continuous with
   *   the ground it cuts out of.
   *
   * Both are `smoothstep`, not `step`: a hard line here is the single most
   * obvious tell that a terrain is splat-mapped.
   */
  #buildTerrain(): void {
    const materials = this.#materials;
    // Segment count is a fixed 1.35 m spacing at the highest tier. Finer than
    // that and the normal map is doing the work anyway; coarser and the basin
    // rim faceted visibly against the fog.
    const segments = Math.max(48, Math.round(192 * Math.sqrt(this.#scatterDensity)));
    this.#terrainSegments = segments;
    const geometry = this.field.buildGeometry(TERRAIN_SIZE, segments);

    // The worn track. This is a *composition* element written into the ground
    // material rather than placed as geometry, and it is the strongest leading
    // line available: a bare, waterlogged trail that enters at the lower right
    // of the default frame, doglegs, and dies at the campfire. The eye follows
    // a value change along a curve without being told to, which is the whole
    // trick — the fence run does the same job in the other half of the frame
    // and the two converge on the fire.
    //
    // Two segments, not one, and they meet at an angle. A dead-straight path
    // across a moor reads as a road, and a road implies an engineer.
    const track = saturate(
      trackMask(TRACK[0], TRACK[1], 2.5, 0.9).max(trackMask(TRACK[1], TRACK[2], 2.1, 0.7)),
    );

    const grassCoverage = saturate(
      smoothstep(-1.4, 0.6, positionWorld.y)
        .mul(smoothstep(0.62, 0.88, normalWorldGeometry.y))
        // The track is scoured to mud. Not to zero: a trail through dead grass
        // keeps a ragged crown of it, and a hard-edged bare strip is exactly
        // the "painted decal" tell this whole terrain setup exists to avoid.
        .mul(oneMinus(track.mul(0.88))),
    );

    // Stylized, not photographic. The mask above is unchanged — height, slope
    // and the worn track still decide where cover grows — but it now selects
    // between authored flat colours instead of driving a two-archetype
    // triplanar PBR blend at 24 texture fetches per fragment.
    //
    // On a 260 m plane that blend was never going to work: one 2.2 m tile
    // stretched across the whole moor, broken up by macro noise that could hide
    // the periodicity but not the mush, with a 14-repeats-per-metre detail
    // normal aliasing on top of it. Flat colour with a crisp, noise-broken
    // boundary is both cheaper and more legible at this scale, and it matches
    // the low-poly geometry it sits on instead of fighting it.
    // See `render/materials/StylizedTerrain`.
    const material =
      materials?.createStylizedTerrain({ coverage: grassCoverage, name: 'moor.ground' }) ??
      new THREE.MeshStandardNodeMaterial({ color: 0x3a3a34, roughness: 0.9 });

    const ground = new THREE.Mesh(geometry, material);
    ground.name = 'ground';
    ground.receiveShadow = true;
    // The ground casts too: the basin rim shadowing its own floor at this sun
    // elevation is a large part of why the hollow reads as a hollow.
    ground.castShadow = true;
    this.#root.add(ground);
    this.#disposables.push(geometry, material);
  }

  /**
   * The west escarpment: real rock geometry, not a splat layer.
   *
   * Five overlapping displaced ellipsoids, flattened at the base so they sit
   * into the bank rather than balancing on it, all sharing one triplanar
   * `rock` material so the whole formation is a single draw call per mesh with
   * no UV seams anywhere. A sixth, `mossyRock`, breaks the tonal uniformity —
   * moss green is one of the three accent colours the art direction allows.
   */
  #buildEscarpment(): void {
    const materials = this.#materials;
    if (materials === null) return;

    const cliff = materials.get('rock');
    const mossy = materials.get('mossyRock');
    const rng = createRng(`${SCATTER_SEED}.escarpment`);

    const blocks: readonly {
      x: number;
      z: number;
      s: readonly [number, number, number];
    }[] = [
      { x: -13.5, z: -6.0, s: [5.4, 4.2, 6.8] },
      { x: -11.2, z: 1.8, s: [4.2, 3.4, 5.0] },
      { x: -15.8, z: 4.6, s: [6.0, 5.2, 5.6] },
      { x: -12.6, z: -13.5, s: [4.8, 4.6, 5.4] },
      { x: -17.4, z: -3.2, s: [7.2, 6.4, 8.0] },
      { x: -9.6, z: -19.0, s: [3.8, 3.0, 4.4] },
    ];

    blocks.forEach((block, index) => {
      const geometry = generateRockGeometry({
        seed: `${SCATTER_SEED}.cliff.${index}`,
        detail: 4,
        radius: 1,
        displacement: 0.42,
        scale: block.s,
        // Flat-bottomed: the underside is never seen and a flat base means the
        // rock can be sunk into the terrain without a visible intersection arc.
        flattenBase: 0.55,
      });
      const mesh = new THREE.Mesh(geometry, index === 2 ? mossy : cliff);
      mesh.name = `escarpment.${index}`;
      // Sunk by half its height, up from a third.
      //
      // A third was not enough, and the symptom was the frame's most persistent
      // artefact: hard-edged black and pale wedges scattered across the cliff
      // faces, which survived two separate rounds of shadow-bias increases
      // because they were never shadow acne. They were the *terrain mesh*
      // poking through the rock. `TerrainField`'s west bank is a smooth ramp
      // rising 4.6 m over 14 m, so a rock keyed to the height at its own centre
      // has ground well above its base at the uphill end, and the ground's
      // triangles cut straight out through the rock's faces. Sinking half the
      // height buries the intersection, and the rocks still read as bedrock
      // breaking through the bank rather than as boulders resting on it.
      mesh.position.set(block.x, this.field.heightAt(block.x, block.z) - block.s[1] * 0.5, block.z);
      mesh.rotation.y = rng.next() * Math.PI * 2;
      mesh.castShadow = true;
      mesh.receiveShadow = true;

      // The escarpment carries no colliders. The terrain does.
      //
      // `buildSceneColliders` fits an axis-aligned box to a mesh's bounding
      // box, and these are displaced ellipsoids up to 14 x 16 m across, sunk
      // half their height into the bank. Six of them derive six boxes that
      // together blanket the whole west bank from x = -21.8 to -7.0 and up to
      // 8 m tall — a wall of invisible geometry across the only approach to the
      // Den of Evil, with the `from-den` arrival point *inside* it. The
      // measured consequences: `findClearSpot` found no open ground within 5 m
      // and placed the player inside the box anyway, and the camera arm was
      // pinned at its floor on the frame that introduces the moor.
      //
      // A box cannot describe this shape. It cannot describe the cave mouth cut
      // into block 4 at all — a hole is precisely what an AABB has no way to
      // express. And the shape is already described somewhere better:
      // `TerrainField`'s west bank is a smooth ramp rising 4.6 m over 14 m and
      // the rocks are keyed into it (see the sinking note above), so the
      // heightfield the player actually walks on already carries the bank's
      // form. What is lost is the ability to bump into an individual boulder
      // face; what is gained is a reachable Den.
      mesh.userData['noCollide'] = true;

      this.#root.add(mesh);
      this.#disposables.push(geometry);
    });
  }

  /**
   * The ruined wall: parallax-mapped masonry, stepped down to a broken end.
   *
   * Each course is a box, and each box gets its own material instance with
   * `tiling` set from its world length so that the stone course height is
   * constant along the wall. Sharing one material would stretch the masonry on
   * the long blocks and squash it on the short ones, which is exactly the
   * artefact that makes modular kit walls look cheap.
   *
   * Parallax on a box is worth its cost here specifically because the wall is
   * seen at a raking angle from the default camera — parallax occlusion buys
   * almost nothing head-on and almost everything at 20° off the surface.
   */
  #buildRuinedWall(): void {
    const materials = this.#materials;
    if (materials === null) return;

    // Runs roughly north-south, left of frame, angled so its top edge is a
    // second leading line converging on the fire.
    const origin = new THREE.Vector3(-5.6, 0, -11.4);
    const yaw = THREE.MathUtils.degToRad(24);
    const courses: readonly {
      along: number;
      height: number;
      length: number;
      depth: number;
    }[] = [
      { along: -4.6, height: 3.4, length: 3.0, depth: 0.86 },
      { along: -1.5, height: 2.9, length: 3.2, depth: 0.86 },
      { along: 1.6, height: 1.9, length: 3.0, depth: 0.82 },
      { along: 4.1, height: 1.05, length: 2.0, depth: 0.78 },
      { along: 5.9, height: 0.5, length: 1.6, depth: 0.7 },
    ];

    const wallRng = createRng(`${SCATTER_SEED}.wallcrown`);
    const group = new THREE.Group();
    group.name = 'ruinedWall';
    group.rotation.y = yaw;
    group.position.set(origin.x, 0, origin.z);
    this.#root.add(group);

    // One metre of wall = one texture repeat. The masonry set is authored at
    // roughly a metre of coursing per tile, so this keeps stones life-sized.
    const REPEATS_PER_METRE = 0.85;

    courses.forEach((course, index) => {
      const geometry = new THREE.BoxGeometry(course.length, course.height, course.depth);
      const material = materials.create('wetStone', {
        tiling: course.length * REPEATS_PER_METRE,
        // Deeper parallax than the archetype default: this is rubble-cored
        // masonry with 40 mm of relief between stones, and it is the hero
        // surface of the material close-up shot.
        parallax: {
          scale: 0.06,
          maxSteps: 20,
          minSteps: 8,
          fadeEnd: 18,
          clipSilhouette: false,
        },
        // Deliberately *not* overriding the archetype tint any more. This
        // per-course override at 0.30 was silently defeating the wetStone
        // archetype's own value (now 0.13) — two places setting the same
        // number, one of them out of date, which is how the wall ended up
        // measuring brighter than the sky after the archetype had already been
        // darkened twice. One authority per decision.
      });
      const mesh = new THREE.Mesh(geometry, material);
      mesh.name = `ruinedWall.${index}`;
      const worldX = origin.x + Math.cos(yaw) * course.along;
      const worldZ = origin.z - Math.sin(yaw) * course.along;
      const base = this.field.heightAt(worldX, worldZ);
      // Sunk 0.25 m so the footing disappears into the mud instead of showing
      // a hard line where the box meets the heightfield.
      mesh.position.set(course.along, base + course.height / 2 - 0.25, 0);
      mesh.rotation.z = THREE.MathUtils.degToRad((index % 2 === 0 ? 1 : -1) * (0.8 + index * 0.35));
      mesh.castShadow = true;
      mesh.receiveShadow = true;
      group.add(mesh);
      this.#disposables.push(geometry, material);

      // **Break the top.**
      //
      // Everything above is about the wall's *surface*. None of it addresses
      // the fact that an extruded box has a dead-straight top edge, and against
      // a bright overcast sky that straight edge is the only thing about the
      // wall the eye actually resolves at distance. A silhouette cannot be
      // fixed with a texture: in stylized work the silhouette *is* the read,
      // and a ruin whose outline is a perfect horizontal is not a ruin.
      //
      // So the top course is re-cut into 0.5 m capstones with randomised
      // heights and randomised gaps — some standing proud, some slumped, some
      // missing entirely. Deterministic from the scatter seed, so the frame is
      // the same every capture. Cost is five to seven extra boxes per course
      // sharing the course's own material: no new draw call after batching, and
      // twelve triangles each.
      const stoneWidth = 0.5;
      const stoneCount = Math.max(2, Math.round(course.length / stoneWidth));
      for (let c = 0; c < stoneCount; c++) {
        // Two missing per course on average. A gap in a wall crown is what
        // makes the remaining stones read as stones rather than as crenellation.
        if (wallRng.next() < 0.3) continue;
        const rise = 0.1 + wallRng.next() * 0.24;
        const capGeometry = new THREE.BoxGeometry(
          stoneWidth * (0.82 + wallRng.next() * 0.16),
          rise,
          course.depth * (0.78 + wallRng.next() * 0.2),
        );
        const cap = new THREE.Mesh(capGeometry, material);
        cap.name = `ruinedWall.${index}.cap.${c}`;
        cap.position.set(
          course.along + (c + 0.5 - stoneCount / 2) * stoneWidth,
          base + course.height - 0.25 + rise * 0.5 - 0.04,
          (wallRng.next() - 0.5) * 0.12,
        );
        cap.rotation.set(
          (wallRng.next() - 0.5) * 0.12,
          (wallRng.next() - 0.5) * 0.22,
          (wallRng.next() - 0.5) * 0.16,
        );
        cap.castShadow = true;
        cap.receiveShadow = true;
        group.add(cap);
        this.#disposables.push(capGeometry);
      }
    });
  }

  /* -- campfire ----------------------------------------------------------- */

  /**
   * The focal point, and the entire reason the frame has any colour contrast.
   *
   * Four things make the fire read, and all four are needed:
   *
   * 1. a clustered {@link LightingService} light, for correct falloff on the
   *    Barbarian and the wall;
   * 2. a `THREE.PointLight` in the scene, which is what the volumetric bridge
   *    enumerates — this is the light that scatters in the fog and gives the
   *    fire its halo. Its *radiometric* contribution is left to the clustered
   *    light: this one carries the same intensity so the two agree, and the
   *    clustered rig parks rather than duplicates when both are present;
   * 3. a local fog volume, so the air immediately around the fire is denser
   *    and warmer than the surrounding mist;
   * 4. an emissive core, so the source itself is not a black hole in the
   *    middle of the brightest thing in the picture.
   */
  #buildCampfire(ctx: GameContext): void {
    const materials = this.#materials;
    const x = CAMPFIRE.x;
    const z = CAMPFIRE.y;
    const base = this.field.heightAt(x, z);
    const group = new THREE.Group();
    group.name = 'campfire';
    group.position.set(x, base, z);
    this.#root.add(group);

    // Stone ring.
    if (materials !== null) {
      // Not the shared `rock` material: these nine stones have had a fire in
      // them. Sooted almost black, so the ring reads as a dark socket holding
      // the one bright thing in the picture. A ring of clean bright boulders
      // around a fire competes with the flame for the eye and wins, because it
      // is bigger.
      const stone = materials.create('rock', {
        albedoTint: [0.115, 0.108, 0.105],
        roughnessRange: [0.72, 1],
      });
      this.#disposables.push(stone);
      const rng = createRng(`${SCATTER_SEED}.firering`);
      for (let i = 0; i < 9; i++) {
        const angle = (i / 9) * Math.PI * 2 + rng.next() * 0.22;
        const radius = 0.72 + rng.next() * 0.09;
        const geometry = generateRockGeometry({
          seed: `${SCATTER_SEED}.firestone.${i}`,
          detail: 2,
          radius: 0.14 + rng.next() * 0.06,
          displacement: 0.34,
          flattenBase: 0.4,
        });
        const mesh = new THREE.Mesh(geometry, stone);
        mesh.position.set(Math.cos(angle) * radius, 0.03, Math.sin(angle) * radius);
        mesh.rotation.set(rng.next() * 0.4, rng.next() * 6.28, rng.next() * 0.4);
        mesh.castShadow = true;
        mesh.receiveShadow = true;
        group.add(mesh);
        this.#disposables.push(geometry);
      }
    }

    // Charred logs. Three of them, crossed, at a shallow angle: the fire needs
    // a *structure* to burn out of. A flame with nothing under it is a decal,
    // and a decal is what the previous blown-out ellipse read as.
    if (materials !== null) {
      const charred = materials.create('bark', {
        ...DEAD_WOOD_OVERRIDES,
        // Darker still than the treeline, and glowing where it faces the
        // embers. This is the only emissive PBR surface in the scene and it is
        // what makes the base of the fire read as burning rather than as lit.
        albedoTint: [0.08, 0.075, 0.075],
      });
      this.#disposables.push(charred);
      const logRng = createRng(`${SCATTER_SEED}.firelogs`);
      for (let i = 0; i < 3; i++) {
        const geometry = new THREE.CylinderGeometry(0.055, 0.075, 1.05, 6, 1);
        // Laid down and splayed, one end raised on the ring: a teepee lay.
        geometry.rotateZ(Math.PI / 2);
        const mesh = new THREE.Mesh(geometry, charred);
        mesh.name = `campfire.log.${i}`;
        mesh.position.set(0, 0.1 + logRng.next() * 0.05, 0);
        mesh.rotation.set(0, (i / 3) * Math.PI * 2 + logRng.next() * 0.4, 0.16 + logRng.next() * 0.1);
        mesh.castShadow = true;
        mesh.receiveShadow = true;
        group.add(mesh);
        this.#disposables.push(geometry);
      }
    }

    // The ember bed: a shallow disc under the flame, hot in the middle and
    // dying to black coal at the rim.
    const emberGeometry = new THREE.CircleGeometry(0.42, 20);
    emberGeometry.rotateX(-Math.PI / 2);
    const emberRadial = positionLocal.xz.length().div(0.42).clamp(0, 1);
    // Additive and depth-write-off, like the flame. The bed used to be an
    // opaque disc, and an opaque disc has a polygon edge — which is precisely
    // the "hard-edged horizontal bright bar under the logs" the critique
    // measured. No radiance ramp can hide a silhouette; the silhouette has to
    // stop existing.
    const emberMaterial = new THREE.MeshBasicNodeMaterial({
      toneMapped: true,
      transparent: true,
      depthWrite: false,
      blending: THREE.AdditiveBlending,
    });
    // A profile with *no plateau anywhere*, times a squared radial vanish.
    //
    // `smoothstep(0.18, 0.95)` held a constant peak across the inner 18% of
    // the disc, and a constant peak is a clipped core no matter how bright it
    // is — that is what read as a flat cream ellipse at 1:1. The `pow(0.55)`
    // ramp is monotonic from r = 0 outward, so every single texel of the bed
    // differs from its neighbour and the tone curve has a gradient to roll off.
    // `(1 - r^2)^2` then drives the edge to *exactly* zero rather than to a
    // small non-zero value cut short by geometry.
    const emberFalloff = oneMinus(emberRadial.mul(emberRadial)).clamp(0, 1);
    // Peak scene-linear radiance 5.2. Measured: at 3.8 the fire core resolved to
    // 204/255 while the overcast sky sat at 211, so the focal point of the
    // composition was not the brightest thing in its own frame. A fire has to
    // win that comparison — the sky is a 0.8-value field covering a third of the
    // image and the fire is a forty-pixel emitter, and if the emitter is not
    // clearly hotter the eye goes to the field. The tone curve still rolls it
    // off rather than clipping, which is what the plateau-free radial profile
    // below exists to guarantee.
    const emberBody = mix(vec3(5.2, 1.7, 0.4), vec3(0.14, 0.02, 0.008), emberRadial.pow(0.55));
    // Coal structure: the bed is not a light, it is a hundred embers at
    // different temperatures. Two octaves of world-stable noise, scrolled very
    // slowly, so the hot spots migrate the way a real bed's do.
    const coalSpace = positionLocal.mul(11.0).add(vec3(0, this.#fireTime.mul(0.5), 0));
    const coals = mx_noise_float(coalSpace)
      .mul(0.42)
      .add(mx_noise_float(coalSpace.mul(2.4).add(vec3(3, 0, 9))).mul(0.2))
      .add(1);
    emberMaterial.colorNode = vec4(
      emberBody.mul(emberFalloff.mul(emberFalloff)).mul(saturate(coals)).mul(this.#fireFlicker),
      float(1),
    );
    const ember = new THREE.Mesh(emberGeometry, emberMaterial);
    ember.name = 'campfire.ember';
    ember.position.y = 0.09;
    // After the opaque pass, before the flame.
    ember.renderOrder = 9;
    group.add(ember);
    this.#disposables.push(emberGeometry, emberMaterial);

    // The flame. A tapered open cone with a vertical radiance ramp, drawn
    // additively with no depth write.
    //
    // Additive is correct for an emitter and it is also what gives the flame
    // its shape for free: where two walls of the cone overlap — the middle,
    // seen edge-on — the contributions sum, so the core is roughly twice the
    // radiance of the flanks without any extra geometry. A tapering,
    // vertically-graded, self-overlapping emitter is a readable flame; a
    // constant-radiance blob is not, no matter what value it is set to.
    // Smaller than it was, twice. A 0.9 m plume on a 1.4 m fire ring is a
    // bonfire, not a campfire, and the extra size was most of why the mesh read
    // as a paper cone: the more screen area a soft emitter covers, the more it
    // has to be *structured* to stay convincing.
    const flameGeometry = new THREE.CylinderGeometry(0.012, 0.15, 0.6, 9, 6, true);
    flameGeometry.translate(0, 0.3, 0);
    const flameHeight = positionLocal.y.div(0.6).clamp(0, 1);
    const flameMaterial = new THREE.MeshBasicNodeMaterial({
      transparent: true,
      depthWrite: false,
      side: THREE.DoubleSide,
      blending: THREE.AdditiveBlending,
      toneMapped: true,
    });
    // Base → tip: white-hot orange, to sodium orange, to the deep red where
    // combustion is failing. The last stop is nearly black so the flame ends in
    // air rather than in a cut edge.
    const flameColour = mix(
      mix(vec3(1.18, 0.44, 0.115), vec3(0.68, 0.185, 0.025), smoothstep(0.0, 0.4, flameHeight)),
      vec3(0.12, 0.011, 0.002),
      smoothstep(0.4, 1.0, flameHeight),
    );
    // Two incommensurable travelling waves up the flame, so the licks never
    // repeat on a visible beat, times the global flicker.
    const lick = sin(flameHeight.mul(9.4).sub(this.#fireTime.mul(6.1)))
      .mul(0.5)
      .add(sin(flameHeight.mul(15.7).sub(this.#fireTime.mul(9.3))).mul(0.3))
      .mul(0.5)
      .add(0.75);
    // Alpha starts falling immediately and is gone well before the tip, and it
    // peaks at 0.62, not 1. Both corrections come from the same mistake: an
    // additive emitter at alpha 1 with two overlapping walls sums to twice its
    // own radiance over the whole cone, which is not a flame, it is a lamp
    // shade. A flame is *mostly transparent* — you see the world through the
    // top two-thirds of one — and its opacity is concentrated in the first
    // couple of decimetres above the fuel.
    const flameAlpha = oneMinus(smoothstep(0.02, 0.86, flameHeight))
      .mul(lick)
      .mul(this.#fireFlicker)
      // **The silhouette dissolve.** Everything above shapes the flame's
      // radiance; none of it removes the fact that a swept surface has a
      // tangent silhouette, and a tangent silhouette on an emitter is a hard
      // cut edge — the "flat beige trapezoid with a straight diagonal edge"
      // the critique measured. `|N·V|` is 1 where the cone faces the lens and
      // 0 exactly along that silhouette, so raising it to a power and
      // multiplying it into alpha drives the outline to nothing while leaving
      // the body untouched. It is the same term a soft-particle billboard uses,
      // evaluated per fragment for two ALU and no depth-buffer fetch.
      .mul(smoothstep(0.0, 0.55, normalView.dot(positionViewDirection).abs()).pow(1.35))
      // And the same treatment at the base, where the cone otherwise cuts a
      // straight line through the logs it is supposed to be burning out of.
      // The ember bed underneath now carries the radiance across that gap.
      .mul(smoothstep(0.0, 0.16, flameHeight))
      .mul(0.72);
    flameMaterial.colorNode = vec4(flameColour.mul(flameAlpha), flameAlpha);
    // Break the cone.
    //
    // Everything above fixes the flame's *radiance*; none of it fixes the fact
    // that a cone has a perfectly straight edge, and a perfectly straight edge
    // is why the previous version still read as a paper lampshade rather than
    // as fire. The silhouette has to move. A gradient-noise field sampled in
    // the flame's own space and scrolled downward — so the licks appear to rise
    // — displaces each ring radially, scaled by height so the base stays welded
    // to the fuel and the tip whips. Two octaves: one for the body's lean, one
    // for the fine tongues.
    //
    // Done in the vertex stage, on 9x5 rings: 54 vertices of noise per frame
    // for the entire effect, which is nothing next to the fragment cost of the
    // additive overdraw it sits inside.
    const wobbleSpace = positionLocal
      .mul(vec3(3.1, 1.7, 3.1))
      .add(vec3(0, this.#fireTime.mul(-2.6), 0));
    const wobble = mx_noise_float(wobbleSpace)
      .mul(0.19)
      .add(mx_noise_float(wobbleSpace.mul(2.7).add(vec3(11, 0, 5))).mul(0.11))
      .add(mx_noise_float(wobbleSpace.mul(6.1).add(vec3(4, 0, 17))).mul(0.05));
    flameMaterial.positionNode = positionLocal.add(
      vec3(positionLocal.x, 0, positionLocal.z).mul(wobble.mul(flameHeight.add(0.25)).mul(9)),
    );

    const flame = new THREE.Mesh(flameGeometry, flameMaterial);
    flame.name = 'campfire.flame';
    flame.position.y = 0.1;
    // Renders after the opaque pass, and never occludes the logs behind it.
    flame.renderOrder = 10;
    group.add(flame);
    this.#fireFlame = flame;
    this.#disposables.push(flameGeometry, flameMaterial);

    // Clustered light: the correct one for surface shading.
    this.#fireLight =
      this.#lighting?.addLight({
        kind: 'point',
        name: 'campfire',
        // Raised from 0.45 m to 0.62 m. Inverse-square is brutal at this
        // scale: at 0.45 m the ring stones sat about 0.3 m from a point source
        // and were being lit ten times harder than the ground a metre away, so
        // a ring of *sooted* stone came back as the brightest object in the
        // shot. Lifting the source into the flame body — which is also where it
        // physically belongs — flattens the falloff across the ring.
        position: { x, y: base + 0.62, z },
        color: FIRE_COLOR,
        intensity: FIRE_INTENSITY,
        // 9 m, from 7. The cutoff radius is a culling bound, not a falloff
        // law: at 7 m the clustered rig stopped evaluating the light exactly
        // where the warm pool was still worth about 10% of the ambient, and
        // that hard boundary is visible as a ring on the ground once the
        // intensity is high enough for the pool to reach it at all.
        radius: 9,
        decay: 2,
        castShadow: true,
        // Highest priority in the scene: if the clustered rig ever has to drop
        // a shadowed light, this must not be the one it drops.
        priority: 10,
      }) ?? null;

    // Scene light: the one the volumetric bridge finds. It must carry the real
    // intensity (the bridge filters on `intensity > 0` and reads the value to
    // scale in-scatter) and it must not shade a single surface, because the
    // clustered light above is already doing that. Two full-intensity lights at
    // the same position is a factor-of-two error in the fire's radiometry, and
    // it is what let a 6 m campfire tint a 50 m basin crimson.
    //
    // `Renderer._projectObject` layer-tests lights exactly as it does meshes,
    // so parking this one on a layer the camera does not render removes it from
    // the render list and therefore from shading. `Scene.traverseVisible`,
    // which is how `RenderBridges` gathers volumetric lights, ignores layers
    // entirely — so the bridge still finds it. `visible = false` would not
    // work: that *is* what `traverseVisible` filters on.
    const glow = new THREE.PointLight(new THREE.Color(FIRE_COLOR), FIRE_SCATTER_INTENSITY, 9, 2);
    glow.name = 'campfire.scatter';
    glow.position.set(x, base + 0.62, z);
    glow.castShadow = false;
    glow.layers.set(VOLUMETRIC_ONLY_LAYER);
    ctx.camera.layers.disable(VOLUMETRIC_ONLY_LAYER);
    ctx.scene.add(glow);
    this.#fireGlow = glow;

    // Warm, denser air around the fire. `emissive` is what makes the smoke
    // column above it glow from the inside rather than just being lit.
    this.#volumetrics?.addFogVolume({
      shape: 'sphere',
      name: 'campfire.smoke',
      center: { x, y: base + 1.5, z },
      radius: 3.4,
      densityScale: 1.9,
      color: 0xb8a894,
      emissive: 0x150802,
      feather: 0.7,
      priority: 5,
    });
  }

  /* -- props -------------------------------------------------------------- */

  /**
   * Load the prop set and place it.
   *
   * A failed key costs one prop, never the scene: `preload` reports failures
   * and every placement re-checks before cloning. This matters because the
   * asset set is fetched by a separate script — a fresh clone that has not run
   * `npm run assets` must still boot to a lit, composed frame.
   */
  async #loadProps(): Promise<void> {
    const assets = this.#assets;
    if (assets === null) {
      console.warn('[BloodMoor] no AssetManager: the scene will have terrain but no props.');
      return;
    }

    const keys = Object.values(PROP_KEYS).flat();
    const result = await assets.preload(keys);
    if (result.failed.length > 0) {
      console.warn(
        `[BloodMoor] ${result.failed.length}/${keys.length} prop models failed to load ` +
          `(${result.failed.join(', ')}); those placements are skipped. ` +
          'Run `npm run assets` to fetch them.',
      );
    }

    const available = new Set(result.loaded);
    const pick = (group: readonly AssetKey[]): AssetKey[] =>
      group.filter((key) => available.has(key));

    this.#placeRocks(pick(PROP_KEYS.rocks));
    this.#placeFence(pick(PROP_KEYS.fence));
    this.#placeCampsiteClutter(pick(PROP_KEYS.clutter), pick(PROP_KEYS.logs));
    this.#placeRubble(pick(PROP_KEYS.rubble));
  }

  /**
   * The treeline: procedural dead hardwoods, drawn as four instanced batches.
   *
   * The kit conifers this replaces were the single loudest failure in the
   * frame. They were saturated primary green, which broke the palette rule the
   * whole scene is built on, and — worse, and unfixable by any tint — they were
   * *solid cones*. A cone has no silhouette. The reason a Torchlight or Fable
   * skyline reads at a glance is that the trees on it are dark broken shapes
   * with sky visible through them, and that gap structure is what makes the
   * aerial-perspective gradient behind the ridge legible at all. See
   * {@link module:scene/DeadTree}.
   *
   * Placement is two passes, because the trees do two different jobs:
   *
   * - **the ridge**, scattered, density rising sharply past z = −18 and cut to
   *   near zero over the basin so nothing stands in front of the fire. This is
   *   background: it exists to be a silhouette.
   * - **the framing pair**, placed by hand at the left and right edges of the
   *   midground and scaled up. These are the composition's proscenium arch —
   *   they close the frame at the sides and their limbs lean inward, over the
   *   camera's sightline to the fire. Scattering cannot produce this; it is a
   *   decision, so it is written down as one.
   *
   * Cost: four geometries at roughly 3–5 k triangles each, drawn as four
   * `InstancedMesh` batches plus two framing meshes. At 1080p on a 2020
   * mid-range discrete part this is six draw calls and about 300 k triangles
   * for the entire treeline, an order of magnitude under the per-frame budget a
   * 60 fps target allows for background dressing; the shadow pass doubles the
   * draw count and the geometry is thin enough that its cascade-0 coverage is
   * negligible.
   */
  #placeDeadTrees(): void {
    // **The treeline deliberately does not use the photoscanned bark archetype.**
    //
    // It used to, and the value would not come down. Three separate rounds of
    // `albedoTint` and `envMapIntensity` cuts left the ridge trunks measuring
    // 159/255 against a 209/255 sky — a 1.3 : 1 separation, where a silhouette
    // needs at least 2.5 : 1. The archetype stack is a *surface* description:
    // triplanar bark, detail normal, macro variation, wetness, parallax. Every
    // one of those is a multiplier or an additive term with its own opinion
    // about the final value, which is exactly what is wanted on a rock the
    // player can walk up to and exactly what cannot be tolerated on the one
    // class of geometry in this frame whose correct rendering is a controlled
    // near-black.
    //
    // At 40–200 m none of that surface detail is resolvable anyway — the ridge
    // treeline is two pixels of trunk against sky. So it gets a plain material
    // with one authored albedo and a hard-capped environment response, and the
    // value becomes a number this file owns outright instead of the output of a
    // six-stage pipeline. The geometry's baked grey vertex ramp still
    // multiplies through, which is what grounds the trunks into the mud and
    // stops any two trees being quite the same value.
    const bark = new THREE.MeshStandardNodeMaterial({
      color: new THREE.Color(DEAD_WOOD_ALBEDO[0], DEAD_WOOD_ALBEDO[1], DEAD_WOOD_ALBEDO[2]),
      roughness: 0.94,
      metalness: 0,
    });
    bark.name = 'deadWood.silhouette';
    // The neutral grey ramp baked into the geometry (dark at the base,
    // lightening toward the twigs, ±12% per limb) multiplies in through
    // `NodeMaterial`'s stock vertex-colour path, after the library's colour
    // node. This is why the trunks ground into the mud instead of floating on
    // it, and why no two trees are quite the same value.
    bark.vertexColors = true;
    // 0.28. Bare limbs are the most sky-exposed geometry in the scene — a
    // cylinder always presents some of itself to the zenith, and nothing
    // occludes a tree on a ridge — so at a normal environment response they
    // came back at roughly half the sky's own value: a pale blue-grey tree
    // against a pale grey sky, which is not a silhouette, it is camouflage. The
    // fix is to cut the *environment* term specifically rather than the albedo,
    // because the albedo is what the rim light and the fire still have to work
    // against, and darkening that would kill the rim too.
    // 0.10, from 0.28. Same argument as the albedo and it has to be made in
    // both places: the albedo is what the rim light and the fire work against,
    // the environment term is what the *sky* relights the limbs with, and it is
    // the sky the trees have to stay dark against. Cutting only one of the two
    // leaves the other free to bring the value straight back up, which is what
    // happened last pass.
    // 0.06. The environment term is what the *sky* relights the limbs with, and
    // the sky is precisely what they have to stay dark against. Capping it here
    // is the other half of owning the value outright.
    bark.envMapIntensity = 0.06;
    this.#disposables.push(bark);

    const geometries = DEAD_TREE_VARIANTS.map((variant, index) =>
      generateDeadTreeGeometry({
        ...variant,
        seed: `${SCATTER_SEED}.deadtree.${index}`,
      }),
    );
    for (const geometry of geometries) this.#disposables.push(geometry);

    /* -- the ridge -------------------------------------------------------- */

    const samples = scatter({
      // Concentrated, not spread. At `area: 130` centred on (-4, -40) the
      // scatter was seeding a 260 m square, and the composition camera is a
      // 34 deg lens — the overwhelming majority of those trees landed outside
      // the frustum, which is why a 96-tree "treeline" arrived on screen as
      // four twigs. The band is now sized to the crest the lens actually sees.
      count: Math.round(165 * this.#scatterDensity),
      area: 62,
      center: new THREE.Vector3(4, 0, -36),
      seed: `${SCATTER_SEED}.trees`,
      // Wider than the old 0.75–1.5. Dead trees have no canopy to give the
      // scale away, so a big range reads as varied species and age rather than
      // as a scaling bug — and the tall outliers are what break the top edge of
      // the treeline into something other than a hedge.
      scaleRange: [0.85, 1.9],
      surface: this.field.surface,
      maxSlopeDegrees: 34,
      minSpacing: 2.9,
      density: (x, z) => {
        // Pushed back from (-12, -32). The band has to straddle the *crest* of
        // the background ridge, not its near face: trees on the near face sit
        // below the horizon line and read as scrub, and the entire job of this
        // treeline is to put a broken edge between the land and the sky. The
        // crest of `TerrainField`'s authored ridge is around z = -40, so the
        // density peaks there.
        const ridge = smootherstep(-17, -33, z);
        const clearing = 1 - 0.94 * Math.exp(-((x - 1) ** 2 + (z + 4) ** 2) / 300);
        // Nothing grows on the escarpment. The slope reject alone did not do
        // it: `generateRockGeometry` builds real rock volumes that sit *above*
        // the heightfield, and the heightfield under them is a smooth bank at
        // well under 34°, so trees were being accepted on ground that turns out
        // to have six metres of granite standing on it. The result was limbs
        // erupting out of the cliff face. The exclusion has to be authored
        // against the rock's footprint, because the terrain function does not
        // know the rock is there.
        const escarpment = smootherstep(-1.0, 4.0, x + 8.0);
        return ridge * clearing * escarpment;
      },
    });

    // Deal the samples round-robin by *position hash*, not by index: dealing by
    // index correlates the variant with the scatter's traversal order, which on
    // a jittered grid is a visible diagonal banding of tree shapes.
    const buckets: ScatterSample[][] = geometries.map(() => []);
    for (const sample of samples) {
      const hash = Math.abs(Math.round(sample.position.x * 7.3 + sample.position.z * 13.1));
      const bucket = buckets[hash % geometries.length];
      // Per-instance non-uniform height, hashed from the same position so it is
      // deterministic and uncorrelated with the variant choice. 0.72-1.46 is a
      // wide enough proportion range that two instances of one geometry do not
      // read as the same tree; see `ScatterSample.stretch`.
      const stretch = 0.72 + (((hash * 2654435761) % 1000) / 1000) * 0.74;
      bucket?.push({ ...sample, stretch, index: bucket.length });
    }

    geometries.forEach((geometry, index) => {
      const bucket = buckets[index];
      if (bucket === undefined || bucket.length === 0) return;
      const mesh = buildInstancedMesh(geometry, bark, bucket);
      mesh.name = `deadTrees.${index}`;
      this.#root.add(mesh);
    });

    /* -- the framing pair -------------------------------------------------- */

    // Left and right of the default camera's sightline, close enough to the
    // lens that they are cut by the frame edge. A framing tree that fits
    // entirely inside the frame is a prop; one that runs off the top corner is
    // architecture.
    const framing: readonly {
      x: number;
      z: number;
      variant: number;
      scale: number;
      yaw: number;
    }[] = [
      { x: -7.6, z: 5.4, variant: 1, scale: 1.35, yaw: 2.35 },
      { x: 13.4, z: 2.4, variant: 0, scale: 1.18, yaw: 0.6 },
      // A third, further back and smaller, on the fire's far side: it stops
      // the right-hand framing tree from reading as a lone bookend and gives
      // the midground a second depth step.
      { x: 9.4, z: -14.6, variant: 2, scale: 1.1, yaw: 4.1 },
    ];

    for (const [index, place] of framing.entries()) {
      const geometry = geometries[place.variant];
      if (geometry === undefined) continue;
      const mesh = new THREE.Mesh(geometry, bark);
      mesh.name = `deadTree.framing.${index}`;
      mesh.position.set(place.x, this.field.heightAt(place.x, place.z) - 0.15, place.z);
      mesh.rotation.y = place.yaw;
      mesh.scale.setScalar(place.scale);
      mesh.castShadow = true;
      mesh.receiveShadow = true;
      this.#root.add(mesh);
    }
  }

  /** Boulders, everywhere but the campsite — including the foreground shelf. */
  #placeRocks(keys: readonly AssetKey[]): void {
    if (keys.length === 0) return;
    const samples = scatter({
      count: Math.round(64 * this.#scatterDensity),
      area: 90,
      center: new THREE.Vector3(-2, 0, -8),
      seed: `${SCATTER_SEED}.rocks`,
      scaleRange: [0.5, 2.1],
      surface: this.field.surface,
      maxSlopeDegrees: 52,
      minSpacing: 2.1,
      alignToNormal: true,
      density: (x, z) =>
        (1 - 0.95 * Math.exp(-((x - CAMPFIRE.x) ** 2 + (z - CAMPFIRE.y) ** 2) / 22)) *
        // Same escarpment carve-out as the treeline, and for the same reason:
        // the heightfield under the cliff is a gentle bank, so the slope reject
        // happily accepts placements that turn out to be inside six metres of
        // granite.
        smootherstep(1.0, 6.0, x + 8.0),
    });
    this.#instance(keys, samples, 'rock', {
      alignToGround: 1,
      shadow: true,
      sink: 0.12,
      targetHeight: 0.95,
    });
  }

  /**
   * The leading line.
   *
   * Not scattered — walked. Fence posts are placed along an explicit polyline
   * from the lower-left foreground back toward the fire, with a gate section
   * partway along and two deliberate gaps where the run is "collapsed". A
   * scattered fence is not a fence; the whole point of this object is that the
   * eye follows it.
   *
   * Each section is given a random lean of up to 9° and a random 4° yaw
   * deviation. Perfectly aligned fence posts read as a CAD model.
   */
  #placeFence(keys: readonly AssetKey[]): void {
    if (keys.length === 0) return;
    const rng = createRng(`${SCATTER_SEED}.fence`);
    // **Re-laid so that it is actually a leading line.**
    //
    // The previous run went (13.5, 15.5) -> (8.9, -12.8): almost exactly down
    // the camera's own bearing at a constant 4 m to its right, which projects
    // to a near-vertical line hard against the right frame edge, receding away
    // from the fire. That is not a leading line, it is a fence post border.
    //
    // A leading line has to *converge* on the focal point in screen space. This
    // path is authored in the camera's basis rather than in world XZ: each
    // control point sits at a chosen depth along the view axis and a chosen
    // fraction of the frame half-width to the right of centre, and that
    // fraction falls monotonically — 100%, 83%, 65%, 50% — while the depth
    // rises, so the run enters at the lower-right corner and drives diagonally
    // up-and-left to arrive at the fire, which sits at 32% right.
    //
    // It stops about 2.3 m short of the ring. The eye completes a line that is
    // interrupted just before its destination far more strongly than one that
    // is drawn all the way there, and a fence running into a fire is a fire
    // hazard rather than a composition.
    const path: readonly THREE.Vector2[] = [
      new THREE.Vector2(11.2, 9.6),
      new THREE.Vector2(9.6, 6.0),
      new THREE.Vector2(8.4, 2.6),
      new THREE.Vector2(7.0, -0.4),
      new THREE.Vector2(5.5, -2.6),
    ];
    const gate = keys.find((key) => key.includes('gate'));
    const plain = keys.find((key) => !key.includes('gate')) ?? keys[0];
    if (plain === undefined) return;

    const samples: ScatterSample[] = [];
    const chosen: AssetKey[] = [];
    let index = 0;
    const SECTION = 1.9;

    for (let segment = 0; segment + 1 < path.length; segment++) {
      const a = path[segment];
      const b = path[segment + 1];
      if (a === undefined || b === undefined) continue;
      const span = a.distanceTo(b);
      const steps = Math.max(1, Math.round(span / SECTION));
      for (let step = 0; step < steps; step++) {
        // Two gaps, at fixed indices rather than random ones: a ruin needs its
        // damage composed too, and these two are where the eye should skip.
        if (index === 4 || index === 9) {
          index++;
          continue;
        }
        const t = step / steps;
        const x = THREE.MathUtils.lerp(a.x, b.x, t);
        const z = THREE.MathUtils.lerp(a.y, b.y, t);
        const heading = Math.atan2(b.x - a.x, b.y - a.y);
        samples.push({
          position: new THREE.Vector3(x, this.field.heightAt(x, z) - 0.08, z),
          normal: this.field.normalAt(x, z),
          scale: 0.95 + rng.next() * 0.12,
          rotation: heading + (rng.next() - 0.5) * 0.14,
          index,
        });
        chosen.push(index === 6 && gate !== undefined ? gate : plain);
        index++;
      }
    }

    this.#instance(chosen, samples, 'fence', {
      alignToGround: 0.35,
      shadow: true,
      targetHeight: 1.25,
      lean: 0.16,
      seed: `${SCATTER_SEED}.fence.lean`,
      perSampleKeys: true,
    });
  }

  /**
   * Barrels, crates and firewood — in three authored groups, not a scatter.
   *
   * The previous version dropped eleven props into an 11 m disc with a 0.95 m
   * minimum spacing, and a Poisson disc is the mathematical definition of "no
   * grouping": every prop is the same distance from every other prop, so there
   * is no hierarchy, no negative space and nothing for the eye to rest on. It
   * photographed as exactly what it was, assets scattered on a hillside.
   *
   * Composition needs the opposite of uniformity. Three groups with empty
   * ground between them, each doing a different job:
   *
   * - **supplies**, stacked hard against the wall, which welds the largest
   *   man-made mass in the frame to the camp and gives the left-hand third
   *   somewhere to terminate;
   * - **the fire ring**, exactly four seat-height props at ~1.6 m, placed on
   *   the arc *away* from the lens so they read as occupancy without ever
   *   standing between the camera and the focal point;
   * - **the wreck**, a tight knot where the fence line bends, which gives the
   *   leading line a waypoint instead of letting it run uninterrupted.
   *
   * Every group is then filtered against a keep-out capsule running from the
   * Barbarian back toward the camera, so no prop can land on the sightline to
   * the subject. `backlit-silhouette`'s pose is deterministic, so the salmon
   * barrel parked in front of the hero in that shot was a reproducible bug and
   * this is a reproducible fix.
   */
  #placeCampsiteClutter(clutter: readonly AssetKey[], logs: readonly AssetKey[]): void {
    if (clutter.length > 0) {
      const rng = createRng(`${SCATTER_SEED}.clusters`);
      const samples: ScatterSample[] = [];
      let index = 0;

      const place = (x: number, z: number, scale: number): void => {
        if (isOnHeroSightline(x, z)) return;
        samples.push({
          position: new THREE.Vector3(x, this.field.heightAt(x, z), z),
          normal: this.field.normalAt(x, z),
          scale,
          rotation: rng.next() * Math.PI * 2,
          index: index++,
        });
      };

      // 1. Supplies against the wall. A stack reads as a stack because its
      //    members are close enough to overlap in silhouette; 0.55-0.8 m apart
      //    at this prop scale is touching.
      const supplyCount = Math.max(3, Math.round(5 * this.#scatterDensity));
      for (let i = 0; i < supplyCount; i++) {
        const along = (i / Math.max(1, supplyCount - 1) - 0.5) * 2.4;
        place(
          -3.9 + Math.cos(WALL_YAW) * along + (rng.next() - 0.5) * 0.5,
          -9.1 - Math.sin(WALL_YAW) * along + (rng.next() - 0.5) * 0.5,
          0.9 + rng.next() * 0.3,
        );
      }

      // 2. Four seats round the fire. Bearings are measured from the direction
      //    of the camera, so the open arc is always the one the lens looks
      //    through however the fire is later moved.
      const toCamera = Math.atan2(CAMERA.position.z - CAMPFIRE.y, CAMERA.position.x - CAMPFIRE.x);
      for (const offset of FIRE_SEAT_BEARINGS) {
        const angle = toCamera + offset;
        const radius = 1.55 + rng.next() * 0.25;
        place(
          CAMPFIRE.x + Math.cos(angle) * radius,
          CAMPFIRE.y + Math.sin(angle) * radius,
          0.78 + rng.next() * 0.14,
        );
      }

      // 3. The wreck at the fence dogleg.
      for (let i = 0; i < 3; i++) {
        const angle = (i / 3) * Math.PI * 2 + rng.next() * 0.6;
        place(7.9 + Math.cos(angle) * 0.62, 1.6 + Math.sin(angle) * 0.62, 0.85 + rng.next() * 0.35);
      }

      this.#instance(clutter, samples, 'clutter', {
        alignToGround: 0.45,
        shadow: true,
        targetHeight: 0.95,
        lean: 0.1,
        seed: `${SCATTER_SEED}.clutter.lean`,
      });
    }

    if (logs.length > 0) {
      const samples = scatter({
        count: 5,
        // 2.6 m, from 5.5. Firewood belongs *at* the fire; spread over five and
        // a half metres it stopped being a woodpile and became more scatter.
        area: 2.6,
        center: new THREE.Vector3(CAMPFIRE.x + 1.9, 0, CAMPFIRE.y + 1.1),
        seed: `${SCATTER_SEED}.logs`,
        scaleRange: [0.8, 1.1],
        surface: this.field.surface,
        maxSlopeDegrees: 30,
        minSpacing: 0.55,
      }).filter((sample) => !isOnHeroSightline(sample.position.x, sample.position.z));
      this.#instance(logs, samples, 'logs', {
        alignToGround: 0.8,
        shadow: true,
        lean: 0.22,
        targetHeight: 0.55,
      });
    }
  }

  /** Rubble, weighted toward the wall it fell off. */
  /**
   * Masonry debris — at the foot of the wall, and nowhere else.
   *
   * This used to be a 34-instance scatter across a 26 m square centred on the
   * midground, and it was the source of the frame's most persistent-looking
   * "rendering bug": hard-edged black shards littered across the mud and the
   * cliff, which survived two rounds of shadow-bias increases because they were
   * never a shading artefact at all. `prop.rubble.*` are dungeon *floor* pieces
   * — near-flat debris authored to sit on a tile — so scattered across open
   * ground, aligned to the terrain normal and seen at a grazing angle, each one
   * presents as a dark irregular sliver. Fifty of them read as damage to the
   * image.
   *
   * They are correct in the one place they were always meant to be: piled
   * against a collapsing wall, which is where fallen masonry actually goes, and
   * which does real compositional work — it welds the wall's base into the
   * ground instead of leaving it standing on a hard line, and it puts a band of
   * broken detail along the leading edge that carries the eye toward the fire.
   *
   * Walked along the wall's own axis rather than scattered, with the spread
   * biased toward the collapsed (low) end. A ruin sheds most of its stone where
   * it has fallen furthest.
   */
  #placeRubble(keys: readonly AssetKey[]): void {
    if (keys.length === 0) return;
    const rng = createRng(`${SCATTER_SEED}.rubble`);
    const origin = new THREE.Vector2(-5.6, -11.4);
    const yaw = THREE.MathUtils.degToRad(24);
    const samples: ScatterSample[] = [];
    // 7, from 18. `prop.rubble.*` are dungeon *floor* tiles: near-flat plates
    // authored to sit inside a room. Laid on open ground and seen at a grazing
    // angle, each one presents as a dark irregular sliver two or three pixels
    // thick. That is the "hard-edged black polygonal shard" two independent
    // critiques both read as a rendering bug — and it is not one, which is why
    // it survived rounds of shadow-bias and occlusion work. It is the wrong
    // asset at the wrong angle, and the only fix for that is fewer of them,
    // larger, and buried deeper.
    const count = Math.round(7 * this.#scatterDensity);

    for (let i = 0; i < count; i++) {
      // Biased toward the broken end: `t^0.65` clusters samples at the high
      // end of the run, which is where the courses step down to nothing.
      const t = Math.pow(rng.next(), 0.65);
      const along = -5.4 + t * 12.4;
      // Off the wall face, mostly on the camera side so the pile is visible.
      const off = (rng.next() - 0.28) * 2.6;
      const x = origin.x + Math.cos(yaw) * along + Math.sin(yaw) * off;
      const z = origin.y - Math.sin(yaw) * along + Math.cos(yaw) * off;
      samples.push({
        position: new THREE.Vector3(x, this.field.heightAt(x, z), z),
        normal: this.field.normalAt(x, z),
        // Bigger, so what remains reads as a fallen block rather than a chip.
        scale: 0.95 + rng.next() * 0.75,
        rotation: rng.next() * Math.PI * 2,
        index: i,
      });
    }

    this.#instance(keys, samples, 'rubble', {
      alignToGround: 1,
      shadow: true,
      // Sunk further than before. These pieces carry their own base plate, and
      // burying it is what stops the plate reading as a shard in its own right.
      sink: 0.4,
      targetHeight: 0.55,
    });
  }

  #instance(
    keys: readonly AssetKey[],
    samples: readonly ScatterSample[],
    label: string,
    options: {
      alignToGround: number;
      shadow: boolean;
      /** Metres the *unscaled* model should stand, before per-sample scale. */
      targetHeight: number;
      sink?: number;
      lean?: number;
      seed?: string;
      perSampleKeys?: boolean;
    },
  ): void {
    const assets = this.#assets;
    if (assets === null || keys.length === 0 || samples.length === 0) return;

    const rng = createRng(options.seed ?? `${SCATTER_SEED}.${label}`);
    const up = new THREE.Vector3(0, 1, 0);
    const align = new THREE.Quaternion();
    const group = new THREE.Group();
    group.name = `props.${label}`;
    this.#root.add(group);

    let placed = 0;
    samples.forEach((sample, i) => {
      const key = options.perSampleKeys ? keys[i] : keys[i % keys.length];
      if (key === undefined) return;
      const gltf = assets.peek(key);
      if (gltf === undefined || !isGLTF(gltf)) return;

      const object = gltf.scene.clone(true);
      this.#weatherMaterials(object);
      object.position.copy(sample.position);
      object.position.y -= options.sink ?? 0;
      // Normalise by measured height, not by trusting the file. These props
      // come from three different KayKit packs authored at three different
      // scales, and a barrel taller than the Barbarian destroys the one thing
      // a figure in frame is there to establish. Measuring once per key and
      // scaling to an authored target height makes the whole set agree
      // regardless of what units each pack shipped in.
      object.scale.setScalar(sample.scale * this.#normalisingScale(key, gltf, options.targetHeight));

      // Partial alignment to the ground normal. Full alignment makes a tree on
      // a bank lean like a mast; none of it makes a rock float on one edge.
      // The right amount is per-prop-class, hence the parameter.
      align.setFromUnitVectors(up, sample.normal);
      object.quaternion.slerp(align, options.alignToGround);
      object.rotateY(sample.rotation);
      if (options.lean !== undefined && options.lean > 0) {
        object.rotateX((rng.next() - 0.5) * options.lean);
        object.rotateZ((rng.next() - 0.5) * options.lean);
      }

      if (options.shadow) {
        object.traverse((child) => {
          if (child instanceof THREE.Mesh) {
            child.castShadow = true;
            child.receiveShadow = true;
          }
        });
      }
      group.add(object);
      placed++;
    });

    if (placed === 0) {
      console.warn(`[BloodMoor] no "${label}" props were placed; check the asset keys.`);
    }
  }

  /**
   * Scale factor that makes `key`'s model stand `targetHeight` metres.
   *
   * Measured once per key from the model's own bounding box and cached: the
   * box computation walks every vertex, and it is the same answer for all
   * eighty copies of a tree.
   *
   * A prop with no measurable height (an empty scene, a degenerate mesh) gets
   * 1 and a warning rather than a division by zero.
   */
  #normalisingScale(key: AssetKey, gltf: GLTFLike, targetHeight: number): number {
    const cached = this.#scaleCache.get(key);
    if (cached !== undefined) return cached;

    const box = new THREE.Box3().setFromObject(gltf.scene);
    const height = box.max.y - box.min.y;
    const footprint = Math.max(box.max.x - box.min.x, box.max.z - box.min.z);

    let factor = 1;
    if (!Number.isFinite(height) || height <= 1e-4) {
      console.warn(`[BloodMoor] "${key}" has no measurable height; leaving it unscaled.`);
    } else {
      // Normalising on height alone is a trap: a flat object — a plank stack, a
      // spill of rubble — is a few centimetres tall and a metre wide, so
      // scaling its *height* to 0.6 m scales its footprint to five metres and
      // drops a grey slab across the foreground. Dividing by whichever of
      // height and footprint is more constraining keeps flat props flat.
      //
      // 2.6 is the aspect a prop is allowed to reach before its footprint
      // starts governing: taller-than-wide objects (fences, trees) are still
      // sized by height, and wider-than-tall ones by width.
      const governing = Math.max(height, footprint / 2.6);
      factor = THREE.MathUtils.clamp(targetHeight / governing, 0.15, 6);
    }
    this.#scaleCache.set(key, factor);
    return factor;
  }

  /**
   * Bring a KayKit prop into the Blood Moor's palette.
   *
   * The prop packs are palette-atlas models authored for a bright, saturated,
   * toy-like look — mid-grey stone, clean tan wood, primary-coloured cloth.
   * Dropped unmodified into an overcast grimdark frame they do two bad things
   * at once: they read as several stops brighter than the mud they stand on,
   * which is what makes them clip to white, and they are the wrong hue for the
   * art direction.
   *
   * So every unique prop material is cloned once and weathered: albedo pulled
   * down and toward the scene's cold grey, roughness pushed up (everything out
   * here is wet or rotten, nothing is polished), and the environment
   * contribution trimmed so the sky does not relight them past the terrain.
   * Cloning is what makes it safe — the originals belong to the
   * `AssetManager`'s cache and are shared with anything else that loads the
   * same model.
   *
   * The cache is keyed on the *original* material, so fifty barrels sharing one
   * material still produce exactly one weathered clone and one shader.
   */
  #weatherMaterials(object: THREE.Object3D, grade: WeatheringGrade = PROP_WEATHERING): void {
    object.traverse((child) => {
      if (!(child instanceof THREE.Mesh)) return;
      const source = child.material;
      const list = Array.isArray(source) ? source : [source];
      const weathered = list.map((material) => {
        const existing = this.#materialCache.get(material);
        if (existing !== undefined) return existing;
        const clone = weatherMaterial(material, grade);
        this.#materialCache.set(material, clone);
        this.#disposables.push(clone);
        return clone;
      });
      child.material = Array.isArray(source) ? weathered : (weathered[0] ?? source);
    });
  }

  /* -- character ---------------------------------------------------------- */

  /**
   * The Barbarian, idling.
   *
   * This is the proof that the 41-joint rig loads, binds and animates, and it
   * is also the scale reference the whole composition needs — a moor with no
   * figure in it has no size.
   *
   * The clip is chosen by name match rather than by index because the export
   * order of 76 clips is not something to depend on. If nothing matches
   * `/idle/i` the first clip is used and the fallback is logged, so a silent
   * T-pose is impossible.
   */
  async #loadHero(): Promise<void> {
    const assets = this.#assets;
    if (assets === null) return;

    try {
      const gltf = await assets.loadGLTF(CHARACTER_KEY);
      assets.pin(CHARACTER_KEY);

      const hero = gltf.scene;
      const x = HERO.x;
      const z = HERO.y;
      hero.name = 'barbarian';
      // Normalised to a human 1.85 m by measurement, exactly as the props are.
      // The character packs are authored at their own scale, and a figure that
      // is not human-sized destroys the one thing having a figure in frame is
      // for — it is the only object in the picture whose real size the viewer
      // already knows.
      const measured = new THREE.Box3().setFromObject(hero);
      const height = measured.max.y - measured.min.y;
      if (Number.isFinite(height) && height > 1e-3) {
        hero.scale.setScalar(THREE.MathUtils.clamp(HERO_HEIGHT / height, 0.05, 20));
      } else {
        console.warn('[BloodMoor] the Barbarian has no measurable height; leaving it unscaled.');
      }
      hero.position.set(x, this.field.heightAt(x, z), z);
      // Facing the fire, so the camera sees a three-quarter back view: the
      // silhouette reads and the warm light rims one shoulder.
      hero.rotation.y = Math.atan2(CAMPFIRE.x - x, CAMPFIRE.y - z);
      hero.traverse((child) => {
        if (child instanceof THREE.Mesh || child instanceof THREE.SkinnedMesh) {
          child.castShadow = true;
          child.receiveShadow = true;
          // Skinned bounds are computed from the bind pose, so a character in
          // an animated pose gets frustum-culled at the edge of frame.
          child.frustumCulled = false;
        }
      });
      // The Barbarian goes through the same shader-side desaturation as the
      // props, at half strength. He arrived as the brightest, most chromatic
      // object in the frame — a cream blob in a cold blue-grey moor — which is
      // the single fastest way to make a character look pasted into a scene
      // rather than standing in it. Half strength, not full, because he is the
      // subject: he still has to hold more chroma than the mud he stands on,
      // and the fire still has to rim him.
      this.#weatherMaterials(hero, HERO_WEATHERING);
      this.#root.add(hero);
      this.#hero = hero;

      const clips = gltf.animations;
      if (clips.length === 0) {
        console.warn('[BloodMoor] the Barbarian GLTF has no animation clips; pose is the bind pose.');
        return;
      }
      this.#heroClips = clips;
      if (this.#options.controlHero === false) {
        console.info(`[BloodMoor] Barbarian: ${clips.length} clips, handed to the player module`);
        return;
      }
      const idle = clips.find((clip) => /idle/i.test(clip.name)) ?? clips[0];
      if (idle === undefined) return;
      if (!/idle/i.test(idle.name)) {
        console.warn(
          `[BloodMoor] no clip matching /idle/i among ${clips.length}; ` +
            `falling back to "${idle.name}".`,
        );
      }
      const mixer = new THREE.AnimationMixer(hero);
      mixer.clipAction(idle).play();
      this.#mixer = mixer;
      console.info(`[BloodMoor] Barbarian: ${clips.length} clips, playing "${idle.name}"`);
    } catch (error) {
      console.warn(
        `[BloodMoor] could not load "${CHARACTER_KEY}"; the frame has no figure in it ` +
          'and therefore no scale reference. Run `npm run assets`.',
        error,
      );
    }
  }

  /* -- occlusion ---------------------------------------------------------- */

  /**
   * Fold the GTAO buffer into every node material under this scene.
   *
   * Applied once, after everything is built, and idempotent per material via a
   * `Set` — `applyOcclusion` composes onto the existing `aoNode`, so calling it
   * twice on a material shared between fifty props would multiply the AO in
   * fifty times.
   */
  #applyOcclusion(): void {
    const ibl = this.#ibl;
    if (ibl === null) {
      console.warn('[BloodMoor] no IBLService: ambient occlusion is not applied to scene materials.');
      return;
    }
    const seen = new Set<THREE.Material>();
    this.#root.traverse((object) => {
      if (!(object instanceof THREE.Mesh)) return;
      const materials = Array.isArray(object.material) ? object.material : [object.material];
      for (const material of materials) {
        if (seen.has(material)) continue;
        seen.add(material);
        if (material instanceof THREE.NodeMaterial) ibl.applyOcclusion(material);
      }
    });
  }

  /**
   * The cave mouth on the west bank: the way down into the Den of Evil.
   *
   * A hole in a hillside is one of the hardest things to make read, because
   * "dark" and "hole" look identical until something tells the eye which it is.
   * Four elements do that here, and none of them is optional:
   *
   * - a **jamb** of two heavy boulders, close enough together to frame a gap
   *   narrower than the player is tall. Scale is the whole cue: an opening wider
   *   than it is high reads as a gully, not as a mouth.
   * - a **lintel** boulder bridging them, so the top edge is a hard line. An
   *   opening with a soft top edge reads as a shadow on a slope.
   * - a **throat**: a near-black recessed box set back 3 m behind the jamb, with
   *   its own material at 0.012 albedo. It is what makes the gap read as
   *   *depth* rather than as a dark patch of the same rock.
   * - **spoil**, a scatter of small rock around the base, because a cave mouth
   *   is a place material came out of.
   *
   * The whole assembly is turned to face east-south-east, toward the player's
   * approach across the moor, so the opening is visible from the spawn rather
   * than only from directly in front of it.
   */
  #buildDenMouth(): void {
    const materials = this.#materials;
    const x = DEN_MOUTH.portal.x;
    const z = DEN_MOUTH.portal.y;
    const base = this.field.heightAt(x, z);

    const group = new THREE.Group();
    group.name = 'moor.denMouth';
    group.position.set(x, base, z);
    // Facing the player's approach, not an axis.
    group.rotation.y = Math.atan2(HERO.x - x, HERO.y - z);
    this.#root.add(group);

    const rock =
      materials?.create('rock', {
        // Darker and flatter than the escarpment's cliff rock, on purpose. The
        // jamb's job is to be the *frame* of a hole: at the escarpment's value
        // and gloss it came back as a bright boulder pile catching the sky, and
        // a bright frame around a dark gap reads as rocks, not as a mouth. The
        // roughness floor is raised for the same reason — a specular highlight
        // on the jamb is the single strongest cue that this is a solid object.
        albedoTint: [0.075, 0.073, 0.078],
        albedoSaturation: 0.45,
        roughnessRange: [0.86, 1],
        wetnessExposure: 0.25,
      }) ?? new THREE.MeshStandardNodeMaterial({ color: 0x121212, roughness: 0.97 });
    rock.name = 'moor.denMouth.rock';
    this.#disposables.push(rock);

    const rng = createRng(`${SCATTER_SEED}.denmouth`);
    const jamb: { dx: number; dz: number; scale: [number, number, number]; radius: number }[] = [
      { dx: -2.5, dz: 0.35, scale: [1.15, 2.5, 1.35], radius: 1.5 },
      { dx: 2.55, dz: 0.15, scale: [1.2, 2.35, 1.3], radius: 1.55 },
      { dx: -1.1, dz: -2.4, scale: [1.5, 1.9, 1.2], radius: 1.7 },
      { dx: 1.35, dz: -2.6, scale: [1.45, 2.0, 1.25], radius: 1.65 },
    ];
    for (const [index, block] of jamb.entries()) {
      const geometry = generateRockGeometry({
        seed: `${SCATTER_SEED}.denmouth.${index}`,
        detail: 3,
        radius: block.radius,
        displacement: 0.4,
        scale: block.scale,
        flattenBase: 0.3,
      });
      this.#disposables.push(geometry);
      const mesh = new THREE.Mesh(geometry, rock);
      mesh.name = `moor.denMouth.jamb.${index}`;
      mesh.position.set(block.dx, this.field.heightAt(x + block.dx, z + block.dz) - base, block.dz);
      mesh.rotation.y = rng.next() * Math.PI * 2;
      mesh.castShadow = true;
      mesh.receiveShadow = true;
      group.add(mesh);
    }

    // The lintel, bridging the jamb at 2.6 m — head height plus a margin, so
    // the opening is unmistakably something a person walks into.
    const lintelGeometry = generateRockGeometry({
      seed: `${SCATTER_SEED}.denmouth.lintel`,
      detail: 3,
      radius: 1.6,
      displacement: 0.34,
      scale: [2.1, 0.62, 1.05],
      flattenBase: 0.15,
    });
    this.#disposables.push(lintelGeometry);
    const lintel = new THREE.Mesh(lintelGeometry, rock);
    lintel.name = 'moor.denMouth.lintel';
    lintel.position.set(0, 2.75, -0.6);
    lintel.castShadow = true;
    lintel.receiveShadow = true;
    group.add(lintel);

    /* the throat */
    const throatMaterial = new THREE.MeshStandardNodeMaterial({
      color: new THREE.Color(0.012, 0.013, 0.016),
      roughness: 1,
      metalness: 0,
    });
    throatMaterial.name = 'moor.denMouth.throat';
    // The throat must not be relit by the sky. It is the one surface in this
    // scene whose correct rendering is a controlled near-black, and the
    // environment term is exactly what would take it back up to slope grey.
    throatMaterial.envMapIntensity = 0.02;
    this.#disposables.push(throatMaterial);

    // Sized to be *hidden* by the jamb from every direction except through the
    // gap. The previous pass enlarged it to 6.2 x 4.4 x 7.5 so the darkness
    // would read from an oblique approach, and it did — as a black cube floating
    // above the rocks, because a recess wider and taller than the thing it is
    // recessed into is not a recess, it is a box. Back inside the jamb, and the
    // obliqueness problem is solved by the hood below instead.
    const throatGeometry = new THREE.BoxGeometry(4.4, 3.2, 6.5);
    this.#disposables.push(throatGeometry);
    const throat = new THREE.Mesh(throatGeometry, throatMaterial);
    throat.name = 'moor.denMouth.throat';
    throat.position.set(0, 1.05, -3.9);
    throat.receiveShadow = true;
    // Never a collider: the player walks into it and the portal takes over.
    throat.userData['noCollide'] = true;
    group.add(throat);

    /* the hood */

    // A slab over the top of the throat, and the piece that makes the whole
    // assembly read from above.
    //
    // The default third-person camera looks *down* at the player, so the
    // approach to any ground-level opening is an approach from above — and from
    // above, a jamb and a lintel frame nothing at all. Without a roof the eye
    // sees straight over the lintel into the recess, and the recess reads as
    // whatever it geometrically is: a dark box lying in the grass. The hood is
    // 5 m of rock across the top of it.
    const hoodGeometry = generateRockGeometry({
      seed: `${SCATTER_SEED}.denmouth.hood`,
      detail: 3,
      radius: 2.6,
      displacement: 0.36,
      scale: [1.95, 0.72, 1.7],
      flattenBase: 0.1,
    });
    this.#disposables.push(hoodGeometry);
    const hood = new THREE.Mesh(hoodGeometry, rock);
    hood.name = 'moor.denMouth.hood';
    hood.position.set(0, 3.3, -3.6);
    hood.rotation.y = 0.24;
    hood.castShadow = true;
    hood.receiveShadow = true;
    group.add(hood);

    /* spoil */
    const spoilGeometry = generateRockGeometry({
      seed: `${SCATTER_SEED}.denmouth.spoil`,
      detail: 2,
      radius: 0.34,
      displacement: 0.42,
      scale: [1, 0.62, 0.9],
      flattenBase: 0.5,
    });
    this.#disposables.push(spoilGeometry);
    // The arrival point, in this group's local frame, and the ground it needs
    // kept clear.
    //
    // The spoil heap's scatter annulus (1.6-5.8 m) contains the `from-den`
    // arrival point at 3.5 m, and it duly dropped a rock on it: a player
    // climbing out of the Den landed with `moor.denMouth.spoil#20` level with
    // his head, 0.6 m from the camera pivot, so the arm collapsed to its floor
    // on the frame that is supposed to show him the moor. A spoil heap outside
    // a mine is exactly the right dressing for this spot; a spoil heap *on the
    // doorstep* is not, and the fix is the doorstep, not the heap.
    //
    // The group is *rotated* to face the player's approach, so the world-space
    // offset has to be turned back into the group's frame before it can be
    // compared with a sample's local position. Skipping that rotation is the
    // kind of mistake that produces a filter which runs, rejects samples, and
    // clears the wrong patch of ground — measured once already.
    const yaw = group.rotation.y;
    const dx = DEN_MOUTH.approach.x - x;
    const dz = DEN_MOUTH.approach.y - z;
    const clearX = dx * Math.cos(yaw) - dz * Math.sin(yaw);
    const clearZ = dx * Math.sin(yaw) + dz * Math.cos(yaw);
    const CLEAR_RADIUS = 2.4;

    const spoil: ScatterSample[] = [];
    // Attempts, not iterations: rejecting a sample must not cost a rock, or
    // clearing the doorstep quietly thins the whole heap.
    for (let attempt = 0; attempt < 220 && spoil.length < 22; attempt++) {
      const angle = rng.next() * Math.PI * 2;
      const radius = 1.6 + rng.next() * 4.2;
      const sx = Math.cos(angle) * radius;
      const sz = Math.sin(angle) * radius * 0.7 + 1.4;
      if (Math.hypot(sx - clearX, sz - clearZ) < CLEAR_RADIUS) continue;
      spoil.push({
        position: new THREE.Vector3(sx, this.field.heightAt(x + sx, z + sz) - base + 0.05, sz),
        normal: new THREE.Vector3(0, 1, 0),
        scale: 0.45 + rng.next() * 1.1,
        rotation: rng.next() * Math.PI * 2,
        index: spoil.length,
      });
    }
    const rubble = buildInstancedMesh(spoilGeometry, rock, spoil);
    rubble.name = 'moor.denMouth.spoil';
    group.add(rubble);
  }

  /* -- zone contract ------------------------------------------------------ */

  /**
   * The moor's subtree. Everything this scene builds hangs off it, which is
   * what lets `world/ZoneManager` unload the zone by detaching one node.
   */
  get root(): THREE.Object3D {
    return this.#root;
  }

  /**
   * Arrival points.
   *
   * `from-camp` is south-east of the composition's focal point, on open ground
   * clear of the six spawned skeletons — a player who steps out of the gate and
   * is immediately inside an encounter has been ambushed by the level, not by
   * the enemies. `from-den` is at the cave mouth on the west bank.
   */
  get entryPoints(): readonly ZoneEntryPoint[] {
    return [
      { id: 'from-camp', position: this.#ground(DEN_MOUTH.campApproach.x, DEN_MOUTH.campApproach.y) },
      { id: 'from-den', position: this.#ground(DEN_MOUTH.approach.x, DEN_MOUTH.approach.y) },
      { id: 'spawn', position: this.#ground(HERO.x, HERO.y) },
    ];
  }

  /** Back to the camp, and down into the Den. */
  get portals(): readonly PortalSpec[] {
    return [
      {
        id: 'moor-to-camp',
        targetZone: 'encampment',
        targetEntry: 'gate',
        position: this.#ground(DEN_MOUTH.campPortal.x, DEN_MOUTH.campPortal.y),
        radius: 2.8,
        height: 3.5,
        label: 'the Rogue Encampment',
        verb: 'return to',
      },
      {
        id: 'moor-to-den',
        targetZone: 'denOfEvil',
        targetEntry: 'cave-mouth',
        position: this.#ground(DEN_MOUTH.portal.x, DEN_MOUTH.portal.y),
        radius: 2.4,
        height: 3.2,
        label: 'the Den of Evil',
      },
    ];
  }

  /**
   * The terrain heightfield and every prop collider on the moor.
   *
   * Sampled from the same {@link TerrainField} the visual mesh is displaced by,
   * at the same segment count: at equal resolution the collider surface and the
   * rendered surface are the same surface, and at unequal resolution they are
   * a character whose feet sink into a ridge.
   */
  buildColliders(physics: PhysicsWorld): void {
    physics.buildTerrain(this.field, TERRAIN_SIZE, this.#terrainSegments);
    physics.buildSceneColliders(this.#root);
  }

  /** A zone-local point on the terrain surface. */
  #ground(x: number, z: number): { x: number; y: number; z: number } {
    return { x, y: this.field.heightAt(x, z), z };
  }

  /* -- diagnostics -------------------------------------------------------- */

  get hero(): THREE.Object3D | null {
    return this.#hero;
  }

  /** The Barbarian's clips, for whoever is animating him. */
  get heroClips(): readonly THREE.AnimationClip[] {
    return this.#heroClips;
  }

  /** Width of the ground mesh in metres. The physics heightfield matches it. */
  get terrainSize(): number {
    return TERRAIN_SIZE;
  }

  /** Segment count the ground mesh was built at. See {@link terrainSize}. */
  get terrainSegments(): number {
    return this.#terrainSegments;
  }

  /** Where the composition wants the camera. Capture shots start from here. */
  static get defaultCamera(): {
    position: THREE.Vector3;
    target: THREE.Vector3;
    fov: number;
  } {
    return {
      position: CAMERA.position.clone(),
      target: CAMERA.target.clone(),
      fov: CAMERA.fov,
    };
  }
}

/**
 * The weathering multiplier applied to every prop albedo.
 *
 * Cold and dark, with the blue channel held highest so that what survives the
 * desaturation is the *shade* direction rather than the wood tone. Tuned
 * against the terrain: props land within about a stop of the mud they stand
 * on, which is what stops them clipping out of a frame exposed for the ground.
 */
/**
 * One weathering recipe. Two exist: one for the world, one for the subject.
 *
 * Separating them is the difference between "everything is desaturated" and an
 * *art direction*. A prop's job is to sit inside the palette and never be
 * looked at; the Barbarian's job is to be looked at, which means he gets the
 * same treatment at half the strength — enough that he belongs in the world,
 * not so much that he stops being the figure in it.
 */
/**
 * How far every prop albedo is dragged toward its own luminance.
 *
 * **0.30, down from 0.72, and this is the number that decides whether the
 * frame has a palette or just a filter.**
 *
 * The previous value came from the right diagnosis and the wrong remedy. The
 * kit props put all of their colour in an atlas and leave the base-colour
 * *factor* white, so multiplying the factor can darken a prop but never
 * desaturate one, and the moor filled with salmon-pink barrels. Desaturating
 * the *sample* fixed that. But 0.72 with a near-neutral grey multiply does not
 * produce palette discipline — it produces palette *collapse*: wood, iron,
 * cloth, masonry and the hero all land on one hue at one value, which is grey
 * mush by construction and reads as untextured plastic at 1:1.
 *
 * Discipline means deliberate separation inside a limited range, not
 * compression to a point. So the chroma stays (0.30), and the control moves to
 * three cheaper, *selective* operators that cannot flatten material identity:
 *
 * 1. a luminance-driven two-stop tint (cold shades, warm lights), which is what
 *    restores the wood-vs-iron read;
 * 2. a `warmClamp` on residual red chroma, which enforces "only the fire is
 *    warm" without touching value or material distinction;
 * 3. a low value on both tint stops, which is what actually keeps props out of
 *    the highlights.
 */
const WEATHERING_DESATURATION = 0.3;

/**
 * The world grade: barrels, crates, fence, rubble, lumber.
 *
 * `envMapIntensity` 0.22 (from 0.55) and `roughnessFloor` 0.60 (from 0.42) are
 * one fix, not two: under a full-overcast IBL the environment is the dominant
 * light source, and a prop that both catches it strongly and has a tight
 * specular lobe ends up the brightest object in a grimdark moor. Between them
 * they take a white picket fence out of the highlights without touching its
 * albedo, which is the correct place to fix "the sky is relighting my props".
 */
const PROP_WEATHERING: WeatheringGrade = {
  desaturation: WEATHERING_DESATURATION,
  // Measured off the capture, not chosen. At (0.20, 0.225, 0.285) ->
  // (0.44, 0.395, 0.315) the barrels and crates came back as near-white plastic
  // at 1:1, because the KayKit atlas cells are already bright and a 0.44
  // multiply on a 0.9 cell is still a 0.4 albedo — brighter than the mud they
  // stand on, and far brighter than anything in a grimdark moor may be while
  // not being the fire. Halved. The *ramp between* these two is what does the
  // art direction; their absolute level is what keeps props out of the
  // highlights. They are independent knobs on the same pair of numbers.
  shadowTint: new THREE.Color(0.105, 0.12, 0.155),
  lightTint: new THREE.Color(0.235, 0.21, 0.165),
  warmClamp: 0.55,
  grime: 0.4,
  wear: 0.16,
  roughnessFloor: 0.6,
  // 0.5, *not* the 0.22 the critique asked for, and this is a correction to the
  // critique rather than to the previous pass. `envMapIntensity` in three gates
  // ambient **diffuse** as well as ambient specular. Under a full overcast sky
  // the environment is essentially the entire light budget on a prop, so
  // cutting this is not "stop the props catching the sky", it is "take away
  // 60% of the light on the props" — measured, it took the barrels and the
  // rubble down to rgb(24, 29, 33), i.e. it re-created the black-shard defect
  // this pass exists to remove, by a completely different mechanism.
  //
  // The right lever for value is the albedo, which is what `shadowTint` and
  // `lightTint` above now are: dark enough that a prop cannot compete with the
  // fire, while still being *lit*. The high `roughnessFloor` is what stops the
  // sky showing up as a specular highlight, and that one is kept.
  envMapIntensity: 0.5,
};

/**
 * The subject grade.
 *
 * Desaturation drops to 0.14 and the warm clamp to almost nothing. This is
 * standard stylized-AAA practice and it is the opposite of what the previous
 * pass did: the subject is the colour anchor of the frame and should hold more
 * chroma than anything else in it except the fire. At 0.44 desaturation the
 * Barbarian was a featureless beige blob — no fur, no straps, no belt — because
 * every material distinction the atlas gives him had been averaged away.
 *
 * Value stays low-ish and `envMapIntensity` stays modest so he still belongs to
 * the world; the separation he needs comes from the rim light, not from being
 * brighter than the mud.
 */
const HERO_WEATHERING: WeatheringGrade = {
  desaturation: 0.14,
  shadowTint: new THREE.Color(0.17, 0.19, 0.245),
  lightTint: new THREE.Color(0.42, 0.375, 0.3),
  warmClamp: 0.12,
  grime: 0.26,
  wear: 0.22,
  roughnessFloor: 0.34,
  envMapIntensity: 0.62,
};

/**
 * The dead-wood grade, applied to the whole treeline.
 *
 * Very dark and slightly cold. These trees exist to be a silhouette against a
 * bright overcast sky, and a silhouette is a *value* relationship: anything
 * above roughly 0.25 albedo starts to catch the sky and the shape dissolves.
 * The blue channel is held highest so what little the limbs do catch reads as
 * skylight rather than as wood.
 *
 * Roughness is floored high — dead standing timber is fibrous and has no
 * specular lobe worth the name — and `porosity` is raised so the wetness system
 * darkens it in the rain rather than glossing it.
 */
/**
 * The treeline's authored albedo. See `#placeDeadTrees` for why it is a
 * constant here rather than a `SurfaceSpec.albedoTint`.
 *
 * 0.030. Against an overcast sky landing at 209/255 this puts a lit trunk face
 * in the 60-75/255 band, which is the ~3 : 1 value separation a silhouette
 * needs to read as a shape rather than as a smudge.
 */
const DEAD_WOOD_ALBEDO: readonly [number, number, number] = [0.03, 0.029, 0.036];

const DEAD_WOOD_OVERRIDES: Partial<SurfaceSpec> = {
  // 0.075. Measured, not guessed: at 0.135 the ridge trees were landing around
  // 0.45 in display space against a 0.75 sky, which is a mid-grey tree — and a
  // mid-grey tree is not a silhouette, it is a smudge. A silhouette is a *value*
  // relationship and it needs at least a 3:1 separation from what is behind it.
  // 0.034. Measured off the capture rather than guessed at, twice now: at
  // 0.062 the ridge trunks came back at 159/255 against a 209/255 sky, a
  // 1.3 : 1 separation, which is camouflage and not a silhouette. A bare tree
  // in front of a bright overcast sky is one of the few objects in a frame
  // whose correct rendering is *almost black*, and stylized work needs at
  // least 2.5 : 1 there because the silhouette carries the entire read.
  albedoTint: [0.034, 0.033, 0.04],
  roughnessRange: [0.78, 1],
  normalStrength: 1.35,
  porosity: 0.86,
  // Tightened from the bark default of 9 m. At 9 m the macro octave was larger
  // than a whole tree, so every limb of a given tree got the same offset and
  // the variation showed up as tree-to-tree rather than as bark. At 2.4 m it
  // breaks up along the trunk, which is where bark variation lives.
  macro: {
    metres: 2.4,
    albedoAmount: 0.34,
    roughnessAmount: 0.12,
    tint: [0.02, 0.019, 0.024],
    tintAmount: 0.4,
  },
};

/**
 * The rim light's colour and strength. See `#buildLightRig`.
 *
 * A blue-white well above the overcast sky's own correlated temperature: the
 * rim has to read as a *different* light from the ambient it is separating
 * against, and at the sky's own colour it would simply add to it.
 */
const RIM_COLOR = 0x9fc2e8;
/**
 * 2.6, up from 0.7.
 *
 * The old value was chosen as if the rim were competing with a sun. It is not:
 * under full overcast the sky dome delivers most of the irradiance on every
 * surface in the frame, and a 0.7-intensity directional against an ambient of
 * comparable magnitude cannot separate anything — it lifts the whole object by
 * a few percent and produces no edge at all. That is why nine capture shots had
 * no determinable light direction and why `backlit-silhouette`, the shot that
 * exists to prove rim separation, had none.
 *
 * A rim light's job is a *ratio*, not a level: the lit edge has to measure
 * roughly twice the background immediately outside the silhouette. Against this
 * ambient that needs an intensity in the 2–3 range. It is still shadowless and
 * still one N·L term, so the cost is unchanged.
 */
const RIM_INTENSITY = 2.6;

/**
 * The shadow fill. Deep blue-teal — enough to put a hue in the darkest quarter
 * of the histogram and no more. Grimdark is not black; black is where detail
 * and mood both go to die.
 *
 * 0.34. Cut to 0.15 with the rim raise and measured straight back up: at 0.15
 * the basin floor in front of the escarpment went to rgb(2, 3, 27) over an
 * eighth of the frame, i.e. a hard black wedge with no hue in it at all, which
 * is the exact failure this light exists to prevent. The rim/fill *ratio* is
 * what makes a silhouette read, and at 2.6 : 0.34 that ratio is 7.6 : 1 —
 * far more separation than the old 0.7 : 0.4 (1.75 : 1) while still leaving
 * enough light in the shade for it to keep its blue.
 */
const SHADOW_FILL_COLOR = 0x4d6d8c;
const SHADOW_FILL_INTENSITY = 0.34;

/** Sodium-lamp orange would be wrong; wood fire is redder and dirtier. */
const FIRE_COLOR = 0xff7a30;

/**
 * Peak firelight intensity.
 *
 * Set relative to the sun, not in absolute units: under a fully overcast sky
 * the key is soft and low, and the fire needs to be roughly comparable at two
 * metres for the warm/cold contrast to exist at all. Much higher and the
 * campsite blows out against the moor; much lower and the whole frame is one
 * temperature.
 *
 * 2.6, down from 9.5. Two separate errors were stacking: this light was being
 * added twice (see `#buildCampfire`), and the value itself was set while the
 * sun was extinguished to 3% of clear sky, so the fire was being tuned against
 * nothing. With the key restored and the double-count gone, a campfire that
 * stops being the dominant illuminant at about six metres is the right size of
 * source — which is what a ~1 kW wood fire actually is.
 *
 * **4.5, up from 2.3.** The reasoning above is right and the arithmetic was
 * wrong. `decay: 2` is inverse-square, so an intensity of 2.3 delivers 2.3 at
 * one metre and 0.14 at four — against an overcast IBL contributing on the
 * order of 1.5 to the same surface. That is a fire that is the dominant
 * illuminant inside a 1.2 m bubble and invisible outside it, which is exactly
 * what the captures showed: crates a metre from the flame reading cold grey,
 * and therefore no warm/cold premise anywhere in the frame.
 *
 * 8.4 puts the crossover — where firelight equals ambient — at about 2.4 m and
 * keeps the fire a *visible* fraction of the light out to 4–5 m, which is the
 * radius of the camp. It does not blow the ring stones out because those were
 * already re-tinted to 0.115 albedo and the source was lifted to 0.62 m.
 */
const FIRE_INTENSITY = 4.5;

/**
 * The same fire's in-scatter strength, kept separate on purpose.
 *
 * The volumetric bridge reads a light's intensity to scale in-scattering
 * through the froxel volume, and scattering integrates along the whole ray
 * rather than falling off with one inverse square. Feeding it the shading
 * intensity is what once let a 6 m campfire tint a 50 m basin crimson. Surface
 * shading and participating-media scattering are two different transport
 * problems and they get two different numbers.
 */
const FIRE_SCATTER_INTENSITY = 3.0;

/**
 * Layer for objects that exist only so a screen-space or volumetric pass can
 * find them, and that must never be drawn or shaded by the main camera.
 *
 * Currently just the campfire's volumetric proxy light. Kept out of layer 0 so
 * `Renderer._projectObject` drops it from the render list.
 */
const VOLUMETRIC_ONLY_LAYER = 3;

type GLTFLike = { scene: THREE.Group; animations: THREE.AnimationClip[] };

/**
 * Boundary check for `AssetManager.peek`, whose return type is the union of
 * everything it can cache. There is no `GLTF` class to `instanceof` against —
 * it is a plain object from the loader — so this is a structural test.
 */
function isGLTF(value: unknown): value is GLTFLike {
  return (
    typeof value === 'object' &&
    value !== null &&
    'scene' in value &&
    (value as { scene: unknown }).scene instanceof THREE.Object3D
  );
}

function countRenderables(root: THREE.Object3D): number {
  let count = 0;
  root.traverse((object) => {
    if (object instanceof THREE.Mesh) count++;
  });
  return count;
}

/**
 * Dispose geometry and materials this scene *created*.
 *
 * Cloned props share their geometry and materials with the `AssetManager`'s
 * cache, so disposing them here would corrupt the cache for the next scene.
 * Anything owned is tracked in `#disposables` instead; this only unlinks.
 */
function disposeTree(root: THREE.Object3D): void {
  // Was `root.clear()` alone, which detached every child *without releasing
  // anything*. That is a leak on its own, and it was worse than it looks:
  // `ZoneManager.#unload` calls `zone.dispose()` and only then walks the tree
  // with `disposeZoneTree`, so by the time the manager looked there was nothing
  // left to find and it reported `disposed: 0` for the moor on every unload. The
  // zone's two directional lights, their targets and every non-shared material
  // under the root survived each visit, and a camp -> moor -> den -> camp lap
  // accumulated another set.
  //
  // `disposeZoneTree` is the right tool and always was: it honours the
  // `userData.shared` marker that `PropKit` stamps onto cached geometry and
  // textures, so the cache-corruption hazard this function's comment was
  // guarding against does not arise. Calling it here keeps the standalone path
  // (capture harnesses, tests) correct too, and the manager's second walk over
  // an emptied root is a harmless no-op.
  disposeZoneTree(root);
  root.clear();
}
