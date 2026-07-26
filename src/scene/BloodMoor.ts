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
 *   background   ridge at z ≈ −34, bare trees on the skyline, aerial
 *                perspective and haze doing the separation
 *   midground    the ruined masonry wall (left), the campfire (right of
 *                centre) — the focal point, and the only warm light in the
 *                image — with the Barbarian standing between them
 *   foreground   a rock shelf and rubble in the lower left, dark and in
 *                shadow, framing the eye into the frame
 *   leading line the broken fence running from the lower-left foreground
 *                diagonally back toward the campfire
 * ```
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
  dot,
  mix,
  normalWorldGeometry,
  positionWorld,
  saturate,
  smoothstep,
  texture as textureNode,
  uv,
  vec3,
} from 'three/tsl';

import { AssetManagerKey, type AssetKey, type AssetManager } from '../assets/AssetManager';
import {
  SimplexNoise,
  createRng,
  generateRockGeometry,
  scatter,
  type ScatterSample,
} from '../assets/Procedural';
import type { GameContext, GameModule } from '../core/types';
import { SCATTER_SEED, auditServices } from '../render/FrameGraph';
import { IBLKey, type IBLService } from '../render/IBL';
import { LightingKey, type LightHandle, type LightingService } from '../render/Lighting';
import { MaterialLibraryKey, type MaterialLibraryService } from '../render/MaterialLibrary';
import { VolumetricsKey, type VolumetricsService } from '../render/Volumetrics';
import type { RenderSettings } from '../render/RenderSettings';

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

/** Where the Barbarian stands: between the camera and the fire, off-centre. */
const HERO = new THREE.Vector2(1.1, 0.4);

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
  trees: ['nature.tree.a', 'nature.tree.b'],
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
}

export class BloodMoor implements GameModule {
  readonly name = 'scene.bloodMoor';

  readonly field = new TerrainField();

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
  #fireEmber: THREE.Mesh | null = null;
  #mixer: THREE.AnimationMixer | null = null;
  #hero: THREE.Object3D | null = null;
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

    this.#configureCamera(ctx.camera);
    this.#buildTerrain();
    this.#buildEscarpment();
    this.#buildRuinedWall();
    this.#buildCampfire(ctx);

    await this.#loadProps();
    await this.#loadHero();

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
    if (this.#fireLight !== null || this.#fireGlow !== null) {
      const flicker =
        1 + 0.085 * Math.sin(t * 7.31) + 0.055 * Math.sin(t * 11.97 + 1.7) + 0.03 * Math.sin(t * 19.3);
      this.#fireLight?.setIntensity(FIRE_INTENSITY * flicker);
      if (this.#fireGlow !== null) this.#fireGlow.intensity = FIRE_INTENSITY * flicker;
      if (this.#fireEmber !== null) this.#fireEmber.scale.setScalar(0.96 + 0.06 * flicker);
    }

    // `setTime`, not `update(dt)`: absolute time makes the pose a pure function
    // of the frame index, so `stepFrames(n)` lands on the same pose every run.
    this.#mixer?.setTime(t);
  }

  dispose(): void {
    this.#fireLight?.release();
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
    const geometry = this.field.buildGeometry(TERRAIN_SIZE, segments);

    const grassCoverage = saturate(
      smoothstep(-1.4, 0.6, positionWorld.y).mul(smoothstep(0.62, 0.88, normalWorldGeometry.y)),
    );

    const material =
      materials?.createBlended({
        base: 'wetMud',
        overlay: 'deadGrass',
        weight: grassCoverage,
        depth: 0.18,
        // Hex anti-tiling off, macro variation on. Hex tiling breaks the
        // repeat by sampling three rotated copies per pixel and blending them,
        // which works beautifully on a surface a few metres across and fails
        // loudly on a 260 m ground plane: at this scale the hex cells are
        // eight metres wide and the blend seams between them are visible as a
        // honeycomb across the whole moor. The macro-variation octave breaks
        // the same repeat at a fraction of the cost and with no cell structure
        // to give itself away.
        baseOverrides: { antiTile: 'macro' },
        overlayOverrides: { antiTile: 'macro' },
      }) ?? new THREE.MeshStandardNodeMaterial({ color: 0x3a3a34, roughness: 0.9 });

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

    const blocks: readonly { x: number; z: number; s: readonly [number, number, number] }[] = [
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
      // Sunk by a third of its height: the formation reads as bedrock breaking
      // through the bank, not as boulders resting on it.
      mesh.position.set(block.x, this.field.heightAt(block.x, block.z) - block.s[1] * 0.34, block.z);
      mesh.rotation.y = rng.next() * Math.PI * 2;
      mesh.castShadow = true;
      mesh.receiveShadow = true;
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
    const courses: readonly { along: number; height: number; length: number; depth: number }[] = [
      { along: -4.6, height: 3.4, length: 3.0, depth: 0.86 },
      { along: -1.5, height: 2.9, length: 3.2, depth: 0.86 },
      { along: 1.6, height: 1.9, length: 3.0, depth: 0.82 },
      { along: 4.1, height: 1.05, length: 2.0, depth: 0.78 },
      { along: 5.9, height: 0.5, length: 1.6, depth: 0.7 },
    ];

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
        // The masonry set is bright limestone. Tinted down and toward the
        // scene's cold grey so a two-metre wall is not the brightest object in
        // an overcast frame, which it otherwise is by a wide margin.
        albedoTint: [0.30, 0.32, 0.35],
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
      const stone = materials.get('rock');
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

    // The ember bed. `MeshBasicNodeMaterial` and not an emissive PBR material:
    // this surface is a light source, it has no diffuse response worth
    // computing, and a basic material guarantees it lands in the HDR buffer at
    // the value written rather than at whatever the shading model returns.
    const emberGeometry = new THREE.SphereGeometry(0.34, 16, 10);
    emberGeometry.scale(1, 0.55, 1);
    const emberMaterial = new THREE.MeshBasicNodeMaterial({
      // Well above 1.0 on purpose. This is the one thing in the frame allowed
      // to blow out, it is what the bloom pyramid picks up, and it is what
      // gives the tone curve a highlight to roll off.
      color: new THREE.Color(4.2, 1.35, 0.3),
      toneMapped: false,
    });
    const ember = new THREE.Mesh(emberGeometry, emberMaterial);
    ember.name = 'campfire.ember';
    ember.position.y = 0.16;
    group.add(ember);
    this.#fireEmber = ember;
    this.#disposables.push(emberGeometry, emberMaterial);

    // Clustered light: the correct one for surface shading.
    this.#fireLight =
      this.#lighting?.addLight({
        kind: 'point',
        name: 'campfire',
        position: { x, y: base + 0.45, z },
        color: FIRE_COLOR,
        intensity: FIRE_INTENSITY,
        radius: 9,
        decay: 2,
        castShadow: true,
        // Highest priority in the scene: if the clustered rig ever has to drop
        // a shadowed light, this must not be the one it drops.
        priority: 10,
      }) ?? null;

    // Scene light: the one the volumetric bridge finds. Zero-intensity would
    // make it invisible to the bridge's `intensity > 0` filter, so it carries
    // the real value and the two lights are kept in step in `update`.
    const glow = new THREE.PointLight(new THREE.Color(FIRE_COLOR), FIRE_INTENSITY, 9, 2);
    glow.name = 'campfire.scatter';
    glow.position.set(x, base + 0.45, z);
    glow.castShadow = false;
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

    this.#placeTrees(pick(PROP_KEYS.trees));
    this.#placeRocks(pick(PROP_KEYS.rocks));
    this.#placeFence(pick(PROP_KEYS.fence));
    this.#placeCampsiteClutter(pick(PROP_KEYS.clutter), pick(PROP_KEYS.logs));
    this.#placeRubble(pick(PROP_KEYS.rubble));
  }

  /**
   * Bare trees, concentrated on the ridge.
   *
   * The density function is the composition: near-zero in the basin (a tree in
   * front of the fire would block the focal point), rising sharply past
   * z = −18 so the ridge carries a treeline. Placement is rejected above 34° of
   * slope, which keeps trees off the escarpment.
   *
   * Scale varies 0.75–1.5 and yaw is fully random. Both matter more than they
   * sound: identical trees at identical scale read as a texture, and the eye
   * finds the repeat instantly.
   */
  #placeTrees(keys: readonly AssetKey[]): void {
    if (keys.length === 0) return;
    const samples = scatter({
      count: Math.round(78 * this.#scatterDensity),
      area: 120,
      center: new THREE.Vector3(-4, 0, -30),
      seed: `${SCATTER_SEED}.trees`,
      scaleRange: [0.75, 1.5],
      surface: this.field.surface,
      maxSlopeDegrees: 34,
      minSpacing: 3.4,
      density: (x, z) => {
        const ridge = smootherstep(-12, -30, z);
        const clearing = 1 - 0.9 * Math.exp(-((x - 1) ** 2 + (z + 4) ** 2) / 260);
        return ridge * clearing;
      },
    });
    this.#instance(keys, samples, 'tree', {
      alignToGround: 0.25,
      shadow: true,
      targetHeight: 5.4,
    });
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
      density: (x, z) => 1 - 0.95 * Math.exp(-((x - CAMPFIRE.x) ** 2 + (z - CAMPFIRE.y) ** 2) / 22),
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
    const path: readonly THREE.Vector2[] = [
      new THREE.Vector2(13.5, 15.5),
      new THREE.Vector2(10.2, 8.4),
      new THREE.Vector2(8.4, 1.2),
      new THREE.Vector2(7.6, -5.6),
      new THREE.Vector2(8.9, -12.8),
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

  /** Barrels, crates and firewood, clustered at the camp rather than strewn. */
  #placeCampsiteClutter(clutter: readonly AssetKey[], logs: readonly AssetKey[]): void {
    if (clutter.length > 0) {
      const samples = scatter({
        count: Math.round(11 * this.#scatterDensity) + 3,
        area: 11,
        center: new THREE.Vector3(CAMPFIRE.x - 1.4, 0, CAMPFIRE.y - 2.6),
        seed: `${SCATTER_SEED}.clutter`,
        scaleRange: [0.85, 1.15],
        surface: this.field.surface,
        maxSlopeDegrees: 26,
        minSpacing: 0.95,
      });
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
        area: 5.5,
        center: new THREE.Vector3(CAMPFIRE.x + 1.5, 0, CAMPFIRE.y + 1.4),
        seed: `${SCATTER_SEED}.logs`,
        scaleRange: [0.8, 1.1],
        surface: this.field.surface,
        maxSlopeDegrees: 30,
        minSpacing: 0.8,
      });
      this.#instance(logs, samples, 'logs', {
        alignToGround: 0.8,
        shadow: true,
        lean: 0.22,
        targetHeight: 0.55,
      });
    }
  }

  /** Rubble, weighted toward the wall it fell off. */
  #placeRubble(keys: readonly AssetKey[]): void {
    if (keys.length === 0) return;
    const samples = scatter({
      count: Math.round(34 * this.#scatterDensity),
      area: 26,
      center: new THREE.Vector3(-5.0, 0, -10.0),
      seed: `${SCATTER_SEED}.rubble`,
      scaleRange: [0.6, 1.35],
      surface: this.field.surface,
      maxSlopeDegrees: 40,
      minSpacing: 1.1,
      alignToNormal: true,
    });
    this.#instance(keys, samples, 'rubble', {
      alignToGround: 1,
      shadow: true,
      sink: 0.08,
      targetHeight: 0.6,
    });
  }

  /**
   * Clone a prop per sample and place it.
   *
   * Clones share geometry and materials with the cached GLTF, so N copies of a
   * tree cost N draw calls but one upload. That is the right trade at this
   * count: `InstancedMesh` would collapse the draws but a GLTF prop is a small
   * hierarchy of several meshes with different materials, and flattening it
   * per-material for ~250 objects buys less than it costs in complexity here.
   * Phase 3's streaming zone is where instancing earns its keep.
   */
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
  #weatherMaterials(object: THREE.Object3D): void {
    object.traverse((child) => {
      if (!(child instanceof THREE.Mesh)) return;
      const source = child.material;
      const list = Array.isArray(source) ? source : [source];
      const weathered = list.map((material) => {
        const existing = this.#materialCache.get(material);
        if (existing !== undefined) return existing;
        const clone = weatherMaterial(material);
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
      this.#root.add(hero);
      this.#hero = hero;

      const clips = gltf.animations;
      if (clips.length === 0) {
        console.warn('[BloodMoor] the Barbarian GLTF has no animation clips; pose is the bind pose.');
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

  /* -- diagnostics -------------------------------------------------------- */

  get hero(): THREE.Object3D | null {
    return this.#hero;
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
const WEATHERING_TINT = new THREE.Color(0.30, 0.31, 0.34);

/** How far a prop albedo is pulled toward its own luminance. */
const WEATHERING_DESATURATION = 0.66;

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
 */
const FIRE_INTENSITY = 9.5;


/**
 * Rebuild one prop material as a weathered node material.
 *
 * The colour correction has to happen in the shader, not on
 * `material.color`. These packs put all of their colour in a palette atlas and
 * leave the base-colour *factor* white, so multiplying the factor can darken a
 * prop but can never desaturate one — and saturation is the actual problem: a
 * primary-green conifer and a primary-red banner do not belong in a frame whose
 * whole thesis is that the only saturated thing in it is the fire.
 *
 * So the sampled albedo is pulled two-thirds of the way to its own luminance
 * and then multiplied by {@link WEATHERING_TINT}. Hue survives as a trace,
 * which is what keeps the props from becoming grey cutouts, and everything
 * lands within about a stop of the mud it stands on.
 */
function weatherMaterial(source: THREE.Material): THREE.Material {
  // Duck-typed rather than `instanceof`. The `three` -> `three/webgpu` alias in
  // `vite.config.ts` should make the two identical, but a material that arrives
  // from a second copy of three would fail an `instanceof` check silently and
  // pass straight through unweathered — which is a bug that only shows up as a
  // colour that is subtly wrong. The `isMeshStandardMaterial` flag is three's
  // own cross-realm type tag and cannot lie.
  const standard = isStandardLike(source) ? source : null;
  if (standard === null) return source.clone();

  const material = new THREE.MeshStandardNodeMaterial();
  material.name = `${standard.name}.weathered`;
  material.side = standard.side;
  material.transparent = standard.transparent;
  material.alphaTest = standard.alphaTest;
  material.normalMap = standard.normalMap;
  material.normalScale.copy(standard.normalScale);
  material.roughness = THREE.MathUtils.clamp(standard.roughness * 0.6 + 0.42, 0.5, 1);
  material.metalness = Math.min(standard.metalness, 0.2);
  material.envMapIntensity = 0.75;
  material.vertexColors = standard.vertexColors;

  const base = standard.map;
  if (base !== null) {
    const sampled = textureNode(base, uv()).rgb;
    const luma = dot(sampled, vec3(0.2126, 0.7152, 0.0722));
    material.colorNode = mix(sampled, vec3(luma), WEATHERING_DESATURATION).mul(
      vec3(WEATHERING_TINT.r, WEATHERING_TINT.g, WEATHERING_TINT.b),
    );
    material.map = base;
  } else {
    material.color.copy(standard.color).multiply(WEATHERING_TINT);
  }
  return material;
}

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

/** three's own cross-realm type tag for the standard/physical material family. */
function isStandardLike(
  material: THREE.Material,
): material is THREE.MeshStandardMaterial | THREE.MeshPhysicalMaterial {
  const tagged = material as unknown as {
    isMeshStandardMaterial?: boolean;
    isMeshPhysicalMaterial?: boolean;
  };
  return tagged.isMeshStandardMaterial === true || tagged.isMeshPhysicalMaterial === true;
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
  root.clear();
}
