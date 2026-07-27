/**
 * @module scene/DenOfEvil
 *
 * The generated cave, realised as geometry, physics and light.
 *
 * {@link module:world/DungeonGenerator} produces the *layout* — a grid, an
 * entrance, chambers, spawn points — and knows nothing about three.js. This
 * module is the other half: it turns that grid into a cave a player can walk
 * through, fight in and see.
 *
 * ## Four surfaces, and why the walls are built twice
 *
 * ```
 *   floor    two triangles per walkable cell, height from noise
 *   skin     marching-squares contour, extruded and noise-displaced  <- what you see
 *   backing  greedy-meshed boxes filling the solid cells             <- what you hit
 *   ceiling  the floor mesh again, raised and inverted
 * ```
 *
 * The walls exist twice on purpose, and it is the central decision in this file.
 *
 * A cave wall built from the grid directly is a staircase: every surface is
 * axis-aligned and one metre wide, and no rock texture rescues it. Marching
 * squares fixes that — sampling the "is this floor" field at cell *centres* and
 * cutting the contour through the midpoints between them turns every 90-degree
 * step into a 45-degree cut — and then a noise displacement applied as a **pure
 * function of the vertex's own position** roughens it into rock. Purity is what
 * makes the seams disappear: two independently emitted quads that happened to
 * share a corner still share it after displacement, so the surface is continuous
 * without any of the loop-chaining a real contour extraction would need.
 *
 * But that skin is a *surface*. It has no volume, its normals are ambiguous
 * where the contour is thin, and the camera arm sphere-cast would happily pass
 * through it. So the solid cells behind it are also filled with boxes — greedy
 * meshed, see {@link DenOfEvil.prototype} `#computeBacking` — and those are what
 * carry the collision and what the camera stops against. Together they give a
 * wall with a rock face on the inside and three metres of literal rock behind it.
 *
 * ## Light
 *
 * The brief is "genuinely dark and oppressive but still navigable", which is a
 * contradiction unless the darkness is *structured*. It is done with three
 * tiers and nothing else:
 *
 * - a **hemisphere fill** at 0.5, cold above and warm below, plus the `env.cave`
 *   HDRI at 0.28. Neither is lighting so much as a floor under black: together
 *   they keep the near floor legible and rock reading as rock, and they are what
 *   the first pass got wrong. At 0.22 and 0.16 the capture came back black
 *   except for one torch — which is not oppressive, it is a zone the player
 *   cannot fight in.
 * - **fourteen guttering torches**, warm, radius 10 m, one per chamber and one
 *   per depth band, never within 7 cells of each other. Fourteen is deliberately
 *   more than the eight-point light budget: `LightingService` culls by importance
 *   every frame, so over-registering lets the arbitration bind whichever are
 *   nearest as the player moves, instead of leaving 20 m of absolute black
 *   between five fixed ones.
 * - **luminous fungus**, cold blue-green, weak, in the stretches between the
 *   torches. Cold against warm is what makes the warm read as warm; without it
 *   the torch pools sit in undifferentiated black and the cave has two states
 *   rather than a gradient.
 *
 * The sun is switched off outright and the sky background replaced, because a
 * directional light does not care about a ceiling and an overcast dome inside a
 * cave is the single most immersion-breaking thing this zone could do. Both are
 * restored on unload — see {@link DenOfEvil.dispose}.
 *
 * ## Cost
 *
 * At the default 88x88 grid with roughly 2 600 walkable cells:
 *
 * | element     | draws | triangles | note                                    |
 * |-------------|-------|-----------|-----------------------------------------|
 * | floor       | 1     | ~7 400    | dilated by one cell so no gap at the wall |
 * | wall skin   | 1     | ~9 600    | ~600 contour segments x 4 rings         |
 * | wall backing| 1     | ~1 600    | greedy-meshed slabs                      |
 * | ceiling     | 1     | ~9 000    | |
 * | stalagmites | 1     | ~28 000   | instanced, 1 280 tris each              |
 * | bone piles  | 1     | ~9 000    | instanced                               |
 * | fungus      | 1     | ~1 900    | instanced                               |
 * | torches     | 28    | ~2 200    | 14 x (ember + plume)                    |
 * | rubble      | ~4    | ~6 000    | kit props                               |
 *
 * About 31 draw calls and 75 k triangles for the whole cave, on two photoscanned
 * material sets. The collider count is the number that actually deserves
 * scrutiny, and it is the reason the backing is greedy-meshed: emitted per cell
 * it is roughly 2 900 boxes, merged into row runs 646, and greedy-meshed in two
 * dimensions it is the figure `verify-zones` reports.
 */

import * as THREE from 'three/webgpu';
import { mergeGeometries } from 'three/examples/jsm/utils/BufferGeometryUtils.js';

import { AssetManagerKey, type AssetKey, type AssetManager } from '../assets/AssetManager';
import { SimplexNoise, buildInstancedMesh, createRng, generateRockGeometry, type ScatterSample } from '../assets/Procedural';
import type { SpawnPoint } from '../ai/EnemyDirector';
import type { GameContext } from '../core/types';
import type { PhysicsWorld } from '../physics/PhysicsWorld';
import { COLLISION_GROUPS } from '../physics/Layers';
import { IBLKey, type IBLService } from '../render/IBL';
import { LightingKey, type LightHandle, type LightingService } from '../render/Lighting';
import { MaterialLibraryKey, type MaterialLibraryService } from '../render/MaterialLibrary';
import { TimeOfDayKey, type TimeOfDay } from '../render/TimeOfDay';
import { VolumetricsKey, type VolumetricsService } from '../render/Volumetrics';
import {
  CaveCell,
  generateDungeon,
  worldOfCell,
  type DungeonLayout,
  type GridPoint,
} from '../world/DungeonGenerator';
import type { PortalSpec, Zone, ZoneEntryPoint } from '../world/Zone';
import { Fire } from './Fire';
import { PropKit } from './PropKit';
import type { WeatheringGrade } from './Weathering';

/* -------------------------------------------------------------------------- */
/* Constants                                                                  */
/* -------------------------------------------------------------------------- */

const SEED = 'd2rim.denOfEvil';

/** Ceiling height above the floor in a corridor, in metres. */
const CEILING_BASE = 3.4;
/** Extra ceiling height per cell of wall clearance, so chambers feel bigger. */
const CEILING_PER_CLEARANCE = 0.5;
const CEILING_MAX_BONUS = 3.4;

/** Peak horizontal roughening of the wall skin, in metres. */
const WALL_ROUGHNESS = 0.38;
/** Vertical rings in the extruded wall skin. More is smoother and costs quads. */
const WALL_RINGS = 5;
/**
 * How far the solid backing extends behind the visible face, in cells.
 *
 * 3 m of rock, and the intuition that a thinner shell would be cheaper is
 * exactly wrong — it was measured both ways. A shell two cells thick around an
 * irregular cave boundary fragments into *more* greedy rectangles than one three
 * cells thick (631 colliders against 579 on the default seed), because the
 * thicker shell gives the rectangle merge room to find large squares where the
 * thin one only ever finds slivers. So 3 is both the safer wall and the cheaper
 * one.
 */
const BACKING_DEPTH = 3;

/** How many torches to register. More than the light budget, on purpose. */
const TORCH_COUNT = 14;
const TORCH_COLOR = 0xff8330;
const FUNGUS_COLOR = 0x5fd6c8;

/**
 * The cave grade: much darker and colder than the surface.
 *
 * `envMapIntensity` is cut hard because the only environment in here is the cave
 * HDRI at 0.16 — a prop that catches it strongly would be brighter than the rock
 * it stands on, which is the exact tell that a prop was authored for a different
 * lighting environment.
 */
const CAVE_WEATHERING: WeatheringGrade = {
  desaturation: 0.42,
  shadowTint: new THREE.Color(0.07, 0.075, 0.09),
  lightTint: new THREE.Color(0.185, 0.17, 0.15),
  warmClamp: 0.6,
  grime: 0.55,
  wear: 0.1,
  roughnessFloor: 0.68,
  envMapIntensity: 0.25,
};

const PROP_KEYS: readonly AssetKey[] = [
  'prop.rubble.large',
  'prop.rubble.half',
  'prop.barrel.small',
  'prop.crate.small',
];

/* -------------------------------------------------------------------------- */
/* Floor and ceiling height                                                   */
/* -------------------------------------------------------------------------- */

/**
 * The cave floor's height function.
 *
 * Sampled by the visual floor mesh, by every prop and stalagmite placement, and
 * by the Rapier heightfield — one function, three consumers, for the same reason
 * every other zone in this project has exactly one.
 *
 * The amplitude is deliberately small (±0.22 m at two scales). A cave floor is
 * uneven, but the generator's whole width guarantee is about *horizontal*
 * navigability, and a floor that undulates by a metre would quietly reintroduce
 * impassable geometry the flood fill cannot see.
 */
export class CaveFloor {
  readonly #noise: SimplexNoise;

  constructor(seed: string | number = `${SEED}.floor`) {
    this.#noise = new SimplexNoise(seed);
  }

  heightAt(x: number, z: number): number {
    const noise = this.#noise;
    return (
      noise.noise2D(x * 0.055, z * 0.055) * 0.22 +
      noise.noise2D(x * 0.21, z * 0.21) * 0.075 +
      noise.noise2D(x * 0.63, z * 0.63) * 0.028
    );
  }

  normalAt(x: number, z: number, epsilon = 0.5): THREE.Vector3 {
    const dx = this.heightAt(x + epsilon, z) - this.heightAt(x - epsilon, z);
    const dz = this.heightAt(x, z + epsilon) - this.heightAt(x, z - epsilon);
    return new THREE.Vector3(-dx, 2 * epsilon, -dz).normalize();
  }

  readonly surface = (x: number, z: number): { y: number; normal: THREE.Vector3 } => ({
    y: this.heightAt(x, z),
    normal: this.normalAt(x, z),
  });
}

/* -------------------------------------------------------------------------- */
/* Module                                                                     */
/* -------------------------------------------------------------------------- */

export interface DenOfEvilOptions {
  /** Seed passed straight to {@link generateDungeon}. */
  readonly seed?: string | number;
  /** Grid size, in cells. Square: the Rapier heightfield is square. */
  readonly cells?: number;
  /** Metres per cell. */
  readonly cellSize?: number;
  /** Override the generated spawn count. */
  readonly spawnCount?: number;
}

export class DenOfEvil implements Zone {
  readonly name = 'scene.denOfEvil';
  readonly zoneId = 'denOfEvil';
  readonly displayName = 'The Den of Evil';

  /**
   * The cave's trim on the frame. The largest of the three, and it has to be.
   *
   * ### What the honest capture showed
   *
   * `captures/zones-before/denOfEvil.png`: **55.45% of the frame below luma
   * 0.02**. Over half the picture was not dark, it was *absent* — no detail, no
   * hue, nothing for the eye to navigate by. Median luma 0.016, third quartile
   * 0.046, and a mean saturation of 0.835 because the only thing left in the
   * frame was the orange of a torch on black. A player cannot fight twenty
   * skeletons in that.
   *
   * ### Why exposure alone is the wrong instrument
   *
   * The torch pool already peaks near 0.57 and the fire itself is above that.
   * The two-plus stops needed to drag the median to a navigable 0.08 would put
   * the torches into hard clipping, and a clipped highlight in a dark frame is
   * the one artefact that reads worse than the darkness did. So the lift comes
   * mostly from `gamma`, which is a *power* on the midtones and shadows and
   * leaves the top of the range where it is:
   *
   * - `gamma [1.60, 1.52, 1.42]` — above 1 brightens, see `ColorGradeSettings`
   *   — raises 0.016 to roughly 0.076 and 0.046 to about 0.15, while the torch
   *   pool's 0.57 moves only to about 0.70, so nothing near the flame goes near
   *   clipping. Red is lifted *most* and blue least, which sounds backwards and
   *   is not: the recovered population is rock lit by nothing, and pulling its
   *   red up furthest is what keeps the torch pools reading as the warm places
   *   rather than as the only lit ones.
   * - `lift` puts a small coloured floor under the blacks. Grimdark shadows
   *   have to retain hue; a shadow with nothing in it has no hue to retain, and
   *   this is the term that guarantees the darkest part of the cave is a cold
   *   blue-grey rather than a hole in the screen.
   * - `stops: +0.45` on top, which is as far as the torches will take.
   * - `contrast: 0.94` about a pivot at **0.10** — the cave's actual midtone
   *   after the gamma. Any contrast above 1 here re-crushes exactly the
   *   population this whole trim exists to recover.
   * - `saturation: 0.8` against a measured 0.835: the orange has to stop being
   *   the only colour in the room.
   *
   * The cave must stay oppressive. The target is not a lit room — it is a frame
   * where the darkness is *legible*, with the torch pools still clearly the
   * safe places and the rock between them readable enough to fight in.
   */
  readonly grade = {
    stops: 0.45,
    grade: {
      temperature: -900,
      tint: -0.01,
      lift: [0.006, 0.009, 0.018] as const,
      gamma: [1.6, 1.52, 1.42] as const,
      gain: [1.02, 0.99, 0.96] as const,
      saturation: 0.8,
      contrast: 0.94,
      contrastPivot: 0.1,
      vignette: 0.24,
    },
  };

  /**
   * The generated layout.
   *
   * Public and readonly because phase 5's quest system needs it: the objective
   * ("clear the Den of Evil") is a count against {@link DungeonLayout.spawnPoints},
   * and the objective's *location* is {@link DungeonLayout.deepestChamber}.
   * Neither needs the scene.
   */
  readonly layout: DungeonLayout;
  readonly field = new CaveFloor();

  readonly #root = new THREE.Group();
  readonly #owned: { dispose(): void }[] = [];
  readonly #fires: Fire[] = [];
  readonly #lights: { handle: LightHandle | null; base: number; fire: Fire | null }[] = [];
  readonly #entry: ZoneEntryPoint[] = [];
  readonly #portals: PortalSpec[] = [];
  readonly #spawns: SpawnPoint[] = [];
  /** Greedy-meshed rectangles filling the solid rock, in cells. */
  readonly #backing: { col0: number; col1: number; row0: number; row1: number }[] = [];

  #kit: PropKit | null = null;
  #materials: MaterialLibraryService | null = null;
  #ibl: IBLService | null = null;
  #lighting: LightingService | null = null;
  #volumetrics: VolumetricsService | null = null;
  #timeOfDay: TimeOfDay | null = null;
  #ctx: GameContext | null = null;
  #caveEnvironment: THREE.Texture | null = null;
  #previousBackground: THREE.Color | THREE.Texture | null = null;

  constructor(options: DenOfEvilOptions = {}) {
    this.#root.name = 'DenOfEvil';
    const cells = options.cells ?? 88;
    this.layout = generateDungeon({
      seed: options.seed ?? SEED,
      cols: cells,
      rows: cells,
      cellSize: options.cellSize ?? 1,
      ...(options.spawnCount === undefined ? {} : { spawnCount: options.spawnCount }),
    });

    this.#buildNavigationPoints();
    for (const spawn of this.layout.spawnPoints) {
      this.#spawns.push({
        variant: spawn.variant,
        x: spawn.x,
        z: spawn.z,
        patrol: spawn.patrol,
      });
    }
  }

  get root(): THREE.Object3D {
    return this.#root;
  }

  get entryPoints(): readonly ZoneEntryPoint[] {
    return this.#entry;
  }

  get portals(): readonly PortalSpec[] {
    return this.#portals;
  }

  get enemySpawns(): readonly SpawnPoint[] {
    return this.#spawns;
  }

  /** Where the phase-5 quest objective goes: the centre of the deepest chamber. */
  get objective(): { x: number; y: number; z: number } {
    const chamber = this.layout.deepestChamber;
    return {
      x: chamber.centerWorld.x,
      y: this.field.heightAt(chamber.centerWorld.x, chamber.centerWorld.z),
      z: chamber.centerWorld.z,
    };
  }

  /** Ceiling height above the floor at a cell, driven by how open it is. */
  ceilingAt(col: number, row: number): number {
    const clearance = this.layout.clearance[row * this.layout.cols + col] ?? 0;
    return CEILING_BASE + Math.min(CEILING_MAX_BONUS, clearance * CEILING_PER_CLEARANCE);
  }

  /* -- lifecycle ---------------------------------------------------------- */

  async init(ctx: GameContext): Promise<void> {
    this.#ctx = ctx;
    ctx.scene.add(this.#root);

    const assets = ctx.services.tryGet<AssetManager>(AssetManagerKey) ?? null;
    this.#materials = ctx.services.tryGet(MaterialLibraryKey) ?? null;
    this.#ibl = ctx.services.tryGet(IBLKey) ?? null;
    this.#lighting = ctx.services.tryGet(LightingKey) ?? null;
    this.#volumetrics = ctx.services.tryGet(VolumetricsKey) ?? null;
    this.#timeOfDay = ctx.services.tryGet(TimeOfDayKey) ?? null;

    await this.#materials?.ready();
    await this.#enterInterior(ctx, assets);

    this.#computeBacking();
    this.#buildFloor();
    this.#buildCeiling();
    this.#buildWallSkin();
    this.#buildWallBacking();
    this.#buildStalagmites();
    this.#buildBones();
    this.#buildFungus();
    this.#buildTorches();

    if (assets !== null) {
      this.#kit = new PropKit(assets, { grade: CAVE_WEATHERING, castShadow: false });
      await this.#kit.load(PROP_KEYS);
      this.#placeDebris();
    }

    this.#applyOcclusion();
    console.info(
      `[DenOfEvil] seed ${this.layout.seed} (attempt ${this.layout.attempt}): ` +
        `${this.layout.floorCells} walkable cells, ${this.layout.chambers.length} chambers, ` +
        `${this.layout.spawnPoints.length} spawns, depth ${this.layout.maxDepth}`,
    );
  }

  update(ctx: GameContext): void {
    const t = ctx.time.elapsed;
    for (const entry of this.#lights) {
      const flicker = entry.fire === null ? 1 : entry.fire.update(t);
      entry.handle?.setIntensity(entry.base * flicker);
    }
    // The sky module reinstalls its own environment whenever it rebuilds, which
    // it does as the time of day advances. One reference comparison per frame is
    // a much cheaper guarantee than trying to suppress the rebuild.
    const ibl = this.#ibl;
    if (ibl !== null && this.#caveEnvironment !== null && ibl.environment !== this.#caveEnvironment) {
      ibl.setEnvironment(this.#caveEnvironment);
    }
  }

  /**
   * Give the surface back its sky.
   *
   * Everything {@link #enterInterior} switched off is switched back on here, and
   * the sun is restored by asking {@link TimeOfDay} to recompute rather than by
   * remembering the old configuration — the astronomy is the authority on what
   * the sun should be, and a cached copy of it goes stale the moment the clock
   * advances while the player is underground.
   */
  dispose(): void {
    for (const entry of this.#lights) entry.handle?.release();
    this.#lights.length = 0;
    for (const fire of this.#fires) fire.dispose();
    this.#fires.length = 0;
    this.#kit?.dispose();
    this.#kit = null;

    const ctx = this.#ctx;
    if (ctx !== null) {
      ctx.scene.background = this.#previousBackground;
      const scene = ctx.scene as unknown as { backgroundNode: unknown };
      scene.backgroundNode = this.#savedBackgroundNode;
    }
    this.#ibl?.setIntensity(1);
    this.#caveEnvironment = null;
    this.#lighting?.setAmbient({ intensity: 0 });
    this.#volumetrics?.setParams({
      density: 0.014,
      height: 1.5,
      heightFalloff: 0.085,
      sunScatteringScale: 1,
      ambientScatteringScale: 1,
      volumeDistance: 64,
    });
    // Restores the sun's direction, colour, intensity and shadow state in one
    // call, from the astronomy rather than from a snapshot.
    this.#timeOfDay?.refresh();

    for (const resource of this.#owned) resource.dispose();
    this.#owned.length = 0;
    this.#ctx = null;
  }

  #savedBackgroundNode: unknown = null;

  /* -- physics ------------------------------------------------------------ */

  /**
   * Two collider families, matching the two wall representations.
   *
   * The floor is a Rapier heightfield sampled from {@link CaveFloor} at exactly
   * the generator's grid resolution, so the collider surface and the rendered
   * surface are the same surface. It extends under the rock as well, which costs
   * nothing and means a player who somehow ends up behind a wall still has
   * ground under him rather than falling out of the world.
   *
   * The walls are the greedy-meshed row runs, as cuboids on the prop layer. On
   * the prop layer specifically, not terrain: `findClearSpot` and `groundHeight`
   * treat terrain as "the floor" and props as "things in the way", and a wall
   * that registered as floor would let the spawn search place the player on top
   * of the rock.
   */
  buildColliders(physics: PhysicsWorld): void {
    const { cols, cellSize } = this.layout;
    physics.buildTerrain(this.field, cols * cellSize, cols);

    const half = this.layout.halfExtent;
    for (const run of this.#backing) {
      const width = (run.col1 - run.col0 + 1) * cellSize;
      const depth = (run.row1 - run.row0 + 1) * cellSize;
      const centreX = ((run.col0 + run.col1 + 1) / 2) * cellSize - half.x;
      const centreZ = ((run.row0 + run.row1 + 1) / 2) * cellSize - half.z;
      const top = CEILING_BASE + CEILING_MAX_BONUS + 1.5;
      const bottom = -2.5;
      const height = top - bottom;
      const desc = physics.rapier.ColliderDesc.cuboid(width / 2, height / 2, depth / 2)
        .setTranslation(centreX, bottom + height / 2, centreZ)
        .setCollisionGroups(COLLISION_GROUPS.prop)
        .setFriction(0.9)
        .setRestitution(0);
      physics.addCollider(desc, { kind: 'prop', label: `den.wall.${run.row0}.${run.col0}` });
    }

    // The props are derived the ordinary way, but the cave shell must not be:
    // fitting a box to the wall-skin mesh's bounds would produce one collider
    // the size of the entire cave.
    physics.buildSceneColliders(this.#root, {
      exclude: /den\.(floor|ceiling|wall|fungus)|fire\.|ember|plume/i,
      maxExtent: 12,
    });
  }

  /* -- interior lighting --------------------------------------------------- */

  async #enterInterior(ctx: GameContext, assets: AssetManager | null): Promise<void> {
    const scene = ctx.scene as unknown as { backgroundNode: unknown };
    this.#savedBackgroundNode = scene.backgroundNode;
    this.#previousBackground = ctx.scene.background;
    // The sky dome is a node on the scene, not an object in it, so hiding it is
    // an assignment rather than a traversal. A near-black background rather than
    // pure black: an absolutely black backdrop makes any pixel where the ceiling
    // has a gap read as a hole in the render rather than as darkness.
    scene.backgroundNode = null;
    ctx.scene.background = new THREE.Color(0x05070a);

    // A directional light does not know it is indoors. Nothing else will stop it.
    this.#lighting?.setSun({ intensity: 0, castShadow: false });
    // ... and with no sun, the hemisphere fill becomes the only thing keeping
    // upward-facing surfaces from being identical to downward-facing ones.
    // 0.5, up from 0.22, and the sky/ground pair is deliberately split cold
    // over warm rather than being one dim grey.
    //
    // The first pass measured "oppressive" and got it: at 0.22 with the IBL at
    // 0.16 the capture was black except for one torch and a fungus dot — the
    // floor was not visible at all, which is not atmosphere, it is a zone the
    // player cannot fight in. Darkness in a cave has to be *structured*: the
    // near floor legible, the middle distance falling off fast, the far end
    // black. The hemisphere fill is what buys the first of those three, and it
    // is the cheapest light in the engine.
    this.#lighting?.setAmbient({
      // The colours matter more than the intensity, and that is what the two
      // previous passes got wrong. A hemisphere light multiplies its colour by
      // its intensity, so 0x1c242e — already a near-black blue — at 0.5 is still
      // near-black, and raising the intensity alone chases a number that cannot
      // reach. These are mid-tones cut by intensity instead: cold from above,
      // a warm bounce off the cave floor, and enough of both that a player can
      // see where the floor is at four metres.
      skyColor: 0x5c6b7a,
      groundColor: 0x46352a,
      intensity: 0.72,
    });

    this.#volumetrics?.setParams({
      // Four times the moor's. Cave air is damp and the torch pools want
      // something to be visible in.
      density: 0.052,
      // Flat, not height-falling: fog in a cave fills the volume.
      height: 40,
      heightFalloff: 0.01,
      sunScatteringScale: 0,
      ambientScatteringScale: 0.35,
      lightScatteringScale: 1.4,
      volumeDistance: 34,
    });

    if (assets === null || !assets.has('env.cave')) return;
    try {
      const texture = await assets.loadEnvironment('env.cave');
      assets.pin('env.cave');
      this.#caveEnvironment = texture;
      this.#ibl?.setEnvironment(texture);
      // 0.45. Still less than half the surface's, but this is the term that
      // gives the rock its *form* — the hemisphere fill is flat by construction
      // and cannot describe a surface, so cutting the IBL too far leaves walls
      // that are lit but shapeless.
      this.#ibl?.setIntensity(0.45);
    } catch (error) {
      console.warn('[DenOfEvil] env.cave failed to load; falling back to local light only:', error);
      this.#ibl?.setIntensity(0.35);
    }
  }

  /* -- navigation points --------------------------------------------------- */

  /**
   * The entry point and the portal back out.
   *
   * They are deliberately not the same place. The portal sits on the entrance
   * cell; the arrival point is the reachable floor cell six steps further in.
   * Landing a player inside the volume he just came through means the first
   * thing the new zone does is offer to send him back, which reads as the
   * transition having failed.
   */
  #buildNavigationPoints(): void {
    const layout = this.layout;
    const mouth = layout.entranceWorld;
    // `openCellAtDepth`, not `cellAtDepth`: the arrival point is a place a
    // character stands and a camera orbits, so it needs room around it and not
    // merely floor under it. See the note on `cellAtDepth`.
    const inner = openCellAtDepth(layout, 6) ?? layout.entrance;
    const innerWorld = worldOfCell(layout, inner.col, inner.row);
    const floorY = (x: number, z: number): number => this.field.heightAt(x, z);

    this.#entry.push(
      {
        id: 'cave-mouth',
        position: { x: innerWorld.x, y: floorY(innerWorld.x, innerWorld.z), z: innerWorld.z },
        yaw: Math.atan2(-innerWorld.x, -innerWorld.z),
      },
      {
        id: 'deepest-chamber',
        position: this.objective,
        yaw: 0,
      },
    );

    this.#portals.push({
      id: 'den-exit',
      targetZone: 'bloodMoor',
      targetEntry: 'from-den',
      position: { x: mouth.x, y: floorY(mouth.x, mouth.z), z: mouth.z },
      radius: 2.4,
      height: 3,
      label: 'the Blood Moor',
      verb: 'climb back out to',
    });
  }

  /* -- geometry: floor and ceiling ------------------------------------------ */

  /**
   * Two triangles per cell, over the walkable set dilated by one.
   *
   * The dilation matters: the wall skin's contour runs through the midpoints
   * between floor and rock cell centres, i.e. roughly along the cell boundary,
   * and after roughening it can sit up to {@link WALL_ROUGHNESS} outside that.
   * A floor that stopped exactly at the walkable cells would show a gap at the
   * base of the wall, and a gap at the base of a wall is a hole into space.
   */
  #buildFloor(): void {
    const geometry = this.#gridSurface(1, (x, z) => this.field.heightAt(x, z), false);
    this.#owned.push(geometry);
    const material =
      this.#materials?.create('rock', {
        albedoTint: [0.15, 0.145, 0.14],
        roughnessRange: [0.72, 1],
        tiling: 0.42,
        // Cave floors are damp everywhere; the wetness system is the cheapest
        // way to say so and it is already paid for.
        wetnessExposure: 0.85,
        porosity: 0.7,
      }) ?? new THREE.MeshStandardNodeMaterial({ color: 0x1b1a18, roughness: 0.95 });
    material.name = 'den.floor';
    this.#owned.push(material);

    const mesh = new THREE.Mesh(geometry, material);
    mesh.name = 'den.floor';
    mesh.receiveShadow = true;
    mesh.castShadow = false;
    mesh.userData['noCollide'] = true;
    this.#root.add(mesh);
  }

  #buildCeiling(): void {
    const geometry = this.#gridSurface(
      2,
      (x, z, col, row) => this.field.heightAt(x, z) + this.ceilingAt(col, row),
      true,
    );
    this.#owned.push(geometry);
    const material =
      this.#materials?.create('rock', {
        albedoTint: [0.075, 0.072, 0.075],
        roughnessRange: [0.82, 1],
        tiling: 0.3,
      }) ?? new THREE.MeshStandardNodeMaterial({ color: 0x0d0d0e, roughness: 0.98 });
    material.name = 'den.ceiling';
    this.#owned.push(material);

    const mesh = new THREE.Mesh(geometry, material);
    mesh.name = 'den.ceiling';
    mesh.receiveShadow = true;
    mesh.castShadow = false;
    mesh.userData['noCollide'] = true;
    this.#root.add(mesh);
  }

  /**
   * Build a triangulated surface over the walkable set dilated by `grow` cells.
   *
   * `flip` reverses the winding, which is what turns the floor mesh into a
   * ceiling: the same triangles seen from underneath.
   */
  #gridSurface(
    grow: number,
    height: (x: number, z: number, col: number, row: number) => number,
    flip: boolean,
  ): THREE.BufferGeometry {
    const { cols, rows, cellSize, cells } = this.layout;
    const half = this.layout.halfExtent;
    const mask = dilate(cells, cols, rows, grow);

    const positions: number[] = [];
    const uvs: number[] = [];
    const corner = (col: number, row: number): [number, number, number] => {
      const x = col * cellSize - half.x;
      const z = row * cellSize - half.z;
      // Sampled at the *corner*, from the same pure function, so neighbouring
      // cells agree on the shared vertex and the surface has no cracks.
      const clampedCol = Math.min(cols - 1, Math.max(0, col));
      const clampedRow = Math.min(rows - 1, Math.max(0, row));
      return [x, height(x, z, clampedCol, clampedRow), z];
    };

    for (let row = 0; row < rows; row++) {
      for (let col = 0; col < cols; col++) {
        if (mask[row * cols + col] !== CaveCell.Floor) continue;
        const a = corner(col, row);
        const b = corner(col + 1, row);
        const c = corner(col + 1, row + 1);
        const d = corner(col, row + 1);
        const quad = flip ? [a, c, b, a, d, c] : [a, b, c, a, c, d];
        for (const point of quad) positions.push(point[0], point[1], point[2]);
        // Planar UVs in metres. Triplanar materials ignore them, but a material
        // that samples `uv()` without the attribute present is a hard failure on
        // the WebGL2 path, and this costs eight floats per quad.
        for (const point of quad) uvs.push(point[0] * 0.25, point[2] * 0.25);
      }
    }

    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3));
    geometry.setAttribute('uv', new THREE.Float32BufferAttribute(uvs, 2));
    geometry.computeVertexNormals();
    geometry.computeBoundingBox();
    geometry.computeBoundingSphere();
    return geometry;
  }

  /* -- geometry: walls ------------------------------------------------------ */

  /**
   * The visible rock face: marching squares, extruded, roughened.
   *
   * See the module header for why this is separate from the collision. The one
   * implementation note worth repeating here is that the displacement function
   * takes only the undisplaced position, which is what lets independently
   * emitted quads stay welded without any contour chaining.
   */
  #buildWallSkin(): void {
    const { cols, rows, cellSize, cells } = this.layout;
    const half = this.layout.halfExtent;
    const noise = new SimplexNoise(`${SEED}.wall`);

    const centre = (col: number, row: number): [number, number] => [
      (col + 0.5) * cellSize - half.x,
      (row + 0.5) * cellSize - half.z,
    ];
    const isFloor = (col: number, row: number): boolean =>
      col >= 0 && row >= 0 && col < cols && row < rows && cells[row * cols + col] === CaveCell.Floor;

    /** Roughen a contour point. Pure in `(x, z, y)`, hence seamless. */
    const displace = (x: number, z: number, y: number): [number, number] => {
      const dx = noise.noise4D(x * 0.29, y * 0.35, z * 0.29, 0) * WALL_ROUGHNESS;
      const dz = noise.noise4D(x * 0.29, y * 0.35, z * 0.29, 13.7) * WALL_ROUGHNESS;
      const fine =
        noise.noise4D(x * 1.15, y * 0.9, z * 1.15, 41.2) * WALL_ROUGHNESS * 0.28;
      return [x + dx + fine, z + dz - fine];
    };

    const positions: number[] = [];
    const uvs: number[] = [];

    for (let row = 0; row < rows - 1; row++) {
      for (let col = 0; col < cols - 1; col++) {
        const code =
          (isFloor(col, row) ? 1 : 0) |
          (isFloor(col + 1, row) ? 2 : 0) |
          (isFloor(col + 1, row + 1) ? 4 : 0) |
          (isFloor(col, row + 1) ? 8 : 0);
        const segments = MARCHING_SQUARES[code];
        if (segments === undefined || segments.length === 0) continue;

        const tl = centre(col, row);
        const br = centre(col + 1, row + 1);
        // Edge midpoints of the dual cell, in world units.
        const edge: [number, number][] = [
          [(tl[0] + br[0]) / 2, tl[1]], // top
          [br[0], (tl[1] + br[1]) / 2], // right
          [(tl[0] + br[0]) / 2, br[1]], // bottom
          [tl[0], (tl[1] + br[1]) / 2], // left
        ];

        // Floor and ceiling for this column, from the nearest walkable cell so
        // the wall meets both surfaces rather than floating between them.
        const [refCol, refRow] = nearestFloorCell(cells, cols, rows, col, row) ?? [col, row];
        const cx = (tl[0] + br[0]) / 2;
        const cz = (tl[1] + br[1]) / 2;
        const baseY = this.field.heightAt(cx, cz) - 0.35;
        const topY = this.field.heightAt(cx, cz) + this.ceilingAt(refCol, refRow) + 0.5;

        for (const [ea, eb] of segments) {
          const p0 = edge[ea];
          const p1 = edge[eb];
          if (p0 === undefined || p1 === undefined) continue;

          for (let ring = 0; ring < WALL_RINGS - 1; ring++) {
            const t0 = ring / (WALL_RINGS - 1);
            const t1 = (ring + 1) / (WALL_RINGS - 1);
            const y0 = baseY + (topY - baseY) * t0;
            const y1 = baseY + (topY - baseY) * t1;
            const a = displace(p0[0], p0[1], y0);
            const b = displace(p1[0], p1[1], y0);
            const c = displace(p1[0], p1[1], y1);
            const d = displace(p0[0], p0[1], y1);
            const quad: [number, number, number][] = [
              [a[0], y0, a[1]],
              [b[0], y0, b[1]],
              [c[0], y1, c[1]],
              [a[0], y0, a[1]],
              [c[0], y1, c[1]],
              [d[0], y1, d[1]],
            ];
            for (const point of quad) {
              positions.push(point[0], point[1], point[2]);
              uvs.push((point[0] + point[2]) * 0.25, point[1] * 0.25);
            }
          }
        }
      }
    }

    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3));
    geometry.setAttribute('uv', new THREE.Float32BufferAttribute(uvs, 2));
    // Faceted, not smoothed. Non-indexed geometry gives per-face normals, and
    // faceted is what rock at this scale looks like — it also matches
    // `generateRockGeometry`, so the stalagmites and the walls agree.
    geometry.computeVertexNormals();
    geometry.computeBoundingBox();
    geometry.computeBoundingSphere();
    this.#owned.push(geometry);

    const material =
      this.#materials?.create('rock', {
        albedoTint: [0.13, 0.125, 0.13],
        roughnessRange: [0.7, 1],
        tiling: 0.36,
        wetnessExposure: 0.5,
      }) ?? new THREE.MeshStandardNodeMaterial({ color: 0x191818, roughness: 0.95 });
    material.name = 'den.wallskin';
    // Double-sided: the contour is a surface with no consistent outward side
    // where it pinches, and the solid backing behind it is what actually stops
    // the camera. One-sided here would produce dropouts at exactly the awkward
    // spots. It casts no shadow — the backing does that, correctly and cheaply.
    material.side = THREE.DoubleSide;
    this.#owned.push(material);

    const mesh = new THREE.Mesh(geometry, material);
    mesh.name = 'den.wallskin';
    mesh.castShadow = false;
    mesh.receiveShadow = true;
    mesh.userData['noCollide'] = true;
    this.#root.add(mesh);
  }

  /**
   * Greedy rectangle meshing over the solid cells near the cave.
   *
   * The naive emission — one box per solid cell within {@link BACKING_DEPTH} of
   * the walkable set — produced **2 900 boxes** on the default grid, and merging
   * only horizontally still left 646. Both numbers are colliders as well as
   * geometry, and 646 static cuboids is an absurd price for a volume the player
   * never sees.
   *
   * Full 2D greedy meshing is the standard fix and is barely longer than the row
   * pass: walk the grid, and at the first unconsumed cell extend right as far as
   * the run holds, then extend *down* as far as every row of that width still
   * holds, and consume the rectangle. Long stretches of cave wall collapse into
   * a handful of slabs. Measured on the default seed it takes the count to
   * roughly a fifth of the row-merged figure, for identical occupied volume.
   */
  #computeBacking(): void {
    const { cols, rows, cells } = this.layout;
    const near = dilate(cells, cols, rows, BACKING_DEPTH);
    const solid = new Uint8Array(cells.length);
    for (let i = 0; i < cells.length; i++) {
      if (near[i] === CaveCell.Floor && cells[i] === CaveCell.Wall) solid[i] = 1;
    }

    this.#backing.length = 0;
    const consumed = new Uint8Array(cells.length);
    const free = (col: number, row: number): boolean =>
      solid[row * cols + col] === 1 && consumed[row * cols + col] === 0;

    for (let row = 0; row < rows; row++) {
      for (let col = 0; col < cols; col++) {
        if (!free(col, row)) continue;

        let col1 = col;
        while (col1 + 1 < cols && free(col1 + 1, row)) col1++;

        let row1 = row;
        outer: while (row1 + 1 < rows) {
          for (let c = col; c <= col1; c++) {
            if (!free(c, row1 + 1)) break outer;
          }
          row1++;
        }

        for (let r = row; r <= row1; r++) {
          for (let c = col; c <= col1; c++) consumed[r * cols + c] = 1;
        }
        this.#backing.push({ col0: col, col1, row0: row, row1 });
      }
    }
  }

  #buildWallBacking(): void {
    const { cellSize } = this.layout;
    const half = this.layout.halfExtent;
    const parts: THREE.BufferGeometry[] = [];
    const top = CEILING_BASE + CEILING_MAX_BONUS + 1.5;
    const bottom = -2.5;

    for (const run of this.#backing) {
      const width = (run.col1 - run.col0 + 1) * cellSize;
      const depth = (run.row1 - run.row0 + 1) * cellSize;
      const box = new THREE.BoxGeometry(width, top - bottom, depth);
      box.translate(
        ((run.col0 + run.col1 + 1) / 2) * cellSize - half.x,
        (top + bottom) / 2,
        ((run.row0 + run.row1 + 1) / 2) * cellSize - half.z,
      );
      parts.push(box);
    }
    if (parts.length === 0) return;

    const merged = mergeGeometries(parts, false);
    for (const part of parts) part.dispose();
    if (merged === null) return;
    merged.computeBoundingBox();
    merged.computeBoundingSphere();
    this.#owned.push(merged);

    const material =
      this.#materials?.create('rock', {
        albedoTint: [0.085, 0.082, 0.085],
        roughnessRange: [0.8, 1],
        tiling: 0.3,
      }) ?? new THREE.MeshStandardNodeMaterial({ color: 0x111111, roughness: 0.97 });
    material.name = 'den.wallbacking';
    this.#owned.push(material);

    const mesh = new THREE.Mesh(merged, material);
    mesh.name = 'den.wallbacking';
    // This is the shadow caster. It is a closed solid, so its shadows are
    // correct where the double-sided skin's would not be.
    mesh.castShadow = true;
    mesh.receiveShadow = true;
    // Its colliders are built explicitly in `buildColliders` from the same runs;
    // deriving one from these bounds would produce a single cave-sized box.
    mesh.userData['noCollide'] = true;
    this.#root.add(mesh);
  }

  /* -- dressing ------------------------------------------------------------- */

  /** Stalagmites, hugging the walls where they would actually form. */
  #buildStalagmites(): void {
    const geometry = generateRockGeometry({
      seed: `${SEED}.stalagmite`,
      detail: 2,
      radius: 0.5,
      displacement: 0.3,
      scale: [0.34, 2.3, 0.34],
      flattenBase: 0.55,
    });
    this.#owned.push(geometry);
    const material =
      this.#materials?.create('rock', {
        albedoTint: [0.16, 0.155, 0.15],
        roughnessRange: [0.6, 0.95],
        wetnessExposure: 0.9,
      }) ?? new THREE.MeshStandardNodeMaterial({ color: 0x1e1c1a, roughness: 0.9 });
    material.name = 'den.stalagmite';
    this.#owned.push(material);

    // Clearance between 1 and 2.6 cells is "near a wall but not in it", which is
    // exactly the band a stalagmite belongs in: out in the middle of a chamber
    // it is an obstacle, and inside the wall it is invisible.
    const samples = this.#scatterOnFloor(`${SEED}.stalagmites`, 22, {
      minClearance: 1,
      maxClearance: 2.6,
      minDepth: 0,
      scaleRange: [0.55, 1.5],
    });
    if (samples.length === 0) return;
    const mesh = buildInstancedMesh(geometry, material, samples);
    mesh.name = 'den.stalagmites';
    mesh.castShadow = true;
    this.#root.add(mesh);
  }

  /** Bone piles: the previous occupants, and the reason the den has a name. */
  #buildBones(): void {
    const geometry = bonePileGeometry(`${SEED}.bones`);
    this.#owned.push(geometry);
    const material =
      this.#materials?.create('rock', {
        // Bone is the one thing down here allowed to be lighter than the rock:
        // it is what the eye finds first, which is the point of putting it here.
        albedoTint: [0.36, 0.34, 0.29],
        roughnessRange: [0.55, 0.85],
      }) ?? new THREE.MeshStandardNodeMaterial({ color: 0x3d382f, roughness: 0.8 });
    material.name = 'den.bones';
    this.#owned.push(material);

    const samples = this.#scatterOnFloor(`${SEED}.bonepiles`, 16, {
      minClearance: 1.4,
      maxClearance: 12,
      minDepth: 4,
      scaleRange: [0.7, 1.35],
    });
    if (samples.length === 0) return;
    const mesh = buildInstancedMesh(geometry, material, samples);
    mesh.name = 'den.bones';
    mesh.castShadow = true;
    this.#root.add(mesh);
  }

  /**
   * Luminous fungus: the cold half of the palette.
   *
   * Unlit emissive geometry, plus three weak cold point lights in the deepest
   * stretches. It is the only cold light source in the act, and it exists
   * because "warm" is a relationship, not a colour — the torches only read as
   * warm if there is something cold nearby for them to be warm *against*.
   */
  #buildFungus(): void {
    const geometry = new THREE.IcosahedronGeometry(0.1, 1);
    this.#owned.push(geometry);
    const material = new THREE.MeshBasicNodeMaterial({
      color: new THREE.Color(FUNGUS_COLOR).multiplyScalar(1.4),
      toneMapped: true,
      transparent: true,
      opacity: 0.95,
      depthWrite: true,
    });
    material.name = 'den.fungus';
    this.#owned.push(material);

    const samples = this.#scatterOnFloor(`${SEED}.fungus`, 90, {
      minClearance: 0.8,
      maxClearance: 3.2,
      minDepth: 8,
      scaleRange: [0.5, 1.7],
    });
    if (samples.length > 0) {
      const mesh = buildInstancedMesh(geometry, material, samples);
      mesh.name = 'den.fungus';
      mesh.castShadow = false;
      mesh.receiveShadow = false;
      mesh.userData['noCollide'] = true;
      this.#root.add(mesh);
    }

    // Three lights, not ninety. The emissive geometry does the reading; the
    // lights only need to keep the rock around a patch from being pure black.
    const picks = samples.filter((_, i) => i % Math.max(1, Math.floor(samples.length / 5)) === 0);
    for (const sample of picks.slice(0, 4)) {
      const handle =
        this.#lighting?.addLight({
          kind: 'point',
          name: 'den.fungus.glow',
          position: {
            x: sample.position.x,
            y: sample.position.y + 0.35,
            z: sample.position.z,
          },
          color: FUNGUS_COLOR,
          intensity: 3.2,
          radius: 6,
          decay: 2,
          castShadow: false,
          priority: 2,
        }) ?? null;
      this.#lights.push({ handle, base: 3.2, fire: null });
    }
  }

  /**
   * Five torches: one at the mouth, one per chamber along the route, one at the
   * back.
   *
   * Chosen from the layout rather than scattered, because their job is
   * navigational. A torch in a chamber is a landmark that tells a player where
   * they have been; a torch at a random point on the floor is a light.
   */
  #buildTorches(): void {
    const layout = this.layout;
    const material =
      this.#materials?.create('bark', {
        albedoTint: [0.12, 0.1, 0.085],
        roughnessRange: [0.85, 1],
      }) ?? new THREE.MeshStandardNodeMaterial({ color: 0x1a1613, roughness: 0.96 });
    material.name = 'den.torch';
    this.#owned.push(material);

    // Ten, not five.
    //
    // `LightingService` culls per frame by importance and binds at most eight
    // point lights, so registering more torches than the budget is not
    // overspending — it is letting the arbitration pick the nearest ones as the
    // player moves. Five torches in an 88 m cave left 20 m gaps of absolute
    // black between them; ten puts one roughly every chamber *and* every depth
    // band, which is what makes the route a chain of lit places.
    // The entrance, and the arrival point six steps in — a player who lands in
    // an unlit stretch has no way to know which direction the cave goes.
    const spots: GridPoint[] = [layout.entrance];
    const seen = new Set<number>([layout.entrance.row * layout.cols + layout.entrance.col]);
    const add = (point: GridPoint | null): void => {
      if (point === null) return;
      const index = point.row * layout.cols + point.col;
      if (seen.has(index)) return;
      // Never two torches in each other's pool: that is one bright place, not two.
      for (const other of spots) {
        if ((other.col - point.col) ** 2 + (other.row - point.row) ** 2 < 49) return;
      }
      seen.add(index);
      spots.push(point);
    };

    const arrival = cellAtDepth(layout, 7);
    if (arrival !== null) {
      seen.add(arrival.row * layout.cols + arrival.col);
      spots.push(arrival);
    }

    for (const chamber of [...layout.chambers].sort((a, b) => a.depth - b.depth)) {
      if (spots.length >= TORCH_COUNT) break;
      if (chamber.depth < 4) continue;
      add(chamber.center);
    }
    for (let band = 1; spots.length < TORCH_COUNT && band < TORCH_COUNT * 2; band++) {
      add(cellAtDepth(layout, Math.round((layout.maxDepth * band) / (TORCH_COUNT * 2))));
    }

    const parts: THREE.BufferGeometry[] = [];
    const height = 1.55;
    for (const spot of spots) {
      const world = worldOfCell(layout, spot.col, spot.row);
      const y = this.field.heightAt(world.x, world.z);
      const post = new THREE.CylinderGeometry(0.055, 0.075, height, 6);
      post.translate(world.x, y + height / 2, world.z);
      parts.push(post);
      const basket = new THREE.CylinderGeometry(0.17, 0.11, 0.24, 7, 1, true);
      basket.translate(world.x, y + height + 0.08, world.z);
      parts.push(basket);
    }

    const merged = mergeGeometries(parts, false);
    for (const part of parts) part.dispose();
    if (merged !== null) {
      merged.computeVertexNormals();
      this.#owned.push(merged);
      const mesh = new THREE.Mesh(merged, material);
      mesh.name = 'den.torches';
      mesh.castShadow = true;
      mesh.receiveShadow = true;
      mesh.userData['noCollide'] = true;
      this.#root.add(mesh);
    }

    for (const [i, spot] of spots.entries()) {
      const world = worldOfCell(layout, spot.col, spot.row);
      const y = this.field.heightAt(world.x, world.z) + height + 0.12;
      const fire = new Fire({
        radius: 0.16,
        height: 0.5,
        intensity: 1.5,
        phase: i * 2.13,
        name: `den.torch.${i}`,
      });
      fire.group.position.set(world.x, y, world.z);
      this.#root.add(fire.group);
      this.#fires.push(fire);

      const handle =
        this.#lighting?.addLight({
          kind: 'point',
          name: `den.torch.${i}`,
          position: { x: world.x, y: y + 0.12, z: world.z },
          color: TORCH_COLOR,
          intensity: 13,
          // 10 m against a guaranteed 4 m corridor width: the pool reaches the
          // walls on both sides and dies before the next torch, which is what
          // makes the route read as a chain of lit places rather than as a
          // uniformly lit tunnel.
          radius: 10,
          decay: 2,
          // Only the first casts. A shadowed point light is a cube map — six
          // full re-renders of the cave shell — and the arbitration binds one
          // shadowed point at a time regardless, so asking for more buys
          // nothing and costs a shadow-map allocation.
          castShadow: i === 0,
          priority: 14 - i,
        }) ?? null;
      this.#lights.push({ handle, base: 13, fire });
    }
  }

  /** Rubble and broken containers, on the floor, away from the walls. */
  #placeDebris(): void {
    const kit = this.#kit;
    if (kit === null) return;
    const rng = createRng(`${SEED}.debris`);
    const samples = this.#scatterOnFloor(`${SEED}.debris`, 26, {
      minClearance: 1.2,
      maxClearance: 12,
      minDepth: 2,
      scaleRange: [0.8, 1.3],
    });
    const keys = PROP_KEYS.filter((key) => kit.has(key));
    if (keys.length === 0) return;
    for (const [i, sample] of samples.entries()) {
      const key = keys[rng.int(0, keys.length - 1)];
      if (key === undefined) continue;
      kit.placeSized(
        key,
        0.75 * sample.scale,
        {
          x: sample.position.x,
          y: sample.position.y,
          z: sample.position.z,
          yaw: sample.rotation,
          tilt: (rng.next() - 0.5) * 0.22,
          name: `den.debris.${i}`,
        },
        this.#root,
      );
    }
  }

  /* -- helpers -------------------------------------------------------------- */

  /**
   * Pick placements on walkable cells, filtered by how open and how deep.
   *
   * A dedicated sampler rather than `Procedural.scatter`, because everything
   * placed in this zone wants to be filtered by the *layout's* fields — wall
   * clearance and distance from the entrance — and those live on the grid, not
   * on a continuous surface function.
   */
  #scatterOnFloor(
    seed: string,
    count: number,
    filter: {
      minClearance: number;
      maxClearance: number;
      minDepth: number;
      scaleRange: [number, number];
    },
  ): ScatterSample[] {
    const layout = this.layout;
    const rng = createRng(seed);
    const candidates: number[] = [];
    for (let i = 0; i < layout.cells.length; i++) {
      if (layout.cells[i] !== CaveCell.Floor) continue;
      const clearance = layout.clearance[i] as number;
      if (clearance < filter.minClearance || clearance > filter.maxClearance) continue;
      if ((layout.distance[i] as number) < filter.minDepth) continue;
      candidates.push(i);
    }
    if (candidates.length === 0) return [];

    const samples: ScatterSample[] = [];
    const used = new Set<number>();
    const attempts = Math.min(count * 12, candidates.length * 4);
    for (let attempt = 0; attempt < attempts && samples.length < count; attempt++) {
      const pick = candidates[rng.int(0, candidates.length - 1)];
      if (pick === undefined || used.has(pick)) continue;
      used.add(pick);
      const col = pick % layout.cols;
      const row = (pick - col) / layout.cols;
      const world = worldOfCell(layout, col, row);
      // Jitter within the cell so the placements are not visibly on a lattice.
      const x = world.x + (rng.next() - 0.5) * layout.cellSize * 0.7;
      const z = world.z + (rng.next() - 0.5) * layout.cellSize * 0.7;
      samples.push({
        position: new THREE.Vector3(x, this.field.heightAt(x, z), z),
        normal: new THREE.Vector3(0, 1, 0),
        scale:
          filter.scaleRange[0] + rng.next() * (filter.scaleRange[1] - filter.scaleRange[0]),
        stretch: 0.8 + rng.next() * 0.55,
        rotation: rng.next() * Math.PI * 2,
        index: samples.length,
      });
    }
    return samples;
  }

  #applyOcclusion(): void {
    const ibl = this.#ibl;
    if (ibl === null) return;
    const seen = new Set<THREE.Material>();
    this.#root.traverse((object) => {
      if (!(object instanceof THREE.Mesh)) return;
      const list = Array.isArray(object.material) ? object.material : [object.material];
      for (const material of list) {
        if (seen.has(material)) continue;
        seen.add(material);
        if (material.name.startsWith('fire.') || material.name === 'den.fungus') continue;
        if (material instanceof THREE.NodeMaterial) ibl.applyOcclusion(material);
      }
    });
  }
}

/* -------------------------------------------------------------------------- */
/* Grid utilities                                                             */
/* -------------------------------------------------------------------------- */

/**
 * The marching-squares case table.
 *
 * Corner bits are `TL=1, TR=2, BR=4, BL=8`, set when that cell is walkable.
 * Values are pairs of edge indices — `0=top, 1=right, 2=bottom, 3=left` of the
 * *dual* cell — so each entry is the contour crossing that configuration.
 *
 * Cases 5 and 10 are the saddles, where the filled corners are diagonal and the
 * contour is genuinely ambiguous. They are resolved as two separate cuts (both
 * corners isolated) rather than as a join, because joining them welds two
 * passages together through a one-cell diagonal — which would silently violate
 * the generator's width guarantee at exactly the place the player would try to
 * squeeze through.
 */
const MARCHING_SQUARES: readonly (readonly (readonly [number, number])[])[] = [
  [], // 0
  [[3, 0]], // 1  TL
  [[0, 1]], // 2  TR
  [[3, 1]], // 3  TL TR
  [[1, 2]], // 4  BR
  [
    [3, 0],
    [1, 2],
  ], // 5  TL BR (saddle)
  [[0, 2]], // 6  TR BR
  [[3, 2]], // 7  TL TR BR
  [[2, 3]], // 8  BL
  [[2, 0]], // 9  TL BL
  [
    [0, 1],
    [2, 3],
  ], // 10 TR BL (saddle)
  [[2, 1]], // 11 TL TR BL
  [[1, 3]], // 12 BR BL
  [[0, 1]], // 13 TL BR BL
  [[3, 0]], // 14 TR BR BL
  [], // 15
];

/** Cells within `radius` of a walkable cell, as a walkable mask. */
function dilate(cells: Uint8Array, cols: number, rows: number, radius: number): Uint8Array {
  if (radius <= 0) return cells;
  const out = new Uint8Array(cells.length);
  const r2 = radius * radius;
  const r = Math.ceil(radius);
  for (let row = 0; row < rows; row++) {
    for (let col = 0; col < cols; col++) {
      if (cells[row * cols + col] !== CaveCell.Floor) continue;
      for (let dr = -r; dr <= r; dr++) {
        for (let dc = -r; dc <= r; dc++) {
          if (dc * dc + dr * dr > r2) continue;
          const c = col + dc;
          const rr = row + dr;
          if (c < 0 || rr < 0 || c >= cols || rr >= rows) continue;
          out[rr * cols + c] = CaveCell.Floor;
        }
      }
    }
  }
  return out;
}

/** The walkable cell nearest `(col, row)`, searched outward. Returns `[col, row]`. */
function nearestFloorCell(
  cells: Uint8Array,
  cols: number,
  rows: number,
  col: number,
  row: number,
): [number, number] | null {
  for (let radius = 0; radius <= 3; radius++) {
    for (let dr = -radius; dr <= radius; dr++) {
      for (let dc = -radius; dc <= radius; dc++) {
        const c = col + dc;
        const r = row + dr;
        if (c < 0 || r < 0 || c >= cols || r >= rows) continue;
        if (cells[r * cols + c] === CaveCell.Floor) return [c, r];
      }
    }
  }
  return null;
}

/**
 * A reachable cell approximately `depth` steps from the entrance.
 *
 * Used to place the arrival point a fixed distance inside the mouth, and to
 * backfill torch positions when the cave has fewer chambers than torches.
 *
 * `minClearance` filters to cells at least that far (in cells) from the nearest
 * wall, and it matters for anything a *character* is put on. Depth alone picks
 * whichever cell the scan reaches first, which in a cave built out of corridors
 * is usually one pressed against a wall: the arrival point landed the player
 * 0.5 m off `den.wall.70.43`, close enough that the camera's shoulder-offset
 * pivot started inside the masonry and the arm collapsed to its floor on the
 * frame the player arrives. Standing a character somewhere is a claim about the
 * space around them, not just about where their feet go.
 */
export function cellAtDepth(
  layout: DungeonLayout,
  depth: number,
  minClearance = 0,
): GridPoint | null {
  let best: GridPoint | null = null;
  let bestError = Infinity;
  for (let i = 0; i < layout.cells.length; i++) {
    if (layout.cells[i] !== CaveCell.Floor) continue;
    const d = layout.distance[i] as number;
    if (d < 0) continue;
    if ((layout.clearance[i] as number) < minClearance) continue;
    const error = Math.abs(d - depth);
    if (error < bestError) {
      bestError = error;
      const col = i % layout.cols;
      best = { col, row: (i - col) / layout.cols };
      if (error === 0) break;
    }
  }
  return best;
}

/**
 * The most open cell near a target depth, degrading gracefully.
 *
 * A generated cave cannot promise a 2 m-clear cell exists at depth six, so this
 * asks for the roomiest and settles for less rather than returning null — an
 * arrival point that does not exist is a worse outcome than a snug one.
 */
export function openCellAtDepth(layout: DungeonLayout, depth: number): GridPoint | null {
  for (const clearance of [2.2, 1.8, 1.4, 1.0, 0]) {
    const cell = cellAtDepth(layout, depth, clearance);
    if (cell !== null) return cell;
  }
  return null;
}

/**
 * A heap of long bones and a skull-sized lump.
 *
 * There is no bone asset in the pack, and a cave called the Den of Evil with no
 * remains in it is a cave with no history. Six tapered cylinders at scattered
 * angles plus two low spheres is enough: at the light level in here what reads
 * is the pale value and the tangle of straight lines, not the anatomy.
 */
function bonePileGeometry(seed: string): THREE.BufferGeometry {
  const rng = createRng(seed);
  // Every part is converted to non-indexed before merging.
  //
  // `mergeGeometries` requires the whole batch to agree about whether an index
  // buffer exists, and three's primitives do not: `CylinderGeometry` is indexed,
  // `IcosahedronGeometry` is not. Mixing them throws — into `console.error`, not
  // up the stack — so the merge returns null, the bone piles silently never
  // appear, and the only symptom is a cave that is missing its dressing.
  const parts: THREE.BufferGeometry[] = [];
  const push = (geometry: THREE.BufferGeometry): void => {
    if (geometry.getIndex() === null) {
      parts.push(geometry);
      return;
    }
    const flat = geometry.toNonIndexed();
    geometry.dispose();
    parts.push(flat);
  };
  for (let i = 0; i < 6; i++) {
    const length = 0.32 + rng.next() * 0.3;
    const bone = new THREE.CylinderGeometry(0.028, 0.034, length, 5);
    bone.rotateZ(Math.PI / 2 + (rng.next() - 0.5) * 0.5);
    bone.rotateY(rng.next() * Math.PI * 2);
    bone.translate((rng.next() - 0.5) * 0.34, 0.035 + rng.next() * 0.06, (rng.next() - 0.5) * 0.34);
    push(bone);
    // The knuckle at each end, which is what stops a cylinder reading as a stick.
    for (const end of [-1, 1]) {
      const knuckle = new THREE.IcosahedronGeometry(0.045, 0);
      knuckle.translate((rng.next() - 0.5) * 0.34 + (end * length) / 2, 0.045, (rng.next() - 0.5) * 0.34);
      push(knuckle);
    }
  }
  const skull = new THREE.IcosahedronGeometry(0.1, 1);
  skull.scale(1, 0.82, 1.15);
  skull.translate((rng.next() - 0.5) * 0.2, 0.08, (rng.next() - 0.5) * 0.2);
  push(skull);

  const merged = mergeGeometries(parts, false);
  for (const part of parts) part.dispose();
  const geometry = merged ?? new THREE.IcosahedronGeometry(0.12, 1);
  geometry.computeVertexNormals();
  geometry.computeBoundingBox();
  geometry.computeBoundingSphere();
  return geometry;
}
