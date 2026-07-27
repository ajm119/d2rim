/**
 * @module scene/RogueEncampment
 *
 * Act I's hub: the one place in the act that is warm, lit and safe.
 *
 * ## The art-direction claim
 *
 * Every other zone in this act is built on the rule that *the fire is the only
 * warm thing in the frame*. The encampment is where that rule pays off. It is
 * the same cold overcast sky, the same mud, the same desaturated props — and
 * then eight separate fires inside one palisade ring. The camp does not read as
 * safe because it is bright; it reads as safe because it is the only place where
 * the warm light wins, and it wins by *quantity* rather than by any of it being
 * individually stronger than the campfire out on the moor.
 *
 * That is why the light plan is what it is:
 *
 * - one large **bonfire** at the centre, shadow-casting, radius 17 m. It is the
 *   only shadow-casting local light in the zone, so the tents and the palisade
 *   throw long spokes outward and the camp has a legible centre.
 * - the **forge**, a second warm pool with no shadow and a hotter, tighter
 *   colour — a coal bed is nearer white than a wood fire and much smaller.
 * - six **torch posts** on the inner ring at 6 m intervals, weak (radius 7 m) and
 *   individually unremarkable. Their job is to stop the ground between the
 *   bonfire and the palisade going black, which is what would otherwise make the
 *   camp read as a bonfire in a field rather than as an enclosure.
 *
 * ## Traversal
 *
 * The brief is "flat, walkable, no props blocking movement", and it is enforced
 * structurally rather than by inspection. The ground is flat to ±0.1 m inside
 * the palisade (see {@link CampGround}), every prop is placed on a ring at a
 * documented radius, and the whole inner disc of radius {@link INNER_CLEAR} is
 * left empty except for the bonfire itself. There is a continuous 4 m annulus
 * between the tent ring and the palisade, and a straight run from the gate to
 * the fire.
 *
 * ## The NPC anchors
 *
 * Phase 5 adds Akara, Kashya, Charsi, Gheed and Warriv. This module does *not*
 * build them; it stages the camp around where they will stand and publishes
 * those spots as {@link CAMP_NPC_ANCHORS}. See {@link RogueEncampment.placeNpc}
 * for the API — the contract is that phase 5 names an anchor and hands over a
 * figure, and never needs to know where the tents ended up.
 *
 * ## Cost
 *
 * Measured by `tools/verify-zones.mjs`: **68 meshes, 64 710 triangles,
 * 249 colliders, 8 registered lights (5 bound)**.
 *
 * | element        | draws | note                                              |
 * |----------------|-------|---------------------------------------------------|
 * | ground         | 1     | 128x128 displaced plane, one blended material     |
 * | palisade       | 1     | one instanced batch of ~198 logs                  |
 * | hearth ring    | 1     | instanced boulders                                |
 * | built geometry | 6     | forge, anvil, two carts, gate, torch posts, fire logs — merged per material |
 * | fires          | 16    | 8 x (ember disc + plume)                          |
 * | kit props      | ~43   | tents, barrels, crates, racks, banners            |
 *
 * The ground is the single largest triangle consumer at ~33 k, and it is the one
 * mesh that is worth it: it is the surface every other cost is measured against.
 * The shadow pass roughly doubles the draw count for the shadow-casting subset,
 * of which there is exactly one light. That is an order of magnitude inside the
 * budget a 60 Hz target allows for a hub, and the dominant *draw* term is the
 * kit props — the right thing for it to be, because they are the only part a
 * player walks up to.
 *
 * Colliders are 246 props plus one terrain heightfield: one per palisade log,
 * which is deliberate. A single ring collider would be cheaper and would also
 * make the wall a cylinder the player slides along instead of a row of posts.
 */

import * as THREE from 'three/webgpu';
import { positionWorld, saturate, smoothstep } from 'three/tsl';
import { mergeGeometries } from 'three/examples/jsm/utils/BufferGeometryUtils.js';

import { AssetManagerKey, type AssetKey, type AssetManager } from '../assets/AssetManager';
import {
  SimplexNoise,
  buildInstancedMesh,
  createRng,
  generateRockGeometry,
  type ScatterSample,
} from '../assets/Procedural';
import type { GameContext } from '../core/types';
import type { PhysicsWorld } from '../physics/PhysicsWorld';
import { IBLKey, type IBLService } from '../render/IBL';
import { LightingKey, type LightHandle, type LightingService } from '../render/Lighting';
import { MaterialLibraryKey, type MaterialLibraryService } from '../render/MaterialLibrary';
import type { RenderSettings } from '../render/RenderSettings';
import type { NpcAnchor, PortalSpec, Zone, ZoneEntryPoint } from '../world/Zone';
import { Fire, type FireOptions } from './Fire';
import { PropKit } from './PropKit';
import type { WeatheringGrade } from './Weathering';

/* -------------------------------------------------------------------------- */
/* Layout                                                                     */
/* -------------------------------------------------------------------------- */

/** Palisade radius, in metres. The camp is 48 m across inside the wall. */
export const PALISADE_RADIUS = 24;
/**
 * Radius kept completely clear around the bonfire.
 *
 * Not decoration: this is the muster ground. Everything the player does in the
 * hub happens between here and the tent ring, so nothing may be placed inside it.
 */
export const INNER_CLEAR = 6.5;
/** Radius the tents, carts and workstations sit on. */
export const STRUCTURE_RING = 12.5;
/** Half-width of the gate opening, in radians of arc. */
const GATE_HALF_ARC = 0.2;
/**
 * Terrain extent in metres.
 *
 * 260, matching the Blood Moor, and not the 170 this started at. At 170 the
 * ground edge sat about 60 m beyond the palisade and cut a hard brown line
 * across the sky from any camera above head height — the camp read as a diorama
 * on a table. The horizon has to be far enough out that the atmosphere's aerial
 * perspective has faded it before it ends.
 */
const TERRAIN_SIZE = 260;
const TERRAIN_SEGMENTS = 128;
const SEED = 'd2rim.encampment';

/** Warm, but restrained: this is firelight, not a stage wash. */
const BONFIRE_COLOR = 0xff9a44;
const FORGE_COLOR = 0xffb066;
const TORCH_COLOR = 0xff8a38;

/**
 * The prop grade for the camp.
 *
 * Slightly lighter than the Blood Moor's — the camp's props are *lit*, and
 * pulling them as far down as the moor's would leave the one warm place in the
 * act looking like the cold one with lamps in it. The warm clamp is kept high
 * regardless: the fires are still the only saturated warm thing here, and a
 * chromatic tent would compete with them.
 */
const CAMP_WEATHERING: WeatheringGrade = {
  desaturation: 0.26,
  shadowTint: new THREE.Color(0.115, 0.13, 0.165),
  lightTint: new THREE.Color(0.285, 0.25, 0.195),
  warmClamp: 0.45,
  grime: 0.34,
  wear: 0.18,
  roughnessFloor: 0.55,
  envMapIntensity: 0.5,
};

const PROP_KEYS = {
  tents: ['prop.tent'],
  barrels: ['prop.barrel.large', 'prop.barrel.small', 'prop.barrel.stack', 'prop.keg'],
  crates: ['prop.crate.large', 'prop.crate.small', 'prop.crate.stack', 'prop.crate.open'],
  smithy: ['prop.weaponrack', 'prop.swordshield', 'prop.bucket'],
  camp: ['prop.sack', 'prop.lumber', 'prop.banner', 'prop.stool', 'prop.table'],
} as const satisfies Record<string, readonly AssetKey[]>;

/* -------------------------------------------------------------------------- */
/* Ground                                                                     */
/* -------------------------------------------------------------------------- */

/**
 * The camp's height function.
 *
 * Flat where it matters and rolling where it does not. Inside the palisade the
 * amplitude is capped at ±0.1 m — enough that the ground is not a plane, far too
 * little to catch a foot or to make a tent float — and outside it the noise is
 * allowed up to ±2.4 m so the camp reads as sitting in country rather than on a
 * table. The transition is a smoothstep across the palisade line itself, which
 * is also where the wall hides it.
 *
 * The same object is sampled by the visual mesh, by every prop placement and by
 * the Rapier heightfield. One height function, three consumers: duplicating it
 * is the classic way to get a tent hovering half a metre off its own shadow.
 */
export class CampGround {
  readonly #noise: SimplexNoise;

  constructor(seed: string | number = `${SEED}.ground`) {
    this.#noise = new SimplexNoise(seed);
  }

  heightAt(x: number, z: number): number {
    const noise = this.#noise;
    const distance = Math.hypot(x, z);
    // 0 inside the camp, 1 well outside it.
    const outside = smootherstep(PALISADE_RADIUS - 3, PALISADE_RADIUS + 26, distance);

    const rolling =
      noise.noise2D(x * 0.011, z * 0.011) * 3.1 +
      noise.noise2D(x * 0.031, z * 0.031) * 1.1 +
      noise.noise2D(x * 0.085, z * 0.085) * 0.26;
    // Trodden ground: fine, shallow, and everywhere. It is what stops the camp
    // floor reading as a plane under a raking firelight.
    const trodden = noise.noise2D(x * 0.42, z * 0.42) * 0.06 + noise.noise2D(x * 0.9, z * 0.9) * 0.035;

    // The rim.
    //
    // Authored, not noise, and it fixes a real defect rather than adding a
    // flourish. With the ground merely rolling out to the terrain edge, the far
    // half of a 260 m plane is seen almost edge-on from any camera above head
    // height, and 130 m of mud at grazing incidence paints a bright banded strip
    // straight across the sky — the horizon becomes a smear of stretched texture
    // instead of a silhouette. Raising the surround into a shallow bowl puts a
    // real skyline in front of that strip.
    //
    // It is also the right *content* answer, which is why it is a rim and not a
    // fog tweak: a camp under siege is sited in dead ground, and the rogues did
    // not pitch this one in the middle of an open plain.
    const rim = smootherstep(42, 118, distance) * 13.5;

    return rolling * outside + rim + trodden;
  }

  normalAt(x: number, z: number, epsilon = 0.6): THREE.Vector3 {
    const dx = this.heightAt(x + epsilon, z) - this.heightAt(x - epsilon, z);
    const dz = this.heightAt(x, z + epsilon) - this.heightAt(x, z - epsilon);
    return new THREE.Vector3(-dx, 2 * epsilon, -dz).normalize();
  }

  readonly surface = (x: number, z: number): { y: number; normal: THREE.Vector3 } => ({
    y: this.heightAt(x, z),
    normal: this.normalAt(x, z),
  });

  buildGeometry(size: number, segments: number): THREE.BufferGeometry {
    const geometry = new THREE.PlaneGeometry(size, size, segments, segments);
    geometry.rotateX(-Math.PI / 2);
    const position = geometry.getAttribute('position');
    for (let i = 0; i < position.count; i++) {
      position.setY(i, this.heightAt(position.getX(i), position.getZ(i)));
    }
    position.needsUpdate = true;
    geometry.computeVertexNormals();
    geometry.computeBoundingBox();
    geometry.computeBoundingSphere();
    return geometry;
  }
}

function smootherstep(edge0: number, edge1: number, x: number): number {
  const t = Math.min(1, Math.max(0, (x - edge0) / (edge1 - edge0)));
  return t * t * t * (t * (t * 6 - 15) + 10);
}

/* -------------------------------------------------------------------------- */
/* NPC anchors                                                                */
/* -------------------------------------------------------------------------- */

const GROUND = new CampGround();

/** Ground height at a camp position. Used to author the anchors below. */
function at(x: number, z: number): { x: number; y: number; z: number } {
  return { x, y: GROUND.heightAt(x, z), z };
}

/** Face a point from a position. */
function facing(from: { x: number; z: number }, tx: number, tz: number): number {
  return Math.atan2(tx - from.x, tz - from.z);
}

/**
 * The five Rogue Encampment characters, staged but not built.
 *
 * Every anchor is placed *outside* {@link INNER_CLEAR} and *inside*
 * {@link STRUCTURE_RING}, on the annulus between the muster ground and the
 * structures, facing roughly inward. That is the arrangement that lets a player
 * walk the ring once and meet everyone without any of them standing in a
 * doorway.
 *
 * These are exported as data, not as a method, so a test or a quest script can
 * read them without constructing the zone.
 */
export const CAMP_NPC_ANCHORS: readonly NpcAnchor[] = [
  {
    id: 'akara',
    displayName: 'Akara',
    role: 'High Priestess of the Sightless Eye — quest giver, healer, vendor',
    position: at(-9.4, -5.2),
    yaw: facing({ x: -9.4, z: -5.2 }, 0, 0),
    clearRadius: 1.4,
    note: 'Outside the large west tent, facing the bonfire. The senior figure, so she gets the deepest position in the camp and the widest clear ground.',
  },
  {
    id: 'kashya',
    displayName: 'Kashya',
    role: 'Captain of the Rogues — quest giver, mercenary hire',
    position: at(6.8, 8.6),
    yaw: facing({ x: 6.8, z: 8.6 }, 0, 20),
    clearRadius: 1.2,
    note: 'By the gate watchpost, facing OUT toward the gate rather than in. She is the camp guard; a captain with her back to the door is a staging mistake.',
  },
  {
    id: 'charsi',
    displayName: 'Charsi',
    role: 'Blacksmith — repairs, weapon and armour vendor',
    position: at(9.6, -3.4),
    yaw: facing({ x: 9.6, z: -3.4 }, 13.4, -4.6),
    clearRadius: 1.2,
    note: 'At the forge, facing her anvil, with the hearth on her right. The forge fire is the second warmest light in the camp and she stands in it.',
  },
  {
    id: 'gheed',
    displayName: 'Gheed',
    role: 'Travelling merchant — general goods, gambling',
    position: at(-6.4, 8.4),
    yaw: facing({ x: -6.4, z: 8.4 }, 0, 0),
    clearRadius: 1.1,
    note: "Beside his wagon on the gate approach, so he is the first person a player passing through meets. Facing the fire, i.e. inward at arriving traffic.",
  },
  {
    id: 'warriv',
    displayName: 'Warriv',
    role: 'Caravan master — travel between acts',
    position: at(-11.2, 1.8),
    yaw: facing({ x: -11.2, z: 1.8 }, 0, 0),
    clearRadius: 1.3,
    note: 'At the caravan cart on the west side, clear of the gate. His interaction leaves the act, so he is deliberately not on the through-route.',
  },
];

/* -------------------------------------------------------------------------- */
/* Module                                                                     */
/* -------------------------------------------------------------------------- */

export interface RogueEncampmentOptions {
  readonly settings?: RenderSettings;
}

export class RogueEncampment implements Zone {
  readonly name = 'scene.rogueEncampment';
  readonly zoneId = 'encampment';
  readonly displayName = 'Rogue Encampment';

  /**
   * The camp's trim on the frame. See `ZoneGrade`.
   *
   * ### What the honest capture showed
   *
   * Measured off `captures/zones-before/encampment.png`, the frame's first
   * three quartiles sat between luma **0.103 and 0.125** — seventy-two percent
   * of the picture inside a band two hundredths wide. That is not a dark frame,
   * it is a *flat* one: there is no tonal structure in it at all. Worse, the
   * **coldest** decile measured −0.025 on the blue-minus-red axis, meaning the
   * single coldest tenth of the camp was still warm. The brief asks for "a warm
   * pool against a cold surround" and the capture had no cold in it anywhere.
   *
   * ### What the grade could and could not do about it
   *
   * Two rounds of this were measured, and the first one is worth recording
   * because it is the more instructive:
   *
   * | round | change | plateau | warm/cold separation |
   * |-------|--------|---------|----------------------|
   * | before | — | 0.103-0.125 | 0.101, both ends warm |
   * | 1 | +0.8 stop, hard blue lift, cool white point | 0.177-0.211 | **0.086** |
   * | 2 | as 1, plus a cold hemisphere fill in the scene | 0.190-0.230 | 0.074 |
   *
   * Round 1 exposed the frame correctly and made the separation *worse*. That
   * is the whole lesson: a colour grade is a function of the pixel, so it can
   * move every hue in the frame together but it cannot pull two populations
   * apart that are one population. Every surface in the camp was lit by fire
   * and only by fire, so brightening it produced a brighter monochrome and
   * cooling it produced a cooler monochrome. The separation had to come from a
   * second *light* — see the hemisphere fill in `init` — and the grade's job
   * afterwards is only to keep out of its way.
   *
   * So this is deliberately restrained where the first attempt was not:
   *
   * - `stops: +0.55`, not +0.8. A correctly exposed night camp is still dark,
   *   and round 1's extra quarter-stop bought a washed grey field.
   * - `temperature: -700`, barely past the base look's −520. Round 1's −1250
   *   cooled the *firelight* as hard as it cooled the ambient, which is exactly
   *   backwards: it took the amber out of the one warm thing in the picture.
   * - `gain` neutral. The colour in this frame now comes from two real light
   *   sources with two real colours; a per-channel gain would flatten them back
   *   toward each other.
   * - `saturation: 1.0` rather than a cut. The original 0.498 mean saturation
   *   was a *cast*, not chroma, and it went away when the fill arrived; cutting
   *   saturation on top of that only removes the fire's amber.
   * - The lift keeps a cold pedestal under the shadows so the darkest corners
   *   of the palisade stay blue-grey rather than going neutral.
   */
  readonly grade = {
    stops: 0.55,
    grade: {
      temperature: -700,
      tint: -0.01,
      lift: [-0.010, 0.0, 0.040] as const,
      gamma: [1.0, 1.0, 1.04] as const,
      gain: [1.0, 1.0, 1.0] as const,
      saturation: 1.0,
      contrast: 1.18,
      contrastPivot: 0.17,
      vignette: 0.22,
    },
  };

  /** Shared with the physics heightfield and every prop placement. */
  readonly field = new CampGround();
  readonly npcAnchors = CAMP_NPC_ANCHORS;

  readonly #root = new THREE.Group();
  readonly #owned: { dispose(): void }[] = [];
  readonly #fires: Fire[] = [];
  readonly #lights: { handle: LightHandle | null; base: number; fire: Fire }[] = [];

  #kit: PropKit | null = null;
  #materials: MaterialLibraryService | null = null;
  #ibl: IBLService | null = null;
  #lighting: LightingService | null = null;

  constructor(_options: RogueEncampmentOptions = {}) {
    this.#root.name = 'RogueEncampment';
  }

  get root(): THREE.Object3D {
    return this.#root;
  }

  get terrainSize(): number {
    return TERRAIN_SIZE;
  }

  get terrainSegments(): number {
    return TERRAIN_SEGMENTS;
  }

  /**
   * Where travel can put the player down.
   *
   * `camp-centre` is the game's opening position: south of the bonfire, looking
   * at it. `gate` is where arrivals from the Blood Moor land — deliberately 4 m
   * inside the gate portal so stepping back through is a decision rather than an
   * accident.
   */
  readonly entryPoints: readonly ZoneEntryPoint[] = [
    // Deliberately off the centre line.
    //
    // The terrain heightfield is 128 cells across a 260 m square centred on the
    // origin, so `x = 0` falls exactly on a cell boundary — and a downward
    // raycast at a heightfield seam is the one place Rapier's query can come
    // back empty over solid ground. `PhysicsWorld.groundHeight` returned null at
    // (0, 8.2) while returning a sane value 1 m either side, which is a
    // diagnostic-only failure here (the character controller resolves contacts
    // by shape cast, not by ray) but would be a real one for foot IK or for any
    // future system that asks "how high is the floor under the player".
    //
    // Authoring spawns off the seam costs nothing and removes the whole class.
    { id: 'camp-centre', position: at(1.4, 8.2), yaw: Math.PI },
    { id: 'gate', position: at(0.9, 19.5), yaw: Math.PI },
    { id: 'forge', position: at(9.6, -6.2), yaw: 0 },
  ];

  readonly portals: readonly PortalSpec[] = [
    {
      id: 'camp-gate',
      targetZone: 'bloodMoor',
      targetEntry: 'from-camp',
      position: at(0, PALISADE_RADIUS + 0.5),
      radius: 2.6,
      height: 3.5,
      label: 'the Blood Moor',
      verb: 'go out to',
    },
  ];

  /* -- lifecycle ---------------------------------------------------------- */

  async init(ctx: GameContext): Promise<void> {
    ctx.scene.add(this.#root);

    const assets = ctx.services.tryGet<AssetManager>(AssetManagerKey) ?? null;
    this.#materials = ctx.services.tryGet(MaterialLibraryKey) ?? null;
    this.#ibl = ctx.services.tryGet(IBLKey) ?? null;
    this.#lighting = ctx.services.tryGet(LightingKey) ?? null;

    // Build nothing that uses a photoscanned set before the set has landed. The
    // library swaps textures in asynchronously, and a material built early
    // renders its untextured procedural fallback for as long as it takes —
    // invisible in a live session, permanent in a capture that steps a fixed
    // number of frames as fast as it can.
    await this.#materials?.ready();

    // The one thing in the camp that is not firelight.
    //
    // Measured off `captures/zones-before/encampment.png`: the *coldest* decile
    // of the frame scored −0.025 on the blue-minus-red axis, i.e. the coldest
    // tenth of the picture was still warm, and the first three quartiles all
    // sat inside a 0.02-wide luma band. The camp was lit by eight fires and
    // nothing else, so every surface in it — the palisade behind the tents, the
    // mud at the gate, the far side of the muster ground — was the same hue at
    // the same value. No exposure or grade can separate two populations that
    // are one population, and a warm pool needs something cold to be a pool in.
    //
    // A hemisphere fill is the right shape for it: the sky half is the cold
    // night above the palisade and the ground half is the mud bouncing a little
    // of the fire back up, which is physically what a camp on open ground at
    // night does. Kept well under the bonfire's own contribution so the fire
    // stays the reason to stand in the middle — this is a fill, not a key.
    this.#lighting?.setAmbient({
      skyColor: 0x3f5f9c,
      groundColor: 0x2b2620,
      intensity: 0.85,
    });

    this.#buildGround();
    this.#buildPalisade();
    this.#buildBonfire();
    this.#buildForge();
    this.#buildTorchPosts();
    this.#buildCarts();

    if (assets !== null) {
      this.#kit = new PropKit(assets, { grade: CAMP_WEATHERING });
      await this.#kit.load(Object.values(PROP_KEYS).flat());
      this.#placeTents();
      this.#placeSupplies();
      this.#placeSmithy();
    } else {
      console.warn('[RogueEncampment] no AssetManager: the camp will be built geometry only');
    }

    this.#applyOcclusion();
  }

  update(ctx: GameContext): void {
    const t = ctx.time.elapsed;
    for (const entry of this.#lights) {
      const flicker = entry.fire.update(t);
      entry.handle?.setIntensity(entry.base * flicker);
    }
  }

  dispose(): void {
    // Hand the fill back. `LightingService` outlives every zone, so an ambient
    // set here and not cleared here would light the Blood Moor with the camp's
    // night sky — the same class of leak `ZoneManager` reverts the exposure
    // trim for, and one the manager cannot see.
    this.#lighting?.setAmbient({ intensity: 0 });
    for (const entry of this.#lights) entry.handle?.release();
    this.#lights.length = 0;
    for (const fire of this.#fires) fire.dispose();
    this.#fires.length = 0;
    this.#kit?.dispose();
    this.#kit = null;
    for (const resource of this.#owned) resource.dispose();
    this.#owned.length = 0;
    // The tree itself is walked and freed by `ZoneManager` through
    // `disposeZoneTree`, which is why nothing above detaches it.
  }

  /* -- physics ------------------------------------------------------------ */

  buildColliders(physics: PhysicsWorld): void {
    physics.buildTerrain(this.field, TERRAIN_SIZE, TERRAIN_SEGMENTS);
    physics.buildSceneColliders(this.#root, {
      // The camp's own exclusion list. `ground` keeps the terrain mesh out (it
      // is already a heightfield), and the fire cards are light, not matter.
      exclude: /ground|fire\.|ember|plume|banner|glow/i,
      // The palisade logs are 2.6 m of 0.16 m post: far past the slenderness
      // threshold, so they are fitted with upright cylinders rather than boxes.
      // A box per log would seal the wall as a solid ring, which is what is
      // wanted — but it would also do it 0.3 m outside the visible surface.
      slenderRatio: 1.6,
      trunkRadiusFactor: 0.85,
    });
  }

  /* -- NPC placement API --------------------------------------------------- */

  /** Look up one of {@link CAMP_NPC_ANCHORS} by id. */
  anchor(id: string): NpcAnchor | null {
    return CAMP_NPC_ANCHORS.find((entry) => entry.id === id) ?? null;
  }

  /**
   * Stand a figure on a named anchor.
   *
   * The contract for phase 5: load and scale the model however the NPC system
   * wants, then hand it here. This positions it on the anchor's ground height,
   * turns it to the anchor's facing, and parents it into the zone so it is torn
   * down with the zone — an NPC left parented to `ctx.scene` survives travel and
   * turns up standing in the middle of the Den of Evil.
   *
   * ```ts
   * const camp = zones.active as RogueEncampment;
   * const akara = await loadNpcModel('character.rogue');
   * camp.placeNpc('akara', akara);
   * ```
   *
   * @returns the anchor used, or `null` if the id is unknown.
   */
  placeNpc(id: string, object: THREE.Object3D): NpcAnchor | null {
    const anchor = this.anchor(id);
    if (anchor === null) {
      console.warn(
        `[RogueEncampment] no NPC anchor "${id}" ` +
          `(have: ${CAMP_NPC_ANCHORS.map((a) => a.id).join(', ')})`,
      );
      return null;
    }
    object.position.set(anchor.position.x, anchor.position.y, anchor.position.z);
    object.rotation.y = anchor.yaw;
    this.#root.add(object);
    return anchor;
  }

  /* -- ground ------------------------------------------------------------- */

  #buildGround(): void {
    const materials = this.#materials;
    const geometry = this.field.buildGeometry(TERRAIN_SIZE, TERRAIN_SEGMENTS);
    this.#owned.push(geometry);

    // Two archetypes, blended by distance from the fire — and the *absence* of
    // an `albedoTint` here is the load-bearing detail.
    //
    // The first two passes tinted `wetMud` by hand and the camp came back a
    // single warm brown from the bonfire out to the horizon. The Blood Moor's
    // ground, which reads correctly cold, passes no tint at all: the archetype
    // spec is already the art direction, and overriding its albedo was
    // overwriting the one thing that had been calibrated. So the tint is gone
    // and the archetypes are used as shipped.
    //
    // The blend is the camp's own idea, though. Two hundred people have walked
    // the inside of this palisade to bare mud; ten metres outside it, the moor's
    // dead grass is still there. That boundary is legible from any angle and it
    // is the cheapest possible way to say "this ground is used".
    const trodden = saturate(
      smoothstep(PALISADE_RADIUS - 7, PALISADE_RADIUS + 11, positionWorld.xz.length()),
    );

    const material =
      materials?.createBlended({
        base: 'wetMud',
        overlay: 'deadGrass',
        weight: trodden,
        depth: 0.16,
        // Hex anti-tiling is wrong at this scale for the same reason it is wrong
        // on the moor: its cells are metres wide on a 260 m plane and the blend
        // seams show up as a honeycomb. Macro variation breaks the repeat with
        // no cell structure to give itself away.
        baseOverrides: { antiTile: 'macro' },
        overlayOverrides: { antiTile: 'macro' },
      }) ?? new THREE.MeshStandardNodeMaterial({ color: 0x2a2622, roughness: 0.95 });
    material.name = 'camp.ground';
    this.#owned.push(material);

    const ground = new THREE.Mesh(geometry, material);
    ground.name = 'camp.ground';
    ground.receiveShadow = true;
    ground.castShadow = false;
    ground.userData['noCollide'] = true;
    this.#root.add(ground);
  }

  /* -- palisade ----------------------------------------------------------- */

  /**
   * The defensive ring: sharpened logs, driven at slight random angles.
   *
   * One instanced batch. The variation that stops it reading as a fence comes
   * from per-instance yaw, lean and height rather than from geometry variants —
   * at 24 m from the centre nobody resolves the difference between two log
   * profiles, but everybody sees a wall whose top edge is a straight line.
   *
   * The gate is an *arc* gap rather than a removed post, because a gap measured
   * in posts changes width when the post count changes.
   */
  #buildPalisade(): void {
    const materials = this.#materials;
    const rng = createRng(`${SEED}.palisade`);

    const logGeometry = palisadeLogGeometry();
    this.#owned.push(logGeometry);
    const material =
      materials?.create('bark', {
        albedoTint: [0.34, 0.3, 0.26],
        roughnessRange: [0.78, 1],
        tiling: 1.5,
      }) ?? new THREE.MeshStandardNodeMaterial({ color: 0x3a3128, roughness: 0.95 });
    material.name = 'camp.palisade';
    this.#owned.push(material);

    const samples: ScatterSample[] = [];
    const count = 198;
    let index = 0;
    for (let i = 0; i < count; i++) {
      const angle = (i / count) * Math.PI * 2;
      // The gate faces +z, i.e. angle = 0 in the (sin, cos) convention below.
      const fromGate = Math.abs(Math.atan2(Math.sin(angle), Math.cos(angle)));
      if (fromGate < GATE_HALF_ARC) continue;

      const jitter = (rng.next() - 0.5) * 0.55;
      const radius = PALISADE_RADIUS + (rng.next() - 0.5) * 0.35;
      const x = Math.sin(angle) * radius + jitter;
      const z = Math.cos(angle) * radius + jitter;
      samples.push({
        position: new THREE.Vector3(x, this.field.heightAt(x, z) - 0.25, z),
        normal: new THREE.Vector3(0, 1, 0),
        scale: 0.9 + rng.next() * 0.28,
        stretch: 0.88 + rng.next() * 0.3,
        // Facing outward, plus a lean. A palisade leans away from the camp.
        rotation: angle + (rng.next() - 0.5) * 0.5,
        index: index++,
      });
    }

    const wall = buildInstancedMesh(logGeometry, material, samples);
    wall.name = 'camp.palisade';
    this.#root.add(wall);

    this.#buildGate();
  }

  /** Two heavy uprights and a lintel, so the gap in the ring reads as a door. */
  #buildGate(): void {
    const materials = this.#materials;
    const material =
      materials?.create('plank', {
        albedoTint: [0.3, 0.26, 0.22],
        roughnessRange: [0.75, 1],
        tiling: 0.8,
      }) ?? new THREE.MeshStandardNodeMaterial({ color: 0x342c24, roughness: 0.95 });
    material.name = 'camp.gate';
    this.#owned.push(material);

    const half = Math.sin(GATE_HALF_ARC) * PALISADE_RADIUS + 0.55;
    const postHeight = 4.4;
    const parts: THREE.BufferGeometry[] = [];

    for (const side of [-1, 1]) {
      const x = side * half;
      const z = PALISADE_RADIUS;
      const post = new THREE.BoxGeometry(0.46, postHeight, 0.46);
      post.translate(x, this.field.heightAt(x, z) + postHeight / 2 - 0.3, z);
      parts.push(post);
    }
    const lintel = new THREE.BoxGeometry(half * 2 + 0.9, 0.46, 0.4);
    lintel.translate(0, this.field.heightAt(0, PALISADE_RADIUS) + postHeight - 0.55, PALISADE_RADIUS);
    parts.push(lintel);

    // A short palisade return on each side of the opening, angled inward, so
    // the gate is a short passage rather than a hole in a circle. This is the
    // difference between "a wall with a gap" and "a gate".
    for (const side of [-1, 1]) {
      for (let i = 0; i < 3; i++) {
        const x = side * (half + 0.2);
        const z = PALISADE_RADIUS - 0.9 - i * 0.95;
        const log = new THREE.CylinderGeometry(0.15, 0.17, 2.9, 6);
        log.translate(x, this.field.heightAt(x, z) + 1.3, z);
        parts.push(log);
      }
    }

    const merged = mergeGeometries(parts, false);
    for (const part of parts) part.dispose();
    if (merged === null) return;
    merged.computeVertexNormals();
    this.#owned.push(merged);

    const gate = new THREE.Mesh(merged, material);
    gate.name = 'camp.gate';
    gate.castShadow = true;
    gate.receiveShadow = true;
    this.#root.add(gate);
  }

  /* -- bonfire ------------------------------------------------------------ */

  /**
   * The centre of the camp, and the reason it reads as warm.
   *
   * A ring of hearth boulders, a leaning stack of half-burnt timbers, and a
   * {@link Fire} at 1.9 m — roughly three times the moor campfire, because this
   * one is a beacon and that one is a huddle.
   */
  #buildBonfire(): void {
    const materials = this.#materials;
    const base = this.field.heightAt(0, 0);
    const group = new THREE.Group();
    group.name = 'camp.bonfire';
    group.position.set(0, base, 0);
    this.#root.add(group);

    /* hearth ring */
    const rng = createRng(`${SEED}.hearth`);
    const stoneGeometry = generateRockGeometry({
      seed: `${SEED}.hearthstone`,
      detail: 2,
      radius: 0.5,
      displacement: 0.32,
      scale: [1, 0.7, 0.92],
      flattenBase: 0.42,
    });
    this.#owned.push(stoneGeometry);
    const stone =
      materials?.create('rock', {
        albedoTint: [0.12, 0.115, 0.11],
        roughnessRange: [0.74, 1],
      }) ?? new THREE.MeshStandardNodeMaterial({ color: 0x1e1c1a, roughness: 0.95 });
    stone.name = 'camp.hearthstone';
    this.#owned.push(stone);

    const stones: ScatterSample[] = [];
    for (let i = 0; i < 14; i++) {
      const angle = (i / 14) * Math.PI * 2 + rng.next() * 0.18;
      const radius = 1.95 + rng.next() * 0.16;
      stones.push({
        position: new THREE.Vector3(Math.cos(angle) * radius, 0.05, Math.sin(angle) * radius),
        normal: new THREE.Vector3(0, 1, 0),
        scale: 0.7 + rng.next() * 0.42,
        rotation: rng.next() * Math.PI * 2,
        index: i,
      });
    }
    const ring = buildInstancedMesh(stoneGeometry, stone, stones);
    ring.name = 'camp.hearthring';
    group.add(ring);

    /* timber stack */
    const charred =
      materials?.create('bark', {
        albedoTint: [0.07, 0.062, 0.058],
        roughnessRange: [0.85, 1],
      }) ?? new THREE.MeshStandardNodeMaterial({ color: 0x141210, roughness: 0.98 });
    charred.name = 'camp.timber';
    this.#owned.push(charred);

    const logs: THREE.BufferGeometry[] = [];
    const logRng = createRng(`${SEED}.timber`);
    for (let i = 0; i < 9; i++) {
      const angle = (i / 9) * Math.PI * 2 + logRng.next() * 0.3;
      const lean = 0.42 + logRng.next() * 0.18;
      const length = 2.3 + logRng.next() * 0.5;
      const log = new THREE.CylinderGeometry(0.09, 0.13, length, 6, 1);
      log.rotateZ(lean);
      log.rotateY(-angle);
      log.translate(Math.cos(angle) * 0.62, length * 0.42, Math.sin(angle) * 0.62);
      logs.push(log);
    }
    const merged = mergeGeometries(logs, false);
    for (const log of logs) log.dispose();
    if (merged !== null) {
      merged.computeVertexNormals();
      this.#owned.push(merged);
      const stack = new THREE.Mesh(merged, charred);
      stack.name = 'camp.timber';
      stack.castShadow = true;
      stack.receiveShadow = true;
      group.add(stack);
    }

    this.#addFire(group, { x: 0, y: 0.22, z: 0 }, {
      radius: 1.35,
      height: 3.1,
      intensity: 1.15,
      name: 'camp.bonfire.flame',
    }, {
      color: BONFIRE_COLOR,
      // 21 and 12 m, down from 34 and 17 m. The larger figures reached the
      // palisade and lit the entire camp floor warm — see the ground tint above
      // for why that is fatal. At 12 m the pool covers the muster ground and the
      // near half of the tent ring and stops, so the camp has a warm centre
      // inside a cold enclosure rather than being uniformly warm.
      intensity: 21,
      radius: 12,
      y: 1.5,
      castShadow: true,
      priority: 20,
    });
  }

  /* -- forge -------------------------------------------------------------- */

  /**
   * Charsi's forge: a stone hearth, a chimney breast, a coal bed and an anvil.
   *
   * There is no anvil or forge in the asset set, so both are built. An anvil is
   * a shape everyone recognises and almost nobody can describe, so it is worth
   * being explicit: a heavy waisted body, a flat face wider than the waist, and
   * one horn. Four boxes and a cone get all three, and at the size a player sees
   * it that is the whole silhouette.
   */
  #buildForge(): void {
    const materials = this.#materials;
    const originX = 13.4;
    const originZ = -4.6;
    const base = this.field.heightAt(originX, originZ);

    const group = new THREE.Group();
    group.name = 'camp.forge';
    group.position.set(originX, base, originZ);
    group.rotation.y = Math.atan2(-originX, -originZ);
    this.#root.add(group);

    const masonry =
      materials?.create('wetStone', {
        albedoTint: [0.19, 0.18, 0.175],
        roughnessRange: [0.68, 0.96],
        tiling: 1.1,
        wetnessExposure: 0.3,
      }) ?? new THREE.MeshStandardNodeMaterial({ color: 0x2a2724, roughness: 0.9 });
    masonry.name = 'camp.forge.stone';
    this.#owned.push(masonry);

    const stoneParts: THREE.BufferGeometry[] = [];
    const hearth = new THREE.BoxGeometry(2.7, 1.05, 1.5);
    hearth.translate(0, 0.52, 0);
    stoneParts.push(hearth);
    // Chimney breast, tapering, set at the back of the hearth.
    const breast = new THREE.BoxGeometry(1.5, 2.3, 0.7);
    breast.translate(0, 2.15, -0.42);
    stoneParts.push(breast);
    const flue = new THREE.CylinderGeometry(0.32, 0.44, 1.5, 8);
    flue.translate(0, 4.0, -0.42);
    stoneParts.push(flue);
    const lip = new THREE.BoxGeometry(2.9, 0.16, 1.7);
    lip.translate(0, 1.1, 0);
    stoneParts.push(lip);

    const stoneMerged = mergeGeometries(stoneParts, false);
    for (const part of stoneParts) part.dispose();
    if (stoneMerged !== null) {
      stoneMerged.computeVertexNormals();
      this.#owned.push(stoneMerged);
      const mesh = new THREE.Mesh(stoneMerged, masonry);
      mesh.name = 'camp.forge.body';
      mesh.castShadow = true;
      mesh.receiveShadow = true;
      group.add(mesh);
    }

    /* the anvil, on a stump */
    const iron =
      materials?.create('ironRusted', {
        albedoTint: [0.13, 0.115, 0.105],
        roughnessRange: [0.42, 0.78],
      }) ?? new THREE.MeshStandardNodeMaterial({ color: 0x201c1a, roughness: 0.6, metalness: 0.7 });
    iron.name = 'camp.forge.iron';
    this.#owned.push(iron);

    const anvilParts: THREE.BufferGeometry[] = [];
    const foot = new THREE.BoxGeometry(0.62, 0.14, 0.34);
    foot.translate(0, 0.07, 0);
    anvilParts.push(foot);
    const waist = new THREE.BoxGeometry(0.3, 0.28, 0.22);
    waist.translate(0, 0.28, 0);
    anvilParts.push(waist);
    const face = new THREE.BoxGeometry(0.86, 0.19, 0.3);
    face.translate(0, 0.51, 0);
    anvilParts.push(face);
    const horn = new THREE.ConeGeometry(0.14, 0.46, 8);
    horn.rotateZ(-Math.PI / 2);
    horn.translate(0.66, 0.51, 0);
    anvilParts.push(horn);
    const anvilMerged = mergeGeometries(anvilParts, false);
    for (const part of anvilParts) part.dispose();

    const stumpGeometry = new THREE.CylinderGeometry(0.34, 0.38, 0.55, 9);
    stumpGeometry.translate(0, 0.27, 0);
    this.#owned.push(stumpGeometry);
    const bark =
      materials?.create('bark', { albedoTint: [0.26, 0.22, 0.19] }) ??
      new THREE.MeshStandardNodeMaterial({ color: 0x2e2620, roughness: 0.95 });
    bark.name = 'camp.forge.stump';
    this.#owned.push(bark);

    const anvilGroup = new THREE.Group();
    anvilGroup.name = 'camp.forge.anvil';
    anvilGroup.position.set(-2.5, 0, 1.4);
    anvilGroup.rotation.y = 0.5;
    const stump = new THREE.Mesh(stumpGeometry, bark);
    stump.castShadow = true;
    stump.receiveShadow = true;
    anvilGroup.add(stump);
    if (anvilMerged !== null) {
      anvilMerged.computeVertexNormals();
      this.#owned.push(anvilMerged);
      const anvil = new THREE.Mesh(anvilMerged, iron);
      anvil.name = 'camp.forge.anvil.iron';
      anvil.position.y = 0.55;
      anvil.castShadow = true;
      anvil.receiveShadow = true;
      anvilGroup.add(anvil);
    }
    group.add(anvilGroup);

    /* the coal bed */
    this.#addFire(group, { x: 0, y: 1.16, z: 0.1 }, {
      radius: 0.62,
      height: 0.9,
      intensity: 1.45,
      phase: 3.7,
      name: 'camp.forge.coals',
    }, {
      color: FORGE_COLOR,
      intensity: 13,
      radius: 8.5,
      y: 0.6,
      castShadow: false,
      priority: 12,
    });
  }

  /* -- torch posts --------------------------------------------------------- */

  /**
   * Six weak lights on the inner ring.
   *
   * Individually they are nothing — radius 7 m, a fifth of the bonfire's output.
   * Collectively they are what turns a bonfire in a field into an enclosure,
   * because they put a floor under the light level everywhere between the muster
   * ground and the wall. Phase-offset so the ring does not pulse in unison,
   * which is the tell that gives away a shared flicker curve.
   */
  #buildTorchPosts(): void {
    const materials = this.#materials;
    const bark =
      materials?.create('bark', {
        albedoTint: [0.24, 0.2, 0.17],
        roughnessRange: [0.8, 1],
      }) ?? new THREE.MeshStandardNodeMaterial({ color: 0x2b2420, roughness: 0.96 });
    bark.name = 'camp.torchpost';
    this.#owned.push(bark);

    const parts: THREE.BufferGeometry[] = [];
    const positions: { x: number; z: number; y: number }[] = [];
    const count = 6;
    const postHeight = 2.55;
    const ringRadius = 17.5;

    for (let i = 0; i < count; i++) {
      // Offset half a step so no post stands in the gate's throat.
      const angle = ((i + 0.5) / count) * Math.PI * 2;
      const x = Math.sin(angle) * ringRadius;
      const z = Math.cos(angle) * ringRadius;
      const y = this.field.heightAt(x, z);
      positions.push({ x, y, z });

      const post = new THREE.CylinderGeometry(0.09, 0.12, postHeight, 7);
      post.translate(x, y + postHeight / 2, z);
      parts.push(post);
      // The iron basket the fuel sits in, as a short open cylinder.
      const basket = new THREE.CylinderGeometry(0.24, 0.16, 0.34, 8, 1, true);
      basket.translate(x, y + postHeight + 0.12, z);
      parts.push(basket);
    }

    const merged = mergeGeometries(parts, false);
    for (const part of parts) part.dispose();
    if (merged !== null) {
      merged.computeVertexNormals();
      this.#owned.push(merged);
      const posts = new THREE.Mesh(merged, bark);
      posts.name = 'camp.torchposts';
      posts.castShadow = true;
      posts.receiveShadow = true;
      this.#root.add(posts);
    }

    for (const [i, point] of positions.entries()) {
      this.#addFire(this.#root, { x: point.x, y: point.y + postHeight + 0.16, z: point.z }, {
        radius: 0.2,
        height: 0.6,
        intensity: 1.3,
        phase: i * 1.37,
        name: `camp.torch.${i}`,
      }, {
        color: TORCH_COLOR,
        intensity: 4.2,
        radius: 5.5,
        y: 0.12,
        castShadow: false,
        priority: 4,
      });
    }
  }

  /* -- carts --------------------------------------------------------------- */

  /**
   * Gheed's wagon and Warriv's caravan cart.
   *
   * Also not in the asset set, so built: a plank bed, two sideboards, two
   * spoked wheels and a pair of shafts resting on the ground. The wheels are
   * where a built cart usually goes wrong — a solid disc reads as a millstone —
   * so they are a rim, a hub and six spokes, which is about 200 triangles and
   * the entire difference.
   */
  #buildCarts(): void {
    const materials = this.#materials;
    const planks =
      materials?.create('plank', {
        albedoTint: [0.3, 0.255, 0.21],
        roughnessRange: [0.72, 1],
        tiling: 1.2,
      }) ?? new THREE.MeshStandardNodeMaterial({ color: 0x322a22, roughness: 0.94 });
    planks.name = 'camp.cart';
    this.#owned.push(planks);

    const carts: { x: number; z: number; yaw: number; name: string }[] = [
      { x: -8.2, z: 9.6, yaw: -0.9, name: 'camp.cart.gheed' },
      { x: -13.4, z: 1.4, yaw: 1.35, name: 'camp.cart.warriv' },
    ];

    for (const cart of carts) {
      const parts: THREE.BufferGeometry[] = [];
      const bed = new THREE.BoxGeometry(2.9, 0.16, 1.6);
      bed.translate(0, 0.92, 0);
      parts.push(bed);
      for (const side of [-1, 1]) {
        const board = new THREE.BoxGeometry(2.9, 0.62, 0.1);
        board.translate(0, 1.28, side * 0.78);
        parts.push(board);
      }
      const headboard = new THREE.BoxGeometry(0.1, 0.8, 1.6);
      headboard.translate(-1.45, 1.36, 0);
      parts.push(headboard);
      for (const side of [-1, 1]) {
        const shaft = new THREE.BoxGeometry(2.4, 0.11, 0.11);
        shaft.rotateZ(-0.32);
        shaft.translate(2.1, 0.5, side * 0.5);
        parts.push(shaft);
      }
      for (const side of [-1, 1]) {
        parts.push(cartWheel(0.62, side * 0.86, 0.62, 0.2));
      }

      const merged = mergeGeometries(parts, false);
      for (const part of parts) part.dispose();
      if (merged === null) continue;
      merged.computeVertexNormals();
      this.#owned.push(merged);

      const mesh = new THREE.Mesh(merged, planks);
      mesh.name = cart.name;
      mesh.position.set(cart.x, this.field.heightAt(cart.x, cart.z), cart.z);
      mesh.rotation.y = cart.yaw;
      mesh.castShadow = true;
      mesh.receiveShadow = true;
      this.#root.add(mesh);
    }
  }

  /* -- kit props ----------------------------------------------------------- */

  /**
   * Six tents on the structure ring, turned to face the fire.
   *
   * The ring is broken deliberately: the gate approach (roughly z > +14) is left
   * clear so a player entering sees the bonfire, not a tent wall. Composition
   * survives being walked around only if the sightline it depends on is a hole
   * in the layout rather than a camera angle.
   */
  #placeTents(): void {
    const kit = this.#kit;
    if (kit === null || !kit.has('prop.tent')) return;
    const rng = createRng(`${SEED}.tents`);
    // Bearings in radians, measured from +z (the gate) clockwise. The 0.0-0.7
    // band around the gate is intentionally empty.
    const bearings = [1.15, 1.95, 2.75, 3.55, 4.35, 5.25];
    for (const [i, bearing] of bearings.entries()) {
      const radius = STRUCTURE_RING + (rng.next() - 0.5) * 1.6;
      const x = Math.sin(bearing) * radius;
      const z = Math.cos(bearing) * radius;
      kit.placeSized(
        'prop.tent',
        3.2,
        {
          x,
          y: this.field.heightAt(x, z),
          z,
          // Doorway toward the fire, plus a little slop so the ring is not a
          // machined hexagon.
          yaw: Math.atan2(-x, -z) + (rng.next() - 0.5) * 0.3,
          name: `camp.tent.${i}`,
        },
        this.#root,
        'xz',
      );
    }
  }

  /** Stores: barrels, crates, sacks and timber, clustered against structures. */
  #placeSupplies(): void {
    const kit = this.#kit;
    if (kit === null) return;
    const rng = createRng(`${SEED}.supplies`);

    /** Clusters, each documented by what it is a store *for*. */
    const clusters: { x: number; z: number; radius: number; count: number; keys: AssetKey[] }[] = [
      // Gheed's stock, beside his wagon on the gate approach.
      { x: -6.8, z: 11.2, radius: 2.1, count: 7, keys: [...PROP_KEYS.crates, 'prop.sack'] },
      // The camp's own stores, behind the tent ring on the west side.
      { x: -14.6, z: -7.4, radius: 2.6, count: 8, keys: [...PROP_KEYS.barrels, 'prop.sack'] },
      // Warriv's caravan load.
      { x: -15.8, z: 3.6, radius: 1.9, count: 5, keys: [...PROP_KEYS.crates, 'prop.lumber'] },
      // Fuel and stock for the forge.
      { x: 15.6, z: -8.4, radius: 2.2, count: 6, keys: [...PROP_KEYS.barrels, 'prop.lumber'] },
      // Seating around the fire — the only cluster inside the structure ring,
      // and it is deliberately outside `INNER_CLEAR`.
      { x: 4.6, z: -6.2, radius: 1.4, count: 3, keys: ['prop.stool', 'prop.table'] },
    ];

    let index = 0;
    for (const cluster of clusters) {
      for (let i = 0; i < cluster.count; i++) {
        const key = cluster.keys[rng.int(0, cluster.keys.length - 1)];
        if (key === undefined || !kit.has(key)) continue;
        const angle = rng.next() * Math.PI * 2;
        const r = Math.sqrt(rng.next()) * cluster.radius;
        const x = cluster.x + Math.cos(angle) * r;
        const z = cluster.z + Math.sin(angle) * r;
        kit.placeSized(
          key,
          key.includes('stack') ? 1.35 : key.includes('table') ? 0.78 : 0.92,
          {
            x,
            y: this.field.heightAt(x, z),
            z,
            yaw: rng.next() * Math.PI * 2,
            tilt: (rng.next() - 0.5) * 0.05,
            name: `camp.supply.${index++}`,
          },
          this.#root,
        );
      }
    }

    // Two banners on the gate uprights: the only thing in the camp allowed to
    // carry a real hue, and it is the Sightless Eye's, not the fire's.
    const half = Math.sin(GATE_HALF_ARC) * PALISADE_RADIUS + 0.55;
    for (const side of [-1, 1]) {
      const x = side * half;
      const z = PALISADE_RADIUS - 0.35;
      kit.placeSized(
        'prop.banner',
        1.9,
        { x, y: this.field.heightAt(x, z) + 1.5, z, yaw: side > 0 ? 0.2 : -0.2, name: 'camp.banner' },
        this.#root,
      );
    }
  }

  /** Charsi's tools: a weapon rack, a bucket to quench in, arms leaning about. */
  #placeSmithy(): void {
    const kit = this.#kit;
    if (kit === null) return;
    const placements: { key: AssetKey; x: number; z: number; size: number; yaw: number }[] = [
      { key: 'prop.weaponrack', x: 11.2, z: -8.4, size: 1.7, yaw: 2.2 },
      { key: 'prop.weaponrack', x: 14.6, z: -9.6, size: 1.7, yaw: 2.5 },
      { key: 'prop.bucket', x: 11.5, z: -2.2, size: 0.5, yaw: 0.9 },
      { key: 'prop.swordshield', x: 12.4, z: -1.4, size: 1.0, yaw: -0.6 },
    ];
    for (const item of placements) {
      kit.placeSized(
        item.key,
        item.size,
        {
          x: item.x,
          y: this.field.heightAt(item.x, item.z),
          z: item.z,
          yaw: item.yaw,
          name: `camp.smithy.${item.key}`,
        },
        this.#root,
      );
    }
  }

  /* -- helpers ------------------------------------------------------------- */

  /** Build a fire and the light that belongs to it, welded to the same flicker. */
  #addFire(
    parent: THREE.Object3D,
    at3: { x: number; y: number; z: number },
    fire: FireOptions,
    light: {
      color: number;
      intensity: number;
      radius: number;
      y: number;
      castShadow: boolean;
      priority: number;
    },
  ): void {
    const effect = new Fire(fire);
    effect.group.position.set(at3.x, at3.y, at3.z);
    parent.add(effect.group);
    this.#fires.push(effect);

    // The light's world position has to account for the parent's transform: the
    // forge group is rotated and offset, and `LightingService` takes world
    // coordinates. Resolving it here rather than trusting the local position is
    // why the forge's glow is in the forge and not 13 m away at the origin.
    parent.updateMatrixWorld(true);
    const world = effect.group.localToWorld(new THREE.Vector3(0, light.y, 0));

    const handle =
      this.#lighting?.addLight({
        kind: 'point',
        name: fire.name ?? 'camp.fire',
        position: { x: world.x, y: world.y, z: world.z },
        color: light.color,
        intensity: light.intensity,
        radius: light.radius,
        decay: 2,
        castShadow: light.castShadow,
        priority: light.priority,
      }) ?? null;

    this.#lights.push({ handle, base: light.intensity, fire: effect });
  }

  /**
   * Fold the GTAO buffer into every material this zone owns.
   *
   * Materials the library builds are not hooked up automatically —
   * `IBLService.applyOcclusion` is an explicit call so a material that wants its
   * own `aoNode` can opt out — and a zone that forgets it is a zone with no
   * contact shadows, which is exactly the cue that makes props sit on ground.
   */
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
        // The fire cards are unlit by construction; occluding them would be
        // applying shadow to a light source.
        if (material.name.startsWith('fire.')) continue;
        if (material instanceof THREE.NodeMaterial) ibl.applyOcclusion(material);
      }
    });
  }
}

/* -------------------------------------------------------------------------- */
/* Built geometry                                                             */
/* -------------------------------------------------------------------------- */

/**
 * One sharpened palisade log: a slightly tapered shaft with a conical point.
 *
 * Seven radial segments. A log seen from 10 m at 0.16 m radius is about twelve
 * pixels wide, and the difference between seven sides and sixteen is invisible
 * at four times the vertex cost — multiplied by 198 instances.
 */
function palisadeLogGeometry(): THREE.BufferGeometry {
  const height = 3.1;
  const radius = 0.16;
  const shaft = new THREE.CylinderGeometry(radius * 0.88, radius, height, 7, 1);
  shaft.translate(0, height / 2, 0);
  const tip = new THREE.ConeGeometry(radius * 0.88, radius * 3.2, 7);
  tip.translate(0, height + radius * 1.6, 0);
  const merged = mergeGeometries([shaft, tip], false);
  shaft.dispose();
  tip.dispose();
  const result = merged ?? new THREE.CylinderGeometry(radius, radius, height, 7);
  result.computeVertexNormals();
  result.computeBoundingBox();
  result.computeBoundingSphere();
  return result;
}

/** A rim, a hub and six spokes, lying in the XY plane and offset along z. */
function cartWheel(radius: number, z: number, y: number, x: number): THREE.BufferGeometry {
  const parts: THREE.BufferGeometry[] = [];
  const rim = new THREE.TorusGeometry(radius, 0.07, 5, 14);
  parts.push(rim);
  const hub = new THREE.CylinderGeometry(0.11, 0.11, 0.2, 7);
  hub.rotateX(Math.PI / 2);
  parts.push(hub);
  for (let i = 0; i < 6; i++) {
    const spoke = new THREE.BoxGeometry(0.05, radius * 1.86, 0.05);
    spoke.rotateZ((i / 6) * Math.PI);
    parts.push(spoke);
  }
  const merged = mergeGeometries(parts, false);
  for (const part of parts) part.dispose();
  const wheel = merged ?? new THREE.TorusGeometry(radius, 0.07, 5, 14);
  wheel.translate(x, y, z);
  return wheel;
}
