/**
 * @module rpg/Loot
 *
 * Drop tables, and the physical loot they produce on the ground.
 *
 * ### Two halves, deliberately separated
 *
 * The top half is pure: a table per enemy type, and {@link rollDrops}, which
 * turns a seed and a table into gold and items. No three.js, no engine, no
 * events — so the drop rates can be measured over a hundred thousand simulated
 * kills in a unit test rather than estimated by playing.
 *
 * The bottom half is {@link LootSystem}, a `GameModule` that listens for
 * `combat:death`, rolls the table, and puts a marker in the world. It resolves
 * combat through the `ServiceLocator` and degrades to doing nothing when combat
 * is absent, so it can be registered in any engine configuration.
 *
 * ### Colour is not decoration
 *
 * The item's quality colour is carried on the ground label, not only in the
 * inventory. In Diablo that colour is the entire pickup decision — a player
 * scanning a floor of white labels for one blue one is reading the game's most
 * important signal, and moving that signal into a menu removes it.
 *
 * ### Determinism
 *
 * Every roll is seeded from the kill: `killSeed(combatantId, killIndex)`. Two
 * runs that kill the same skeletons in the same order produce the same floor.
 * That is what makes a loot bug reproducible.
 */

import * as THREE from 'three/webgpu';

import { CombatantsKey, type CombatantRegistry } from '../combat/Combatant';
import { mulberry32 } from '../combat/DamageModel';
import { serviceKey } from '../core/ServiceLocator';
import type { GameContext, GameModule } from '../core/types';
import {
  generateItem,
  GOLD_COLOUR,
  ITEM_COLOURS,
  type Item,
  type ItemCategory,
  type ItemQuality,
} from './ItemGenerator';

/* -------------------------------------------------------------------------- */
/* Events                                                                      */
/* -------------------------------------------------------------------------- */

declare module '../core/EventBus' {
  interface GameEvents {
    /** Something hit the floor. One event per drop, not one per kill. */
    'loot:dropped': {
      id: number;
      kind: 'item' | 'gold';
      label: string;
      colour: string;
      position: THREE.Vector3;
    };
    /** The player picked something up. */
    'loot:pickedUp': { id: number; kind: 'item' | 'gold'; label: string; gold: number };
    /** A pickup was refused because the bag is full. */
    'loot:refused': { id: number; label: string; reason: 'inventory-full' };
  }
}

/* -------------------------------------------------------------------------- */
/* Drop tables                                                                 */
/* -------------------------------------------------------------------------- */

export interface DropTable {
  /** Enemy variant this table is keyed to. */
  readonly id: string;
  /** Monster level. Feeds both the experience award and the item level. */
  readonly monsterLevel: number;
  /** Base experience before the level-difference taper. */
  readonly experience: number;
  /** Probability in `[0, 1]` that gold drops at all. */
  readonly goldChance: number;
  readonly goldMin: number;
  readonly goldMax: number;
  /** Probability in `[0, 1]` that an item drops. */
  readonly itemChance: number;
  /** Multiplier on the magic/rare/unique odds. */
  readonly qualityBias: number;
  /** Restricts which base categories can drop. Omitted means anything. */
  readonly categories?: readonly ItemCategory[];
}

function table(definition: DropTable): DropTable {
  return definition;
}

/**
 * The Act I tables.
 *
 * Tuned so that clearing the Den of Evil's full spawn count yields roughly two
 * or three items worth picking up and a few hundred gold: enough that the walk
 * back to Gheed is worth making, not so much that the first shopping trip
 * trivialises the act. The mage drops least often and best, which is the usual
 * shape — the enemy that is annoying to fight has to pay for itself.
 */
export const DROP_TABLES: Readonly<Record<string, DropTable>> = {
  minion: table({
    id: 'minion',
    monsterLevel: 2,
    experience: 60,
    goldChance: 0.55,
    goldMin: 3,
    goldMax: 14,
    itemChance: 0.22,
    qualityBias: 1,
  }),
  warrior: table({
    id: 'warrior',
    monsterLevel: 3,
    experience: 110,
    goldChance: 0.65,
    goldMin: 6,
    goldMax: 22,
    itemChance: 0.32,
    qualityBias: 1.15,
  }),
  rogue: table({
    id: 'rogue',
    monsterLevel: 3,
    experience: 125,
    goldChance: 0.6,
    goldMin: 5,
    goldMax: 20,
    itemChance: 0.3,
    qualityBias: 1.2,
  }),
  mage: table({
    id: 'mage',
    monsterLevel: 4,
    experience: 165,
    goldChance: 0.7,
    goldMin: 10,
    goldMax: 30,
    itemChance: 0.38,
    qualityBias: 1.5,
  }),
};

/** The table used when an enemy variant has none of its own. */
export const DEFAULT_DROP_TABLE: DropTable = DROP_TABLES['minion'] as DropTable;

export function dropTableFor(variant: string): DropTable {
  return DROP_TABLES[variant] ?? DEFAULT_DROP_TABLE;
}

/** Parse `"warrior#7"` — the shape `EnemyBase` builds its label in. */
export function variantFromLabel(label: string): string {
  const hash = label.indexOf('#');
  return hash === -1 ? label : label.slice(0, hash);
}

/**
 * A reproducible seed for one kill.
 *
 * Mixing the combatant id with a monotonically increasing kill index means two
 * skeletons of the same variant killed in the same session roll differently,
 * while a replay of that session rolls identically.
 */
export function killSeed(combatantId: number, killIndex: number): number {
  return (Math.imul(combatantId + 1, 0x9e3779b1) ^ Math.imul(killIndex + 1, 0x85ebca6b)) >>> 0;
}

export interface DropRollOptions {
  readonly seed: number;
  /** Overrides the table's monster level as the item level. */
  readonly itemLevel?: number;
  /** Extra multiplier on quality, e.g. from a quest or a boss. */
  readonly qualityBias?: number;
}

export interface DropRoll {
  readonly gold: number;
  readonly items: readonly Item[];
}

/**
 * Roll one enemy's drops.
 *
 * Gold is rolled before items, unconditionally, so that changing the item
 * chance does not shift the gold stream for an already-recorded seed.
 */
export function rollDrops(table: DropTable, options: DropRollOptions): DropRoll {
  const rng = mulberry32(options.seed);
  const itemLevel = Math.max(1, Math.round(options.itemLevel ?? table.monsterLevel));

  const goldRoll = rng();
  const goldAmount = rng();
  const gold =
    goldRoll < table.goldChance
      ? Math.max(1, Math.round(table.goldMin + goldAmount * (table.goldMax - table.goldMin)))
      : 0;

  const items: Item[] = [];
  if (rng() < table.itemChance) {
    items.push(
      generateItem({
        // A second, decorrelated seed so that the item roll cannot be inferred
        // from the gold roll — which matters the moment anyone tunes one of
        // them and expects the other to hold still.
        seed: (options.seed ^ 0x632be5ab) >>> 0,
        itemLevel,
        qualityBias: table.qualityBias * (options.qualityBias ?? 1),
        ...(table.categories !== undefined ? { categories: table.categories } : {}),
      }),
    );
  }
  return { gold, items };
}

/**
 * Measured drop statistics over `samples` seeded rolls.
 *
 * Exists so the tests can assert on the *distribution* rather than on one
 * lucky seed. A drop rate that is right on average and wrong for the first
 * thousand kills is still wrong.
 */
export function sampleDropRates(
  table: DropTable,
  samples: number,
  startSeed = 1,
): { goldRate: number; itemRate: number; qualities: Record<ItemQuality, number> } {
  const qualities: Record<ItemQuality, number> = { normal: 0, magic: 0, rare: 0, unique: 0 };
  let goldDrops = 0;
  let itemDrops = 0;
  for (let i = 0; i < samples; i++) {
    const roll = rollDrops(table, { seed: startSeed + i });
    if (roll.gold > 0) goldDrops++;
    for (const item of roll.items) {
      itemDrops++;
      qualities[item.quality]++;
    }
  }
  return { goldRate: goldDrops / samples, itemRate: itemDrops / samples, qualities };
}

/* -------------------------------------------------------------------------- */
/* Ground loot                                                                 */
/* -------------------------------------------------------------------------- */

/** One thing lying on the floor. */
export interface GroundLoot {
  readonly id: number;
  readonly kind: 'item' | 'gold';
  /** Null for gold piles. */
  readonly item: Item | null;
  readonly gold: number;
  /** What the label reads. */
  readonly label: string;
  /** D2 quality colour, or gold's own. */
  readonly colour: string;
  readonly position: THREE.Vector3;
  /** The in-world marker, when the system has a scene to put it in. */
  readonly object: THREE.Object3D | null;
}

/** How close the player must be to pick something up, in metres. */
export const PICKUP_RADIUS = 2.2;

/** How far away a label stays legible before it is culled, in metres. */
export const LABEL_RANGE = 14;

export const LootSystemKey = serviceKey<LootSystem>('rpg.loot');

/** Callback the RPG layer supplies to actually take an item. */
export interface LootReceiver {
  /** @returns whether the item was taken. False means the bag is full. */
  takeItem(item: Item): boolean;
  takeGold(amount: number): void;
}

export interface LootSystemOptions {
  /** Drop marker scale, metres. */
  readonly markerSize?: number;
  /** Whether to build in-world markers at all. Off in headless unit tests. */
  readonly markers?: boolean;
}

/**
 * Turns enemy deaths into things on the floor, and the Interact key into
 * things in the bag.
 *
 * The marker geometry and material are created once and shared across every
 * drop, so a floor covered in loot costs one geometry and two materials rather
 * than one of each per pile.
 */
export class LootSystem implements GameModule {
  readonly name = 'rpg.loot';

  readonly #options: Required<LootSystemOptions>;
  readonly #entries: GroundLoot[] = [];
  readonly #unsubscribe: Array<() => void> = [];
  readonly #scratch = new THREE.Vector3();

  #ctx: GameContext | null = null;
  #receiver: LootReceiver | null = null;
  #root: THREE.Group | null = null;
  #geometry: THREE.BufferGeometry | null = null;
  #itemMaterials = new Map<string, THREE.Material>();
  #nextId = 1;
  #kills = 0;

  constructor(options: LootSystemOptions = {}) {
    this.#options = {
      markerSize: options.markerSize ?? 0.22,
      markers: options.markers ?? true,
    };
  }

  /** Everything currently on the floor. */
  get entries(): readonly GroundLoot[] {
    return this.#entries;
  }

  /** Who receives picked-up loot. Set by the RPG system at boot. */
  setReceiver(receiver: LootReceiver | null): void {
    this.#receiver = receiver;
  }

  init(ctx: GameContext): void {
    this.#ctx = ctx;
    ctx.services.register(LootSystemKey, this);

    if (this.#options.markers) {
      this.#root = new THREE.Group();
      this.#root.name = 'rpg.loot';
      ctx.scene.add(this.#root);
      // An octahedron rather than a sphere: it reads as "a thing", not as a
      // particle, at the two-pixel size it occupies from across a room, and it
      // costs eight triangles.
      this.#geometry = new THREE.OctahedronGeometry(this.#options.markerSize, 0);
    }

    this.#unsubscribe.push(
      ctx.events.on('combat:death', (payload) => {
        if (payload.faction !== 'enemy') return;
        this.onEnemyDeath(payload.combatant, payload.label);
      }),
      // Loot is zone-local: a pile dropped in the Den must not be waiting in
      // the camp. The zone system tears down its own subtree, so anything
      // parented to `ctx.scene` — which these markers are, so they survive a
      // zone that is mid-teardown — has to be cleared explicitly.
      ctx.events.on('zone:unloading', () => this.clear()),
    );
  }

  update(ctx: GameContext): void {
    if (ctx.input.wasPressed('Interact')) this.pickUpNearest();
    // A slow spin makes a two-pixel marker findable in peripheral vision, which
    // is the entire job of the marker.
    const spin = ctx.time.elapsed * 1.4;
    for (const entry of this.#entries) {
      if (entry.object !== null) entry.object.rotation.y = spin;
    }
  }

  dispose(): void {
    for (const off of this.#unsubscribe) off();
    this.#unsubscribe.length = 0;
    this.clear();
    this.#root?.removeFromParent();
    this.#root = null;
    this.#geometry?.dispose();
    this.#geometry = null;
    for (const material of this.#itemMaterials.values()) material.dispose();
    this.#itemMaterials.clear();
    this.#ctx?.services.unregister(LootSystemKey);
    this.#ctx = null;
  }

  /* -- drops --------------------------------------------------------------- */

  /**
   * Roll and place one enemy's drops.
   *
   * Public so the drive harness can trigger a drop without staging a kill, and
   * so a test can call it with a stub registry.
   */
  onEnemyDeath(combatantId: number, label: string): DropRoll {
    const variant = variantFromLabel(label);
    const table = dropTableFor(variant);
    const seed = killSeed(combatantId, this.#kills++);
    const roll = rollDrops(table, { seed });

    const position = this.#positionOf(combatantId);
    if (roll.gold > 0) {
      this.spawnGold(roll.gold, position, 0);
    }
    let index = roll.gold > 0 ? 1 : 0;
    for (const item of roll.items) this.spawnItem(item, position, index++);
    return roll;
  }

  /** Put an item on the floor at `position`. */
  spawnItem(item: Item, position: THREE.Vector3, spread = 0): GroundLoot {
    return this.#spawn({
      kind: 'item',
      item,
      gold: 0,
      label: item.name,
      colour: ITEM_COLOURS[item.quality],
      position,
      spread,
    });
  }

  /** Put a gold pile on the floor. */
  spawnGold(amount: number, position: THREE.Vector3, spread = 0): GroundLoot {
    return this.#spawn({
      kind: 'gold',
      item: null,
      gold: Math.max(1, Math.round(amount)),
      label: `${Math.max(1, Math.round(amount))} Gold`,
      colour: GOLD_COLOUR,
      position,
      spread,
    });
  }

  /* -- pickup -------------------------------------------------------------- */

  /** The closest pickup within {@link PICKUP_RADIUS} of the player, or null. */
  nearest(): GroundLoot | null {
    const player = this.#playerPosition();
    if (player === null) return null;
    let best: GroundLoot | null = null;
    let bestDistance = PICKUP_RADIUS * PICKUP_RADIUS;
    for (const entry of this.#entries) {
      const distance = entry.position.distanceToSquared(player);
      if (distance <= bestDistance) {
        best = entry;
        bestDistance = distance;
      }
    }
    return best;
  }

  /** Take the nearest pickup. @returns what was taken, or null. */
  pickUpNearest(): GroundLoot | null {
    const entry = this.nearest();
    if (entry === null) return null;
    return this.pickUp(entry.id) ? entry : null;
  }

  /**
   * Take a specific pickup.
   *
   * A full bag refuses the pickup and leaves the item on the floor, rather
   * than dropping it into the void — and says why, so the HUD can tell the
   * player instead of leaving them pressing E at an item that will not move.
   */
  pickUp(id: number): boolean {
    const index = this.#entries.findIndex((entry) => entry.id === id);
    if (index === -1) return false;
    const entry = this.#entries[index] as GroundLoot;
    const receiver = this.#receiver;

    if (entry.kind === 'gold') {
      receiver?.takeGold(entry.gold);
    } else if (entry.item !== null) {
      const taken = receiver === null ? true : receiver.takeItem(entry.item);
      if (!taken) {
        this.#ctx?.events.emit('loot:refused', {
          id: entry.id,
          label: entry.label,
          reason: 'inventory-full',
        });
        return false;
      }
    }

    this.#entries.splice(index, 1);
    entry.object?.removeFromParent();
    this.#ctx?.events.emit('loot:pickedUp', {
      id: entry.id,
      kind: entry.kind,
      label: entry.label,
      gold: entry.gold,
    });
    return true;
  }

  /** Remove everything from the floor. Called on zone unload and on dispose. */
  clear(): void {
    for (const entry of this.#entries) entry.object?.removeFromParent();
    this.#entries.length = 0;
  }

  /* -- internals ----------------------------------------------------------- */

  #spawn(spec: {
    kind: 'item' | 'gold';
    item: Item | null;
    gold: number;
    label: string;
    colour: string;
    position: THREE.Vector3;
    spread: number;
  }): GroundLoot {
    // Fan multiple drops from one kill apart so two labels do not sit on top of
    // one another and read as one unreadable smear.
    const angle = spec.spread * 2.4;
    const radius = spec.spread === 0 ? 0 : 0.45 + spec.spread * 0.18;
    const position = spec.position
      .clone()
      .add(new THREE.Vector3(Math.cos(angle) * radius, 0, Math.sin(angle) * radius));

    const entry: GroundLoot = {
      id: this.#nextId++,
      kind: spec.kind,
      item: spec.item,
      gold: spec.gold,
      label: spec.label,
      colour: spec.colour,
      position,
      object: this.#buildMarker(spec.colour, position),
    };
    this.#entries.push(entry);
    this.#ctx?.events.emit('loot:dropped', {
      id: entry.id,
      kind: entry.kind,
      label: entry.label,
      colour: entry.colour,
      position: position.clone(),
    });
    return entry;
  }

  #buildMarker(colour: string, position: THREE.Vector3): THREE.Object3D | null {
    const geometry = this.#geometry;
    const root = this.#root;
    if (geometry === null || root === null) return null;

    let material = this.#itemMaterials.get(colour);
    if (material === undefined) {
      material = new THREE.MeshStandardMaterial({
        color: new THREE.Color(colour),
        emissive: new THREE.Color(colour),
        // Bright enough to read against the dark grade without blowing out the
        // bloom threshold, which would turn every drop into a white blob.
        emissiveIntensity: 1.6,
        roughness: 0.4,
        metalness: 0,
      });
      this.#itemMaterials.set(colour, material);
    }

    const mesh = new THREE.Mesh(geometry, material);
    mesh.position.copy(position);
    mesh.position.y += this.#options.markerSize + 0.05;
    mesh.castShadow = false;
    mesh.receiveShadow = false;
    root.add(mesh);
    return mesh;
  }

  #positionOf(combatantId: number): THREE.Vector3 {
    const registry = this.#ctx?.services.tryGet<CombatantRegistry>(CombatantsKey);
    const combatant = registry?.all.find((entry) => entry.id === combatantId);
    if (combatant === undefined) return this.#playerPosition() ?? new THREE.Vector3();
    return combatant.footPosition(this.#scratch).clone();
  }

  #playerPosition(): THREE.Vector3 | null {
    const registry = this.#ctx?.services.tryGet<CombatantRegistry>(CombatantsKey);
    const player = registry?.first('player');
    if (player === undefined || player === null) return null;
    return player.footPosition(this.#scratch).clone();
  }
}
