/**
 * @module rpg/Inventory
 *
 * Diablo II's grid inventory and its equipment paper doll.
 *
 * ### Why a grid rather than a list
 *
 * Because the grid *is* the decision. A list inventory with 40 slots asks the
 * player nothing; a 10x4 grid where a Broad Sword costs six cells and a ring
 * costs one asks them, on every drop, whether the sword is worth two rings —
 * and it asks without a single line of UI text. Every rule below exists to make
 * that question answerable and honest:
 *
 * - An item occupies a `width x height` rectangle. Placement is validated
 *   against real occupancy, never against a free-slot count.
 * - {@link InventoryGrid.add} is a **first-fit** scan in reading order, so
 *   auto-pickup is deterministic and a save/load round-trip cannot shuffle the
 *   bag.
 * - Dropping an item onto exactly one other item swaps them, which is D2's
 *   cursor behaviour; onto two or more, the drop is refused, because there is
 *   no sensible thing to put on the cursor.
 *
 * ### Equipment
 *
 * Ten slots, two of which are rings. Slot validation is by item *slot* rather
 * than category, and the two-handed rule is enforced in both directions: a
 * two-handed weapon evicts the offhand, and equipping an offhand evicts a
 * two-handed weapon. Getting only one direction right is the standard bug and
 * it presents as a shield the player can see but is not wearing.
 *
 * Stat application is deliberately *not* done here. {@link Equipment.modifiers}
 * returns the summed bag and `rpg/Stats` derives from it, so there is exactly
 * one place where "what is my attack rating" is answered and no possibility of
 * an equip/unequip pair leaving a residue.
 */

import type { DamageRange } from '../combat/DamageModel';
import { cloneItem, isBroken, sumItemModifiers, type Item, type ItemSlot } from './ItemGenerator';
import type { ModifierTotals } from './Stats';
import { zeroModifiers } from './Stats';

/* -------------------------------------------------------------------------- */
/* Grid                                                                        */
/* -------------------------------------------------------------------------- */

/** D2's inventory is 10 columns by 4 rows. */
export const INVENTORY_WIDTH = 10;
export const INVENTORY_HEIGHT = 4;

/** An item and where its top-left cell sits. */
export interface Placement {
  readonly item: Item;
  readonly x: number;
  readonly y: number;
}

/** Outcome of a drag-and-drop onto the grid. */
export interface DropResult {
  readonly placed: boolean;
  /** The item pushed onto the cursor by a swap, if any. */
  readonly displaced: Item | null;
  /** Why a refused drop was refused. Drives the UI's error feedback. */
  readonly reason: 'ok' | 'out-of-bounds' | 'blocked' | 'not-found';
}

export interface InventorySnapshot {
  readonly width: number;
  readonly height: number;
  readonly items: readonly { readonly x: number; readonly y: number; readonly item: Item }[];
}

export class InventoryGrid {
  readonly width: number;
  readonly height: number;

  /**
   * Occupancy, one entry per cell, holding the index into {@link #placements}
   * or -1. An index rather than a boolean so that "what is under the cursor"
   * is O(1) — the UI asks that on every pointer move.
   */
  readonly #cells: Int16Array;
  readonly #placements: Placement[] = [];

  constructor(width = INVENTORY_WIDTH, height = INVENTORY_HEIGHT) {
    this.width = Math.max(1, Math.round(width));
    this.height = Math.max(1, Math.round(height));
    this.#cells = new Int16Array(this.width * this.height).fill(-1);
  }

  /** Every placed item, in placement order. */
  get placements(): readonly Placement[] {
    return this.#placements;
  }

  get items(): Item[] {
    return this.#placements.map((entry) => entry.item);
  }

  get count(): number {
    return this.#placements.length;
  }

  /** Cells not covered by any item. */
  get freeCells(): number {
    let free = 0;
    for (let i = 0; i < this.#cells.length; i++) if (this.#cells[i] === -1) free++;
    return free;
  }

  /** The item covering `(x, y)`, or null. */
  itemAt(x: number, y: number): Item | null {
    const index = this.#indexAt(x, y);
    if (index === -1) return null;
    return this.#placements[index]?.item ?? null;
  }

  /** Where `item` sits, or null if it is not in this grid. */
  placementOf(item: Item): Placement | null {
    return this.#placements.find((entry) => entry.item === item) ?? null;
  }

  contains(item: Item): boolean {
    return this.#placements.some((entry) => entry.item === item);
  }

  /**
   * Items whose footprint intersects `item` placed at `(x, y)`.
   *
   * `ignore` excludes an item from the test, which is what makes "move this
   * item one cell to the right" work: without it, an item always collides with
   * itself.
   */
  overlapping(item: Item, x: number, y: number, ignore?: Item | null): Item[] {
    const found: Item[] = [];
    if (!this.#inBounds(item, x, y)) return found;
    for (let dy = 0; dy < item.height; dy++) {
      for (let dx = 0; dx < item.width; dx++) {
        const index = this.#indexAt(x + dx, y + dy);
        if (index === -1) continue;
        const other = this.#placements[index]?.item;
        if (other === undefined || other === ignore) continue;
        if (!found.includes(other)) found.push(other);
      }
    }
    return found;
  }

  /** Whether `item` fits at `(x, y)` with nothing already there. */
  canPlace(item: Item, x: number, y: number, ignore?: Item | null): boolean {
    if (!this.#inBounds(item, x, y)) return false;
    return this.overlapping(item, x, y, ignore).length === 0;
  }

  /** Place `item` at `(x, y)`. @returns whether it was placed. */
  place(item: Item, x: number, y: number): boolean {
    if (this.contains(item)) return false;
    if (!this.canPlace(item, x, y)) return false;
    this.#write(item, x, y);
    return true;
  }

  /**
   * First-fit insertion in reading order.
   *
   * Reading order, not "nearest to where it dropped": a deterministic scan is
   * what makes a save round-trip and an automated test reproducible, and the
   * player who cares about layout drags things anyway.
   */
  add(item: Item): Placement | null {
    for (let y = 0; y <= this.height - item.height; y++) {
      for (let x = 0; x <= this.width - item.width; x++) {
        if (!this.canPlace(item, x, y)) continue;
        this.#write(item, x, y);
        return { item, x, y };
      }
    }
    return null;
  }

  /** Whether {@link add} would succeed, without mutating anything. */
  hasRoomFor(item: Item): boolean {
    for (let y = 0; y <= this.height - item.height; y++) {
      for (let x = 0; x <= this.width - item.width; x++) {
        if (this.canPlace(item, x, y)) return true;
      }
    }
    return false;
  }

  /** Remove and return whatever covers `(x, y)`. */
  removeAt(x: number, y: number): Item | null {
    const item = this.itemAt(x, y);
    if (item === null) return null;
    this.remove(item);
    return item;
  }

  remove(item: Item): boolean {
    const index = this.#placements.findIndex((entry) => entry.item === item);
    if (index === -1) return false;
    this.#placements.splice(index, 1);
    this.#rebuild();
    return true;
  }

  /**
   * Drop a *held* item onto `(x, y)`.
   *
   * The D2 cursor contract: an empty target places it, a target covering
   * exactly one item swaps (the displaced item goes to the cursor), and a
   * target covering two or more is refused. `held` must not already be in this
   * grid — take it out first with {@link removeAt}.
   */
  drop(held: Item, x: number, y: number): DropResult {
    if (!this.#inBounds(held, x, y)) {
      return { placed: false, displaced: null, reason: 'out-of-bounds' };
    }
    const collisions = this.overlapping(held, x, y);
    if (collisions.length === 0) {
      this.#write(held, x, y);
      return { placed: true, displaced: null, reason: 'ok' };
    }
    if (collisions.length > 1) {
      return { placed: false, displaced: null, reason: 'blocked' };
    }
    const displaced = collisions[0] as Item;
    this.remove(displaced);
    this.#write(held, x, y);
    return { placed: true, displaced, reason: 'ok' };
  }

  /**
   * Move an item already in the grid from one cell to another.
   *
   * Distinct from {@link drop} because the moving item must not collide with
   * itself, and because a refused move must leave the grid *exactly* as it was
   * — a move implemented as remove-then-place corrupts the bag whenever the
   * placement fails.
   */
  move(from: { x: number; y: number }, to: { x: number; y: number }): DropResult {
    const item = this.itemAt(from.x, from.y);
    if (item === null) return { placed: false, displaced: null, reason: 'not-found' };
    if (!this.#inBounds(item, to.x, to.y)) {
      return { placed: false, displaced: null, reason: 'out-of-bounds' };
    }
    const collisions = this.overlapping(item, to.x, to.y, item);
    if (collisions.length > 1) return { placed: false, displaced: null, reason: 'blocked' };

    this.remove(item);
    if (collisions.length === 1) {
      const displaced = collisions[0] as Item;
      this.remove(displaced);
      this.#write(item, to.x, to.y);
      // Put the displaced item back if it still fits somewhere; otherwise hand
      // it to the caller, which is the cursor.
      const rehomed = this.add(displaced);
      return { placed: true, displaced: rehomed === null ? displaced : null, reason: 'ok' };
    }
    this.#write(item, to.x, to.y);
    return { placed: true, displaced: null, reason: 'ok' };
  }

  clear(): void {
    this.#placements.length = 0;
    this.#cells.fill(-1);
  }

  toJSON(): InventorySnapshot {
    return {
      width: this.width,
      height: this.height,
      items: this.#placements.map((entry) => ({
        x: entry.x,
        y: entry.y,
        item: cloneItem(entry.item),
      })),
    };
  }

  /**
   * Restore from a snapshot.
   *
   * Placements that no longer fit — a save from a build with a wider grid, or a
   * corrupted file — are re-added by first fit rather than dropped, so a bad
   * save costs the player their layout and never their items.
   */
  load(snapshot: InventorySnapshot): void {
    this.clear();
    const overflow: Item[] = [];
    for (const entry of snapshot.items) {
      const item = cloneItem(entry.item);
      if (!this.place(item, entry.x, entry.y)) overflow.push(item);
    }
    for (const item of overflow) this.add(item);
  }

  /* -- internals ---------------------------------------------------------- */

  #inBounds(item: Item, x: number, y: number): boolean {
    return (
      Number.isInteger(x) &&
      Number.isInteger(y) &&
      x >= 0 &&
      y >= 0 &&
      x + item.width <= this.width &&
      y + item.height <= this.height
    );
  }

  #indexAt(x: number, y: number): number {
    if (x < 0 || y < 0 || x >= this.width || y >= this.height) return -1;
    return this.#cells[y * this.width + x] ?? -1;
  }

  #write(item: Item, x: number, y: number): void {
    const index = this.#placements.length;
    this.#placements.push({ item, x, y });
    for (let dy = 0; dy < item.height; dy++) {
      for (let dx = 0; dx < item.width; dx++) {
        this.#cells[(y + dy) * this.width + (x + dx)] = index;
      }
    }
  }

  #rebuild(): void {
    this.#cells.fill(-1);
    for (let index = 0; index < this.#placements.length; index++) {
      const entry = this.#placements[index];
      if (entry === undefined) continue;
      for (let dy = 0; dy < entry.item.height; dy++) {
        for (let dx = 0; dx < entry.item.width; dx++) {
          this.#cells[(entry.y + dy) * this.width + (entry.x + dx)] = index;
        }
      }
    }
  }
}

/* -------------------------------------------------------------------------- */
/* Equipment                                                                   */
/* -------------------------------------------------------------------------- */

export type EquipmentSlot =
  | 'head'
  | 'body'
  | 'weapon'
  | 'offhand'
  | 'gloves'
  | 'boots'
  | 'belt'
  | 'ring1'
  | 'ring2'
  | 'amulet';

/** Paper-doll order, top to bottom, which the UI lays out in. */
export const EQUIPMENT_SLOTS: readonly EquipmentSlot[] = [
  'head',
  'amulet',
  'body',
  'weapon',
  'offhand',
  'gloves',
  'belt',
  'boots',
  'ring1',
  'ring2',
];

/** What a UI calls each slot. */
export const EQUIPMENT_SLOT_LABELS: Readonly<Record<EquipmentSlot, string>> = {
  head: 'Helm',
  amulet: 'Amulet',
  body: 'Armor',
  weapon: 'Weapon',
  offhand: 'Shield',
  gloves: 'Gloves',
  belt: 'Belt',
  boots: 'Boots',
  ring1: 'Ring',
  ring2: 'Ring',
};

/** Which item slot each equipment slot accepts. */
const SLOT_ACCEPTS: Readonly<Record<EquipmentSlot, ItemSlot>> = {
  head: 'head',
  amulet: 'amulet',
  body: 'body',
  weapon: 'weapon',
  offhand: 'offhand',
  gloves: 'gloves',
  belt: 'belt',
  boots: 'boots',
  ring1: 'ring',
  ring2: 'ring',
};

/** Equipment slots an item of `slot` may go into, in preference order. */
export function slotsFor(slot: ItemSlot): EquipmentSlot[] {
  if (slot === 'ring') return ['ring1', 'ring2'];
  const match = EQUIPMENT_SLOTS.filter((entry) => SLOT_ACCEPTS[entry] === slot);
  return match;
}

export type EquipmentSnapshot = Partial<Record<EquipmentSlot, Item>>;

/** What an equip displaced, so the caller can put it back in the bag. */
export interface EquipResult {
  readonly equipped: boolean;
  /** Items removed to make room. Zero, one, or two (two-hander over both hands). */
  readonly displaced: readonly Item[];
  readonly slot: EquipmentSlot | null;
  readonly reason: 'ok' | 'wrong-slot' | 'occupied';
}

export class Equipment {
  readonly #slots = new Map<EquipmentSlot, Item>();

  get(slot: EquipmentSlot): Item | null {
    return this.#slots.get(slot) ?? null;
  }

  /** Every equipped item, in paper-doll order. */
  items(): Item[] {
    const out: Item[] = [];
    for (const slot of EQUIPMENT_SLOTS) {
      const item = this.#slots.get(slot);
      if (item !== undefined) out.push(item);
    }
    return out;
  }

  /** Whether `item` is mechanically allowed in `slot`, ignoring occupancy. */
  accepts(item: Item, slot: EquipmentSlot): boolean {
    return SLOT_ACCEPTS[slot] === item.slot;
  }

  /**
   * Where this item would go if equipped now.
   *
   * Prefers an empty slot, which is what makes the second ring land on the free
   * finger rather than replacing the first.
   */
  preferredSlot(item: Item): EquipmentSlot | null {
    const candidates = slotsFor(item.slot);
    for (const slot of candidates) if (!this.#slots.has(slot)) return slot;
    return candidates[0] ?? null;
  }

  /**
   * Equip `item`, evicting whatever it has to.
   *
   * The two-handed rule runs in both directions here, in one place, so the two
   * halves cannot drift apart.
   */
  equip(item: Item, slot?: EquipmentSlot): EquipResult {
    const target = slot ?? this.preferredSlot(item);
    if (target === null) return { equipped: false, displaced: [], slot: null, reason: 'wrong-slot' };
    if (!this.accepts(item, target)) {
      return { equipped: false, displaced: [], slot: target, reason: 'wrong-slot' };
    }

    const displaced: Item[] = [];
    const existing = this.#slots.get(target);
    if (existing !== undefined) displaced.push(existing);

    if (target === 'weapon' && item.twoHanded) {
      const offhand = this.#slots.get('offhand');
      if (offhand !== undefined) {
        displaced.push(offhand);
        this.#slots.delete('offhand');
      }
    }
    if (target === 'offhand') {
      const weapon = this.#slots.get('weapon');
      if (weapon !== undefined && weapon.twoHanded) {
        displaced.push(weapon);
        this.#slots.delete('weapon');
      }
    }

    this.#slots.set(target, item);
    return { equipped: true, displaced, slot: target, reason: 'ok' };
  }

  unequip(slot: EquipmentSlot): Item | null {
    const item = this.#slots.get(slot);
    if (item === undefined) return null;
    this.#slots.delete(slot);
    return item;
  }

  /** Which slot holds `item`, or null. */
  slotOf(item: Item): EquipmentSlot | null {
    for (const slot of EQUIPMENT_SLOTS) {
      if (this.#slots.get(slot) === item) return slot;
    }
    return null;
  }

  clear(): void {
    this.#slots.clear();
  }

  /* -- derived readouts ---------------------------------------------------- */

  /** Summed modifiers of everything equipped and unbroken. */
  modifiers(): ModifierTotals {
    return sumItemModifiers(this.items());
  }

  /**
   * The equipped weapon's damage range, or `undefined` for a bare-handed
   * character — which is the signal `deriveStats` reads to fall back to
   * {@link module:rpg/Stats.UNARMED_DAMAGE}.
   */
  weaponDamage(): DamageRange | undefined {
    const weapon = this.#slots.get('weapon');
    if (weapon === undefined || weapon.damage === undefined) return undefined;
    if (isBroken(weapon)) return undefined;
    return weapon.damage;
  }

  /** Summed base defence of armour pieces, before `enhancedDefense`. */
  armourDefense(): number {
    let total = 0;
    for (const item of this.items()) {
      if (isBroken(item)) continue;
      total += item.defense ?? 0;
    }
    return total;
  }

  /** Innate block chance of the equipped shield, `[0, 1]`. */
  blockChance(): number {
    const offhand = this.#slots.get('offhand');
    if (offhand === undefined || isBroken(offhand)) return 0;
    return offhand.block ?? 0;
  }

  toJSON(): EquipmentSnapshot {
    const out: EquipmentSnapshot = {};
    for (const slot of EQUIPMENT_SLOTS) {
      const item = this.#slots.get(slot);
      if (item !== undefined) out[slot] = cloneItem(item);
    }
    return out;
  }

  load(snapshot: EquipmentSnapshot): void {
    this.clear();
    for (const slot of EQUIPMENT_SLOTS) {
      const item = snapshot[slot];
      if (item === undefined) continue;
      const copy = cloneItem(item);
      // Validated on the way in: a hand-edited save that puts a sword on the
      // amulet slot loses the sword rather than corrupting the derivation.
      if (this.accepts(copy, slot)) this.#slots.set(slot, copy);
    }
  }
}

/* -------------------------------------------------------------------------- */
/* Requirements                                                                */
/* -------------------------------------------------------------------------- */

/** What a character must satisfy to wear an item. */
export interface EquipRequirements {
  readonly level: number;
  readonly strength: number;
  readonly dexterity: number;
}

/** Whether the character meets an item's requirements. */
export function meetsRequirements(item: Item, character: EquipRequirements): boolean {
  return (
    character.level >= item.requiredLevel &&
    character.strength >= item.requiredStrength &&
    character.dexterity >= item.requiredDexterity
  );
}

/** The specific requirement that fails first, for the UI's red tooltip line. */
export function unmetRequirement(
  item: Item,
  character: EquipRequirements,
): 'level' | 'strength' | 'dexterity' | null {
  if (character.level < item.requiredLevel) return 'level';
  if (character.strength < item.requiredStrength) return 'strength';
  if (character.dexterity < item.requiredDexterity) return 'dexterity';
  return null;
}

/** An empty modifier bag, for callers with no equipment at all. */
export function emptyModifiers(): ModifierTotals {
  return zeroModifiers();
}
